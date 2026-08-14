from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from prototypes import DiTConfiguration


class PatchEmbed(nn.Module):
    def __init__(self, input_size: int, patch_size: int, channels: int, dimension: int) -> None:
        super().__init__()
        if input_size % patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(channels, dimension, patch_size, patch_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.proj(image).flatten(2).transpose(1, 2)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.reshape(*values.shape[:-1], -1, 2).unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(start_dim=-2)


class RoPE(nn.Module):
    def __init__(self, dimension: int, visual_tokens: int, condition_tokens: int = 0) -> None:
        super().__init__()
        if dimension % 4:
            raise ValueError("RoPE dimension must be divisible by four")
        side = math.isqrt(visual_tokens)
        if side * side != visual_tokens:
            raise ValueError("visual token count must be a square")

        frequency_dimension = dimension // 2
        frequencies = 1 / (
            10000
            ** (torch.arange(0, frequency_dimension, 2).float() / frequency_dimension)
        )
        positions = torch.arange(side).float()
        horizontal, vertical = torch.meshgrid(positions, positions, indexing="ij")
        visual_angles = torch.cat(
            [
                horizontal.flatten()[:, None] * frequencies,
                vertical.flatten()[:, None] * frequencies,
            ],
            dim=-1,
        )
        angles = torch.cat(
            [visual_angles, torch.zeros(condition_tokens, frequency_dimension)],
            dim=0,
        ).repeat_interleave(2, dim=-1)
        self.register_buffer("freqs_cos", angles.cos())
        self.register_buffer("freqs_sin", angles.sin())

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.freqs_cos + rotate_half(values) * self.freqs_sin


class SwiGLUFFN(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dimension, hidden_dimension)
        self.w2 = nn.Linear(dimension, hidden_dimension)
        self.w3 = nn.Linear(hidden_dimension, dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.w3(functional.silu(self.w1(values)) * self.w2(values))


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.eps = epsilon
        self.weight = nn.Parameter(torch.ones(dimension))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalize(values).to(values.dtype) * self.weight


class NormAttention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("dimension must be divisible by heads")
        self.num_heads = heads
        self.dimension = dimension
        self.head_dimension = dimension // heads
        self.q = nn.Linear(dimension, dimension)
        self.k = nn.Linear(dimension, dimension)
        self.v = nn.Linear(dimension, dimension)
        self.proj = nn.Linear(dimension, dimension)
        self.q_norm = RMSNorm(self.head_dimension)
        self.k_norm = RMSNorm(self.head_dimension)

    def forward(
        self,
        values: torch.Tensor,
        rope: RoPE,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, token_count, _ = values.shape
        query = self.q(values).reshape(
            batch_size, token_count, self.num_heads, self.head_dimension
        ).transpose(1, 2)
        key = self.k(values).reshape(
            batch_size, token_count, self.num_heads, self.head_dimension
        ).transpose(1, 2)
        value = self.v(values).reshape(
            batch_size, token_count, self.num_heads, self.head_dimension
        ).transpose(1, 2)
        query, key = rope(self.q_norm(query)), rope(self.k_norm(key))
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
        )
        return self.proj(attended.transpose(1, 2).reshape(batch_size, token_count, self.dimension))


class GaussianFourierEmbedding(nn.Module):
    def __init__(
        self,
        dimension: int,
        token_count: int,
        embedding_size: int,
    ) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.learnable_tokens = nn.Parameter(
            torch.randn(token_count, dimension) / dimension**0.5
        )

    def forward(
        self,
        timesteps: torch.Tensor,
        return_base_embed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        angles = timesteps[:, None] * self.W[None, :] * 2 * math.pi
        embedding = self.mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))
        if return_base_embed:
            base_embedding = embedding.unsqueeze(1)
            return base_embedding, self.learnable_tokens + base_embedding
        return self.learnable_tokens + embedding.unsqueeze(1)


class ConditionEmbedder(nn.Module):
    def __init__(self, dimension: int, context_dimension: int) -> None:
        super().__init__()
        self.norm = RMSNorm(context_dimension)
        self.proj = nn.Linear(context_dimension, dimension)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(context))


def modulate(
    values: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return values * (1 + scale) + shift


class DDTEncoderBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dimension)
        self.norm2 = RMSNorm(dimension)
        self.attn = NormAttention(dimension, heads)
        self.mlp = SwiGLUFFN(dimension, int(2 / 3 * dimension * mlp_ratio))

    def forward(
        self,
        values: torch.Tensor,
        rope: RoPE,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = values + self.attn(self.norm1(values), rope, attention_mask)
        return values + self.mlp(self.norm2(values))


class DDTDecoderBlock(DDTEncoderBlock):
    def __init__(self, dimension: int, heads: int, mlp_ratio: float) -> None:
        super().__init__(dimension, heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 6 * dimension))

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        rope: RoPE,
    ) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_modulation(condition).chunk(6, dim=-1)
        )
        values = values + gate_attention * self.attn(
            modulate(self.norm1(values), shift_attention, scale_attention), rope
        )
        return values + gate_mlp * self.mlp(
            modulate(self.norm2(values), shift_mlp, scale_mlp)
        )


class DDTFinalLayer(nn.Module):
    def __init__(self, dimension: int, patch_size: int, channels: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dimension)
        self.linear = nn.Linear(dimension, patch_size * patch_size * channels)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dimension, 2 * dimension))

    def forward(self, values: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(condition).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(values), shift, scale))


class DiTwDDTHead(nn.Module):
    def __init__(self, configuration: DiTConfiguration) -> None:
        super().__init__()
        self.configuration = configuration
        self.in_channels = configuration.in_channels
        self.enc_hidden_size, self.dec_hidden_size = configuration.hidden_size
        self.num_enc_blocks, self.num_dec_blocks = configuration.depth
        self.s_patch_size, self.x_patch_size = configuration.patch_size
        enc_heads, dec_heads = configuration.num_heads
        self.num_cond_tokens = configuration.time_tokens + configuration.context_tokens

        self.s_embedder = PatchEmbed(
            configuration.input_size,
            self.s_patch_size,
            self.in_channels,
            self.enc_hidden_size,
        )
        self.x_embedder = PatchEmbed(
            configuration.input_size,
            self.x_patch_size,
            self.in_channels,
            self.dec_hidden_size,
        )
        self.s_projector = (
            nn.Linear(self.enc_hidden_size, self.dec_hidden_size)
            if self.enc_hidden_size != self.dec_hidden_size
            else nn.Identity()
        )
        self.t_embedder = GaussianFourierEmbedding(
            self.enc_hidden_size,
            configuration.time_tokens,
            configuration.time_embedding_size,
        )
        self.ctx_embedder = ConditionEmbedder(
            self.enc_hidden_size,
            configuration.context_dimension,
        )
        self.blocks = nn.ModuleList(
            [
                *(
                    DDTEncoderBlock(
                        self.enc_hidden_size,
                        enc_heads,
                        configuration.mlp_ratio,
                    )
                    for block_index in range(self.num_enc_blocks)
                ),
                *(
                    DDTDecoderBlock(
                        self.dec_hidden_size,
                        dec_heads,
                        configuration.mlp_ratio,
                    )
                    for block_index in range(self.num_dec_blocks)
                ),
            ]
        )
        self.final_layer = DDTFinalLayer(
            self.dec_hidden_size,
            self.x_patch_size,
            self.in_channels,
        )
        self.enc_rope = RoPE(
            self.enc_hidden_size // enc_heads,
            self.s_embedder.num_patches,
            self.num_cond_tokens,
        )
        self.dec_rope = RoPE(
            self.dec_hidden_size // dec_heads,
            self.x_embedder.num_patches,
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for embedder in (self.s_embedder, self.x_embedder):
            nn.init.xavier_uniform_(embedder.proj.weight.flatten(1))
            nn.init.zeros_(embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks[self.num_enc_blocks :]:
            nn.init.zeros_(block.adaln_modulation[-1].weight)
            nn.init.zeros_(block.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaln_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def build_sequence(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_tokens = self.s_embedder(latents)
        base_time, time_tokens = self.t_embedder(timesteps, return_base_embed=True)
        context_tokens = self.ctx_embedder(context)
        return torch.cat([source_tokens, time_tokens, context_tokens], dim=1), base_time

    def build_attention_mask(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            sequence.shape[0], 1, 1, sequence.shape[1], device=sequence.device, dtype=sequence.dtype
        )

    def unpatchify(self, tokens: torch.Tensor, patch_size: int) -> torch.Tensor:
        side = math.isqrt(tokens.shape[1])
        return tokens.reshape(
            tokens.shape[0], side, side, patch_size, patch_size, self.in_channels
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            tokens.shape[0], self.in_channels, side * patch_size, side * patch_size
        )

    def forward_model(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        sequence, base_time = self.build_sequence(latents, timesteps, context)
        attention_mask = self.build_attention_mask(sequence)
        for block in self.blocks[: self.num_enc_blocks]:
            sequence = block(sequence, self.enc_rope, attention_mask)
        condition = self.s_projector(
            functional.silu(base_time + sequence[:, : self.s_embedder.num_patches])
        )
        target_tokens = self.x_embedder(latents)
        for block in self.blocks[self.num_enc_blocks :]:
            target_tokens = block(target_tokens, condition, self.dec_rope)
        return self.unpatchify(
            self.final_layer(target_tokens, condition),
            self.x_patch_size,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor] | dict[str, torch.Tensor]:
        prediction = self.forward_model(
            sample,
            timestep.float() / self.configuration.max_timestep,
            encoder_hidden_states,
        )
        return {"sample": prediction} if return_dict else (prediction,)
