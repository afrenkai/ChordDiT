from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from transforms import CenterSquareCropTransform


class ImageProcessingMixin:
    def build_vae_transform(self: ImageProcessingMixin) -> transforms.Compose:
        operations: list[object] = []
        if self.use_center_crop:
            operations.append(CenterSquareCropTransform())
            interpolation = InterpolationMode.LANCZOS
        else:
            interpolation = InterpolationMode.BILINEAR
        operations.extend(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=interpolation,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        return transforms.Compose(operations)

    def prepare_image_tensor(
        self: ImageProcessingMixin,
        image: Image.Image | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image_tensor = self.vae_transform(image)
        elif torch.is_tensor(image):
            if image.ndim not in (3, 4) or image.numel() == 0:
                raise ValueError("image tensor must be a non-empty CHW or BCHW tensor")
            image_tensor = image.float().to(self.fallback_device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.max() > 1:
                image_tensor = image_tensor / 255
            image_tensor = image_tensor * 2 - 1
        else:
            raise TypeError("image must be a PIL.Image or a torch.Tensor")

        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if self.use_center_crop and image_tensor.shape[-2] != image_tensor.shape[-1]:
            side = min(image_tensor.shape[-2:])
            top = (image_tensor.shape[-2] - side) // 2
            left = (image_tensor.shape[-1] - side) // 2
            image_tensor = image_tensor[..., top : top + side, left : left + side]
        return image_tensor.to(device=self.execution_device, dtype=self.compute_dtype)

    def encode_image_to_latent(
        self: ImageProcessingMixin,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.vae.encode(
                pixel_values.to(device=self.execution_device, dtype=self.compute_dtype)
            ).latent_dist.mode()
            * self.vae.config.scaling_factor
        )

    def decode_latent_to_image(
        self: ImageProcessingMixin,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.vae.decode(
                latents.to(device=self.execution_device, dtype=self.compute_dtype)
                / self.vae.config.scaling_factor
            )
            .sample.clamp(-1, 1)
            .add(1)
            .div(2)
        )

    def encode_prompt(
        self: ImageProcessingMixin,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        tokenized = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        attention_mask = tokenized.attention_mask if self.use_attention_mask else None
        hidden_states = self.text_encoder(
            input_ids=tokenized.input_ids.to(self.execution_device),
            attention_mask=(
                attention_mask.to(self.execution_device)
                if attention_mask is not None
                else None
            ),
        ).last_hidden_state
        return hidden_states.to(device=self.execution_device, dtype=self.compute_dtype)

    def tensor_to_pil(
        self: ImageProcessingMixin,
        tensor: torch.Tensor,
    ) -> list[Image.Image]:
        to_pil = transforms.ToPILImage()
        return [to_pil(sample) for sample in tensor.detach().cpu().clamp(0, 1)]
