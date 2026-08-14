from __future__ import annotations

from collections.abc import Mapping

import torch

from prototypes import EditParameters


class ChordEditMixin:
    def prepare_edit_params(
        self: ChordEditMixin,
        config: Mapping[str, object],
    ) -> EditParameters:
        required = ("noise_samples", "n_steps", "t_start", "t_end", "t_delta", "step_scale")
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"edit_config is missing required keys: {missing}")

        start = max(0.0, min(1.0, float(config["t_start"])))
        delta = max(0.0, min(1.0, float(config["t_delta"])))
        if delta >= start:
            delta = max(0.0, start - 1 / max(1, self.max_denoiser_timestep))
        return EditParameters(
            noise_samples=max(1, int(config["noise_samples"])),
            n_steps=max(1, int(config["n_steps"])),
            t_start=start,
            t_end=max(0.0, min(start, float(config["t_end"]))),
            t_delta=delta,
            step_scale=float(config["step_scale"]),
            cleanup=bool(config.get("cleanup", False)),
        )

    def prepare_noise_list(
        self: ChordEditMixin,
        latents: torch.Tensor,
        seed: int,
        sample_count: int,
    ) -> list[torch.Tensor]:
        generator = torch.Generator(device=self.fallback_device).manual_seed(seed)
        return [
            torch.randn(
                latents.shape,
                device=self.fallback_device,
                dtype=self.compute_dtype,
                generator=generator,
            ).to(latents.device)
            for sample_index in range(sample_count)
        ]

    def time_to_index(
        self: ChordEditMixin,
        batch_size: int,
        time: float,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        timestep = round(self.max_denoiser_timestep * time)
        timestep = max(0, min(self.max_denoiser_timestep, timestep))
        return torch.full(
            (batch_size,), timestep, device=self.fallback_device, dtype=dtype
        ).to(device)

    def get_alpha_sigma(
        self: ChordEditMixin,
        tensor: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alphas_cumprod = self.scheduler.alphas_cumprod.to(
            device=self.fallback_device,
            dtype=torch.float32,
        )
        fallback_timesteps = timesteps.to(self.fallback_device)
        alpha = alphas_cumprod[fallback_timesteps].sqrt().view(-1, 1, 1, 1)
        sigma = (1 - alphas_cumprod[fallback_timesteps]).sqrt().view(-1, 1, 1, 1)
        alpha = alpha.to(device=tensor.device, dtype=tensor.dtype)
        sigma = sigma.to(device=tensor.device, dtype=tensor.dtype)
        return alpha.clamp_min(torch.finfo(alpha.dtype).eps), sigma

    def predict_x0(
        self: ChordEditMixin,
        anchor: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha, sigma = self.get_alpha_sigma(anchor, timesteps)
        noisy_sample = alpha * anchor + sigma * noise
        prediction = self.denoiser(
            sample=noisy_sample,
            timestep=timesteps,
            encoder_hidden_states=condition,
            return_dict=False,
        )[0]
        return self.prediction_to_clean(noisy_sample, alpha, sigma, prediction)

    def prediction_to_clean(
        self: ChordEditMixin,
        noisy_sample: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        if self.prediction_type == "x":
            return prediction
        return (noisy_sample - sigma * prediction) / alpha

    def estimate_update(
        self: ChordEditMixin,
        anchor: torch.Tensor,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        noise: list[torch.Tensor] | torch.Tensor,
        time: float,
        delta: float,
    ) -> torch.Tensor:
        batch_size = anchor.shape[0]
        time_index = self.time_to_index(batch_size, time, device=anchor.device)
        previous_time_index = self.time_to_index(
            batch_size,
            max(0.0, time - delta),
            device=anchor.device,
        )
        noise_values = [noise] if torch.is_tensor(noise) else noise
        if not noise_values:
            raise ValueError("noise must contain at least one sample")

        alpha, sigma = self.get_alpha_sigma(anchor, time_index)
        previous_alpha, previous_sigma = self.get_alpha_sigma(anchor, previous_time_index)
        sample_count = len(noise_values)
        noise_stack = torch.stack(noise_values)
        anchor_batch = anchor.unsqueeze(0).expand(sample_count, -1, -1, -1, -1)
        noisy_sample = alpha.unsqueeze(0) * anchor_batch + sigma.unsqueeze(0) * noise_stack
        previous_sample = (
            previous_alpha.unsqueeze(0) * anchor_batch
            + previous_sigma.unsqueeze(0) * noise_stack
        )
        samples = torch.stack(
            [noisy_sample, noisy_sample, previous_sample, previous_sample],
            dim=1,
        ).reshape(sample_count * 4 * batch_size, *anchor.shape[1:])
        conditions = torch.cat(
            [source_embedding, target_embedding, source_embedding, target_embedding],
            dim=0,
        ).repeat(sample_count, 1, 1)
        timesteps = torch.cat(
            [time_index, time_index, previous_time_index, previous_time_index],
            dim=0,
        ).repeat(sample_count)
        alpha_batch = torch.stack(
            [alpha, alpha, previous_alpha, previous_alpha],
            dim=1,
        ).reshape(sample_count * 4 * batch_size, 1, 1, 1)
        sigma_batch = torch.stack(
            [sigma, sigma, previous_sigma, previous_sigma],
            dim=1,
        ).reshape(sample_count * 4 * batch_size, 1, 1, 1)

        prediction = self.denoiser(
            sample=samples,
            timestep=timesteps,
            encoder_hidden_states=conditions,
            return_dict=False,
        )[0]
        predicted_x0 = self.prediction_to_clean(samples, alpha_batch, sigma_batch, prediction)
        predicted_x0 = predicted_x0.reshape(sample_count, 4, batch_size, *anchor.shape[1:])
        source_at_time, target_at_time, source_previous, target_previous = predicted_x0.unbind(dim=1)
        source_update = (target_at_time - source_at_time).mean(dim=0)
        previous_update = (target_previous - source_previous).mean(dim=0)
        denominator = time + delta
        if denominator <= 1e-6:
            return source_update
        return (delta * source_update + time * previous_update) / denominator

    def run_edit(
        self: ChordEditMixin,
        source: torch.Tensor,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        noise: list[torch.Tensor],
        parameters: EditParameters,
    ) -> torch.Tensor:
        if parameters.n_steps == 1:
            times = [parameters.t_start]
        else:
            times = torch.linspace(
                parameters.t_start,
                parameters.t_end,
                parameters.n_steps,
                device=self.fallback_device,
            ).to(source.device).tolist()

        edited = source
        for time in times:
            edited = edited + parameters.step_scale * self.estimate_update(
                edited,
                source_embedding,
                target_embedding,
                noise,
                float(time),
                parameters.t_delta,
            )
        if parameters.cleanup:
            timestep = self.time_to_index(
                source.shape[0],
                parameters.t_end,
                device=source.device,
            )
            edited = self.predict_x0(edited, timestep, target_embedding, noise[0])
        return edited
