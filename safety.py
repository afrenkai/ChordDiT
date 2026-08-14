from __future__ import annotations

import logging

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

LOGGER = logging.getLogger(__name__)


class SafetyCheckerMixin:
    def initialize_safety_checker(self: SafetyCheckerMixin) -> None:
        if not self.safety_checker_id:
            LOGGER.warning("Safety checker requested without an identifier; disabling it")
            self.safety_checker_enabled = False
            return
        try:
            self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                self.safety_checker_id,
                torch_dtype=self.compute_dtype,
            ).to(self.fallback_device)
            self.safety_feature_extractor = CLIPImageProcessor.from_pretrained(
                self.safety_checker_id
            )
        except (OSError, RuntimeError, ValueError) as error:
            LOGGER.warning("Could not initialize safety checker: %s", error)
            self.safety_checker = None
            self.safety_feature_extractor = None
            self.safety_checker_enabled = False

    def apply_safety_checker(
        self: SafetyCheckerMixin,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[bool]]:
        batch_size = images.shape[0]
        if (
            not self.safety_checker_enabled
            or self.safety_checker is None
            or self.safety_feature_extractor is None
        ):
            return images, [False] * batch_size

        pil_images = self.tensor_to_pil(images)
        try:
            clip_input = self.safety_feature_extractor(images=pil_images, return_tensors="pt")
            image_array = np.stack(
                [np.asarray(image, dtype=np.float32) / 255 for image in pil_images]
            )
            flags = self.safety_checker(
                clip_input=clip_input.pixel_values.to(self.fallback_device),
                images=image_array,
            )[1]
        except (RuntimeError, ValueError) as error:
            LOGGER.warning("Safety checker failed: %s", error)
            return images, [False] * batch_size

        has_nsfw = flags.tolist() if torch.is_tensor(flags) else [bool(flag) for flag in flags]
        safe_images = images.clone()
        for image_index, flagged in enumerate(has_nsfw):
            if flagged:
                safe_images[image_index].zero_()
        return safe_images, has_nsfw
