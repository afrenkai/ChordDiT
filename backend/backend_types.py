from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

from fastapi import Form
from pydantic import BaseModel, ConfigDict, Field

from prototypes import (
    DEFAULT_CONFIGURATION,
    DiTConfiguration,
    EditParameters,
    ModelName,
)


class LocalModelPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str = DEFAULT_CONFIGURATION.model_id
    unet_path: str | None = None
    dit_path: str | None = None
    scheduler_path: str | None = None
    vae_path: str | None = None
    tokenizer_path: str | None = None
    text_encoder_path: str | None = None
    dit_config_path: str | None = None

    @classmethod
    def from_environment(cls) -> LocalModelPaths:
        root_value = os.environ.get("CHORD_MODEL_ROOT")
        root = Path(root_value).expanduser() if root_value is not None else None
        values: dict[str, str | None] = {
            "model_id": os.environ.get("CHORD_MODEL_ID", DEFAULT_CONFIGURATION.model_id),
            "dit_path": os.environ.get("CHORD_DIT_PATH"),
            "dit_config_path": os.environ.get("CHORD_DIT_CONFIG"),
        }
        for component in ("unet", "scheduler", "vae", "tokenizer", "text_encoder"):
            default_path = str(root / component) if root is not None else None
            values[f"{component}_path"] = os.environ.get(
                f"CHORD_{component.upper()}_PATH",
                default_path,
            )
        return cls(**values)

    @property
    def uses_pretrained_model(self) -> bool:
        return all(
            path is None
            for path in (
                self.unet_path,
                self.vae_path,
                self.scheduler_path,
                self.tokenizer_path,
                self.text_encoder_path,
            )
        )

    def component_paths(self) -> dict[str, str]:
        paths = {
            "unet_path": self.unet_path,
            "scheduler_path": self.scheduler_path,
            "vae_path": self.vae_path,
            "tokenizer_path": self.tokenizer_path,
            "text_encoder_path": self.text_encoder_path,
        }
        if any(path is None for path in paths.values()):
            raise RuntimeError("Local model component paths are incomplete")
        components = {name: path for name, path in paths.items() if path is not None}
        if self.dit_path is not None:
            components["dit_path"] = self.dit_path
        return components

    def missing_paths(self, model_name: ModelName) -> list[str]:
        component_paths = {
            "unet_path": self.unet_path,
            "scheduler_path": self.scheduler_path,
            "vae_path": self.vae_path,
            "tokenizer_path": self.tokenizer_path,
            "text_encoder_path": self.text_encoder_path,
        }
        missing = [
            name
            for name, path in component_paths.items()
            if path is not None and not Path(path).is_dir()
        ]
        if not self.uses_pretrained_model:
            missing.extend(
                name for name, path in component_paths.items() if path is None
            )
        if model_name == "dit" and (
            self.dit_path is None or not Path(self.dit_path).is_file()
        ):
            missing.append("dit_path")
        return missing

    def dit_configuration(self, context_dimension: int) -> DiTConfiguration:
        if self.dit_config_path is None:
            return DEFAULT_CONFIGURATION.dit.model_copy(
                update={"context_dimension": context_dimension}
            )
        configuration = json.loads(Path(self.dit_config_path).read_text(encoding="utf-8"))
        return DiTConfiguration.model_validate(configuration)


class LocalInferenceConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True)

    paths: LocalModelPaths
    image_size: int = Field(default=512, ge=64)
    device: str | None = None

    @classmethod
    def from_environment(cls) -> LocalInferenceConfiguration:
        return cls(
            paths=LocalModelPaths.from_environment(),
            image_size=int(os.environ.get("CHORD_IMAGE_SIZE", "512")),
            device=os.environ.get("CHORD_DEVICE"),
        )


class EditForm(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ModelName = "unet"
    source_prompt: str = Field(min_length=1)
    target_prompt: str = Field(min_length=1)
    seed: int = DEFAULT_CONFIGURATION.seed
    noise_samples: int = Field(default=DEFAULT_CONFIGURATION.edit.noise_samples, ge=1, le=16)
    step_scale: float = Field(default=DEFAULT_CONFIGURATION.edit.step_scale, ge=0.1, le=5.0)
    t_start: float = Field(default=DEFAULT_CONFIGURATION.edit.t_start, ge=0.01, le=1.0)
    t_end: float = Field(default=DEFAULT_CONFIGURATION.edit.t_end, ge=0.0, le=0.99)
    t_delta: float = Field(default=DEFAULT_CONFIGURATION.edit.t_delta, ge=0.0, le=0.5)

    @classmethod
    def as_form(
        cls,
        source_prompt: Annotated[str, Form()],
        target_prompt: Annotated[str, Form()],
        model: Annotated[ModelName, Form()] = "unet",
        seed: Annotated[int, Form()] = DEFAULT_CONFIGURATION.seed,
        noise_samples: Annotated[int, Form()] = DEFAULT_CONFIGURATION.edit.noise_samples,
        step_scale: Annotated[float, Form()] = DEFAULT_CONFIGURATION.edit.step_scale,
        t_start: Annotated[float, Form()] = DEFAULT_CONFIGURATION.edit.t_start,
        t_end: Annotated[float, Form()] = DEFAULT_CONFIGURATION.edit.t_end,
        t_delta: Annotated[float, Form()] = DEFAULT_CONFIGURATION.edit.t_delta,
    ) -> EditForm:
        return cls(
            model=model,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            seed=seed,
            noise_samples=noise_samples,
            step_scale=step_scale,
            t_start=t_start,
            t_end=t_end,
            t_delta=t_delta,
        )

    def edit_parameters(self) -> EditParameters:
        return DEFAULT_CONFIGURATION.edit.model_copy(
            update={
                "noise_samples": self.noise_samples,
                "step_scale": self.step_scale,
                "t_start": self.t_start,
                "t_end": self.t_end,
                "t_delta": self.t_delta,
            }
        )


class EditResponse(BaseModel):
    original_image: str
    image: str
    model: ModelName
    source_prompt: str
    target_prompt: str
