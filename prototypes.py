from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch import Tensor

ModelName = Literal["unet", "dit"]
PredictionName = Literal["epsilon", "x"]
ComputeDtypeName = Literal["float32", "float16", "bfloat16"]


class DiTConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_size: int
    in_channels: int
    patch_size: tuple[int, int]
    hidden_size: tuple[int, int]
    depth: tuple[int, int]
    num_heads: tuple[int, int]
    mlp_ratio: float
    time_tokens: int
    context_tokens: int
    context_dimension: int
    time_embedding_size: int
    max_timestep: int


class EditParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    noise_samples: int
    n_steps: int
    t_start: float
    t_end: float
    t_delta: float
    step_scale: float
    cleanup: bool


class ModelConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    seed: int
    image_size: int
    compute_dtype: ComputeDtypeName
    use_attention_mask: bool
    use_center_crop: bool
    use_safety_checker: bool
    safety_checker_id: str | None
    edit: EditParameters
    dit: DiTConfiguration

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.compute_dtype]


class DatasetSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    id: str
    original_image: Image.Image
    edited_image: Image.Image
    original_prompt: str
    edited_prompt: str
    edit_prompt: str
    image_path: str


class EditRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_path: Path
    source_prompt: str
    target_prompt: str
    edit_prompt: str
    edit_id: str | None = None


class ChordEditPipelineOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: list[Image.Image] | Tensor
    latents: Tensor


def load_configuration(path: str | Path | None = None, overrides: list[str] | None = None) -> ModelConfiguration:
    config_directory = Path(__file__).with_name("conf")
    config_name = "config"
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        config_directory = config_path.parent
        config_name = config_path.stem
    with initialize_config_dir(version_base=None, config_dir=str(config_directory)):
        composed: DictConfig = compose(config_name=config_name, overrides=overrides or [])
    values = OmegaConf.to_container(composed, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("Hydra configuration must contain a mapping")
    return ModelConfiguration.model_validate(values)


DEFAULT_CONFIGURATION = load_configuration()
