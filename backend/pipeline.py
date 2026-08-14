from __future__ import annotations

from threading import Lock

import torch
from PIL import Image

from backend.backend_types import EditForm, LocalInferenceConfiguration
from chord import ChordEditPipeline
from dit import DiTwDDTHead
from prototypes import DiTConfiguration, ModelName


class PipelineStore:
    def __init__(self) -> None:
        self.configuration = LocalInferenceConfiguration.from_environment()
        self.pipelines: dict[ModelName, ChordEditPipeline] = {}
        self.lock = Lock()

    def load(self, model_name: ModelName) -> ChordEditPipeline:
        with self.lock:
            pipeline = self.pipelines.get(model_name)
            if pipeline is None:
                pipeline = self.build_pipeline(model_name)
                self.pipelines[model_name] = pipeline
            return pipeline

    def build_pipeline(self, model_name: ModelName) -> ChordEditPipeline:
        paths = self.configuration.paths
        missing = paths.missing_paths(model_name)
        if missing:
            raise RuntimeError(
                f"Missing local model directories for {model_name}: {', '.join(missing)}. "
                "Set CHORD_MODEL_ROOT or the CHORD_*_PATH variables."
            )

        dit: DiTwDDTHead | None = None
        dit_config: DiTConfiguration | None = None
        if model_name == "dit":
            if paths.dit_path is None:
                raise RuntimeError("CHORD_DIT_PATH is required for the DiT backend")
            dit_config = paths.dit_configuration(768)
            if paths.dit_config_path is None:
                dit_config = dit_config.model_copy(
                    update={"input_size": self.configuration.image_size // 8}
                )
            dit = DiTwDDTHead(dit_config)
            checkpoint = torch.load(paths.dit_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model", checkpoint)
            dit.load_state_dict(state_dict, strict=True)

        options = {
            "model_type": model_name,
            "dit": dit,
            "dit_config": dit_config,
            "image_size": self.configuration.image_size,
            "device": self.configuration.device,
            "torch_dtype": torch.float32,
            "compute_dtype": torch.float32,
        }
        if paths.uses_pretrained_model:
            return ChordEditPipeline.from_pretrained(paths.model_id, **options)
        return ChordEditPipeline.from_local_weights(paths.component_paths(), **options)

    def edit(self, image: Image.Image, form: EditForm) -> Image.Image:
        result = self.load(form.model)(
            image,
            source_prompt=form.source_prompt.strip(),
            target_prompt=form.target_prompt.strip(),
            edit_config=form.edit_parameters().model_dump(),
            seed=form.seed,
            output_type="pil",
        )
        output = result.images[0]
        if not isinstance(output, Image.Image):
            raise TypeError("Pipeline returned a tensor instead of an image")
        return output
