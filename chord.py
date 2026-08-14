from __future__ import annotations

from collections.abc import Mapping

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPTextModel

from chord_utils import ImageProcessingMixin
from devices import select_device, select_fallback_device
from dit import DiTwDDTHead
from prototypes import (
    DEFAULT_CONFIGURATION,
    ChordEditPipelineOutput,
    DiTConfiguration,
    ModelName,
    PredictionName,
)
from run_chord import ChordEditMixin
from safety import SafetyCheckerMixin


class ChordEditPipeline(
    ImageProcessingMixin,
    ChordEditMixin,
    SafetyCheckerMixin,
    DiffusionPipeline,
):
    def __init__(
        self,
        unet: UNet2DConditionModel | None,
        scheduler: DDPMScheduler,
        vae: AutoencoderKL,
        tokenizer: AutoTokenizer,
        text_encoder: CLIPTextModel,
        dit: DiTwDDTHead | None = None,
        prediction_type: PredictionName | None = None,
        default_edit_config: Mapping[str, object] | None = None,
        image_size: int = DEFAULT_CONFIGURATION.image_size,
        device: str | torch.device | None = None,
        compute_dtype: torch.dtype = DEFAULT_CONFIGURATION.torch_dtype,
        use_attention_mask: bool = DEFAULT_CONFIGURATION.use_attention_mask,
        use_center_crop: bool = DEFAULT_CONFIGURATION.use_center_crop,
        use_safety_checker: bool = DEFAULT_CONFIGURATION.use_safety_checker,
        safety_checker_id: str | None = DEFAULT_CONFIGURATION.safety_checker_id,
    ) -> None:
        if (unet is None) == (dit is None):
            raise ValueError("Provide exactly one of unet or dit")
        super().__init__()
        self.register_modules(
            unet=unet,
            dit=dit,
            scheduler=scheduler,
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.execution_device = select_device(device)
        self.fallback_device = select_fallback_device(self.execution_device)
        self.compute_dtype = compute_dtype
        self.denoiser = unet if unet is not None else dit
        self.prediction_type = prediction_type or ("epsilon" if unet is not None else "x")
        self.use_attention_mask = use_attention_mask
        self.default_edit_config: Mapping[str, object] = (
            default_edit_config
            if default_edit_config is not None
            else DEFAULT_CONFIGURATION.edit.model_dump()
        )
        self.image_size = image_size
        self.use_center_crop = use_center_crop
        self.max_denoiser_timestep = scheduler.config.num_train_timesteps - 1
        self.safety_checker_enabled = use_safety_checker
        self.safety_checker_id = safety_checker_id
        self.safety_checker: StableDiffusionSafetyChecker | None = None
        self.safety_feature_extractor: CLIPImageProcessor | None = None

        self.to(self.execution_device)
        self.set_compute_precision()
        self.vae_transform = self.build_vae_transform()
        self.denoiser.eval()
        self.vae.eval()
        self.text_encoder.eval()
        if use_safety_checker:
            self.initialize_safety_checker()

    @classmethod
    def from_local_weights(
        cls,
        component_paths: Mapping[str, str],
        *,
        default_edit_config: Mapping[str, object] | None = None,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
        model_type: ModelName = "unet",
        dit_config: DiTConfiguration | None = None,
        image_size: int = DEFAULT_CONFIGURATION.image_size,
        use_center_crop: bool = DEFAULT_CONFIGURATION.use_center_crop,
        compute_dtype: torch.dtype = DEFAULT_CONFIGURATION.torch_dtype,
        use_attention_mask: bool = DEFAULT_CONFIGURATION.use_attention_mask,
        use_safety_checker: bool = DEFAULT_CONFIGURATION.use_safety_checker,
        safety_checker_id: str | None = DEFAULT_CONFIGURATION.safety_checker_id,
    ) -> ChordEditPipeline:
        scheduler = DDPMScheduler.from_pretrained(component_paths["scheduler_path"])
        text_encoder = CLIPTextModel.from_pretrained(
            component_paths["text_encoder_path"],
            torch_dtype=torch_dtype,
        )
        if model_type == "dit":
            configuration = dit_config or DEFAULT_CONFIGURATION.dit.model_copy(
                update={
                    "context_dimension": text_encoder.config.hidden_size,
                    "max_timestep": scheduler.config.num_train_timesteps,
                }
            )
            dit = DiTwDDTHead(configuration)
            checkpoint = torch.load(
                component_paths["dit_path"],
                map_location="cpu",
                weights_only=True,
            )
            state_dict = checkpoint.get("model", checkpoint)
            dit.load_state_dict(state_dict, strict=True)
            unet = None
            prediction_type = "x"
        else:
            unet = UNet2DConditionModel.from_pretrained(
                component_paths["unet_path"],
                torch_dtype=torch_dtype,
            )
            dit = None
            prediction_type = "epsilon"
        return cls(
            unet=unet,
            dit=dit,
            prediction_type=prediction_type,
            scheduler=scheduler,
            vae=AutoencoderKL.from_pretrained(
                component_paths["vae_path"],
                torch_dtype=torch_dtype,
            ),
            tokenizer=AutoTokenizer.from_pretrained(component_paths["tokenizer_path"]),
            text_encoder=text_encoder,
            default_edit_config=default_edit_config,
            image_size=image_size,
            device=device,
            compute_dtype=compute_dtype,
            use_attention_mask=use_attention_mask,
            use_center_crop=use_center_crop,
            use_safety_checker=use_safety_checker,
            safety_checker_id=safety_checker_id,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        model_type: ModelName = "unet",
        dit: DiTwDDTHead | None = None,
        dit_config: DiTConfiguration | None = None,
        default_edit_config: Mapping[str, object] | None = None,
        device: str | torch.device | None = None,
        torch_dtype: torch.dtype = torch.float32,
        image_size: int = DEFAULT_CONFIGURATION.image_size,
        use_center_crop: bool = DEFAULT_CONFIGURATION.use_center_crop,
        compute_dtype: torch.dtype = DEFAULT_CONFIGURATION.torch_dtype,
        use_attention_mask: bool = False,
        use_safety_checker: bool = DEFAULT_CONFIGURATION.use_safety_checker,
        safety_checker_id: str | None = DEFAULT_CONFIGURATION.safety_checker_id,
    ) -> ChordEditPipeline:
        scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        )
        if model_type == "dit":
            if dit is None:
                if dit_config is None:
                    dit_config = DEFAULT_CONFIGURATION.dit.model_copy(
                        update={
                            "context_dimension": text_encoder.config.hidden_size,
                            "input_size": image_size // 8,
                        }
                    )
                dit = DiTwDDTHead(dit_config)
            unet = None
            prediction_type = "x"
        else:
            unet = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                torch_dtype=torch_dtype,
            )
            prediction_type = "epsilon"
        return cls(
            unet=unet,
            dit=dit,
            prediction_type=prediction_type,
            scheduler=scheduler,
            vae=AutoencoderKL.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch_dtype,
            ),
            tokenizer=AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer"),
            text_encoder=text_encoder,
            default_edit_config=default_edit_config,
            image_size=image_size,
            device=device,
            compute_dtype=compute_dtype,
            use_center_crop=use_center_crop,
            use_attention_mask=use_attention_mask,
            use_safety_checker=use_safety_checker,
            safety_checker_id=safety_checker_id,
        )

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image | torch.Tensor,
        *,
        source_prompt: str,
        target_prompt: str,
        edit_config: Mapping[str, object] | None = None,
        seed: int | None = None,
        output_type: str = "pil",
    ) -> ChordEditPipelineOutput:
        if output_type not in {"pil", "tensor"}:
            raise ValueError("output_type must be 'pil' or 'tensor'")

        config = {**self.default_edit_config, **(edit_config or {})}
        parameters = self.prepare_edit_params(config)
        image_tensor = self.prepare_image_tensor(image)
        source_latents = self.encode_image_to_latent(image_tensor)
        source_embedding = self.encode_prompt([source_prompt])
        target_embedding = self.encode_prompt([target_prompt])
        noise = self.prepare_noise_list(
            source_latents,
            DEFAULT_CONFIGURATION.seed if seed is None else seed,
            parameters.noise_samples,
        )
        edited_latents = self.run_edit(
            source_latents,
            source_embedding,
            target_embedding,
            noise,
            parameters,
        )
        images = self.apply_safety_checker(self.decode_latent_to_image(edited_latents))[0].cpu()
        output_images = self.tensor_to_pil(images) if output_type == "pil" else images
        return ChordEditPipelineOutput(
            images=output_images,
            latents=edited_latents.detach().cpu(),
        )

    def set_compute_precision(self) -> None:
        for module in (self.denoiser, self.vae, self.text_encoder):
            module.to(device=self.execution_device, dtype=self.compute_dtype)
