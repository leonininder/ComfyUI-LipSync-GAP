# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Tuple, Union
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import check_ffmpeg_installed
# from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        # Cache expensive per-resolution utilities so we do not recreate them every call
        self._mask_cache = {}
        self._image_processor_cache = {}
        self._mask_generation_tag = "mouth_mask_v3"
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._target_audio_embed_dim = getattr(self.unet.config, "cross_attention_dim", None)

        # Handle CompVis/ComfyUI VAEs which don't have .config
        if not hasattr(self.vae, "config"):
            # Emulate SD1.5 VAE config
            self.vae.config = FrozenDict({
                "block_out_channels": [128, 256, 512, 512],
                "scaling_factor": 0.18215,
                "shift_factor": 0.0
            })

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _match_audio_embedding_dim(self, audio_embeds: torch.Tensor) -> torch.Tensor:
        if self._target_audio_embed_dim is None:
            return audio_embeds

        current_dim = audio_embeds.shape[-1]
        target_dim = self._target_audio_embed_dim

        if current_dim == target_dim:
            return audio_embeds

        if current_dim > target_dim:
            leading_shape = audio_embeds.shape[:-1]
            embeds = audio_embeds.reshape(-1, 1, current_dim)
            orig_dtype = embeds.dtype
            embeds = embeds.to(torch.float32)
            embeds = F.interpolate(embeds, size=target_dim, mode="linear", align_corners=False)
            embeds = embeds.to(orig_dtype).reshape(*leading_shape, target_dim)
            return embeds

        pad_size = target_dim - current_dim
        pad_shape = list(audio_embeds.shape[:-1]) + [pad_size]
        pad_tensor = torch.zeros(*pad_shape, device=audio_embeds.device, dtype=audio_embeds.dtype)
        return torch.cat([audio_embeds, pad_tensor], dim=-1)

    def decode_latents(self, latents):
        # We must align with whatever VAE we have.
        # If we have a Comfy/LDM VAE (first_stage_model), it expects UNSCALED latents.
        # The pipeline produces SCALED latents (Variance 0.18) from the UNet.
        # So we MUST unscale them (Divide by 0.18) to match the VAE's expectation.
        
        scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        
        latents = latents / scaling_factor + shift_factor

        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        
        # Decoding
        decoded = self.vae.decode(latents)
        if hasattr(decoded, "sample"):
            decoded_latents = decoded.sample
        else:
            decoded_latents = decoded

        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        
        # Generating random noise in FP8 is not supported, so fallback if necessary
        try:
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        except (NotImplementedError, RuntimeError):
            # Fallback to float32 for generation, then cast
            latents = torch.randn(shape, generator=generator, device=rand_device, dtype=torch.float32).to(device)
            latents = latents.to(dtype)
            
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        try:
             latents = latents * self.scheduler.init_noise_sigma
        except (NotImplementedError, RuntimeError):
             # mul_cuda not implemented for Float8_e4m3fn
             orig_dtype = latents.dtype
             latents = latents.float() * self.scheduler.init_noise_sigma
             latents = latents.to(orig_dtype)
             
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        def _ensure_nchw(tensor: torch.Tensor, name: str) -> torch.Tensor:
            """Collapse leading dimensions so tensor becomes (N, C, H, W)."""
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.dim() == 3:
                tensor = tensor.unsqueeze(1)

            if tensor.dim() < 4:
                raise ValueError(f"{name} must be at least 4D (N, C, H, W); received shape {tuple(tensor.shape)}")

            if tensor.dim() > 4:
                tensor = tensor.flatten(0, tensor.dim() - 4)

            return tensor

        mask = _ensure_nchw(mask, "mask")
        masked_image = _ensure_nchw(masked_image, "masked_image")

        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask,
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="nearest",
        )

        if masked_image.shape[0] != mask.shape[0]:
            mask_frames = mask.shape[0]
            image_frames = masked_image.shape[0]
            if image_frames % mask_frames == 0:
                fold = image_frames // mask_frames
                masked_image = masked_image.reshape(
                    mask_frames,
                    fold,
                    masked_image.shape[1],
                    masked_image.shape[-2],
                    masked_image.shape[-1],
                )

                if fold > 1:
                    ref = masked_image[:, :1]
                    delta = masked_image[:, 1:] - ref
                    if torch.max(delta.abs()).item() < 1e-4:
                        masked_image = ref.squeeze(1)
                    else:
                        masked_image = masked_image.mean(dim=1)
                else:
                    masked_image = masked_image.squeeze(1)
            else:
                raise ValueError(
                    f"Masked image batch mismatch: mask has {mask_frames} frames, masked image has {image_frames}"
                )


        # Determine strict VAE input dtype. If pipeline is running in FP8,
        # we still need to provide standard precision (BF16/FP16/FP32) to the VAE.
        vae_input_dtype = dtype
        is_fp8 = dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        if is_fp8:
            if hasattr(self.vae, "dtype") and self.vae.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
                vae_input_dtype = self.vae.dtype
            else:
                vae_input_dtype = torch.float32
        
        masked_image = masked_image.to(device=device, dtype=vae_input_dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        encoder_posterior = self.vae.encode(masked_image)
        if hasattr(encoder_posterior, "latent_dist"):
            dist = encoder_posterior.latent_dist
        else:
            dist = encoder_posterior
            
        if hasattr(dist, "sample"):
            masked_image_latents = dist.sample(generator=generator)
            # Only apply scaling if using Standard Diffusers VAE
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
            masked_image_latents = (masked_image_latents - shift_factor) * scaling_factor
        else:
            # ComfyUI VAE.encode returns sampled latents directly (from LDM).
            # LDM encode returns distribution, but if we have the distribution object here (dist),
            # wait. 'dist' IS encoder_posterior.
            # LDM encode returns DiagonalGaussianDistribution.
            # If so, it has .sample().
            # If we reached 'else' here, it means dist does NOT have .sample().
            # This implies Comfy VAE wrapper returned a Tensor directly?
            # Comfy VAE wrapper 'encode' returns 'vae.encode(pixels)'.
            # If we unwrapped it in nodes.py, we are using raw LDM.
            # Raw LDM returns distribution.
            # So we should NEVER reach this 'else' block if using raw LDM.
            # Unless LDM implementation is weird.
            # Assume if it's a Tensor, it is Unscaled Latents.
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            masked_image_latents = dist * scaling_factor # Scale IT.

        # alignment
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        # Determine strict VAE input dtype. If pipeline is running in FP8,
        # we still need to provide standard precision to the VAE.
        vae_input_dtype = dtype
        is_fp8 = dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
        if is_fp8:
            if hasattr(self.vae, "dtype") and self.vae.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
                vae_input_dtype = self.vae.dtype
            else:
                vae_input_dtype = torch.float32
       
        # Ensure images are on device/dtype for VAE input
        images = images.to(device=device, dtype=vae_input_dtype)
        
        # Encoding: Support both Diffusers VAE (has latent_dist) and LDM/Comfy VAE (returns dist directly)
        encoder_posterior = self.vae.encode(images)
        
        if hasattr(encoder_posterior, "latent_dist"):
            dist = encoder_posterior.latent_dist
        else:
            dist = encoder_posterior
            
        if hasattr(dist, "sample"):
            image_latents = dist.sample(generator=generator)
            # Only apply scaling if using Standard Diffusers VAE
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
            image_latents = (image_latents - shift_factor) * scaling_factor
        else:
            # ComfyUI VAE.encode returns sampled latents directly (Tensor).
            # Assume Unscaled. Scale it.
            scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
            image_latents = dist * scaling_factor
        
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        
        # Cast back to pipeline dtype (FP8)
        image_latents = image_latents.to(dtype=dtype)
        
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
        
        return image_latents

    @staticmethod
    def _normalize_video_frames_input(video_frames: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert incoming frames to float32 numpy array in (F, H, W, C) layout with RGB channels."""
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.detach().cpu().numpy()
        else:
            video_frames = np.asarray(video_frames)

        if video_frames.ndim == 5 and video_frames.shape[0] == 1:
            video_frames = video_frames[0]

        if video_frames.ndim != 4:
            raise ValueError(f"Expected 4D video tensor, received shape {video_frames.shape}")

        if video_frames.shape[-1] not in (1, 3, 4):
            if video_frames.shape[1] in (1, 3, 4):
                video_frames = np.moveaxis(video_frames, 1, -1)
            else:
                raise ValueError(
                    "Unsupported channel layout for video frames: "
                    f"{video_frames.shape}. Expected channels-last or channels-first." )

        channels = video_frames.shape[-1]

        if channels == 1:
            video_frames = np.repeat(video_frames, 3, axis=-1)
        elif channels > 3:
            rgb, alpha = video_frames[..., :3], video_frames[..., 3:]
            if np.max(np.abs(alpha - alpha[..., :1])) < 1e-4:
                video_frames = rgb
            else:
                video_frames = rgb

        return video_frames.astype(np.float32)

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    def _resolve_mask_tensor(self, resolution: int, mask_image_path: str):
        # Resolve mask path relative to project to support both absolute and package-relative paths
        candidate_paths = []
        if mask_image_path:
            if os.path.isabs(mask_image_path):
                candidate_paths.append(mask_image_path)
            else:
                candidate_paths.append(os.path.join(self._project_root, mask_image_path))
                candidate_paths.append(mask_image_path)

        last_error = None
        for candidate in candidate_paths:
            try:
                mtime = os.path.getmtime(candidate)
                cache_key = (self._mask_generation_tag, resolution, os.path.abspath(candidate), mtime)
                cached = self._mask_cache.get(cache_key)
                if cached is not None:
                    return cached.clone(), os.path.abspath(candidate), cache_key

                mask_tensor = load_fixed_mask(resolution, candidate)
                self._mask_cache[cache_key] = mask_tensor
                return mask_tensor.clone(), os.path.abspath(candidate), cache_key
            except (FileNotFoundError, OSError) as err:
                last_error = err
                continue

        raise FileNotFoundError(f"Unable to resolve mask at {mask_image_path}: {last_error}")

    def _get_image_processor(self, resolution: int, device: torch.device, mask_image_path: str) -> ImageProcessor:
        mask_tensor, mask_path, mask_version = self._resolve_mask_tensor(resolution, mask_image_path)
        if isinstance(device, torch.device):
            device_key = device.type
        else:
            device_key = str(device)
        if device_key != "cuda":
            device_key = "cpu"
        cache_key = (resolution, device_key, mask_version)

        cached = self._image_processor_cache.get(cache_key)
        if cached is not None:
            cached.mask_source = mask_path
            return cached

        processor = ImageProcessor(resolution, device=device_key, mask_image=mask_tensor.clone())
        processor.mask_source = mask_path
        self._image_processor_cache[cache_key] = processor
        return processor

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        # decoded_latents comes from VAE and is likely BF16/FP32.
        # pixel_values/masks might be passed as float32 or whatever.
        # weight_dtype might be FP8.
        # We should unify to decoded_latents.dtype to avoid FP8 mix errors.
        
        target_dtype = decoded_latents.dtype
        
        pixel_values = pixel_values.to(device=device, dtype=target_dtype)
        masks = masks.to(device=device, dtype=target_dtype)
        
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def _flatten_video_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return tensor
        if tensor.dim() <= 4:
            return tensor

        trailing = tensor.shape[-3:]
        leading = int(math.prod(tensor.shape[:-3]))
        return tensor.reshape(leading, *trailing)

    @staticmethod
    def _coerce_video_tensor_shape(
        tensor: torch.Tensor,
        target_channels: int,
        target_size: Tuple[int, int],
        *,
        mode: str = "bilinear",
        tensor_name: str = "tensor",
    ) -> torch.Tensor:
        if tensor is None:
            return tensor

        updated = tensor

        if updated.shape[1] != target_channels:
            if updated.shape[1] > target_channels:
                logger.warning(
                    "[ComfyUI-LipSync-GAP] Trimming channels for %s: %s -> %s",
                    tensor_name,
                    updated.shape,
                    (updated.shape[0], target_channels, *updated.shape[2:]),
                )
                updated = updated[:, :target_channels]
            elif updated.shape[1] == 1 and target_channels == 3:
                logger.warning(
                    "[ComfyUI-LipSync-GAP] Expanding grayscale %s to RGB for downstream compatibility.",
                    tensor_name,
                )
                updated = updated.repeat(1, 3, 1, 1)
            else:
                logger.warning(
                    "[ComfyUI-LipSync-GAP] Adjusting channel count for %s via replication: %s -> %s",
                    tensor_name,
                    updated.shape,
                    (updated.shape[0], target_channels, *updated.shape[2:]),
                )
                repeat_factor = math.ceil(target_channels / updated.shape[1])
                updated = updated.repeat(1, repeat_factor, 1, 1)[:, :target_channels]

        if updated.shape[-2:] != target_size:
            logger.warning(
                "[ComfyUI-LipSync-GAP] Resizing %s from %s to %s to keep chunk shapes aligned.",
                tensor_name,
                updated.shape[-2:],
                target_size,
            )
            updated = F.interpolate(updated, size=target_size, mode=mode, align_corners=False)

        return updated

    @staticmethod
    def _guard_bad_frames(
        decoded_frames: torch.Tensor,
        reference_frames: torch.Tensor,
        blend_masks: torch.Tensor,
        prev_frame: Optional[torch.Tensor] = None,
        threshold: float = 3.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detect extreme mouth deviations and clamp to the last valid frame to avoid harsh glitches.
        guarded = decoded_frames.clone()
        device = guarded.device

        mouth_mask = (1 - blend_masks).to(device=device, dtype=guarded.dtype)
        diff = (guarded - reference_frames.to(device=device, dtype=guarded.dtype)) * mouth_mask
        frame_scores = diff.abs().mean(dim=(1, 2, 3))
        finite_scores = frame_scores[torch.isfinite(frame_scores)]

        if finite_scores.numel() == 0:
            baseline = torch.tensor(0.0, device=device, dtype=guarded.dtype)
        else:
            baseline = torch.quantile(finite_scores.float(), 0.5).to(device=device, dtype=guarded.dtype)

        limit = torch.maximum(baseline * threshold + 1e-4, torch.tensor(0.15, device=device, dtype=guarded.dtype))
        limit_value = float(limit.reshape(-1)[0].item())
        last_valid = prev_frame.to(device=device, dtype=guarded.dtype) if prev_frame is not None else None

        for index in range(guarded.shape[0]):
            score = frame_scores[index]
            score_is_finite = bool(torch.isfinite(score).all().item())
            score_value = float(score.float().mean().item())
            if (not score_is_finite) or (score_value > limit_value):
                if index > 0:
                    fallback = guarded[index - 1]
                elif last_valid is not None:
                    fallback = last_valid
                else:
                    fallback = reference_frames[index].to(device=device, dtype=guarded.dtype)
                guarded[index] = fallback.detach()
            last_valid = guarded[index].detach()

        return guarded, last_valid

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    def affine_transform_video(self, video_frames: np.ndarray):
        faces = []
        boxes = []
        affine_matrices = []
        mouth_masks = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for frame in tqdm.tqdm(video_frames):
            face, box, affine_matrix, mouth_mask = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
            if mouth_mask is None:
                default_mask = self.image_processor.mask_blend.detach().cpu().clone().float()
                mouth_masks.append(default_mask)
            else:
                if mouth_mask.dim() == 2:
                    mouth_mask = mouth_mask.unsqueeze(0)
                mouth_masks.append(mouth_mask.detach().cpu().clone().float())

        faces = torch.stack(faces)
        mouth_masks = torch.stack(mouth_masks)
        return faces, boxes, affine_matrices, mouth_masks

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            face = torchvision.transforms.functional.resize(
                face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
            )
            out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
            out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(self, whisper_chunks: list, video_frames: np.ndarray):
        # If the audio is longer than the video, we need to loop the video
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices, mouth_masks = self.affine_transform_video(video_frames)
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_masks = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_masks.append(mouth_masks)
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_masks.append(mouth_masks.flip(0))

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            mouth_masks = torch.cat(loop_masks, dim=0)[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices, mouth_masks = self.affine_transform_video(video_frames)

        return video_frames, faces, boxes, affine_matrices, mouth_masks

    @torch.no_grad()
    def __call__(
        self,
        video_frames: torch.Tensor,
        audio_waveform: torch.Tensor,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        temporal_blend_strength: float = 0.2,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        # check_ffmpeg_installed() # Not needed for tensor-only flow

        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device

        if hasattr(self.audio_encoder, "model_location"):
            print(f"[ComfyUI-LipSync-GAP] Audio encoder expects checkpoint at {self.audio_encoder.model_location}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.image_processor = self._get_image_processor(height, device, mask_image_path)
        if hasattr(self.image_processor, "mask_source"):
            mask_path_display = os.path.abspath(self.image_processor.mask_source)
            print(f"[ComfyUI-LipSync-GAP] Using mouth mask from {mask_path_display}")

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # DEBUG: Save intermediate steps for inspection
        import cv2
        debug_dir = os.path.join(os.path.dirname(__file__), "..", "..", "debug_output")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "debug_mask.jpg"), (self.image_processor.mask_image[0].cpu().numpy() * 255).astype(np.uint8))
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        num_frames = 16 # LatentSync default?
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Handle Audio Features
        # audio_waveform should be prepared (16k)
        if hasattr(self.audio_encoder, "audio2feat"):
             whisper_feature = self.audio_encoder.audio2feat(audio_waveform)
        else:
             # Fallback if interface mismatch
             whisper_feature = self.audio_encoder._audio2feat(audio_waveform)
             
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)

        # Handle Video
        video_frames = self._normalize_video_frames_input(video_frames)
        
        # Audio samples for output
        if isinstance(audio_waveform, str):
            # If path, load it or just leave it? We don't use it for returning.
            audio_samples = None 
        elif isinstance(audio_waveform, torch.Tensor):
            audio_samples = audio_waveform.cpu().numpy()
        else:
            audio_samples = audio_waveform

        # Use loop_video (which also does affine transforms)
        video_frames, faces, boxes, affine_matrices, mouth_masks = self.loop_video(whisper_chunks, video_frames)

        synced_video_frames = []
        expected_rgb_channels: Optional[int] = None
        expected_spatial_size: Optional[Tuple[int, int]] = None
        expected_mask_channels: Optional[int] = None
        prev_guard_frame = None

        num_channels_latents = self.vae.config.latent_channels

        # Update num_frames if loop_video changed it? 
        # Actually loop_video returns the full length frames.
        # The inference loop processes in chunks of `num_frames` (default 16 in Audio2Feature but here hardcoded?)
        # ComfyUI passed num_frames=16 in original arg, but it was just "Sample frames".
        # The code chunks by `num_frames`. Let's use 16 as chunk size.
        chunk_size = 16

        num_inferences = math.ceil(len(whisper_chunks) / chunk_size)
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            
            # Slice the audio embeds
            current_audio_chunk = whisper_chunks[i * chunk_size : (i + 1) * chunk_size]
            if len(current_audio_chunk) == 0: break
            
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(current_audio_chunk)
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                audio_embeds = self._match_audio_embedding_dim(audio_embeds)
                
                # --- AUDIO DEBUG ---
                if i == 0:
                    print(f"[ComfyUI-LipSync-GAP] AudioEmbeds Stats: Shape={audio_embeds.shape}")
                    print(f"Max={audio_embeds.max().item():.4f}, Min={audio_embeds.min().item():.4f}, Mean={audio_embeds.mean().item():.4f}, Std={audio_embeds.std().item():.4f}")
                    if audio_embeds.std().item() < 1e-4:
                        print("[ComfyUI-LipSync-GAP] WARNING: Audio Embeddings seem essentially empty/flat!")
                # -------------------
                
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
                
            # Slice faces and latents
            inference_faces = faces[i * chunk_size : (i + 1) * chunk_size]
            audio_chunk_len = len(current_audio_chunk)
            chunk_length = min(audio_chunk_len, inference_faces.shape[0])
            if chunk_length == 0:
                continue

            if audio_chunk_len != chunk_length:
                current_audio_chunk = current_audio_chunk[:chunk_length]

            if inference_faces.shape[0] != chunk_length:
                inference_faces = inference_faces[:chunk_length]

            latents = self.prepare_latents(
                batch_size,
                chunk_length,
                num_channels_latents,
                height,
                width,
                weight_dtype,
                device,
                generator,
            )
            
            # Prepare masks
            chunk_mouth_masks = mouth_masks[i * chunk_size : (i + 1) * chunk_size]
            # Keep per-frame mouth mask overrides aligned with the truncated chunk
            if chunk_mouth_masks.shape[0] != chunk_length:
                chunk_mouth_masks = chunk_mouth_masks[:chunk_length]

            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces,
                affine_transform=False,
                mouth_masks=chunk_mouth_masks,
            )

            ref_pixel_values = self._flatten_video_batch(ref_pixel_values)
            masked_pixel_values = self._flatten_video_batch(masked_pixel_values)
            masks = self._flatten_video_batch(masks)

            if expected_rgb_channels is None:
                expected_rgb_channels = ref_pixel_values.shape[1]
                expected_spatial_size = ref_pixel_values.shape[-2:]
                expected_mask_channels = masks.shape[1]
            else:
                ref_pixel_values = self._coerce_video_tensor_shape(
                    ref_pixel_values,
                    expected_rgb_channels,
                    expected_spatial_size,
                    tensor_name="reference frames",
                )
                masked_pixel_values = self._coerce_video_tensor_shape(
                    masked_pixel_values,
                    expected_rgb_channels,
                    expected_spatial_size,
                    tensor_name="masked frames",
                )
                masks = self._coerce_video_tensor_shape(
                    masks,
                    expected_mask_channels,
                    expected_spatial_size,
                    mode="nearest",
                    tensor_name="blend mask",
                )

            # DEBUG: Save one batch of inputs
            if i == 0:
                def _prepare_debug_image(tensor: torch.Tensor, *, assume_normalized: bool = True, to_bgr: bool = False) -> np.ndarray:
                    if tensor.is_sparse:
                        tensor = tensor.to_dense()
                    if tensor.dim() > 3:
                        tensor = tensor.squeeze(0)
                    tensor = tensor.detach().float()
                    tensor = tensor.movedim(0, -1)
                    array = tensor.cpu().numpy()
                    if assume_normalized:
                        array = (array * 0.5 + 0.5).clip(0.0, 1.0) * 255.0
                    else:
                        array = np.clip(array, 0.0, 255.0)
                    image = array.astype(np.uint8)
                    if to_bgr:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    return image

                ref_img = _prepare_debug_image(ref_pixel_values[0])
                cv2.imwrite(os.path.join(debug_dir, "debug_ref_input.jpg"), ref_img)

                masked_img = _prepare_debug_image(masked_pixel_values[0])
                cv2.imwrite(os.path.join(debug_dir, "debug_masked_input.jpg"), masked_img)

                face_img = _prepare_debug_image(inference_faces[0], assume_normalized=False, to_bgr=True)
                cv2.imwrite(os.path.join(debug_dir, "debug_face_original.jpg"), face_img)

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            
            # We must not clear the progress bar if we want one global bar, but here we loop chunks.
            # The original code had nested progress bar.
            
            for j, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                unet_input = self.scheduler.scale_model_input(unet_input, t)

                # concat latents, mask, masked_image_latents in the channel dimension
                unet_input = torch.cat(
                    [unet_input, mask_latents, masked_image_latents, ref_latents], dim=1
                )

                # predict the noise residual
                noise_pred = self.unet(
                    unet_input, t, encoder_hidden_states=audio_embeds
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self._flatten_video_batch(decoded_latents)

            if expected_rgb_channels is not None and expected_spatial_size is not None:
                decoded_latents = self._coerce_video_tensor_shape(
                    decoded_latents,
                    expected_rgb_channels,
                    expected_spatial_size,
                    tensor_name="decoded frames",
                )
            
            # --- DEBUG: Save Generated Lips ---
            # Save the raw output of the VAE (the generated face) to local disk for inspection
            images_raw = self.pixel_values_to_images(decoded_latents) # [B, H, W, C]
            
            debug_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "debug_lips")
            os.makedirs(debug_out_dir, exist_ok=True)
            
            for k in range(len(images_raw)):
                frame_idx = i * chunk_size + k
                # Save first few frames only to avoid spam
                if frame_idx < 10: 
                    # Convert RGB to BGR for OpenCV
                    img_save = cv2.cvtColor(images_raw[k], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(debug_out_dir, f"lip_gen_{frame_idx:04d}.jpg"), img_save)
            # ----------------------------------

            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, masks, device, weight_dtype
            )

            decoded_latents, prev_guard_frame = self._guard_bad_frames(
                decoded_latents,
                ref_pixel_values,
                masks,
                prev_guard_frame,
            )

            if temporal_blend_strength > 0.0 and synced_video_frames:
                blend = float(min(max(temporal_blend_strength, 0.0), 1.0))
                prev_chunk = synced_video_frames[-1]
                prev_tail = prev_chunk[-1:].to(device=decoded_latents.device, dtype=decoded_latents.dtype)
                target_first = decoded_latents[:1]

                if prev_tail.shape == target_first.shape:
                    blended_first = torch.lerp(target_first, prev_tail, blend)
                    decoded_latents = torch.cat([blended_first, decoded_latents[1:]], dim=0)

                    updated_tail = torch.lerp(prev_tail, blended_first.detach(), blend)
                    updated_tail = updated_tail.to(device=prev_chunk.device, dtype=prev_chunk.dtype)
                    synced_video_frames[-1] = torch.cat(
                        [prev_chunk[:-1], updated_tail], dim=0
                    )
                else:
                    # Shape mismatch usually indicates a format change across chunks (e.g. differing channel counts).
                    # Skip temporal blending for this boundary rather than raising.
                    target_warning = (
                        f"[ComfyUI-LipSync-GAP] Skip temporal blend: prev_tail shape {tuple(prev_tail.shape)} "
                        f"!= current shape {tuple(target_first.shape)}"
                    )
                    logger.warning(target_warning)
            prev_guard_frame = decoded_latents[-1].detach().to(device=decoded_latents.device, dtype=decoded_latents.dtype)
            synced_video_frames.append(decoded_latents)

        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices)

        if is_train:
            self.unet.train()
            
        # Return tensors (B, H, W, C) - wait restore_video returns numpy usually [F, H, W, 3]
        # ComfyUI expects [F, H, W, C] float (0-1) or uint8 (0-255)
        # We should return tensor float 0-1
        
        output_tensor = torch.from_numpy(synced_video_frames).float() / 255.0
        return output_tensor
