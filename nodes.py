# Copyright (c) 2026 Geekatplay Studio
# Part of the Geekatplay Studio ComfyUI Toolset

import os
import torch
import numpy as np
import folder_paths
import comfy.model_management
import comfy.utils
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.configuration_utils import FrozenDict
from omegaconf import OmegaConf

# Import internal library
from .latentsync.models.unet import UNet3DConditionModel
from .latentsync.whisper.audio2feature import Audio2Feature
from .latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from .latentsync.whisper.whisper import load_model as load_whisper_model
from .latentsync.whisper.whisper import available_models as whisper_available_models
from .latentsync.whisper.whisper import _MODELS as WHISPER_MODEL_URLS
from DeepCache import DeepCacheSDHelper

# Register model folders
if "latentsync" not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("latentsync", os.path.join(folder_paths.models_dir, "latentsync"))

WHISPER_ALIAS_SET = set(whisper_available_models())
WHISPER_DEFAULT_CHOICES = sorted(WHISPER_ALIAS_SET | {"tiny.pt", "small.pt", "medium.pt", "large.pt"})


def _normalize_whisper_selection(selection: str) -> str:
    normalized = (selection or "").strip()
    if normalized.lower().endswith(".pt"):
        normalized = normalized[:-3]
    return normalized.lower()

class GapLipSyncModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_model": (folder_paths.get_filename_list("latentsync"),),
                "whisper_model": (WHISPER_DEFAULT_CHOICES, {"default": "tiny"}),
                "precision": (["fp16", "fp32", "fp8_e4m3fn"], {"default": "fp16"}),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("PERFECT_SYNC_MODELS",)
    RETURN_NAMES = ("models",)
    FUNCTION = "load_models"
    CATEGORY = "Geekatplay Studio/LipSync GAP"

    def load_models(self, unet_model, whisper_model, precision, vae):
        device = comfy.model_management.get_torch_device()
        
        # Determine dtype
        if precision == "fp8_e4m3fn":
            dtype = torch.float8_e4m3fn
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

        # 1. Load UNet
        unet_path = folder_paths.get_full_path("latentsync", unet_model)
        if not unet_path:
            # Fallback check if it's a folder in 'latentsync/unet'
            potential_path = os.path.join(folder_paths.models_dir, "latentsync", "unet", unet_model)
            if os.path.exists(potential_path):
                unet_path = potential_path
            else:
                 # Check if the user selected a file that is actually a config or checkpoint inside a folder
                 unet_path = folder_paths.get_full_path("latentsync", unet_model)
        
        if not unet_path or not os.path.exists(unet_path):
             raise FileNotFoundError(f"UNet model not found: {unet_model}")

        print(f"[ComfyUI-LipSync-GAP] Loading UNet from {unet_path} with precision {precision}")
        
        # Assume unet_path acts as 'pretrained_model_name_or_path'. 
        # LatentSync unet loading usually requires a config. 
        # If unet_path is a .pt file, we need a separate config.
        # But standard Diffusers format relies on folder structure or single file + config.
        
        # The cloned wrapper suggests 'latentsync_unet.pt' and a separate config yaml.
        # We need to handle this.
        
        config_path = os.path.join(os.path.dirname(__file__), "configs", "unet", "stage2_512.yaml") # Default
        
        # Check if the UNet is just a checkpoint or a diffusers folder
        if os.path.isfile(unet_path) and unet_path.endswith('.pt'):
            # Load config
            config = OmegaConf.load(config_path)
            unet, _ = UNet3DConditionModel.from_pretrained(
                OmegaConf.to_container(config.model),
                unet_path,
                device="cpu", # Load to CPU first
            )
        else:
            # Assume local diffusers folder
            unet = UNet3DConditionModel.from_pretrained(unet_path)

        unet = unet.to(dtype=dtype)

        # 2. Load Whisper (AudioEncoder)
        # We use a wrapper around Whisper
        # We need to find the whisper model path
        whisper_folder = os.path.join(folder_paths.models_dir, "latentsync", "whisper")
        whisper_path = os.path.join(whisper_folder, whisper_model)
        
        # Normalize Whisper selection to supported aliases if user chose e.g. "medium.pt"
        whisper_alias = _normalize_whisper_selection(whisper_model)
        alias_known = whisper_alias in WHISPER_ALIAS_SET
        if alias_known:
            # Update target path to the expected filename for logging or manual placement.
            alias_filename = os.path.basename(WHISPER_MODEL_URLS[whisper_alias])
            whisper_path = os.path.join(whisper_folder, alias_filename)

        if not os.path.exists(whisper_path):
             # Try checking if it's in the standard storage
             pass 
        
        print(f"[ComfyUI-LipSync-GAP] Loading AudioEncoder (Whisper: {whisper_model})...")
        # Audio2Feature expects a 'model_path' which is either an alias or a resolved checkpoint path.
        # Convert known selections to aliases so the internal loader can download them automatically when missing.
        audio_encoder_identifier = whisper_alias if alias_known else (whisper_path if os.path.exists(whisper_path) else whisper_model)

        audio_encoder = Audio2Feature(
            model_path=audio_encoder_identifier,
            device="cuda", # AudioEncoder likely needs cuda
            num_frames=16,
            audio_feat_length=[2, 2],
        )

        print("[ComfyUI-LipSync-GAP] Models are ready for inference.")
        return ((unet, audio_encoder, vae, dtype, config_path),)


class GapLipSyncSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models": ("PERFECT_SYNC_MODELS",),
                "images": ("IMAGE",), # Video frames
                "audio": ("AUDIO",),
                "seed": ("INT", {"default": 1247}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "frame_rate": ("INT", {"default": 25, "min": 1, "max": 60}),
                "video_resize_limit": (["None", "1280", "1024", "768", "512"], {"default": "1280"}),
                "audio_offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
                "audio_duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "sample"
    CATEGORY = "Geekatplay Studio/LipSync GAP"

    class VAEWrapper(torch.nn.Module):
        class OpsWrapper:
            def __init__(self, config):
                self.config = config
            def __getattr__(self, name):
                if name in self.config:
                    return self.config[name]
                raise AttributeError(f"'OpsWrapper' object has no attribute '{name}'")
        
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        
        @property
        def dtype(self):
            # Check if underlying VAE has dtype, else infer from parameters
            if hasattr(self.vae, "dtype"):
                return self.vae.dtype
            
            # Support ComfyUI VAE which holds model in first_stage_model
            if hasattr(self.vae, "first_stage_model"):
                try:
                    return next(self.vae.first_stage_model.parameters()).dtype
                except:
                    pass
            
            try:
                return next(self.vae.parameters()).dtype
            except:
                return torch.float32 # Fallback
            
        @property
        def device(self):
            if hasattr(self.vae, "device"):
                return self.vae.device
            try:
                return next(self.vae.parameters()).device
            except:
                return torch.device("cpu")
            
        @property
        def config(self):
            # Define minimal defaults for SD1.5 VAE
            cfg = {
                "latent_channels": 4,
                "block_out_channels": [128, 256, 512, 512],
                "scaling_factor": 0.18215,
                "shift_factor": 0.0,
                "sample_size": 256,
                "in_channels": 3,
                "out_channels": 3
            }
            
            if hasattr(self.vae, "config") and self.vae.config is not None:
                # If the internal model has a config dict, use it
                c = self.vae.config
                if hasattr(c, "items"):
                    cfg.update(c)
                
            return self.OpsWrapper(cfg)

            
        def encode(self, *args, **kwargs):
            return self.vae.encode(*args, **kwargs)
            
        def decode(self, *args, **kwargs):
            return self.vae.decode(*args, **kwargs)
            
        # Forward everything else
        def __getattr__(self, name):
            try:
                # Delegate to torch.nn.Module to find submodules/parameters/buffers first
                return super().__getattr__(name)
            except AttributeError:
                # If not found in self, check the wrapped VAE
                # Caution: accessing self.vae here relies on super().__getattr__('vae') succeeding
                # in the next recursive call, preventing infinite recursion.
                return getattr(self.vae, name)
             
    def sample(self, models, images, audio, seed, steps, guidance_scale, frame_rate, video_resize_limit="1280", audio_offset_seconds=0.0, audio_duration_seconds=0.0):
        unet, audio_encoder, vae, dtype, config_path = models
        
        # --- Video Resize Logic ---
        if video_resize_limit != "None":
            try:
                max_limit = int(video_resize_limit)
                B, H, W, C = images.shape
                
                # Check if resizing is needed
                max_dim = max(H, W)
                if max_dim > max_limit:
                    scale = max_limit / max_dim
                    new_H = int(H * scale)
                    new_W = int(W * scale)
                    
                    # Ensure Even Dimensions (Important for ffmpeg/video codecs)
                    if new_H % 2 != 0: new_H -= 1
                    if new_W % 2 != 0: new_W -= 1
                    
                    print(f"[ComfyUI-LipSync-GAP] Resizing input video: {W}x{H} -> {new_W}x{new_H} (Limit: {max_limit}px)")
                    
                    # Permute to [B, C, H, W] for interpolate
                    images_p = images.permute(0, 3, 1, 2)
                    
                    # Use bilinear interpolation
                    # Note: antialias=True requires standard torch builds, should be safe in Comfy env
                    images_resized = torch.nn.functional.interpolate(
                        images_p, size=(new_H, new_W), mode="bilinear", align_corners=False, antialias=True
                    )
                    
                    # Permute back to [B, H, W, C]
                    images = images_resized.permute(0, 2, 3, 1)
            except Exception as e:
                print(f"[ComfyUI-LipSync-GAP] Warning: Video resize failed: {e}. Proceeding with original size.")
        # --------------------------

        # --- Audio Slicing Logic ---
        if audio_offset_seconds > 0 or audio_duration_seconds > 0:
            sr = audio["sample_rate"]
            input_waveform = audio["waveform"]
            
            start_idx = int(audio_offset_seconds * sr)
            if audio_duration_seconds > 0:
                length_idx = int(audio_duration_seconds * sr)
                end_idx = start_idx + length_idx
            else:
                end_idx = input_waveform.shape[-1]
                
            # Clamp
            start_idx = max(0, min(start_idx, input_waveform.shape[-1]))
            end_idx = max(start_idx, min(end_idx, input_waveform.shape[-1]))
            
            if start_idx == end_idx:
                 print(f"[ComfyUI-LipSync-GAP] Warning: Audio slicing resulted in empty audio. Using original.")
            else:
                 input_waveform = input_waveform[..., start_idx:end_idx]
                 new_audio = audio.copy()
                 new_audio["waveform"] = input_waveform
                 audio = new_audio
                 print(f"[ComfyUI-LipSync-GAP] Sliced audio: Start={audio_offset_seconds}s, Duration={audio_duration_seconds}s")
        # ---------------------------

        # Setup VAE
        if vae is None:
             # Load default or throw warning?
             # For now, let's assume user must pass VAE or we rely on some fallback
             # But standard Comfy workflow allows passing VAE.
             # If passed from VAE Loader, it is an AutoencoderKL object (wrapped by Comfy).
             # We need to unwrap it if it's a Comfy wrapper, or just use it.
             # Comfy VAE object is usually `vae.first_stage_model` if it's SD. 
             # Wait, Comfy VAE input type ("VAE",) passes a Comfy VAE wrapper.
             pass
        
        # Unwrap VAE if it is a Comfy VAE
        if hasattr(vae, "first_stage_model"):
            vae_model = vae.first_stage_model
        else:
            # Fallback: Create a default VAE? Or maybe we can't.
            # Use standard SD 1.5 VAE config if absolutely necessary, but better to fail.
            if vae is None:
                raise ValueError("VAE is required! Please connect a standard VAE Loader (e.g. SD 1.5 VAE).")
            vae_model = vae

        # Ensure VAE is in the correct dtype (e.g. FP16) to match the pipeline
        # This prevents "Input type (Half) and bias type (BFloat16) should be the same" errors
        try:
            # Check if we need to cast
            # basic check: try to see if parameters are in the target dtype
            first_param = next(vae_model.parameters(), None)
            if first_param is not None and first_param.dtype != dtype:
                print(f"[ComfyUI-LipSync-GAP] Casting VAE from {first_param.dtype} to {dtype}")
                vae_model.to(dtype=dtype)
        except Exception as e:
            print(f"[ComfyUI-LipSync-GAP] Warning: Failed to cast VAE: {e}")

        # Wrap VAE to satisfy Diffusers Pipeline requirements (needs .dtype, .device, .config)
        vae_model = self.VAEWrapper(vae_model)

        # Create Scheduler
        scheduler = DDIMScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            clip_sample=False,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            set_alpha_to_one=False,
            steps_offset=1
        )

        # Initialize Pipeline
        pipeline = LipsyncPipeline(
            vae=vae_model,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )
        
        # DeepCache - Experimental - Currently DISABLED to debug "Gray Square" issue
        # To enable, uncomment the lines below
        # try:
        #     helper = DeepCacheSDHelper(pipe=pipeline)
        #     helper.set_params(cache_interval=3, cache_branch_id=0)
        #     helper.enable()
        #     print("[ComfyUI-LipSync-GAP] DeepCache Enabled (Interval=3)")
        # except Exception as e:
        #     print(f"[ComfyUI-LipSync-GAP] Failed to enable DeepCache: {e}")
        
        # Move pipeline to device
        device = comfy.model_management.get_torch_device()
        pipeline.to(device) # This might be heavy if VAE/UNet are big. 
        # Comfy normally manages this. 'to(device)' moves everything.
        
        # Prepare Inputs
        # images is [B, H, W, 3] range [0, 1].
        # LatentSync pipeline internal logic assumes [0, 255] video frames (similar to read_video output).
        # Convert to numpy array (H, W, C, uint8) for the pipeline's CV2/Mediapipe logic
        video_frames = (images * 255).clone().cpu().numpy().astype("uint8")
        
        # Audio
        # Comfy output is {"waveform": ..., "sample_rate": ...}
        # LatentSyncWrapper Manual Resampling Logic Implementation
        
        waveform = audio["waveform"] # [1, C, N] or [C, N]
        sample_rate = audio["sample_rate"]
        
        # Ensure we are on CPU for torchaudio operations to avoid potential GPU issues
        if waveform.device.type != "cpu":
            waveform = waveform.cpu()
            
        # Handle dimensions: Ensure [C, N]
        if waveform.dim() == 3: # [B, C, N] -> [C, N] (Take first batch)
            waveform = waveform.squeeze(0)
            
        # Resample to 16000Hz (Required by Whisper)
        target_sr = 16000
        if sample_rate != target_sr:
            print(f"[ComfyUI-LipSync-GAP] Resampling audio from {sample_rate}Hz to {target_sr}Hz")
            try:
                import torchaudio
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
                waveform = resampler(waveform)
                sample_rate = target_sr
            except Exception as e:
                print(f"[ComfyUI-LipSync-GAP] Error resampling audio: {e}. Proceeding without resampling (might cause issues).")

        # Mix to Mono (Required by Whisper internal model, though check_inputs usually handles it, doing it explicitly is safer)
        if waveform.shape[0] > 1:
            print(f"[ComfyUI-LipSync-GAP] Mixing audio to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Prepare Audio Temp File
        import soundfile as sf
        import uuid
        
        temp_dir = folder_paths.get_temp_directory()
        temp_audio_path = os.path.join(temp_dir, f"latentsync_{uuid.uuid4().hex[:8]}.wav")
        
        # soundfile expects [N, C] or [N]
        # waveform is [1, N]
        waveform_np = waveform.numpy().T # [N, 1]
        
        try:
            sf.write(temp_audio_path, waveform_np, sample_rate)
            print(f"[ComfyUI-LipSync-GAP] Saved prepared audio (16k, Mono) to {temp_audio_path}")
        except Exception as e:
             print(f"[ComfyUI-LipSync-GAP] Failed to save audio using soundfile: {e}. Trying torchaudio.")
             import torchaudio
             torchaudio.save(temp_audio_path, waveform, sample_rate)
        
        # Run Inference
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # LatentSync needs the mask image path. We included one in utils.
        mask_path = os.path.join(os.path.dirname(__file__), "latentsync", "utils", "mask.png")
        
        try:
            result_frames = pipeline(
                video_frames=video_frames,
                audio_waveform=temp_audio_path, # Pass path instead of array
                video_fps=frame_rate,
                audio_sample_rate=16000,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                weight_dtype=dtype,
                mask_image_path=mask_path,
                generator=generator
            )
        finally:
            # Cleanup temp audio
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except:
                    pass
        
        # Cleanup
        # Offload models? 
        # Since we passed them in, we leave them.
        
        # Result audio? Use the input audio (but synced/trimmed? LatentSync pipeline might trim/loop video)
        # The pipeline logic modified video length to match audio chunks.
        # We should return the audio corresponding to the video.
        # `pipeline` returned `(synced_video_frames, audio_samples)`?
        # My modified `pipeline` returned ONLY `output_tensor`.
        # I should have returned audio too?
        # The modified code:
        # `output_tensor = torch.from_numpy(synced_video_frames).float() / 255.0`
        # `return output_tensor`
        
        # The video length is adjusted to audio in the pipeline.
        # So we should return the original audio (prepared 16k) or the original input audio?
        # Ideally the audio that matches the video duration.
        # But audio usually drives the video length here.
        # So passing back the ORIGINAL audio (maybe trimmed) is fine.
                # Convert result_frames (numpy uint8) back to Tensor (float 0-1) for ComfyUI
        if isinstance(result_frames, np.ndarray):
            result_frames = torch.from_numpy(result_frames).float() / 255.0
            
        return (result_frames, audio)

NODE_CLASS_MAPPINGS = {
    "GapLipSyncModelLoader": GapLipSyncModelLoader,
    "GapLipSyncSampler": GapLipSyncSampler,
    # Backward compatibility aliases
    "GapPerfectSyncModelLoader": GapLipSyncModelLoader,
    "GapPerfectSyncSampler": GapLipSyncSampler,
    "PerfectSyncModelLoader": GapLipSyncModelLoader,
    "PerfectSyncSampler": GapLipSyncSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GapLipSyncModelLoader": "LipSync GAP Model Loader",
    "GapLipSyncSampler": "LipSync GAP Sampler",
    "GapPerfectSyncModelLoader": "LipSync GAP Model Loader (Legacy Alias)",
    "GapPerfectSyncSampler": "LipSync GAP Sampler (Legacy Alias)",
    "PerfectSyncModelLoader": "LipSync GAP Model Loader (Legacy)",
    "PerfectSyncSampler": "LipSync GAP Sampler (Legacy)",
}
