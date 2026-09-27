# ComfyUI-LipSync-GAP — audio-driven lip sync for ComfyUI

Turn an existing talking-head clip and a replacement speech track into lip-synced frames inside ComfyUI, using the LatentSync pipeline.

**Experimental fork maintained by Leon.** Based on [Geekatplay Studio's extension](https://github.com/GeekatplayStudio/ComfyUI-LipSync-GAP) and [ByteDance LatentSync](https://github.com/bytedance/LatentSync). This fork adds verified model downloads, explicit audio/output contracts and a reproducible GPU engineering smoke. It does not introduce a new lip-sync model or claim better output quality than upstream.

[First run](#first-run) · [Changes in this fork](#changes-in-this-fork) · [Measured smoke and demo plan](docs/VALIDATION.md) · [Release candidate](docs/RELEASE_CANDIDATE.md)

## Is this for you?

Use this if you already run ComfyUI on an NVIDIA CUDA GPU and want to replace speech in a short video using a node workflow. You provide the video, speech audio, LatentSync checkpoint and compatible VAE. The sampler returns frames and audio; VideoHelperSuite combines them into a file.

This is still an experimental integration. A measured VRAM minimum, end-to-end compatibility matrix and reproducible before/after demo have not yet been established for this fork. Start with a short clip, FP16 and a 512 resize limit. FP8 is exposed as an experimental option, not a verified memory or quality guarantee.

## Changes in this fork

- Verifies pinned model sizes and SHA-256 before atomic download publication and before loading; corrupt files fail explicitly.
- Fixes mask shape, audio slicing/resampling and padded output duration, with temporary-file cleanup on failure.
- Replaces the inherited StyleSync adapter with OpenCV geometry; removes unused training-only components and records their provenance.
- Reuses a Whisper checkpoint already installed under `ComfyUI/models/latentsync/whisper/` instead of downloading the same alias into a second cache.
- Offers `tiny` and `tiny.en`, matching the shipped UNet configuration's 384 audio feature dimensions; rejects incompatible selections before loading the UNet.
- Removes the mandatory import of DeepCache, whose inference integration is currently disabled.
- Aligns the starter workflow to 25 fps and a three-second clip, corrects serialized widget values and uses the documented UNet path.

These changes have regression checks and a bounded 20-step GPU-to-MP4 smoke (public-domain static image plus synthetic tone). Full speech-quality and UI workflow validation remain open. See [validation status](docs/VALIDATION.md) before relying on a production workflow.

## First run

### 1. Install in your ComfyUI environment

From `ComfyUI/custom_nodes/`:

```sh
git clone https://github.com/leonininder/ComfyUI-LipSync-GAP.git
```

Install [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), used by the provided video input/output nodes, and make FFmpeg available for video I/O. The Python package `ffmpeg-python` alone does not install the FFmpeg executable.

On Windows, open a terminal **in this extension's folder** and run `install.bat`. Review the script first: it installs requirements and downloads model assets. It searches for ComfyUI's embedded Python, then falls back to system Python. Check the printed interpreter path; dependencies must be installed into the Python environment that launches ComfyUI. The installer stops with a nonzero exit code when a required download command fails. Pinned assets are checked for size and SHA-256 integrity; successful installation does not establish inference compatibility.

For a virtual environment, activate the environment used by ComfyUI, then run:

```sh
python -m pip install -r requirements.txt
```

Place model files manually as below. No clean-install version matrix has been verified yet; do not treat the minimum package versions as a tested lockfile.

### 2. Check the model files

| Asset | Destination / selection |
|---|---|
| LatentSync UNet | `ComfyUI/models/latentsync/latentsync_unet.pt` |
| Whisper tiny | `ComfyUI/models/latentsync/whisper/tiny.pt` |
| MediaPipe Face Landmarker | `ComfyUI/custom_nodes/ComfyUI-LipSync-GAP/latentsync/weights/face_landmarker.task` |
| Compatible VAE | Select your installed checkpoint in the workflow's `CheckpointLoaderSimple`; its VAE output feeds the model loader |

The installer sources the UNet from [ByteDance/LatentSync-1.6](https://huggingface.co/ByteDance/LatentSync-1.6), Whisper tiny from [ByteDance/LatentSync](https://huggingface.co/ByteDance/LatentSync) and the landmarker from [Google's MediaPipe model storage](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task). Verify model provenance and applicable terms before use. VAE compatibility remains part of the GPU acceptance run; the example's checkpoint filename is a placeholder you must replace.

If a supported Whisper file is missing, the audio encoder may download it to this extension's `checkpoints/whisper/` cache. For reproducible/offline runs, place the file in the documented ComfyUI model folder first.

### 3. Load the starter workflow

1. Restart ComfyUI and open [`examples/workflow_lipsync_gap.json`](examples/workflow_lipsync_gap.json).
2. Choose a short video in `VHS_LoadVideo` and your speech in `LoadAudio`; the placeholder files are not bundled. Use footage you own or have permission to edit.
3. Select your installed checkpoint/VAE in `CheckpointLoaderSimple` and `latentsync_unet.pt`, `tiny`, `fp16` in `GapLipSyncModelLoader`.
4. Keep the video loader, sampler and video output at **25 fps** for this first run. The sample caps input at 75 frames and audio at three seconds.
5. Queue the workflow. Inspect the exported clip for audible synchronization, face artifacts, frame continuity and duration. A completed queue alone is not a quality result.

Node categories remain `Geekatplay Studio/LipSync GAP` for compatibility. The legacy `workflow_perfect_sync.json` is retained as an upstream reference; use the starter workflow above for this fork.

## Verify before inference

Using the Python environment that launches ComfyUI:

```sh
python preflight.py --comfy-root /path/to/ComfyUI --vae /path/to/selected-vae.safetensors
```

The read-only check imports required packages, checks CUDA/FFmpeg/configuration, verifies all three asset hashes and requires a selected VAE file. `PREFLIGHT_OK` means those checks passed, not that the VAE or output quality has been certified. Missing or corrupt assets return exit 1. No download occurs in preflight.

`assets-manifest.json` pins LatentSync **1.6** (~5.07 GB), Whisper tiny (~75.6 MB), and the landmarker (~3.76 MB). The original unversioned LatentSync download is a different checkpoint. The current loader accepts the pinned 1.6 asset; alternative checkpoints are not silently accepted. To install assets explicitly, use `python asset_integrity.py --comfy-root /path/to/ComfyUI --download`. Downloads are published only after size/hash verification and atomic replacement; interrupted temporary files are cleaned. A corrupt existing file is rejected and preserved for diagnosis. Move it aside explicitly before retrying. Hashes establish identity/integrity, not license clearance or model safety.

Output frame count follows the selected audio duration, rounded up to the next video frame; padded Whisper features are not exported as extra frames.

## Troubleshooting

| Symptom | First action |
|---|---|
| Missing VHS nodes | Install VideoHelperSuite and restart ComfyUI |
| Missing GAP nodes | Read the ComfyUI import traceback; check packages in the active interpreter |
| UNet dropdown empty | Check the file under `models/latentsync/`, then restart/refresh |
| Whisper feature mismatch | Use `tiny` or `tiny.en`; larger models are not interchangeable with the shipped config |
| CUDA out of memory | Shorten the clip and reduce the resize limit; no memory target is certified yet |
| Black/gray output or VAE errors | Record the selected VAE, versions and traceback; do not assume successful inference |
| Drift or wrong duration | Check all three fps settings, frame count and audio trim |

For a useful bug report, include OS/GPU, ComfyUI commit, Python/PyTorch/CUDA versions, full traceback, workflow, chosen model filenames and a shareable minimal clip. Never include private media or credentials. Fork-specific reports and fixes belong in this fork's GitHub discussion/PR facilities as available; upstream support remains with its authors.

## Architecture and verification

`Video frames + speech → model loader (UNet, Whisper, VAE) → sampler (audio features, diffusion, face blending) → frames + audio → VideoHelperSuite export`

`nodes.py` adapts ComfyUI inputs; `latentsync/` holds the inherited model/pipeline implementation; `model_selection.py` resolves compatible Whisper checkpoints without requiring GPU imports.

Run the regression suite without model downloads (Torch/OpenCV enable the geometry checks):

```sh
python -m unittest discover -s tests -v
```

See [the acceptance recipe](docs/VALIDATION.md) for the remaining end-to-end work and a reproducible public demo. No quality, throughput or VRAM benchmark is currently published by this fork.

## Credits and licensing

The integration and inherited code credit Geekatplay Studio, ByteDance LatentSync and their upstream dependencies, including Whisper. Leon's changes are the fork-specific items listed above. The inherited author explicitly declared Apache 2.0 in the README; the exact declaration and full upstream notice are retained in `docs/licenses/`. This declaration is evidence for its own integration code, not every third-party file. Confirm upstream notices, vendored components and model terms before redistribution; this README does not grant new rights or replace those licenses.

On 2026-09-27, the [LatentSync code license](https://github.com/ByteDance/LatentSync/blob/main/LICENSE)
was Apache 2.0, while the [Hugging Face checkpoint repository](https://huggingface.co/ByteDance/LatentSync)
displayed `openrail++`. Code and model terms must be checked separately; neither
link resolves the provenance and notices of every file inherited by this fork.

See [vendored license provenance](docs/LICENSE_PROVENANCE.md) for component notices and unresolved redistribution boundaries.
