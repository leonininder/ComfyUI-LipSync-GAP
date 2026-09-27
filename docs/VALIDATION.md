# Validation and public-demo acceptance recipe

## Current evidence

Base revision: `7de1ab56604a8faee5e8c7e26b81fcb5a6fb64bd` (upstream-derived initial beta).

Twenty-three regression tests pass in the existing ComfyUI Python environment: asset integrity/atomic publication, missing-environment preflight, checkpoint selection, resampling failure, mask dimensions, output duration, OpenCV geometry, empty slice rejection and write/generator failure cleanup. These tests do not establish lip-sync quality.

## Measured GPU smoke (2026-09-27)

An existing RTX 5080 (16 GB), Python 3.13.12, PyTorch 2.12.1+cu130, CUDA 13.0, Diffusers 0.41.0.dev0, MediaPipe 1.0.1 and OpenCV 5.0.0 completed a 16-frame, 512x512, 25 fps tensor smoke through the actual node loader and sampler. The pinned LatentSync 1.6 and tiny checkpoint hashes passed. `sd_vae_ft_mse.safetensors` was used from the existing local installation: SHA-256 `a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815`. ComfyUI commit: `e638023d54497dbe0579565e5de4bb7076899592`.

The fixture is the NASA public-domain astronaut still supplied by scikit-image plus a generated 220 Hz sine tone. After the OpenCV replacement, 20 diffusion steps returned 16 finite RGB frames in 105.25 seconds for the sampler call; peak PyTorch allocated CUDA memory was 12,123,103,232 bytes (not total system VRAM). FFmpeg exported an MP4 with 16 video frames, one audio stream and a 0.640000-second duration. SHA-256: `c38d74ceb14f1cc7f7485bbef2b324bac1275161151c49239925727d7efe75cb`. This is a technical smoke, not a speech or visual-quality benchmark. Earlier runs showed a shutdown ModelPatcher warning; the final 20-step run exited 0 without that warning after explicit model reference cleanup. No speech or visual equivalence score is claimed for the changed affine geometry.

Run from the extension directory using the existing ComfyUI interpreter:

```sh
python smoke_runtime.py --comfy-root /path/to/ComfyUI --vae /path/to/sd_vae_ft_mse.safetensors --steps 20
```

This optional smoke also needs scikit-image and FFmpeg/ffprobe; it uses no private media and installs no packages. It writes the test movie under `debug_output/`. It does not import a workflow in the ComfyUI browser or launch a server.

**Pending:** a clean installation, browser workflow import, meaningful speech and video-quality evaluation, a supported release dependency matrix, and license closure for all inherited components. See `LICENSE_PROVENANCE.md`. The existing environment includes a development Diffusers version and different Torch/torchaudio release numbers; one completed fixture is not a general compatibility guarantee.

## Reproducible first demo

Use one consented/licensed three-second, single-speaker frontal clip and a replacement speech file. Publish the input and output only with redistribution permission. Record:

- Git commit, OS, GPU and driver; ComfyUI, VideoHelperSuite, Python, PyTorch and CUDA versions.
- Exact UNet, Whisper and VAE filenames, source URLs and SHA-256 hashes.
- Input hashes, resolution, frame count, audio duration and sample rate.
- Workflow JSON; seed 1247, 20 steps, guidance 1.5, FP16, 25 fps and resize limit 512.
- First-run download/setup time separately from warm inference time; measured peak VRAM.
- Output frame count, duration, speech alignment, visible face/background artifacts and failure log.

Export the original and result side by side with audible speech. Include a difficult case (profile/occlusion) alongside the best case so viewers can judge applicability. Do not generate a simulated output and label it an inference demo.

## Acceptance gates

1. A new environment can install from the documented instructions with all external node dependencies stated.
2. The workflow imports, each placeholder is selectable, and execution completes with the recorded files.
3. Audio/frame duration agrees with the chosen trim and fps; a reviewer inspects sync and artifacts.
4. A second reviewer reproduces the documented result using the same assets and hashes.
5. License/notice review is completed before packaging or redistributing code and model assets.

Only after these pass should the README replace pending status with a dated measured result. Keep the smallest supported dependency matrix first; do not promise every GPU or newer package combination.

## Distribution experiment

Publish one reproducible demo and a short explanation of the fork's checkpoint-selection fix in a relevant ComfyUI community where project sharing is allowed. Keep upstream attribution visible. Measure repository visitors, clone counts, successful first runs and actionable issues before interpreting star changes. Compare equal observation windows; no claim of guaranteed star growth is justified. The repository About description/topics are separate GitHub settings, not changed by README edits.
