# Experimental release candidate: 0.1.0-rc1

Local, unpublished candidate on inherited base `7de1ab56604a8faee5e8c7e26b81fcb5a6fb64bd`. This label is not a published Git tag or PyPI release. Changes are inspectable in the working-tree diff. No supported production release or speech-quality benchmark is claimed.

## Candidate contents

Pinned and atomic asset downloader; read-only JSON preflight; compatible Whisper selection; strict audio resampling/slicing; correct output length and mask shape; failure cleanup; OpenCV alignment replacement; removal of unused training modules; source/notice inventory; 23 regression tests and a real 20-step GPU-to-MP4 fixture.

## Expected first engineering smoke

Use the Python environment that launches ComfyUI, from this extension directory. Install extension requirements into that environment; the optional fixture additionally needs scikit-image. FFmpeg and ffprobe must be on PATH. This workflow does not start a ComfyUI server.

```sh
python -m unittest discover -s tests -v
python preflight.py --comfy-root /path/to/ComfyUI --vae /path/to/sd_vae_ft_mse.safetensors
python smoke_runtime.py --comfy-root /path/to/ComfyUI --vae /path/to/sd_vae_ft_mse.safetensors --steps 20
```

Expected: 23 tests pass with Torch/OpenCV installed; preflight prints `PREFLIGHT_OK` with `inference_verified: false`; smoke prints `GPU_TENSOR_SMOKE_COMPLETED`, shape `[16,512,512,3]`, then export metadata for `debug_output/smoke.mp4`, 16 video frames and 0.640000 seconds. An exact video hash need not match across different drivers or libraries. The full suite requires runtime dependencies including Torch/OpenCV. Skipped tests or import failures do not count as release acceptance.

Measured environment, timing and hashes: [VALIDATION](VALIDATION.md). Approximately 5.151 GB of pinned assets plus a compatible VAE are required; first installation also needs CUDA/Python dependencies. Existing local cache makes downloads zero. One observed RTX 5080 16 GB run consumed 12.12 GB peak PyTorch allocations and took 105.25 seconds in the sampler. Neither this number nor a one-machine run defines a minimum GPU or compatibility matrix.

For actual speech, use the three-second workflow in the README and the [normal speech acceptance recipe](VALIDATION.md#reproducible-first-demo). Expected interface output is audio-duration-matched RGB frames plus the selected original audio. Sync quality and browser workflow import have not yet been accepted; do not advertise the pure-tone fixture as a speech demo.

## Privacy, failure and rollback

The local node does not upload media. Model installation downloads only the URLs in the manifest; first Whisper use can download its official checkpoint. The sampler writes a UUID-named temporary WAV and removes it in `finally`, including audio-write, generator and inference failure. Abrupt process termination can leave files in ComfyUI's temp directory; inspect and remove only your own `latentsync_*.wav` files. The optional smoke intentionally retains its public fixture exports under `debug_output/`.

Integrity failure preserves the existing corrupt file; inspect and explicitly move it aside before retrying. Resampling/empty slice failures stop inference. Missing face is an explicit runtime error. Preserve the previous extension folder before replacing it; rollback by restoring that folder and restarting ComfyUI. This candidate does not alter model files or the user's live install during its review.

## Publication and feedback

Proposed About: “Experimental ComfyUI LatentSync integration with verified model assets, strict audio contracts and a reproducible GPU smoke.” Topics: `comfyui`, `latentsync`, `lip-sync`, `experimental`, `pytorch`. These GitHub settings are proposed and have not been applied.

Comfy forum Custom Nodes draft: “I maintain an experimental fork of Geekatplay's LatentSync integration. This candidate adds hash-checked atomic model downloads and fixes mask, audio-slice and frame-count failures. A public-domain static portrait plus generated tone completes a 20-step GPU-to-MP4 smoke; it is not a speech-quality demo. I would appreciate reproducible installation or workflow reports, with environment details. Upstream credits, remaining limitations and reproduction commands are in the README.” Rules checked 2026-09-27: use the [Custom Nodes category](https://forum.comfy.org/c/general/4); its [category instructions](https://forum.comfy.org/t/about-the-custom-nodes-category/3) require searching existing topics, tagging the node name and including helpful logs. The [forum guidelines](https://forum.comfy.org/guidelines) prohibit spam, duplicate cross-posting and unauthorized media. This is a specific technical node discussion, not a mass promotion campaign. No numeric self-promotion quota is published in those sources. Nothing has been posted.

Report bugs using the repository issue template. Track date/channel, views, clones, completed smoke attempts, environment, failure stage and actionable reports. Stars and adoption remain unknown until measured. Before any broader release, independently review the candidate diff, notices and model terms; reproduce on a fresh environment; import the browser workflow; evaluate a licensed speech fixture. No third-party licensing rights are invented: [source inventory](LICENSE_PROVENANCE.md).
