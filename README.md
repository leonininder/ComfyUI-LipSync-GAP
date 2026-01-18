# ComfyUI-LipSync-GAP (Geekatplay Studio)

ComfyUI-LipSync-GAP is Geekatplay Studio's production-ready port of ByteDance's LatentSync pipeline for ComfyUI. The add-on delivers high-quality lip synchronization, smooth mouth articulation, and VRAM-efficient inference tuned for creators who expect studio-grade output inside ComfyUI workflows.

> **Beta Notice**
> This release remains experimental and under active development; expect rough edges, and please report stability problems as we iterate.

---

## Key Features
- **LatentSync v1.6 Core** – Leverages the official UNet and Whisper audio encoder for accurate lip motion.
- **Geekatplay Optimizations** – In-memory video flow, FP8 support, optional DeepCache integration, and robust whisper alias handling.
- **Smart Masking** – Feathered lip masks and stability guards prevent visual artifacts or frame spikes.
- **Workflow Ready** – Ships with sample ComfyUI workflows and auto-download helpers for first run.

---

## Requirements
- ComfyUI 0.9.0 or newer (Windows portable or Python install).
- NVIDIA GPU with at least 12 GB VRAM (FP8 mode helps on smaller cards).
- Python packages installed via the included installer (diffusers, mediapipe, torch, etc.).

---

## Installation
1. **Clone or download** this repository into `ComfyUI/custom_nodes/` so the folder name is `ComfyUI-LipSync-GAP`.
2. **Run the installer** from the repo folder:
   ```bat
   install.bat
   ```
   The script locates the embedded ComfyUI Python (or your system interpreter) and installs dependencies, LatentSync models, and the Mediapipe face landmarker. If automated model pulls fail, the script prints manual download instructions.

---

## Required Models
The installer attempts to download everything automatically. To set up manually:
- `ComfyUI/models/latentsync/latentsync_unet.pt`
- `ComfyUI/models/latentsync/whisper/<alias>.pt` (tiny/small/medium/large)
- `custom_nodes/ComfyUI-LipSync-GAP/latentsync/weights/face_landmarker.task`

Ensure folders exist before copying files.

---

## Quick Start
1. Launch ComfyUI and restart after installation.
2. Load `examples/workflow_lipsync_gap.json` (rename your existing workflow or use the provided sample).
3. Nodes appear under **Geekatplay Studio/LipSync GAP**:
   - **LipSync GAP Model Loader** – Select UNet, Whisper alias, and precision.
   - **LipSync GAP Sampler** – Connect video frames, audio input, and tweak guidance/steps.
4. Use 20–30 sampler steps, guidance scale ~1.5, and match `frame_rate` to the source video.

---

## Configuration Highlights
- **Precision** – `fp16` for highest fidelity, `fp8_e4m3fn` for lower VRAM usage.
- **Video Resize Limit** – Optional automatic down-scaling keeps inference stable on high-res footage.
- **Audio Trim Controls** – Offset and duration inputs let you sync subsections without pre-editing audio.

---

## Troubleshooting
- **Nodes Missing** – Restart ComfyUI after copying the repo; confirm the folder path matches `custom_nodes/ComfyUI-LipSync-GAP`.
- **Model Not Found** – Verify `latentsync_unet.pt` and Whisper checkpoints are in `ComfyUI/models/latentsync/` (the installer creates folders for you).
- **VRAM Errors** – Switch the Model Loader to `fp8_e4m3fn`, reduce video resolution, or process shorter clips.
- **Audio Too Quiet/Flat** – Confirm you selected the desired Whisper size (`tiny`, `small`, `medium`, `large`). Larger models yield stronger mouth articulation.

---

## Repository Layout
```
ComfyUI-LipSync-GAP/
├─ install.bat                # Dependency + model bootstrap
├─ nodes.py                   # ComfyUI node implementations
├─ latentsync/                # Adapted LatentSync pipeline
├─ checkpoints/               # Placeholder for future packaged assets
├─ examples/                  # Example workflows
└─ README.md                  # This file
```

---

## Credits & License
- **Research & Base Pipeline** – [ByteDance LatentSync](https://github.com/bytedance/LatentSync)
- **Adaptation & Enhancements** – Geekatplay Studio
- **License** – Apache 2.0 (see `LICENSE` files where applicable)

For support, open an issue at [GeekatplayStudio/ComfyUI-LipSync-GAP](https://github.com/GeekatplayStudio/ComfyUI-LipSync-GAP/issues) or join the Geekatplay Studio community channels.
