"""Resolve Whisper checkpoints without importing the GPU runtime."""
from pathlib import Path
try:
    from .asset_integrity import verify_file
except ImportError:
    from asset_integrity import verify_file


def resolve_whisper(selection, model_dir, model_urls):
    alias = (selection or "").strip().lower()
    if alias.endswith(".pt"):
        alias = alias[:-3]
    # stage2_512.yaml expects 384-dimensional Whisper features.
    if alias not in ("tiny", "tiny.en") or alias not in model_urls:
        raise ValueError("This UNet configuration requires Whisper tiny or tiny.en (384 audio features).")
    checkpoint = Path(model_dir) / Path(model_urls[alias]).name
    if checkpoint.exists():
        verify_file(checkpoint, model_urls[alias].split("/")[-2])
        return str(checkpoint.resolve())
    return alias
