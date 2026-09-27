"""Read-only preflight. Readiness is not an inference-quality result."""
import argparse
import importlib
import json
from pathlib import Path
import re
import shutil
from asset_integrity import ROOT, manifest_assets, verify_file


def inspect(comfy_root, vae, check_runtime=True):
    checks = []
    def add(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=str(detail)))
    add('comfy_root', (comfy_root/'folder_paths.py').is_file(), comfy_root)
    try:
        config = (ROOT/'configs/unet/stage2_512.yaml').read_text()
        add('audio_features', bool(re.search(r'cross_attention_dim:\s*384\b', config)), 'Expected 384')
    except (OSError, UnicodeError) as error:
        add('audio_features', False, error)
    add('vae_file', vae is not None and vae.is_file() and vae.stat().st_size > 0,
        vae or 'Specify --vae; file presence is not VAE compatibility')
    add('ffmpeg', shutil.which('ffmpeg') is not None, shutil.which('ffmpeg') or 'Missing from PATH')
    try:
        assets = list(manifest_assets(comfy_root))
    except (OSError, ValueError, KeyError, TypeError) as error:
        add("asset_manifest", False, error)
        assets = []
    for asset, path in assets:
        try:
            verify_file(path, asset['sha256'], asset['size'])
            add(asset['name'], True, 'Pinned SHA-256 and size verified')
        except (OSError, ValueError) as error:
            add(asset['name'], False, error)
    if check_runtime:
        for name in ('torch','torchaudio','diffusers','mediapipe','omegaconf','cv2','soundfile','einops','transformers'):
            try:
                module = importlib.import_module(name)
                add(name, True, getattr(module, '__version__', 'imported'))
                if name == 'torch':
                    add('cuda', module.cuda.is_available(), module.version.cuda)
            except Exception as error:
                add(name, False, f'{type(error).__name__}: {error}')
    return {'status': 'PREFLIGHT_OK' if all(c['ok'] for c in checks) else 'BLOCKED',
            'inference_verified': False, 'checks': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', required=True, type=Path)
    parser.add_argument('--vae', type=Path)
    args = parser.parse_args()
    result = inspect(args.comfy_root, args.vae)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PREFLIGHT_OK' else 1

if __name__ == '__main__':
    raise SystemExit(main())
