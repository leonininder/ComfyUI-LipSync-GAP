"""Pinned asset verification and atomic downloads. No model deserialization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent


def verify_file(path, digest, size=None):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing asset: {path}")
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Asset size mismatch: {path}")
    actual = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            actual.update(chunk)
    if actual.hexdigest() != digest:
        raise ValueError(f"Asset SHA-256 mismatch: {path}")


def download_asset(url, destination, digest, size, opener=urllib.request.urlopen):
    destination = Path(destination)
    # Never silently overwrite a corrupt existing file; preserve it for diagnosis.
    if destination.exists():
        verify_file(destination, digest, size)
        return 'verified-existing'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name+'.', suffix='.part', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output, opener(url, timeout=60) as response:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if size is not None and total > size:
                    raise ValueError('Download exceeds pinned size')
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        verify_file(temporary, digest, size)
        os.replace(temporary, destination)
        return 'downloaded-verified'
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_manifest():
    payload = json.loads((ROOT / 'assets-manifest.json').read_text())
    assets = payload.get('assets') if isinstance(payload, dict) else None
    if not isinstance(assets, list) or not assets:
        raise ValueError('Asset manifest must contain a nonempty assets list')
    names = set()
    for asset in assets:
        if not isinstance(asset, dict) or not all(k in asset for k in ('name','root','relative','sha256','size','url')):
            raise ValueError('Incomplete asset manifest entry')
        if not isinstance(asset['relative'], str) or not asset['relative']:
            raise ValueError('Asset path must be a nonempty string')
        relative = Path(asset['relative'])
        if relative.is_absolute() or relative.drive or '..' in relative.parts:
            raise ValueError('Asset path must stay relative to its declared root')
        if asset['root'] not in ('comfy','extension') or not isinstance(asset['size'], int) or asset['size'] <= 0:
            raise ValueError('Invalid asset root or size')
        if not isinstance(asset['sha256'], str) or len(asset['sha256']) != 64 or any(c not in '0123456789abcdef' for c in asset['sha256']):
            raise ValueError('Expected lowercase SHA-256 digest')
        if not isinstance(asset['url'], str) or not asset['url'].startswith('https://'):
            raise ValueError('Asset URL must use HTTPS')
        if not isinstance(asset['name'],str) or asset['name'] in names:
            raise ValueError('Asset names must be unique strings')
        names.add(asset['name'])
    if names != {'unet', 'whisper', 'landmarker'}:
        raise ValueError('Manifest must contain exactly unet, whisper and landmarker')
    return assets


def verify_named_asset(name, path):
    assets = load_manifest()
    asset = next(a for a in assets if a['name'] == name)
    verify_file(path, asset['sha256'], asset['size'])


def manifest_assets(comfy_root):
    for asset in load_manifest():
        root = Path(comfy_root) if asset['root'] == 'comfy' else ROOT
        yield asset, root / asset['relative']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', required=True, type=Path)
    parser.add_argument('--download', action='store_true', help='Explicitly allow approximately 5.15 GB of model downloads')
    args = parser.parse_args()
    if not (args.comfy_root / 'folder_paths.py').is_file():
        parser.error('--comfy-root must contain folder_paths.py')
    failures = []
    for asset, path in manifest_assets(args.comfy_root):
        try:
            if args.download:
                status = download_asset(asset['url'], path, asset['sha256'], asset['size'])
            else:
                verify_file(path, asset['sha256'], asset['size'])
                status = 'verified'
            print(asset['name'], status)
        except (OSError, ValueError) as error:
            failures.append(str(error))
            print('ERROR:', error)
    return 1 if failures else 0

if __name__ == '__main__':
    raise SystemExit(main())
