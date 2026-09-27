import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from asset_integrity import download_asset, verify_file
from preflight import inspect

DATA = b'valid model bytes'
DIGEST = hashlib.sha256(DATA).hexdigest()

class IntegrityTests(unittest.TestCase):
    def test_atomic_success_and_existing_cache_never_downloaded(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'model.pt'
            self.assertEqual(download_asset('fixture',path,DIGEST,len(DATA),lambda *a,**k:io.BytesIO(DATA)), 'downloaded-verified')
            self.assertEqual(download_asset('fixture',path,DIGEST,len(DATA),lambda *a,**k:self.fail('network used')), 'verified-existing')
            self.assertEqual(list(Path(root).glob('*.part')), [])

    def test_truncated_wrong_hash_and_oversize_never_published(self):
        for data in (DATA[:-1], b'x'*len(DATA), DATA+b'x'):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as root:
                path=Path(root)/'model.pt'
                with self.assertRaises(ValueError):
                    download_asset('fixture',path,DIGEST,len(DATA),lambda *a,**k:io.BytesIO(data))
                self.assertFalse(path.exists())
                self.assertEqual(list(Path(root).iterdir()), [])

    def test_network_failure_cleans_temp(self):
        def fail(*a,**k): raise OSError('network interrupted')
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'model.pt'
            with self.assertRaises(OSError): download_asset('fixture',path,DIGEST,len(DATA),fail)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_corrupt_existing_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'model.pt'; path.write_bytes(b'broken')
            with self.assertRaises(ValueError):
                download_asset('fixture',path,DIGEST,len(DATA),lambda *a,**k:self.fail('network used'))
            self.assertEqual(path.read_bytes(),b'broken')

    def test_missing_environment_cannot_pass(self):
        with tempfile.TemporaryDirectory() as root:
            result=inspect(Path(root),None,check_runtime=False)
            self.assertEqual(result['status'],'BLOCKED')
            self.assertFalse(result['inference_verified'])
            self.assertFalse(next(c['ok'] for c in result['checks'] if c['name']=='vae_file'))

    def test_ready_static_fixture_does_not_claim_inference(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); (root/'folder_paths.py').touch()
            vae=root/'vae'; vae.write_bytes(b'fixture')
            with patch('preflight.manifest_assets',return_value=[]), patch('preflight.shutil.which',return_value='ffmpeg'):
                result=inspect(root,vae,check_runtime=False)
            self.assertEqual(result['status'],'PREFLIGHT_OK')
            self.assertFalse(result['inference_verified'])

class BrokenInstallationTests(unittest.TestCase):
    def test_missing_config_reports_blocked_json(self):
        with tempfile.TemporaryDirectory() as folder, patch('preflight.ROOT',Path(folder)):
            result=inspect(Path(folder),None,check_runtime=False)
            self.assertEqual(result['status'],'BLOCKED')
            self.assertFalse(next(c['ok'] for c in result['checks'] if c['name']=='audio_features'))
            self.assertFalse(result['inference_verified'])

    def test_malformed_manifests_report_blocked_json(self):
        for payload in ('{','{}','{"assets":[]}','{"assets":[{}]}'):
            with self.subTest(payload=payload),tempfile.TemporaryDirectory() as folder,patch('asset_integrity.ROOT',Path(folder)):
                (Path(folder)/'assets-manifest.json').write_text(payload)
                result=inspect(Path(folder),None,check_runtime=False)
                self.assertEqual(result['status'],'BLOCKED')
                self.assertFalse(next(c['ok'] for c in result['checks'] if c['name']=='asset_manifest'))
    def test_manifest_schema_and_required_assets(self):
        import json
        import asset_integrity
        original=json.loads((asset_integrity.ROOT/'assets-manifest.json').read_text())
        import copy
        variants=[]
        x=copy.deepcopy(original);x['assets'].pop();variants.append(x)
        x=copy.deepcopy(original);del x['assets'][0]['sha256'];variants.append(x)
        x=copy.deepcopy(original);x['assets'][0]['root']='unknown';variants.append(x)
        x=copy.deepcopy(original);x['assets'][0]['relative']='../outside';variants.append(x)
        for payload in variants:
            with self.subTest(payload=payload),tempfile.TemporaryDirectory() as folder,patch('asset_integrity.ROOT',Path(folder)):
                (Path(folder)/'assets-manifest.json').write_text(json.dumps(payload))
                result=inspect(Path(folder),None,check_runtime=False)
                self.assertFalse(next(c['ok'] for c in result['checks'] if c['name']=='asset_manifest'))
