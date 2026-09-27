import hashlib
import tempfile
import unittest
from pathlib import Path
from model_selection import resolve_whisper

DIGEST = hashlib.sha256(b"fixture").hexdigest()
URLS = {"tiny": f"https://example.test/{DIGEST}/tiny.pt", "tiny.en": "https://example.test/tiny.en.pt", "small": "https://example.test/small.pt"}

class WhisperSelectionTests(unittest.TestCase):
    def test_existing_checkpoint_prevents_alias_redownload(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "tiny.pt"
            checkpoint.write_bytes(b"fixture")
            self.assertEqual(resolve_whisper(" Tiny.PT ", root, URLS), str(checkpoint.resolve()))

    def test_missing_checkpoint_retains_supported_download_alias(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(resolve_whisper("tiny", root, URLS), "tiny")
            self.assertEqual(resolve_whisper("tiny.en", root, URLS), "tiny.en")

    def test_incompatible_or_unknown_model_rejected(self):
        for selection in ("small", "medium.pt", "large", "../tiny", "", None):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "384"):
                resolve_whisper(selection, ".", URLS)

if __name__ == "__main__":
    unittest.main()
