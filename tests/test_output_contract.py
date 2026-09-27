import unittest
from output_contract import align_output_frames

class OutputContractTests(unittest.TestCase):
    def test_whisper_padding_is_not_exported(self):
        self.assertEqual(len(align_output_frames(list(range(18)),10240,16000,25)),16)
    def test_partial_frame_rounds_up(self):
        self.assertEqual(len(align_output_frames(list(range(20)),10241,16000,25)),17)
    def test_short_inference_and_empty_audio_rejected(self):
        with self.assertRaises(RuntimeError): align_output_frames([1],16000,16000,25)
        with self.assertRaises(ValueError): align_output_frames([1],0,16000,25)
