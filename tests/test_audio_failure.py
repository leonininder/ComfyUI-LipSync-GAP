"""Exercise the sampler's actual resampling branch without importing CUDA modules."""
import ast
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class AudioFailureTests(unittest.TestCase):
    def test_resampling_failure_stops_inference(self):
        tree = ast.parse((Path(__file__).parents[1] / "nodes.py").read_text())
        branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "sample_rate != target_sr")
        code = compile(ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[])),
                       "nodes.py:audio-resampling", "exec")
        def broken_resampler(**kwargs):
            raise OSError("simulated incompatible audio backend")
        fake = types.SimpleNamespace(transforms=types.SimpleNamespace(Resample=broken_resampler))
        with patch.dict(sys.modules, {"torchaudio": fake}):
            with self.assertRaisesRegex(RuntimeError, "inference stopped") as failure:
                exec(code, {"sample_rate": 48000, "target_sr": 16000, "waveform": object()})
        self.assertIsInstance(failure.exception.__cause__, OSError)

if __name__ == "__main__":
    unittest.main()

class AudioBoundaryTests(unittest.TestCase):
    def test_empty_slice_rejected(self):
        tree=ast.parse((Path(__file__).parents[1]/'nodes.py').read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='start_idx == end_idx')
        code=compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),'nodes:empty-slice','exec')
        with self.assertRaisesRegex(ValueError,'slice is empty'):
            exec(code,{'start_idx':16000,'end_idx':16000})

    def test_partial_write_and_generator_failure_remove_temp(self):
        import tempfile,os
        tree=ast.parse((Path(__file__).parents[1]/'nodes.py').read_text())
        body=next(n.body for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='sample')
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Try) and 'sf.write(' in ast.unparse(n))
        end=next(i for i,n in enumerate(body) if i>=start and isinstance(n,ast.Try) and n.finalbody and 'os.remove(temp_audio_path)' in ast.unparse(n))
        code=compile(ast.fix_missing_locations(ast.Module(body=body[start:end+1],type_ignores=[])),'nodes:temp-cleanup','exec')
        for fail_write in (False,True):
            with self.subTest(fail_write=fail_write),tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'audio.wav'
                def writer(*args):
                    path.write_bytes(b'partial')
                    if fail_write:raise OSError('write failed')
                def fail(*args,**kwargs):raise OSError('backend failed')
                context={'sf':types.SimpleNamespace(write=writer),'temp_audio_path':str(path),'waveform_np':object(),'waveform':object(),'sample_rate':16000,'torch':types.SimpleNamespace(Generator=fail),'device':'cuda','seed':1,'os':os}
                with patch.dict(sys.modules,{'torchaudio':types.SimpleNamespace(save=fail)}),self.assertRaises(OSError):exec(code,context)
                self.assertFalse(path.exists())
