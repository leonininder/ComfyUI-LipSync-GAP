"""Regression for the actual mask-preparation method; needs the inference environment."""
import ast
from pathlib import Path
import types
import unittest
try:
    import torch
    import cv2
    import numpy as np
    AVAILABLE=True
except ImportError:
    AVAILABLE=False

@unittest.skipUnless(AVAILABLE, 'requires torch, numpy and OpenCV')
class MaskTests(unittest.TestCase):
    def test_rgb_mask_has_2d_opencv_plane(self):
        source=Path(__file__).parents[1]/'latentsync/utils/image_processor.py'
        tree=ast.parse(source.read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_prepare_mask_variants')
        namespace={'torch':torch,'cv2':cv2,'np':np}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),namespace)
        obj=types.SimpleNamespace(mask_image=torch.zeros(3,64,64))
        obj.mask_image[:,24:40,24:40]=1
        namespace['_prepare_mask_variants'](obj)
        self.assertEqual(tuple(obj.mask_latent_rgb.shape),(3,64,64))
        self.assertEqual(tuple(obj.mask_blend.shape),(1,64,64))
        self.assertTrue(torch.isfinite(obj.mask_blend).all())
