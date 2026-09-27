import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('affine_adapter',Path(__file__).resolve().parents[1]/'latentsync/utils/affine_transform.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class AlignmentContract(unittest.TestCase):
 def test_anchors_and_inverse(self):
  obj=module.AlignRestore(resolution=64)
  src=np.array([[2,3],[20,4],[9,25]],np.float32)
  dst=np.array([[5,8],[45,10],[19,52]],np.float32)
  matrix,_=obj.transformation_from_points(src,dst,smooth=False)
  np.testing.assert_allclose(np.column_stack((src,np.ones(3)))@matrix.T,dst,atol=1e-5)
  inverse=module.cv2.invertAffineTransform(matrix)
  np.testing.assert_allclose(np.column_stack((dst,np.ones(3)))@inverse.T,src,atol=1e-5)
 def test_degenerate_rejected(self):
  obj=module.AlignRestore()
  for points in (np.zeros((3,2)),np.full((3,2),np.nan)):
   with self.assertRaises(ValueError):obj.transformation_from_points(points,obj.face_template)
 def test_restore_preserves_outside_and_is_finite(self):
  obj=module.AlignRestore(resolution=32)
  original=np.full((96,96,3),23,np.uint8)
  crop,matrix=obj.align_warp_face(original,obj.face_template+25,smooth=False)
  self.assertEqual(crop.shape,(obj.face_size[1],obj.face_size[0],3))
  result=obj.restore_img(original,torch.zeros(3,*crop.shape[:2]),matrix)
  self.assertTrue(np.isfinite(result).all())
  np.testing.assert_array_equal(result[0],original[0])
  self.assertGreater(result[45,45,0],23)
 def test_history_is_bounded(self):
  obj=module.AlignRestore()
  src=obj.face_template.copy()
  _,previous=obj.transformation_from_points(src,obj.face_template)
  _,updated=obj.transformation_from_points(src+10,obj.face_template,p_bias=previous)
  np.testing.assert_allclose(updated,src+8,atol=1e-5)
