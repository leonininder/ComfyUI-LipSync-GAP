"""OpenCV affine adapter implemented for this fork, replacing the StyleSync adapter.

API references: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
The crop dimensions/anchor coordinates preserve the inherited model input contract.
This replacement is not a claim about historical redistribution permissions.
"""
import cv2
import numpy as np
import torch


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


class AlignRestore:
    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float16):
        if align_points != 3 or resolution < 1:
            raise ValueError("Alignment requires three anchors and a positive resolution")
        self.device, self.dtype = device, dtype
        scale = resolution * 2.8 / 256
        self.face_template = np.array([[17, 20], [58, 20], [37.5, 40]], np.float32) * scale
        self.face_size = (int(75 * scale), int(100 * scale))
        self.p_bias = None

    def transformation_from_points(self, points1, points0, smooth=True, p_bias=None):
        source, target = _array(points1), _array(points0)
        for points in (source, target):
            if points.shape != (3, 2) or not np.isfinite(points).all():
                raise ValueError("Expected three finite 2D anchors")
            if abs(np.linalg.det(np.column_stack((points, np.ones(3))))) < 1e-5:
                raise ValueError("Alignment anchors must not be collinear")
        # Smooth the measured anchors, not scale/rotation matrix coefficients.
        if smooth and p_bias is not None:
            previous = _array(p_bias)
            if previous.shape != (3, 2) or not np.isfinite(previous).all():
                raise ValueError("Invalid alignment history")
            source = 0.8 * source + 0.2 * previous
        matrix = cv2.getAffineTransform(source.astype(np.float32), target)
        if abs(np.linalg.det(matrix[:, :2])) < 1e-8:
            raise ValueError("Alignment transform is singular")
        return matrix.astype(np.float32), source.copy() if smooth else None

    def align_warp_face(self, img, landmarks3, smooth=True):
        matrix, self.p_bias = self.transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias)
        crop = cv2.warpAffine(img, matrix, self.face_size,
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(127, 127, 127))
        return crop, torch.as_tensor(matrix, device=self.device, dtype=torch.float32).unsqueeze(0)

    def restore_img(self, input_img, face, affine_matrix):
        matrix = _array(affine_matrix).reshape(2, 3)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
            raise ValueError("Cannot restore a nonfinite or singular affine transform")
        pixels = _array(face)
        if pixels.ndim != 3 or pixels.shape[0] != 3 or not np.isfinite(pixels).all():
            raise ValueError("Expected finite normalized CHW face")
        pixels = np.clip((pixels.transpose(1, 2, 0) + 1) * 127.5, 0, 255)
        if (pixels.shape[1], pixels.shape[0]) != self.face_size:
            pixels = cv2.resize(pixels, self.face_size)
        inverse = cv2.invertAffineTransform(matrix)
        size = (input_img.shape[1], input_img.shape[0])
        restored = cv2.warpAffine(pixels, inverse, size, flags=cv2.INTER_LINEAR)
        # Feather inside the crop, keeping all pixels outside its support unchanged.
        alpha = np.ones((self.face_size[1], self.face_size[0]), np.float32)
        margin = max(1, min(self.face_size) // 40)
        alpha[:margin] = 0; alpha[-margin:] = 0
        alpha[:, :margin] = 0; alpha[:, -margin:] = 0
        alpha = cv2.GaussianBlur(alpha, (margin * 2 + 1, margin * 2 + 1), 0)
        alpha = cv2.warpAffine(alpha, inverse, size, flags=cv2.INTER_LINEAR)[..., None]
        return np.clip(restored * alpha + input_img.astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)
