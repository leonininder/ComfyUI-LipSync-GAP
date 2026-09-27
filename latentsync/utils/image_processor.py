# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Optional, Tuple, Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_rgb = cv2.imread(mask_image_path, cv2.IMREAD_COLOR)
    if mask_rgb is None:
        raise FileNotFoundError(f"Mask image not found at {mask_image_path}")

    # Keep behaviour identical to the upstream LatentSync implementation to avoid toggling mask polarity.
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
    mask_rgb = cv2.resize(mask_rgb, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4)
    mask_rgb = mask_rgb.astype(np.float32) / 255.0

    mask_tensor = torch.from_numpy(mask_rgb)
    mask_tensor = rearrange(mask_tensor, "h w c -> c h w")
    return mask_tensor


MOUTH_OUTER_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
MOUTH_INNER_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        self._prepare_mask_variants()

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    def _prepare_mask_variants(self):
        mask_rgb = torch.clamp(self.mask_image.float(), 0.0, 1.0)
        mask_gray = mask_rgb[0].cpu().numpy().astype(np.float32)

        binary_mask = (mask_gray > 0.5).astype(np.float32)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        dilated_mask = np.clip(dilated_mask, 0.0, 1.0)

        soft_mask = cv2.GaussianBlur(dilated_mask, (0, 0), sigmaX=3.0)
        soft_mask = np.clip(soft_mask, 0.0, 1.0)

        self.mask_latent_rgb = torch.from_numpy(np.repeat(dilated_mask[None, ...], 3, axis=0)).float()
        self.mask_blend = torch.from_numpy(soft_mask[None, ...]).float()

    def affine_transform(self, image) -> Tuple[torch.Tensor, list, torch.Tensor, Optional[torch.Tensor]]:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")

        detection = self.face_detector(image)
        if detection[0] is None:
            raise RuntimeError("Face not detected")

        bbox, landmarks3, landmarks_all = detection
        if bbox is None:
            raise RuntimeError("Face not detected")

        # landmarks3 provided directly by FaceDetector
        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face_tensor = rearrange(torch.from_numpy(face), "h w c -> c h w")

        mouth_mask = None
        if landmarks_all is not None and landmarks_all.shape[0] > max(max(MOUTH_OUTER_IDX), max(MOUTH_INNER_IDX)):
            aligned_landmarks = self._transform_landmarks_to_aligned_space(landmarks_all, affine_matrix)
            mouth_mask = self._build_dynamic_mouth_mask(aligned_landmarks)

        return face_tensor, box, affine_matrix, mouth_mask

    def _transform_landmarks_to_aligned_space(
        self, landmarks: np.ndarray, affine_matrix: torch.Tensor
    ) -> np.ndarray:
        matrix = affine_matrix.squeeze(0).detach().cpu().numpy()
        ones = np.ones((landmarks.shape[0], 1), dtype=np.float32)
        coords = np.concatenate([landmarks.astype(np.float32), ones], axis=1)
        aligned = coords @ matrix.T

        scale_x = self.resolution / self.restorer.face_size[0]
        scale_y = self.resolution / self.restorer.face_size[1]
        aligned[:, 0] *= scale_x
        aligned[:, 1] *= scale_y
        return aligned

    def _build_dynamic_mouth_mask(self, aligned_landmarks: np.ndarray) -> Optional[torch.Tensor]:
        try:
            outer = np.round(aligned_landmarks[MOUTH_OUTER_IDX]).astype(np.int32)
            inner = np.round(aligned_landmarks[MOUTH_INNER_IDX]).astype(np.int32)
        except IndexError:
            return None

        h = w = self.resolution
        outer[:, 0] = np.clip(outer[:, 0], 0, w - 1)
        outer[:, 1] = np.clip(outer[:, 1], 0, h - 1)
        inner[:, 0] = np.clip(inner[:, 0], 0, w - 1)
        inner[:, 1] = np.clip(inner[:, 1], 0, h - 1)

        canvas = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(canvas, [outer], 1.0)
        cv2.fillPoly(canvas, [inner], 0.0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        canvas = cv2.dilate(canvas, kernel, iterations=1)
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=1.5, sigmaY=1.5)
        canvas = np.clip(canvas, 0.0, 1.0)
        coverage = float(canvas.mean())
        if 0.01 <= coverage <= 0.22:
            return torch.from_numpy(canvas)

        # Fallback: build a compact elliptical mouth mask from the outer landmarks.
        fallback = np.zeros((h, w), dtype=np.float32)
        x_min, y_min = outer.min(axis=0)
        x_max, y_max = outer.max(axis=0)
        width = max(float(x_max - x_min), 1.0)
        height = max(float(y_max - y_min), 1.0)
        center_x = int(round((x_min + x_max) * 0.5))
        center_y = int(round((y_min + y_max) * 0.5))
        center = (
            int(np.clip(center_x, 0, w - 1)),
            int(np.clip(center_y, 0, h - 1)),
        )
        axes = (
            max(1, int(round(width * 0.55))),
            max(1, int(round(height * 0.65))),
        )
        cv2.ellipse(fallback, center, axes, 0, 0, 360, 1.0, thickness=-1)
        fallback = cv2.GaussianBlur(fallback, (0, 0), sigmaX=1.2, sigmaY=1.2)
        fallback = np.clip(fallback, 0.0, 1.0)
        return torch.from_numpy(fallback)

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False, mouth_mask: Optional[torch.Tensor] = None):
        if affine_transform:
            image, _, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        if mouth_mask is not None:
            mouth_mask = mouth_mask.to(device=pixel_values.device, dtype=pixel_values.dtype)
            if mouth_mask.dim() == 2:
                mouth_mask = mouth_mask.unsqueeze(0)
            blend_mask = mouth_mask
            mask_latent_rgb = mouth_mask.repeat(3, 1, 1)
        else:
            mask_latent_rgb = self.mask_latent_rgb.to(device=pixel_values.device, dtype=pixel_values.dtype)
            blend_mask = self.mask_blend.to(device=pixel_values.device, dtype=pixel_values.dtype)
        masked_pixel_values = pixel_values * mask_latent_rgb
        return pixel_values, masked_pixel_values, blend_mask

    def prepare_masks_and_masked_images(
        self,
        images: Union[torch.Tensor, np.ndarray],
        affine_transform=False,
        mouth_masks: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        if mouth_masks is not None:
            mouth_masks = torch.as_tensor(mouth_masks)
            if mouth_masks.dim() == 3:
                mouth_masks = mouth_masks.unsqueeze(1)
            mouth_masks = mouth_masks.to(dtype=torch.float32)

        results = []
        for idx, image in enumerate(images):
            override_mask = None
            if mouth_masks is not None:
                override_mask = mouth_masks[idx]
            pixel_values, masked_pixel_values, blend_mask = self.preprocess_fixed_mask_image(
                image, affine_transform=affine_transform, mouth_mask=override_mask
            )
            results.append((pixel_values, masked_pixel_values, blend_mask))

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
