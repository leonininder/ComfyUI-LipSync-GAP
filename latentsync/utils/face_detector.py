import mediapipe as mp
import numpy as np
import os
from asset_integrity import verify_named_asset

# Try to import the new Tasks API
try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_TASKS_API = True
except ImportError:
    # Fallback for standard environments
    USE_TASKS_API = False

class FaceDetector:
    def __init__(self, device="cuda"):
        self.device = device
        
        if USE_TASKS_API:
            # New Tasks API Initialization
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights', 'face_landmarker.task')
            
            if not os.path.exists(model_path):
                raise RuntimeError(f"FaceDetector: Model file not found at {model_path}. Please execute the download script or check installation.")

            verify_named_asset("landmarker", model_path)
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            
            self.detector = vision.FaceLandmarker.create_from_options(options)
        else:
            # Legacy Solutions API Initialization
            if not hasattr(mp, 'solutions'):
                 raise ImportError("Mediapipe solutions not found and Tasks API could not be initialized.")
                 
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )

    def __call__(self, frame, threshold=0.5):
        # frame is RGB HWC numpy array
        frame = np.asarray(frame)

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                max_val = np.nanmax(frame)
                needs_scaling = np.isfinite(max_val) and max_val <= 1.0
                scale = 255.0 if needs_scaling else 1.0
                frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        h, w, _ = frame.shape
        
        if USE_TASKS_API:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = self.detector.detect(mp_image)
            
            if not detection_result.face_landmarks:
                return None, None, None
                
            face_landmarks_list = detection_result.face_landmarks[0]
            # face_landmarks_list is a list of NormalizedLandmark objects with x, y, z
            
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            # Compute bbox
            for lm in face_landmarks_list:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            bbox = [x_min, y_min, x_max, y_max]

            landmarks_full = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks_list], dtype=np.float32)
            
            # Compute Keypoints: Left Eye (468), Right Eye (473), Nose (4)
            if len(face_landmarks_list) > 473:
                kp_indices = [468, 473, 4]
            else:
                 kp_indices = [33, 263, 4] # Fallback
            
            keypoints = []
            for idx in kp_indices:
                lm = face_landmarks_list[idx]
                # Keep as float for consistency with original code
                x, y = lm.x * w, lm.y * h
                keypoints.append([x, y])
            
            # Return as numpy array float32
            return bbox, np.array(keypoints, dtype=np.float32), landmarks_full
            
        else:
            # Legacy Code
            results = self.face_mesh.process(frame)
            
            if not results.multi_face_landmarks:
                return None, None, None
                
            face_landmarks = results.multi_face_landmarks[0]
            
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
                
            bbox = [x_min, y_min, x_max, y_max]
            
            kp_indices = [468, 473, 4]
            if len(face_landmarks.landmark) <= 468:
                 kp_indices = [33, 263, 4]
                 
            keypoints = []
            for idx in kp_indices:
                lm = face_landmarks.landmark[idx]
                x, y = lm.x * w, lm.y * h
                keypoints.append([x, y])
                
            landmarks_full = np.array(
                [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark], dtype=np.float32
            )

            return bbox, np.array(keypoints, dtype=np.float32), landmarks_full
