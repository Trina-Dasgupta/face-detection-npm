import cv2
import numpy as np
import mediapipe as mp
import base64
from typing import Dict, Any, List

class FaceAnalyzer:
    def __init__(self):
        # Initialize face detection and MediaPipe models
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.8
        )

    def analyze_face(self, image_data: str) -> Dict[str, Any]:
        """Analyze face visibility and quality in the image."""
        image = self._decode_image(image_data)
        if image is None:
            return self._error_result("Invalid image data")

        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Run face detection first (faster than mesh)
        detection_result = self.mp_face_detection.process(image_rgb)
        if not detection_result.detections:
            return self._error_result("No face detected")

        # Get face mesh for detailed analysis
        mesh_result = self.mp_face_mesh.process(image_rgb)
        if not mesh_result.multi_face_landmarks:
            return self._error_result("Could not analyze facial features")

        landmarks = mesh_result.multi_face_landmarks[0].landmark
        points = np.array([(lm.x * width, lm.y * height) for lm in landmarks])

        # Run comprehensive quality checks
        quality_results = self._check_face_quality(image, points)
        issues: List[str] = []

        # Check lighting
        if quality_results['lighting_score'] < 0.6:
            issues.append("Please move to a brighter area")

        # Check glasses
        if quality_results['has_glasses']:
            issues.append("Please remove sunglasses/glasses")

        # Check face coverage (mask, hand, etc.)
        if quality_results['face_coverage_score'] < 0.85 or quality_results['mouth_visible_score'] < 0.75:
            if quality_results['mouth_visible_score'] < 0.75:
                issues.append("Please remove hand or face covering from mouth area")
            else:
                issues.append("Please ensure face is fully visible")

        # Check face angle/pose
        if quality_results['pose_score'] < 0.8:
            issues.append("Please face the camera directly")

        # Check if one side is hidden/blocked
        if quality_results['symmetry_score'] < 0.85:
            issues.append("Please show your entire face (both sides visible)")

        # Final visibility determination
        is_visible = (
            quality_results['lighting_score'] >= 0.6 and
            not quality_results['has_glasses'] and
            quality_results['face_coverage_score'] >= 0.85 and
            quality_results['pose_score'] >= 0.8 and
            quality_results['symmetry_score'] >= 0.85
        )

        return {
            'isDetected': True,
            'isVisible': is_visible,
            'score': quality_results['overall_score'] * 100,
            'message': '; '.join(issues) if issues else 'Face properly visible',
            'landmarks': points.tolist(),
            'details': {
                'lighting': quality_results['lighting_score'],
                'pose': quality_results['pose_score'],
                'symmetry': quality_results['symmetry_score'],
                'face_coverage': quality_results['face_coverage_score'],
                'has_glasses': quality_results['has_glasses']
            }
        }

    def _detect_face(self, image):
        # Get face detection
        faces = self.face_detector(image)
        if len(faces) == 0:
            return {'success': False, 'message': 'No face detected'}

        # Get facial landmarks
        landmarks = self.landmark_predictor(image, faces[0])
        points = np.array([[p.x, p.y] for p in landmarks.parts()])

        return {
            'success': True,
            'face': faces[0],
            'landmarks': points
        }

    def _check_face_quality(self, image: np.ndarray, landmarks: np.ndarray) -> Dict[str, float]:
        """Run comprehensive face quality checks."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check lighting
        lighting_score = self._check_lighting(gray)
        
        # Check for glasses
        has_glasses = self._detect_glasses(gray, landmarks)
        
        # Check face symmetry and coverage
        symmetry_score = self._check_symmetry(landmarks)
        face_coverage = self._check_face_coverage(landmarks)
        pose_score = self._check_pose(landmarks)
        mouth_visible = self._check_mouth_visibility(landmarks)

        # Calculate overall quality score
        overall_score = np.mean([
            lighting_score,
            1.0 if not has_glasses else 0.0,
            symmetry_score,
            face_coverage,
            pose_score
        ])

        return {
            'lighting_score': lighting_score,
            'has_glasses': has_glasses,
            'symmetry_score': symmetry_score,
            'face_coverage_score': face_coverage,
            'pose_score': pose_score,
            'mouth_visible_score': mouth_visible,
            'overall_score': overall_score
        }

    def _check_lighting(self, gray_image: np.ndarray) -> float:
        """Check if lighting is adequate."""
        mean_brightness = np.mean(gray_image)
        std_brightness = np.std(gray_image)
        
        # Normalize brightness score (ideal range: 100-200)
        brightness_score = np.clip((mean_brightness - 50) / 150, 0, 1)
        # Normalize contrast score (ideal range: 40-100)
        contrast_score = np.clip(std_brightness / 80, 0, 1)
        
        return 0.7 * brightness_score + 0.3 * contrast_score

    def _check_pose(self, landmarks: np.ndarray) -> float:
        """Check if face is front-facing using landmark positions."""
        # Use nose and eyes to estimate pose
        nose_tip = landmarks[4]
        left_eye = np.mean(landmarks[[33, 133]], axis=0)
        right_eye = np.mean(landmarks[[362, 263]], axis=0)
        
        # Calculate face direction vector
        face_normal = nose_tip - np.mean([left_eye, right_eye], axis=0)
        
        # Check horizontal pose (yaw)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        nose_deviation = abs(nose_tip[0] - np.mean([left_eye[0], right_eye[0]]))
        yaw_score = max(0, 1.0 - (nose_deviation / (eye_distance * 0.5)))

        # Check vertical pose (pitch)
        eyes_center = np.mean([left_eye, right_eye], axis=0)
        vertical_angle = abs(np.degrees(np.arctan2(nose_tip[1] - eyes_center[1],
                                                  nose_tip[0] - eyes_center[0])))
        pitch_score = max(0, 1.0 - (abs(vertical_angle - 90) / 45))

        return min(yaw_score, pitch_score)

    def _detect_glasses(self, gray_image: np.ndarray, landmarks: np.ndarray) -> bool:
        """Detect presence of glasses using eye region analysis."""
        def get_eye_region(eye_landmarks):
            eye_region = landmarks[eye_landmarks].astype(np.int32)
            x, y = eye_region[:, 0], eye_region[:, 1]
            return np.s_[min(y):max(y), min(x):max(x)]

        # MediaPipe landmark indices for eyes
        left_eye = [33, 133, 157, 158, 159, 160, 161, 246]  # Left eye perimeter
        right_eye = [362, 263, 384, 385, 386, 387, 388, 466]  # Right eye perimeter

        left_region = get_eye_region(left_eye)
        right_region = get_eye_region(right_eye)

        # Check for strong edges in eye regions (indicates glasses)
        def check_eye_region(region):
            if region[0].start >= region[0].stop or region[1].start >= region[1].stop:
                return 0
            eye_area = gray_image[region]
            edges = cv2.Sobel(eye_area, cv2.CV_64F, 1, 1, ksize=3)
            return np.mean(np.abs(edges)) > 30

        return check_eye_region(left_region) or check_eye_region(right_region)

    def _check_symmetry(self, landmarks: np.ndarray) -> float:
        """Check face symmetry using landmark pairs."""
        # MediaPipe landmark pairs to check (left-right)
        pairs = [
            # Eyes outer corners
            (33, 263),
            # Eyes inner corners
            (133, 362),
            # Eyebrows
            (105, 334),
            # Mouth corners
            (61, 291),
            # Cheeks
            (205, 425),
            # Jaw
            (207, 427)
        ]
        
        center_x = np.mean(landmarks[:, 0])
        scores = []
        
        for left, right in pairs:
            left_point = landmarks[left]
            right_point = landmarks[right]
            
            left_dist = abs(center_x - left_point[0])
            right_dist = abs(right_point[0] - center_x)
            
            if max(left_dist, right_dist) == 0:
                continue
                
            # Calculate symmetry score for this pair
            pair_score = 1 - (abs(left_dist - right_dist) / max(left_dist, right_dist))
            scores.append(pair_score)
        
        return np.mean(scores) if scores else 0.0

    def _check_face_coverage(self, landmarks: np.ndarray) -> float:
        """Check if the face is fully visible and unoccluded."""
        def get_region_visibility(region_landmarks, region_weight=1.0):
            points = landmarks[region_landmarks]
            hull = cv2.convexHull(points.astype(np.float32))
            area = cv2.contourArea(hull)
            
            # Get expected area from bounding box
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            expected_area = (x_max - x_min) * (y_max - y_min)
            
            # Calculate visibility score with weight
            score = area / expected_area if expected_area > 0 else 0
            return score * region_weight

        # Define facial regions to check with weights
        # Higher weights for critical regions (mouth, nose)
        regions = {
            'left_eye': {'landmarks': [33, 133, 157, 158, 159, 160, 161, 246], 'weight': 0.8},
            'right_eye': {'landmarks': [362, 263, 384, 385, 386, 387, 388, 466], 'weight': 0.8},
            'nose': {'landmarks': [1, 2, 3, 4, 5, 6, 168, 197, 195, 5], 'weight': 1.0},
            'mouth': {'landmarks': [61, 146, 91, 181, 84, 17, 314, 405, 321, 291], 'weight': 1.2},  # Higher weight
            'upper_lip': {'landmarks': [37, 39, 40, 185, 267, 269, 270, 409], 'weight': 1.1},  # Added for better mouth detection
            'lower_lip': {'landmarks': [84, 181, 314, 405, 321, 375, 291, 409], 'weight': 1.1},  # Added for better mouth detection
            'jaw': {'landmarks': [132, 58, 172, 136, 150, 149, 176, 148, 152], 'weight': 0.9}
        }

        # Calculate weighted visibility scores
        total_weight = sum(region['weight'] for region in regions.values())
        weighted_scores = []

        for region_info in regions.values():
            weighted_score = get_region_visibility(region_info['landmarks'], region_info['weight'])
            weighted_scores.append(weighted_score)

        # Normalize by total weight
        return sum(weighted_scores) / total_weight

    def _check_mouth_visibility(self, landmarks: np.ndarray) -> float:
        """Check if mouth is visible and not covered by analyzing mouth region landmarks."""
        # Use more points around the mouth for better coverage detection
        mouth_outline = landmarks[[61, 146, 91, 181, 84, 17, 314, 405, 321, 291]]
        inner_mouth = landmarks[[78, 95, 88, 178, 87, 14, 317, 402, 318, 324]]
        
        # Calculate convex hull and area for outer mouth region
        outer_hull = cv2.convexHull(mouth_outline.astype(np.float32))
        outer_area = cv2.contourArea(outer_hull)
        
        # Calculate convex hull and area for inner mouth region
        inner_hull = cv2.convexHull(inner_mouth.astype(np.float32))
        inner_area = cv2.contourArea(inner_hull)
        
        # Get bounding box for expected area
        all_points = np.vstack([mouth_outline, inner_mouth])
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        expected_area = (x_max - x_min) * (y_max - y_min)
        
        if expected_area == 0:
            return 0
        
        # Calculate visibility scores for both regions
        outer_score = outer_area / expected_area
        inner_score = inner_area / (expected_area * 0.3)  # Inner mouth should be ~30% of outer
        
        # Combined score with emphasis on outer mouth visibility
        visibility_score = (outer_score * 0.7 + inner_score * 0.3)
        
        # Additional check for vertical compression (hand covering)
        vertical_ratio = (y_max - y_min) / (x_max - x_min)
        if vertical_ratio < 0.3:  # Mouth appears too compressed
            visibility_score *= 0.5
        
        return visibility_score

    def _error_result(self, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            'isDetected': False,
            'isVisible': False,
            'score': 0,
            'message': message,
            'landmarks': None,
            'details': None
        }

    def _decode_image(self, base64_string):
        """Decode base64 image to OpenCV format"""
        try:
            # Remove header if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            img_data = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            
            # Decode image
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            return None