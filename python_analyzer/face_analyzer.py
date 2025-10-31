import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, Tuple, Optional

class FaceQualityAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh for detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def analyze_face(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze face quality metrics including pose, symmetry, and feature visibility.
        Returns detailed analysis results.
        """
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Get face mesh results
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return {
                "is_visible": False,
                "score": 0,
                "message": "No face detected",
                "details": {}
            }

        landmarks = results.multi_face_landmarks[0].landmark
        
        # Convert landmarks to pixel coordinates
        points = np.array([(lm.x * width, lm.y * height) for lm in landmarks])
        
        # Calculate quality metrics
        pose_scores = self._check_face_pose(points)
        symmetry_scores = self._analyze_face_symmetry(points)
        feature_scores = self._check_feature_visibility(points)
        
        # Aggregate scores and determine visibility
        total_score = self._calculate_final_score(pose_scores, symmetry_scores, feature_scores)
        
        return {
            "is_visible": total_score["is_visible"],
            "score": total_score["score"],
            "message": total_score["message"],
            "details": {
                "pose": pose_scores,
                "symmetry": symmetry_scores,
                "features": feature_scores
            }
        }

    def _check_face_pose(self, points: np.ndarray) -> Dict[str, float]:
        """
        Check face pose using facial landmarks.
        Returns scores for yaw (side-to-side), pitch (up-down), and roll (tilt).
        """
        # Key points for pose estimation (using MediaPipe Face Mesh indices)
        nose_tip = points[4]
        left_eye = np.mean(points[[33, 133]], axis=0)
        right_eye = np.mean(points[[362, 263]], axis=0)
        mouth_left = points[61]
        mouth_right = points[291]
        
        # Calculate face direction vector
        face_normal = nose_tip - np.mean([left_eye, right_eye], axis=0)
        
        # Estimate yaw (side-to-side rotation)
        eye_distance = np.linalg.norm(right_eye - left_eye)
        nose_deviation = np.abs(nose_tip[0] - np.mean([left_eye[0], right_eye[0]]))
        yaw_score = 1.0 - min(1.0, nose_deviation / (eye_distance * 0.5))
        
        # Estimate roll (head tilt)
        eye_angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], 
                                        right_eye[0] - left_eye[0]))
        roll_score = 1.0 - min(1.0, abs(eye_angle) / 20.0)  # Penalize tilt over 20 degrees
        
        # Estimate pitch (up-down rotation)
        mouth_center = np.mean([mouth_left, mouth_right], axis=0)
        vertical_ratio = (nose_tip[1] - np.mean([left_eye[1], right_eye[1]])) / \
                        (mouth_center[1] - nose_tip[1])
        pitch_score = 1.0 - min(1.0, abs(vertical_ratio - 1.0))
        
        return {
            "yaw": yaw_score,
            "pitch": pitch_score,
            "roll": roll_score
        }

    def _analyze_face_symmetry(self, points: np.ndarray) -> Dict[str, float]:
        """
        Analyze facial symmetry using key landmark pairs.
        Returns symmetry scores for different facial regions.
        """
        def calculate_pair_symmetry(left_idx: int, right_idx: int) -> float:
            left_point = points[left_idx]
            right_point = points[right_idx]
            center_x = np.mean(points[:, 0])
            
            left_dist = abs(center_x - left_point[0])
            right_dist = abs(right_point[0] - center_x)
            
            max_dist = max(left_dist, right_dist)
            if max_dist == 0:
                return 1.0
            return 1.0 - abs(left_dist - right_dist) / max_dist

        # Key symmetry pairs (MediaPipe Face Mesh indices)
        eye_pair = calculate_pair_symmetry(33, 263)  # Eyes outer corners
        brow_pair = calculate_pair_symmetry(105, 334)  # Eyebrows
        mouth_pair = calculate_pair_symmetry(61, 291)  # Mouth corners
        
        # Weight the symmetry scores
        weights = {"eyes": 0.4, "brows": 0.3, "mouth": 0.3}
        
        return {
            "eyes": eye_pair,
            "brows": brow_pair,
            "mouth": mouth_pair,
            "overall": (eye_pair * weights["eyes"] + 
                       brow_pair * weights["brows"] + 
                       mouth_pair * weights["mouth"])
        }

    def _check_feature_visibility(self, points: np.ndarray) -> Dict[str, float]:
        """
        Check visibility and quality of key facial features.
        Returns visibility scores for eyes, nose, and mouth.
        """
        # Define key feature regions
        left_eye = points[33:246]  # Left eye region
        right_eye = points[362:466]  # Right eye region
        nose = points[168:200]  # Nose region
        mouth = points[61:291]  # Mouth region
        
        def calculate_feature_visibility(feature_points: np.ndarray) -> float:
            if len(feature_points) < 2:
                return 0.0
            
            # Calculate the area and spread of points
            hull = cv2.convexHull(feature_points.astype(np.float32))
            area = cv2.contourArea(hull)
            
            # Normalize by the expected area for each feature
            max_area = (np.max(feature_points[:, 0]) - np.min(feature_points[:, 0])) * \
                      (np.max(feature_points[:, 1]) - np.min(feature_points[:, 1]))
            
            return min(1.0, area / (max_area + 1e-6))

        return {
            "left_eye": calculate_feature_visibility(left_eye),
            "right_eye": calculate_feature_visibility(right_eye),
            "nose": calculate_feature_visibility(nose),
            "mouth": calculate_feature_visibility(mouth)
        }

    def _calculate_final_score(self, 
                             pose: Dict[str, float],
                             symmetry: Dict[str, float],
                             features: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate final face quality score and determine if the face is properly visible.
        """
        # Weight the different components
        pose_score = (pose["yaw"] * 0.4 + pose["pitch"] * 0.3 + pose["roll"] * 0.3) * 100
        symmetry_score = symmetry["overall"] * 100
        
        # Average feature visibility
        feature_score = np.mean([
            features["left_eye"],
            features["right_eye"],
            features["nose"],
            features["mouth"]
        ]) * 100
        
        # Calculate final weighted score
        final_score = (pose_score * 0.4 + symmetry_score * 0.3 + feature_score * 0.3)
        
        # Define thresholds
        pose_threshold = 75
        symmetry_threshold = 80
        feature_threshold = 70
        
        # Generate status message
        issues = []
        if pose_score < pose_threshold:
            issues.append("Face not properly aligned")
        if symmetry_score < symmetry_threshold:
            issues.append("Face not fully visible")
        if feature_score < feature_threshold:
            issues.append("Facial features not clearly visible")
        
        message = "Face properly visible" if not issues else "; ".join(issues)
        
        return {
            "score": final_score,
            "is_visible": (pose_score >= pose_threshold and 
                         symmetry_score >= symmetry_threshold and 
                         feature_score >= feature_threshold),
            "message": message
        }