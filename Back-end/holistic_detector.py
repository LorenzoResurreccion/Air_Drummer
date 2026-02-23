"""
HolisticDetector component for detecting body landmarks using MediaPipe Holistic.

This module provides a wrapper around MediaPipe's Holistic solution for detecting
face, pose, and hand landmarks from RGB video frames.
"""

import mediapipe as mp
import numpy as np
from typing import Optional


class HolisticDetector:
    """
    Detects body landmarks using MediaPipe Holistic model.
    
    This class manages the MediaPipe Holistic solution lifecycle and provides
    a simple interface for processing frames and obtaining detection results.
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        smooth_landmarks: bool = True
    ):
        """
        Initialize MediaPipe Holistic model with configurable parameters.
        
        Args:
            min_detection_confidence: Minimum confidence [0.0, 1.0] for initial detection
            min_tracking_confidence: Minimum confidence [0.0, 1.0] for tracking across frames
            model_complexity: Complexity of the pose landmark model (0=lite, 1=full, 2=heavy)
            smooth_landmarks: Whether to enable temporal smoothing for stable tracking
        """
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,  # False for video processing
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def detect(self, frame: np.ndarray):
        """
        Process an RGB frame and return detection results.
        
        Args:
            frame: RGB image as numpy array (height, width, 3)
            
        Returns:
            MediaPipe Holistic results containing pose_landmarks, left_hand_landmarks,
            right_hand_landmarks, and face_landmarks. Returns None if detection fails.
        """
        try:
            # Process the frame through MediaPipe Holistic
            results = self.holistic.process(frame)
            return results
        except Exception as e:
            # Handle detection failures gracefully
            print(f"Error during holistic detection: {e}")
            return None
    
    def close(self):
        """
        Release MediaPipe resources.
        
        Should be called when done with detection to properly clean up resources.
        """
        try:
            if self.holistic:
                self.holistic.close()
        except Exception as e:
            print(f"Error during holistic detector cleanup: {e}")
