"""
HolisticDetector component for detecting body landmarks using MediaPipe Holistic.

This module provides a wrapper around MediaPipe's Holistic solution for detecting
pose and hand landmarks from RGB video frames.
"""

import mediapipe as mp
import numpy as np
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.consecutive_failures = 0
        self.failure_threshold = 10
        
        try:
            self.holistic = self.mp_holistic.Holistic(
                static_image_mode=False,  # False for video processing
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                refine_face_landmarks=False  # Disable facial landmark refinement
            )
            logger.info("HolisticDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe Holistic: {e}")
            raise
        
    def detect(self, frame: np.ndarray):
        """
        Process an RGB frame and return detection results.
        
        Args:
            frame: RGB image as numpy array (height, width, 3)
            
        Returns:
            MediaPipe Holistic results containing pose_landmarks, left_hand_landmarks,
            and right_hand_landmarks. Returns None if detection fails.
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to detector")
            return None
            
        try:
            # Process the frame through MediaPipe Holistic
            results = self.holistic.process(frame)
            
            # Reset failure counter on success
            if self.consecutive_failures > 0:
                logger.info(f"Detection recovered after {self.consecutive_failures} failures")
            self.consecutive_failures = 0
            
            return results
            
        except Exception as e:
            # Handle detection failures gracefully
            self.consecutive_failures += 1
            logger.error(f"Error during holistic detection (failure #{self.consecutive_failures}): {e}")
            
            # Warn if consecutive failures exceed threshold
            if self.consecutive_failures >= self.failure_threshold:
                logger.warning(
                    f"MediaPipe processing has failed {self.consecutive_failures} consecutive times. "
                    f"This may indicate a serious issue with the model or input frames."
                )
            
            return None
    
    def close(self):
        """
        Release MediaPipe resources.
        
        Should be called when done with detection to properly clean up resources.
        """
        try:
            if hasattr(self, 'holistic') and self.holistic:
                self.holistic.close()
                logger.info("HolisticDetector resources released successfully")
        except Exception as e:
            logger.error(f"Error during holistic detector cleanup: {e}")
            # Don't raise - cleanup should be best-effort
