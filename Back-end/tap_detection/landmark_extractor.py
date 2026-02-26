"""
Foot landmark extraction from tracking data.

This module provides functionality to extract right foot landmarks from
TrackingData objects produced by the body tracking system. It handles
landmark extraction, validation, and confidence checking.
"""

from typing import Optional

from comp_vision.models import TrackingData, LandmarkData
from tap_detection.tap_models import FootLandmarks


# MediaPipe Holistic pose landmark indices for right foot
RIGHT_ANKLE_INDEX = 28
RIGHT_HEEL_INDEX = 30
RIGHT_FOOT_INDEX = 32


class FootLandmarkExtractor:
    """
    Extracts and validates right foot landmarks from tracking data.
    
    This class is responsible for extracting the three key right foot landmarks
    (heel, ankle, toe) from TrackingData objects and validating their confidence
    scores to ensure reliable tap detection.
    """
    
    def extract(self, tracking_data: TrackingData) -> Optional[FootLandmarks]:
        """
        Extract right foot landmarks from tracking data.
        
        Extracts RIGHT_HEEL (index 30), RIGHT_ANKLE (index 28), and 
        RIGHT_FOOT_INDEX (index 32) from the pose landmarks in the tracking data.
        
        Args:
            tracking_data: TrackingData object from body tracking system
            
        Returns:
            FootLandmarks object if all required landmarks are present,
            None if pose_landmarks is None or missing required indices
        """
        # Handle missing pose landmarks
        if tracking_data.pose_landmarks is None:
            return None
        
        pose_landmarks = tracking_data.pose_landmarks
        
        # Check if all required landmark indices are available
        # MediaPipe Holistic provides 33 pose landmarks (indices 0-32)
        if len(pose_landmarks) < 33:
            return None
        
        # Extract the three required landmarks
        try:
            heel = pose_landmarks[RIGHT_HEEL_INDEX]
            ankle = pose_landmarks[RIGHT_ANKLE_INDEX]
            toe = pose_landmarks[RIGHT_FOOT_INDEX]
        except (IndexError, KeyError):
            # Handle case where indices don't exist
            return None
        
        # Create and return FootLandmarks object
        return FootLandmarks(
            heel=heel,
            ankle=ankle,
            toe=toe,
            timestamp=tracking_data.timestamp
        )
    
    def validate_landmarks(
        self, 
        landmarks: FootLandmarks, 
        confidence_threshold: float = 0.5
    ) -> bool:
        """
        Check if landmarks meet confidence requirements.
        
        Validates that all three foot landmarks (heel, ankle, toe) have
        confidence scores above the specified threshold. This ensures
        reliable tap detection by filtering out poorly tracked landmarks.
        
        Args:
            landmarks: FootLandmarks object to validate
            confidence_threshold: Minimum confidence score (default 0.5)
            
        Returns:
            True if all landmarks have visibility above threshold, False otherwise
        """
        return landmarks.all_reliable(confidence_threshold)
