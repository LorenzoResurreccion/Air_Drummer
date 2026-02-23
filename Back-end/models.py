"""
Data models for full body tracking system.

This module defines the core data structures used to represent tracking data
from MediaPipe Holistic, including landmark positions, confidence scores, and
complete tracking information for each frame.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LandmarkData:
    """
    Individual landmark information.
    
    Attributes:
        x: Normalized x coordinate [0.0, 1.0]
        y: Normalized y coordinate [0.0, 1.0]
        z: Depth coordinate (relative to hip midpoint)
        visibility: Confidence score [0.0, 1.0]
        landmark_type: Landmark identifier (e.g., "LEFT_WRIST", "RIGHT_KNEE")
    """
    x: float
    y: float
    z: float
    visibility: float
    landmark_type: str
    
    def is_valid(self) -> bool:
        """
        Validate that landmark data is within expected ranges.
        
        Returns:
            True if coordinates and visibility are in valid ranges, False otherwise
        """
        return (
            0.0 <= self.x <= 1.0 and
            0.0 <= self.y <= 1.0 and
            0.0 <= self.visibility <= 1.0
        )
    
    def is_reliable(self, threshold: float = 0.5) -> bool:
        """
        Check if landmark has sufficient confidence to be considered reliable.
        
        Args:
            threshold: Minimum confidence threshold (default 0.5)
            
        Returns:
            True if visibility score is above threshold, False otherwise
        """
        return self.visibility >= threshold
    
    def to_dict(self) -> dict:
        """
        Convert landmark data to dictionary format.
        
        Returns:
            Dictionary representation of landmark data
        """
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'visibility': self.visibility,
            'landmark_type': self.landmark_type,
            'is_reliable': self.is_reliable()
        }


@dataclass
class TrackingData:
    """
    Complete tracking information for a single frame.
    
    Attributes:
        timestamp: Frame timestamp in seconds
        frame_number: Sequential frame number
        pose_landmarks: List of pose landmarks (33 points) or None if not detected
        left_hand_landmarks: List of left hand landmarks (21 points) or None if not detected
        right_hand_landmarks: List of right hand landmarks (21 points) or None if not detected
        face_landmarks: List of face landmarks or None if not detected
    """
    timestamp: float
    frame_number: int
    pose_landmarks: Optional[List[LandmarkData]]
    left_hand_landmarks: Optional[List[LandmarkData]]
    right_hand_landmarks: Optional[List[LandmarkData]]
    face_landmarks: Optional[List[LandmarkData]]
    
    def has_body_detected(self) -> bool:
        """
        Check if any body parts were detected in this frame.
        
        Returns:
            True if any landmark collection is not None, False otherwise
        """
        return any([
            self.pose_landmarks is not None,
            self.left_hand_landmarks is not None,
            self.right_hand_landmarks is not None,
            self.face_landmarks is not None
        ])
    
    def get_reliable_landmarks(self, threshold: float = 0.5) -> dict:
        """
        Get all landmarks that meet the reliability threshold.
        
        Args:
            threshold: Minimum confidence threshold (default 0.5)
            
        Returns:
            Dictionary with keys for each body part containing only reliable landmarks
        """
        result = {
            'pose': [],
            'left_hand': [],
            'right_hand': [],
            'face': []
        }
        
        if self.pose_landmarks:
            result['pose'] = [lm for lm in self.pose_landmarks if lm.is_reliable(threshold)]
        
        if self.left_hand_landmarks:
            result['left_hand'] = [lm for lm in self.left_hand_landmarks if lm.is_reliable(threshold)]
        
        if self.right_hand_landmarks:
            result['right_hand'] = [lm for lm in self.right_hand_landmarks if lm.is_reliable(threshold)]
        
        if self.face_landmarks:
            result['face'] = [lm for lm in self.face_landmarks if lm.is_reliable(threshold)]
        
        return result
    
    def to_dict(self) -> dict:
        """
        Convert tracking data to dictionary format.
        
        Returns:
            Dictionary representation of complete tracking data
        """
        return {
            'timestamp': self.timestamp,
            'frame_number': self.frame_number,
            'pose_landmarks': [lm.to_dict() for lm in self.pose_landmarks] if self.pose_landmarks else None,
            'left_hand_landmarks': [lm.to_dict() for lm in self.left_hand_landmarks] if self.left_hand_landmarks else None,
            'right_hand_landmarks': [lm.to_dict() for lm in self.right_hand_landmarks] if self.right_hand_landmarks else None,
            'face_landmarks': [lm.to_dict() for lm in self.face_landmarks] if self.face_landmarks else None,
            'has_body_detected': self.has_body_detected()
        }
