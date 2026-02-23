"""
Tracking data extractor for MediaPipe Holistic results.

This module provides functionality to extract structured tracking data from
MediaPipe Holistic detection results, converting raw landmark data into
TrackingData objects with proper handling of missing landmarks.
"""

from typing import Optional, List
import time
from models import TrackingData, LandmarkData


class TrackingDataExtractor:
    """
    Extracts structured tracking data from MediaPipe Holistic results.
    
    This class processes MediaPipe detection results and converts them into
    TrackingData objects containing normalized coordinates, confidence scores,
    and metadata for each frame.
    """
    
    # MediaPipe pose landmark names for reference
    POSE_LANDMARK_NAMES = [
        "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
        "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
        "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
        "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
        "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
        "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB",
        "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
        "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
        "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
    ]
    
    # MediaPipe hand landmark names for reference
    HAND_LANDMARK_NAMES = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]
    
    def __init__(self):
        """Initialize the tracking data extractor."""
        self._frame_counter = 0
    
    def extract(self, results, timestamp: Optional[float] = None) -> TrackingData:
        """
        Extract structured tracking data from MediaPipe Holistic results.
        
        Args:
            results: MediaPipe Holistic detection results
            timestamp: Frame timestamp in seconds (uses current time if None)
            
        Returns:
            TrackingData object containing all detected landmarks and metadata
        """
        if timestamp is None:
            timestamp = time.time()
        
        self._frame_counter += 1
        
        # Extract landmarks from each body part
        pose_landmarks = self._extract_pose_landmarks(results)
        left_hand_landmarks = self._extract_hand_landmarks(results, hand_type="left")
        right_hand_landmarks = self._extract_hand_landmarks(results, hand_type="right")
        face_landmarks = self._extract_face_landmarks(results)
        
        return TrackingData(
            timestamp=timestamp,
            frame_number=self._frame_counter,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=left_hand_landmarks,
            right_hand_landmarks=right_hand_landmarks,
            face_landmarks=face_landmarks
        )
    
    def _extract_pose_landmarks(self, results) -> Optional[List[LandmarkData]]:
        """
        Extract pose landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe Holistic detection results
            
        Returns:
            List of LandmarkData objects for pose, or None if not detected
        """
        if not results.pose_landmarks:
            return None
        
        landmarks = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmark_name = self.POSE_LANDMARK_NAMES[idx] if idx < len(self.POSE_LANDMARK_NAMES) else f"POSE_{idx}"
            
            landmarks.append(LandmarkData(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=landmark.visibility,
                landmark_type=landmark_name
            ))
        
        return landmarks
    
    def _extract_hand_landmarks(self, results, hand_type: str) -> Optional[List[LandmarkData]]:
        """
        Extract hand landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe Holistic detection results
            hand_type: Either "left" or "right"
            
        Returns:
            List of LandmarkData objects for the specified hand, or None if not detected
        """
        hand_landmarks = None
        
        if hand_type == "left":
            hand_landmarks = results.left_hand_landmarks
        elif hand_type == "right":
            hand_landmarks = results.right_hand_landmarks
        else:
            raise ValueError(f"Invalid hand_type: {hand_type}. Must be 'left' or 'right'")
        
        if not hand_landmarks:
            return None
        
        landmarks = []
        hand_prefix = "LEFT" if hand_type == "left" else "RIGHT"
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmark_name = self.HAND_LANDMARK_NAMES[idx] if idx < len(self.HAND_LANDMARK_NAMES) else f"HAND_{idx}"
            full_name = f"{hand_prefix}_{landmark_name}"
            
            # Hand landmarks don't have visibility, use 1.0 as default confidence
            visibility = getattr(landmark, 'visibility', 1.0)
            
            landmarks.append(LandmarkData(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=visibility,
                landmark_type=full_name
            ))
        
        return landmarks
    
    def _extract_face_landmarks(self, results) -> Optional[List[LandmarkData]]:
        """
        Extract face landmarks from MediaPipe results.
        
        Args:
            results: MediaPipe Holistic detection results
            
        Returns:
            List of LandmarkData objects for face, or None if not detected
        """
        if not results.face_landmarks:
            return None
        
        landmarks = []
        for idx, landmark in enumerate(results.face_landmarks.landmark):
            landmarks.append(LandmarkData(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z,
                visibility=getattr(landmark, 'visibility', 1.0),
                landmark_type=f"FACE_{idx}"
            ))
        
        return landmarks
    
    def get_landmark_position(self, landmark) -> dict:
        """
        Get normalized position coordinates for a landmark.
        
        Args:
            landmark: MediaPipe landmark object
            
        Returns:
            Dictionary containing x, y, z coordinates
        """
        return {
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z
        }
    
    def get_confidence_score(self, landmark) -> float:
        """
        Get confidence score for a landmark.
        
        Args:
            landmark: MediaPipe landmark object
            
        Returns:
            Confidence score (visibility) as float
        """
        return getattr(landmark, 'visibility', 1.0)
    
    def reset_frame_counter(self):
        """Reset the internal frame counter to zero."""
        self._frame_counter = 0
