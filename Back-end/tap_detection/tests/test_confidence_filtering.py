"""
Unit tests for confidence filtering in TapDetector.

Tests that the detector properly handles low confidence landmarks and
missing landmarks by skipping processing and canceling in-progress gestures.

Requirements: 1.4, 2.4, 3.4, 7.1, 7.2, 7.3, 7.4
"""

import pytest
from tap_detection.tap_detector import TapDetector
from tap_detection.tap_models import (
    FootLandmarks,
    TapDetectionConfig,
    DetectionState,
    TapType
)
from comp_vision.models import LandmarkData


def create_foot_landmarks(
    heel_y: float = 0.5,
    toe_y: float = 0.5,
    timestamp: float = 0.0,
    visibility: float = 0.9
) -> FootLandmarks:
    """Helper to create FootLandmarks with specified parameters."""
    heel = LandmarkData(
        x=0.5, y=heel_y, z=0.0,
        visibility=visibility,
        landmark_type="RIGHT_HEEL"
    )
    ankle = LandmarkData(
        x=0.5, y=0.4, z=0.0,
        visibility=visibility,
        landmark_type="RIGHT_ANKLE"
    )
    toe = LandmarkData(
        x=0.5, y=toe_y, z=0.0,
        visibility=visibility,
        landmark_type="RIGHT_FOOT_INDEX"
    )
    
    return FootLandmarks(
        heel=heel,
        ankle=ankle,
        toe=toe,
        timestamp=timestamp
    )


class TestConfidenceFiltering:
    """Test confidence-based filtering of landmarks."""
    
    def test_low_confidence_landmarks_skipped(self):
        """
        Test that frames with low confidence landmarks are skipped.
        
        Requirements 1.4, 2.4, 3.4, 7.2: Skip processing when landmarks
        below confidence threshold
        """
        config = TapDetectionConfig(
            min_displacement=0.05,
            min_velocity=0.25,
            confidence_threshold=0.5
        )
        detector = TapDetector(config)
        
        # Create landmarks with low confidence (below 0.5 threshold)
        low_confidence_frame = create_foot_landmarks(
            heel_y=0.40,  # Significant upward movement
            toe_y=0.50,
            timestamp=0.1,
            visibility=0.3  # Below threshold
        )
        
        # Process frame with low confidence
        tap_event = detector.process_frame(low_confidence_frame)
        
        # Should skip processing and return None
        assert tap_event is None
        # Should remain in IDLE state
        assert detector.get_state() == DetectionState.IDLE
    
    def test_missing_landmarks_skipped(self):
        """
        Test that None landmarks are handled gracefully.
        
        Requirement 7.1: Skip processing when landmarks missing
        """
        config = TapDetectionConfig()
        detector = TapDetector(config)
        
        # Process None landmarks
        tap_event = detector.process_frame(None)
        
        # Should return None without errors
        assert tap_event is None
        # Should remain in IDLE state
        assert detector.get_state() == DetectionState.IDLE
    
    def test_in_progress_gesture_cancelled_on_low_confidence(self):
        """
        Test that in-progress gestures are cancelled when confidence drops.
        
        Requirement 7.3: Cancel in-progress gestures when landmarks
        become unavailable
        """
        config = TapDetectionConfig(
            min_displacement=0.05,
            min_velocity=0.25,
            confidence_threshold=0.5
        )
        detector = TapDetector(config)
        
        # Start a valid heel tap gesture
        frames = [
            create_foot_landmarks(heel_y=0.50, toe_y=0.50, timestamp=0.0, visibility=0.9),
            create_foot_landmarks(heel_y=0.44, toe_y=0.50, timestamp=0.05, visibility=0.9),
            create_foot_landmarks(heel_y=0.40, toe_y=0.50, timestamp=0.10, visibility=0.9),
        ]
        
        for frame in frames:
            detector.process_frame(frame)
        
        # Should be in DETECTING_UPWARD or DETECTING_DOWNWARD state
        assert detector.get_state() in [DetectionState.DETECTING_UPWARD, DetectionState.DETECTING_DOWNWARD]
        
        # Now send frame with low confidence
        low_confidence_frame = create_foot_landmarks(
            heel_y=0.40,
            toe_y=0.50,
            timestamp=0.15,
            visibility=0.3  # Below threshold
        )
        
        tap_event = detector.process_frame(low_confidence_frame)
        
        # Gesture should be cancelled
        assert tap_event is None
        # Should reset to IDLE state
        assert detector.get_state() == DetectionState.IDLE
    
    def test_in_progress_gesture_cancelled_on_missing_landmarks(self):
        """
        Test that in-progress gestures are cancelled when landmarks go missing.
        
        Requirement 7.3: Cancel in-progress gestures when landmarks
        become unavailable
        """
        config = TapDetectionConfig(
            min_displacement=0.05,
            min_velocity=0.25
        )
        detector = TapDetector(config)
        
        # Start a valid toe tap gesture
        frames = [
            create_foot_landmarks(heel_y=0.50, toe_y=0.50, timestamp=0.0, visibility=0.9),
            create_foot_landmarks(heel_y=0.50, toe_y=0.44, timestamp=0.05, visibility=0.9),
            create_foot_landmarks(heel_y=0.50, toe_y=0.40, timestamp=0.10, visibility=0.9),
        ]
        
        for frame in frames:
            detector.process_frame(frame)
        
        # Should be tracking a gesture
        assert detector.get_state() in [DetectionState.DETECTING_UPWARD, DetectionState.DETECTING_DOWNWARD]
        
        # Send None landmarks
        tap_event = detector.process_frame(None)
        
        # Gesture should be cancelled
        assert tap_event is None
        # Should reset to IDLE state
        assert detector.get_state() == DetectionState.IDLE
    
    def test_recovery_after_confidence_returns(self):
        """
        Test that detector resumes normal operation after confidence returns.
        
        Requirement 7.4: Reset state and resume normal detection when
        landmarks return after being unavailable
        """
        config = TapDetectionConfig(
            min_displacement=0.05,
            min_velocity=0.25,
            confidence_threshold=0.5
        )
        detector = TapDetector(config)
        
        # Start with low confidence frames
        low_conf_frames = [
            create_foot_landmarks(heel_y=0.50, toe_y=0.50, timestamp=0.0, visibility=0.3),
            create_foot_landmarks(heel_y=0.45, toe_y=0.50, timestamp=0.05, visibility=0.3),
        ]
        
        for frame in low_conf_frames:
            detector.process_frame(frame)
        
        # Should remain in IDLE
        assert detector.get_state() == DetectionState.IDLE
        
        # Now send frames with good confidence - should detect tap normally
        good_frames = [
            create_foot_landmarks(heel_y=0.50, toe_y=0.50, timestamp=0.10, visibility=0.9),
            create_foot_landmarks(heel_y=0.44, toe_y=0.50, timestamp=0.15, visibility=0.9),
            create_foot_landmarks(heel_y=0.40, toe_y=0.50, timestamp=0.20, visibility=0.9),
            create_foot_landmarks(heel_y=0.42, toe_y=0.50, timestamp=0.25, visibility=0.9),
            create_foot_landmarks(heel_y=0.48, toe_y=0.50, timestamp=0.30, visibility=0.9),
        ]
        
        tap_event = None
        for frame in good_frames:
            result = detector.process_frame(frame)
            if result is not None:
                tap_event = result
        
        # Should detect tap successfully
        assert tap_event is not None
        assert tap_event.tap_type == TapType.HEEL
    
    def test_partial_low_confidence_cancels_gesture(self):
        """
        Test that if any landmark has low confidence, processing is skipped.
        
        Requirement 7.2: Skip processing when any required landmark
        below confidence threshold
        """
        config = TapDetectionConfig(
            min_displacement=0.05,
            min_velocity=0.25,
            confidence_threshold=0.5
        )
        detector = TapDetector(config)
        
        # Start gesture with all landmarks having good confidence
        frames = [
            create_foot_landmarks(heel_y=0.50, toe_y=0.50, timestamp=0.0, visibility=0.9),
            create_foot_landmarks(heel_y=0.44, toe_y=0.50, timestamp=0.05, visibility=0.9),
        ]
        
        for frame in frames:
            detector.process_frame(frame)
        
        # Should be tracking gesture
        assert detector.get_state() in [DetectionState.DETECTING_UPWARD, DetectionState.DETECTING_DOWNWARD]
        
        # Create frame where only one landmark has low confidence
        heel = LandmarkData(x=0.5, y=0.40, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL")
        ankle = LandmarkData(x=0.5, y=0.4, z=0.0, visibility=0.3, landmark_type="RIGHT_ANKLE")  # Low!
        toe = LandmarkData(x=0.5, y=0.50, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX")
        
        mixed_confidence_frame = FootLandmarks(
            heel=heel,
            ankle=ankle,
            toe=toe,
            timestamp=0.10
        )
        
        tap_event = detector.process_frame(mixed_confidence_frame)
        
        # Should cancel gesture
        assert tap_event is None
        assert detector.get_state() == DetectionState.IDLE
    
    def test_debouncing_state_not_affected_by_low_confidence(self):
        """
        Test that debouncing state is maintained even with low confidence.
        
        The debounce period should complete regardless of landmark quality.
        """
        config = TapDetectionConfig(
            min_displacement=0.05,
            min_velocity=0.25,
            confidence_threshold=0.5
        )
        detector = TapDetector(config)
        
        # Complete a valid tap to enter debouncing state
        frames = [
            create_foot_landmarks(heel_y=0.50, toe_y=0.50, timestamp=0.0, visibility=0.9),
            create_foot_landmarks(heel_y=0.44, toe_y=0.50, timestamp=0.05, visibility=0.9),
            create_foot_landmarks(heel_y=0.40, toe_y=0.50, timestamp=0.10, visibility=0.9),
            create_foot_landmarks(heel_y=0.42, toe_y=0.50, timestamp=0.15, visibility=0.9),
            create_foot_landmarks(heel_y=0.48, toe_y=0.50, timestamp=0.20, visibility=0.9),
        ]
        
        tap_event = None
        for frame in frames:
            result = detector.process_frame(frame)
            if result is not None:
                tap_event = result
        
        # Should have detected tap and be in debouncing
        assert tap_event is not None
        assert detector.get_state() == DetectionState.DEBOUNCING
        
        # Send low confidence frame during debounce
        low_conf_frame = create_foot_landmarks(
            heel_y=0.50,
            toe_y=0.50,
            timestamp=0.22,
            visibility=0.3
        )
        
        result = detector.process_frame(low_conf_frame)
        
        # Should still be debouncing (not reset to IDLE)
        assert result is None
        assert detector.get_state() == DetectionState.DEBOUNCING
