"""
Unit tests for tap history management in TapDetector.

Tests verify that TapDetector correctly records tap events with metadata
and maintains a history of the last 10 taps.
"""

import pytest
from tap_detection.tap_detector import TapDetector
from tap_detection.tap_models import (
    TapDetectionConfig,
    FootLandmarks,
    TapType,
    DetectionState
)
from comp_vision.models import LandmarkData


class TestTapHistory:
    """Test tap history recording and management."""
    
    def test_tap_history_initially_empty(self):
        """Test that tap history is empty on initialization."""
        config = TapDetectionConfig()
        detector = TapDetector(config)
        
        history = detector.get_tap_history()
        assert len(history) == 0, "Tap history should be empty initially"
    
    def test_tap_recorded_in_history(self):
        """Test that detected taps are recorded in history."""
        config = TapDetectionConfig()
        detector = TapDetector(config)
        
        # Generate heel tap sequence
        base_time = 0.0
        frames = []
        
        # Initial stationary position
        for i in range(3):
            frames.append(FootLandmarks(
                heel=LandmarkData(x=0.5, y=0.8, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                timestamp=base_time + i * 0.033
            ))
        
        # Upward motion (heel lifts)
        for i in range(5):
            y_pos = 0.8 - (i + 1) * 0.02  # Move up
            frames.append(FootLandmarks(
                heel=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                timestamp=base_time + (i + 3) * 0.033
            ))
        
        # Downward motion (heel returns)
        for i in range(5):
            y_pos = 0.7 + (i + 1) * 0.02  # Move down
            frames.append(FootLandmarks(
                heel=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                timestamp=base_time + (i + 8) * 0.033
            ))
        
        # Process frames
        tap_detected = False
        for frame in frames:
            result = detector.process_frame(frame)
            if result is not None:
                tap_detected = True
        
        assert tap_detected, "Tap should be detected"
        
        # Check history
        history = detector.get_tap_history()
        assert len(history) == 1, "History should contain one tap"
        assert history[0].tap_type == TapType.HEEL, "Tap type should be HEEL"
        assert history[0].peak_displacement > 0, "Peak displacement should be positive"
        assert history[0].duration_ms > 0, "Duration should be positive"
    
    def test_multiple_taps_recorded(self):
        """Test that multiple taps are recorded in history."""
        config = TapDetectionConfig()
        detector = TapDetector(config)
        
        base_time = 0.0
        
        # Detect 3 taps
        for tap_num in range(3):
            tap_start_time = base_time + tap_num * 1.0  # 1 second apart
            
            # Initial position
            for i in range(3):
                frame = FootLandmarks(
                    heel=LandmarkData(x=0.5, y=0.8, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                    ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                    toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                    timestamp=tap_start_time + i * 0.033
                )
                detector.process_frame(frame)
            
            # Upward motion
            for i in range(5):
                y_pos = 0.8 - (i + 1) * 0.02
                frame = FootLandmarks(
                    heel=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                    ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                    toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                    timestamp=tap_start_time + (i + 3) * 0.033
                )
                detector.process_frame(frame)
            
            # Downward motion
            for i in range(5):
                y_pos = 0.7 + (i + 1) * 0.02
                frame = FootLandmarks(
                    heel=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                    ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                    toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                    timestamp=tap_start_time + (i + 8) * 0.033
                )
                detector.process_frame(frame)
        
        # Check history
        history = detector.get_tap_history()
        assert len(history) == 3, f"History should contain 3 taps, got {len(history)}"
    
    def test_tap_history_limited_to_10(self):
        """Test that tap history is limited to last 10 taps."""
        config = TapDetectionConfig()
        detector = TapDetector(config)
        
        base_time = 0.0
        
        # Detect 15 taps (more than the limit of 10)
        for tap_num in range(15):
            tap_start_time = base_time + tap_num * 1.0
            
            # Initial position
            for i in range(3):
                frame = FootLandmarks(
                    heel=LandmarkData(x=0.5, y=0.8, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                    ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                    toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                    timestamp=tap_start_time + i * 0.033
                )
                detector.process_frame(frame)
            
            # Upward motion
            for i in range(5):
                y_pos = 0.8 - (i + 1) * 0.02
                frame = FootLandmarks(
                    heel=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                    ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                    toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                    timestamp=tap_start_time + (i + 3) * 0.033
                )
                detector.process_frame(frame)
            
            # Downward motion
            for i in range(5):
                y_pos = 0.7 + (i + 1) * 0.02
                frame = FootLandmarks(
                    heel=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                    ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                    toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                    timestamp=tap_start_time + (i + 8) * 0.033
                )
                detector.process_frame(frame)
        
        # Check history is limited to 10
        history = detector.get_tap_history()
        assert len(history) == 10, f"History should be limited to 10 taps, got {len(history)}"
    
    def test_tap_metadata_recorded_correctly(self):
        """Test that tap metadata is recorded with correct values."""
        config = TapDetectionConfig()
        detector = TapDetector(config)
        
        base_time = 1.0
        
        # Generate toe tap sequence
        frames = []
        
        # Initial position
        for i in range(3):
            frames.append(FootLandmarks(
                heel=LandmarkData(x=0.5, y=0.8, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                toe=LandmarkData(x=0.5, y=0.85, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                timestamp=base_time + i * 0.033
            ))
        
        # Upward motion (toe lifts)
        for i in range(5):
            y_pos = 0.85 - (i + 1) * 0.02
            frames.append(FootLandmarks(
                heel=LandmarkData(x=0.5, y=0.8, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                toe=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                timestamp=base_time + (i + 3) * 0.033
            ))
        
        # Downward motion
        for i in range(5):
            y_pos = 0.75 + (i + 1) * 0.02
            frames.append(FootLandmarks(
                heel=LandmarkData(x=0.5, y=0.8, z=0.0, visibility=0.9, landmark_type="RIGHT_HEEL"),
                ankle=LandmarkData(x=0.5, y=0.7, z=0.0, visibility=0.9, landmark_type="RIGHT_ANKLE"),
                toe=LandmarkData(x=0.5, y=y_pos, z=0.0, visibility=0.9, landmark_type="RIGHT_FOOT_INDEX"),
                timestamp=base_time + (i + 8) * 0.033
            ))
        
        # Process frames
        for frame in frames:
            detector.process_frame(frame)
        
        # Check metadata
        history = detector.get_tap_history()
        assert len(history) == 1
        
        tap_event = history[0]
        assert tap_event.tap_type == TapType.TOE, "Tap type should be TOE"
        assert tap_event.timestamp >= base_time, "Timestamp should be valid"
        assert tap_event.peak_displacement >= 0.05, "Peak displacement should exceed threshold"
        assert 0 < tap_event.duration_ms < 500, "Duration should be within reasonable range"
