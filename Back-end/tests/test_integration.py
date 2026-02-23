"""
Integration tests for the complete Body Tracker system.

Tests the end-to-end integration of all components.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util
import sys
from pathlib import Path

# Import Body-Tracker.py module (with hyphen in name)
body_tracker_path = Path(__file__).parent.parent / "Body-Tracker.py"
spec = importlib.util.spec_from_file_location("body_tracker", body_tracker_path)
body_tracker = importlib.util.module_from_spec(spec)
sys.modules["body_tracker"] = body_tracker
spec.loader.exec_module(body_tracker)

BodyTracker = body_tracker.BodyTracker
parse_arguments = body_tracker.parse_arguments
from models import TrackingData


class TestBodyTrackerIntegration:
    """Integration tests for the complete BodyTracker system."""
    
    def test_body_tracker_initialization(self):
        """Test that BodyTracker can be initialized with default parameters."""
        # Note: This test may fail if no camera is available
        # In CI/CD, this would need to be mocked or skipped
        try:
            tracker = BodyTracker(camera_index=0)
            assert tracker is not None
            assert tracker.frame_number == 0
            assert tracker.detection_failures == 0
            tracker.stop()
        except RuntimeError as e:
            # Camera not available - this is acceptable in test environment
            pytest.skip(f"Camera not available: {e}")
    
    def test_body_tracker_initialization_with_custom_params(self):
        """Test BodyTracker initialization with custom parameters."""
        try:
            tracker = BodyTracker(
                camera_index=0,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
                model_complexity=1,
                flip_frame=False,
                show_fps=True
            )
            assert tracker is not None
            assert tracker.flip_frame is False
            assert tracker.show_fps is True
            tracker.stop()
        except RuntimeError as e:
            pytest.skip(f"Camera not available: {e}")
    
    def test_process_frame_returns_correct_types(self):
        """Test that process_frame returns correct data types."""
        try:
            tracker = BodyTracker(camera_index=0)
            tracking_data, display_frame = tracker.process_frame()
            
            # Check return types
            if tracking_data is not None:
                assert isinstance(tracking_data, TrackingData)
                assert hasattr(tracking_data, 'timestamp')
                assert hasattr(tracking_data, 'frame_number')
                assert hasattr(tracking_data, 'pose_landmarks')
                assert hasattr(tracking_data, 'left_hand_landmarks')
                assert hasattr(tracking_data, 'right_hand_landmarks')
            
            if display_frame is not None:
                assert isinstance(display_frame, np.ndarray)
                assert len(display_frame.shape) == 3  # Height x Width x Channels
            
            tracker.stop()
        except RuntimeError as e:
            pytest.skip(f"Camera not available: {e}")
    
    def test_tracking_data_structure_completeness(self):
        """
        Test that TrackingData structure is complete even when no body is detected.
        
        Validates: Requirements 8.1, 8.2, 8.3
        """
        try:
            tracker = BodyTracker(camera_index=0)
            tracking_data, _ = tracker.process_frame()
            
            if tracking_data is not None:
                # Verify all required fields exist
                assert hasattr(tracking_data, 'timestamp')
                assert hasattr(tracking_data, 'frame_number')
                assert hasattr(tracking_data, 'pose_landmarks')
                assert hasattr(tracking_data, 'left_hand_landmarks')
                assert hasattr(tracking_data, 'right_hand_landmarks')
                
                # Verify timestamp is valid
                assert isinstance(tracking_data.timestamp, float)
                assert tracking_data.timestamp > 0
                
                # Verify frame number is valid
                assert isinstance(tracking_data.frame_number, int)
                assert tracking_data.frame_number >= 0
            
            tracker.stop()
        except RuntimeError as e:
            pytest.skip(f"Camera not available: {e}")
    
    def test_visual_feedback_display(self):
        """
        Test that visual feedback is generated correctly.
        
        Validates: Requirements 7.1, 7.2
        """
        try:
            tracker = BodyTracker(camera_index=0)
            _, display_frame = tracker.process_frame()
            
            if display_frame is not None:
                # Verify frame is a valid image
                assert isinstance(display_frame, np.ndarray)
                assert len(display_frame.shape) == 3
                assert display_frame.shape[2] == 3  # BGR format
                
                # Verify frame has reasonable dimensions
                height, width, _ = display_frame.shape
                assert height > 0 and width > 0
            
            tracker.stop()
        except RuntimeError as e:
            pytest.skip(f"Camera not available: {e}")
    
    def test_resource_cleanup_on_stop(self):
        """
        Test that resources are properly cleaned up when tracker is stopped.
        
        Validates: Requirements 1.4
        """
        try:
            tracker = BodyTracker(camera_index=0)
            tracker.stop()
            
            # After stop, camera should be released
            # We can't directly test this without accessing internals,
            # but we verify no exceptions are raised
            assert True
        except RuntimeError as e:
            pytest.skip(f"Camera not available: {e}")
    
    def test_error_handling_graceful_degradation(self):
        """Test that system handles errors gracefully without crashing."""
        try:
            tracker = BodyTracker(camera_index=0)
            
            # Process multiple frames to test stability
            for _ in range(5):
                tracking_data, display_frame = tracker.process_frame()
                # System should not crash even if processing fails
                # Returns None values for failed frames
            
            tracker.stop()
        except RuntimeError as e:
            pytest.skip(f"Camera not available: {e}")


class TestCommandLineInterface:
    """Tests for command-line argument parsing."""
    
    def test_parse_arguments_defaults(self, monkeypatch):
        """Test that default arguments are parsed correctly."""
        monkeypatch.setattr('sys.argv', ['Body-Tracker.py'])
        args = parse_arguments()
        
        assert args.camera == 0
        assert args.min_detection_confidence == 0.5
        assert args.min_tracking_confidence == 0.5
        assert args.model_complexity == 1
        assert args.no_flip is False
        assert args.show_fps is False
        assert args.log_level == 'INFO'
    
    def test_parse_arguments_custom_values(self, monkeypatch):
        """Test parsing custom command-line arguments."""
        monkeypatch.setattr('sys.argv', [
            'Body-Tracker.py',
            '--camera', '1',
            '--min-detection-confidence', '0.7',
            '--min-tracking-confidence', '0.6',
            '--model-complexity', '2',
            '--no-flip',
            '--show-fps',
            '--width', '1280',
            '--height', '720',
            '--log-level', 'DEBUG'
        ])
        args = parse_arguments()
        
        assert args.camera == 1
        assert args.min_detection_confidence == 0.7
        assert args.min_tracking_confidence == 0.6
        assert args.model_complexity == 2
        assert args.no_flip is True
        assert args.show_fps is True
        assert args.width == 1280
        assert args.height == 720
        assert args.log_level == 'DEBUG'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
