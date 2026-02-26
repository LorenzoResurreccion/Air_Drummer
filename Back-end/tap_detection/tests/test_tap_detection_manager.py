"""
Unit tests for TapDetectionManager.

Tests the main controller that orchestrates the complete tap detection pipeline.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from comp_vision.models import TrackingData, LandmarkData
from tap_detection.tap_detection_manager import TapDetectionManager
from tap_detection.tap_models import (
    TapDetectionConfig,
    TapType,
    DetectionState
)


def create_test_tracking_data(
    timestamp: float,
    heel_y: float = 0.8,
    ankle_y: float = 0.7,
    toe_y: float = 0.85,
    confidence: float = 0.9
) -> TrackingData:
    """Helper to create TrackingData with right foot landmarks."""
    pose_landmarks = [None] * 33
    
    # RIGHT_ANKLE (index 28)
    pose_landmarks[28] = LandmarkData(
        x=0.5, y=ankle_y, z=0.0,
        visibility=confidence,
        landmark_type="RIGHT_ANKLE"
    )
    
    # RIGHT_HEEL (index 30)
    pose_landmarks[30] = LandmarkData(
        x=0.5, y=heel_y, z=0.0,
        visibility=confidence,
        landmark_type="RIGHT_HEEL"
    )
    
    # RIGHT_FOOT_INDEX (index 32)
    pose_landmarks[32] = LandmarkData(
        x=0.5, y=toe_y, z=0.0,
        visibility=confidence,
        landmark_type="RIGHT_FOOT_INDEX"
    )
    
    return TrackingData(
        timestamp=timestamp,
        frame_number=0,
        pose_landmarks=pose_landmarks,
        left_hand_landmarks=None,
        right_hand_landmarks=None
    )


class TestTapDetectionManagerInitialization:
    """Test TapDetectionManager initialization."""
    
    def test_initialization_with_default_config(self):
        """Test manager initializes with default configuration."""
        manager = TapDetectionManager()
        
        assert manager.config is not None
        assert manager.extractor is not None
        assert manager.detector is not None
        assert manager.debounce_filter is not None
        assert manager.trigger is not None
    
    def test_initialization_with_custom_config(self):
        """Test manager initializes with custom configuration."""
        config = TapDetectionConfig(
            min_displacement=0.08,
            min_velocity=0.3,
            max_tap_duration_ms=300
        )
        
        manager = TapDetectionManager(config=config)
        
        assert manager.config.min_displacement == 0.08
        assert manager.config.min_velocity == 0.3
        assert manager.config.max_tap_duration_ms == 300
    
    def test_initialization_with_custom_debounce(self):
        """Test manager initializes with custom debounce window."""
        manager = TapDetectionManager(debounce_ms=200)
        
        assert manager.debounce_filter.debounce_window_ms == 200


class TestTapDetectionManagerProcessing:
    """Test TapDetectionManager processing pipeline."""
    
    def test_process_tracking_data_with_missing_landmarks(self):
        """Test processing handles missing pose landmarks gracefully."""
        manager = TapDetectionManager()
        
        tracking_data = TrackingData(
            timestamp=time.time(),
            frame_number=0,
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        # Should not raise exception
        manager.process_tracking_data(tracking_data)
        
        # State should remain IDLE
        assert manager.get_state() == DetectionState.IDLE
    
    def test_process_tracking_data_with_low_confidence(self):
        """Test processing filters out low confidence landmarks."""
        manager = TapDetectionManager()
        
        tracking_data = create_test_tracking_data(
            timestamp=time.time(),
            confidence=0.3  # Below default threshold of 0.5
        )
        
        manager.process_tracking_data(tracking_data)
        
        # State should remain IDLE due to low confidence
        assert manager.get_state() == DetectionState.IDLE
    
    def test_process_tracking_data_normal_operation(self):
        """Test processing with valid tracking data."""
        manager = TapDetectionManager()
        
        timestamp = time.time()
        tracking_data = create_test_tracking_data(timestamp)
        
        # Should process without errors
        manager.process_tracking_data(tracking_data)
        
        # State should be IDLE (no tap detected yet)
        assert manager.get_state() == DetectionState.IDLE


class TestTapDetectionManagerStateDelegation:
    """Test TapDetectionManager state delegation methods."""
    
    def test_get_state(self):
        """Test get_state delegates to detector."""
        manager = TapDetectionManager()
        
        state = manager.get_state()
        
        assert isinstance(state, DetectionState)
        assert state == DetectionState.IDLE
    
    def test_get_tap_history_empty(self):
        """Test get_tap_history returns empty list initially."""
        manager = TapDetectionManager()
        
        history = manager.get_tap_history()
        
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_get_trigger_count(self):
        """Test get_trigger_count returns initial count."""
        manager = TapDetectionManager()
        
        count = manager.get_trigger_count()
        
        assert count == 0
    
    def test_is_debouncing_initially_false(self):
        """Test is_debouncing returns False initially."""
        manager = TapDetectionManager()
        
        assert manager.is_debouncing() is False


class TestTapDetectionManagerConfigUpdate:
    """Test TapDetectionManager configuration updates."""
    
    def test_update_config_with_valid_config(self):
        """Test updating configuration with valid parameters."""
        manager = TapDetectionManager()
        
        new_config = TapDetectionConfig(
            min_displacement=0.07,
            min_velocity=0.35,
            max_tap_duration_ms=350
        )
        
        manager.update_config(new_config)
        
        assert manager.config.min_displacement == 0.07
        assert manager.config.min_velocity == 0.35
        assert manager.config.max_tap_duration_ms == 350
    
    def test_update_config_preserves_tap_history(self):
        """Test configuration update preserves tap history."""
        manager = TapDetectionManager()
        
        # Manually add a tap to history for testing
        from tap_detection.tap_models import TapEvent
        test_tap = TapEvent(
            tap_type=TapType.HEEL,
            timestamp=time.time(),
            peak_displacement=0.06,
            duration_ms=150.0
        )
        manager.detector._tap_history.append(test_tap)
        
        # Update configuration
        new_config = TapDetectionConfig(min_displacement=0.07)
        manager.update_config(new_config)
        
        # History should be preserved
        history = manager.get_tap_history()
        assert len(history) == 1
        assert history[0].tap_type == TapType.HEEL
    
    def test_update_config_with_invalid_config_raises_error(self):
        """Test updating with invalid configuration raises ValueError."""
        manager = TapDetectionManager()
        
        invalid_config = TapDetectionConfig(
            min_displacement=0.5  # Out of valid range (0.01-0.15)
        )
        
        with pytest.raises(ValueError):
            manager.update_config(invalid_config)


class TestTapDetectionManagerSocketIO:
    """Test TapDetectionManager Socket.IO integration."""
    
    def test_get_socketio_server(self):
        """Test getting Socket.IO server instance."""
        manager = TapDetectionManager()
        
        sio = manager.get_socketio_server()
        
        assert sio is not None
        # Should be a socketio.Server instance
        import socketio
        assert isinstance(sio, socketio.Server)
    
    def test_initialization_with_custom_socketio_server(self):
        """Test manager can be initialized with custom Socket.IO server."""
        import socketio
        custom_sio = socketio.Server(cors_allowed_origins='http://localhost:3000')
        
        manager = TapDetectionManager(sio=custom_sio)
        
        # Should use the provided server
        assert manager.get_socketio_server() is custom_sio


class TestTapDetectionManagerErrorHandling:
    """Test TapDetectionManager error handling."""
    
    def test_process_tracking_data_handles_exceptions_gracefully(self):
        """Test processing handles unexpected exceptions without crashing."""
        manager = TapDetectionManager()
        
        # Create malformed tracking data that might cause issues
        tracking_data = TrackingData(
            timestamp=time.time(),
            frame_number=0,
            pose_landmarks=[],  # Empty list instead of None or proper landmarks
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        # Should not raise exception
        manager.process_tracking_data(tracking_data)
        
        # Manager should still be operational
        assert manager.get_state() == DetectionState.IDLE
