"""
Unit tests for MovementAnalyzer class.

Tests displacement calculation, velocity calculation, stationary detection,
and buffer wraparound behavior.
"""

import pytest
from tap_detection.movement_analyzer import MovementAnalyzer
from tap_detection.tap_models import FootLandmarks
from comp_vision.models import LandmarkData


def create_foot_landmarks(
    heel_y: float = 0.5,
    ankle_y: float = 0.4,
    toe_y: float = 0.5,
    timestamp: float = 0.0,
    visibility: float = 0.9
) -> FootLandmarks:
    """
    Helper function to create FootLandmarks for testing.
    
    Args:
        heel_y: Y-coordinate for heel (default 0.5)
        ankle_y: Y-coordinate for ankle (default 0.4)
        toe_y: Y-coordinate for toe (default 0.5)
        timestamp: Frame timestamp in seconds (default 0.0)
        visibility: Confidence score for all landmarks (default 0.9)
        
    Returns:
        FootLandmarks instance with specified parameters
    """
    heel = LandmarkData(
        x=0.5, y=heel_y, z=0.0, 
        visibility=visibility, 
        landmark_type="RIGHT_HEEL"
    )
    ankle = LandmarkData(
        x=0.5, y=ankle_y, z=0.0, 
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


class TestMovementAnalyzer:
    """Test suite for MovementAnalyzer class."""
    
    def test_initialization(self):
        """Test MovementAnalyzer initialization with default and custom history size."""
        # Default history size
        analyzer = MovementAnalyzer()
        assert analyzer.history_size == 10
        assert len(analyzer._history) == 0
        
        # Custom history size
        analyzer_custom = MovementAnalyzer(history_size=5)
        assert analyzer_custom.history_size == 5
        assert len(analyzer_custom._history) == 0
    
    def test_update_adds_landmarks_to_history(self):
        """Test that update() adds landmarks to history buffer."""
        analyzer = MovementAnalyzer(history_size=3)
        
        # Add first frame
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        analyzer.update(landmarks1)
        assert len(analyzer._history) == 1
        
        # Add second frame
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.033)
        analyzer.update(landmarks2)
        assert len(analyzer._history) == 2
        
        # Add third frame
        landmarks3 = create_foot_landmarks(heel_y=0.40, timestamp=0.066)
        analyzer.update(landmarks3)
        assert len(analyzer._history) == 3
    
    def test_buffer_wraparound_behavior(self):
        """Test that buffer correctly wraps around when exceeding history_size."""
        analyzer = MovementAnalyzer(history_size=3)
        
        # Add 3 frames (fill buffer)
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.033)
        landmarks3 = create_foot_landmarks(heel_y=0.40, timestamp=0.066)
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        analyzer.update(landmarks3)
        
        assert len(analyzer._history) == 3
        assert analyzer._history[0].heel.y == 0.5
        
        # Add 4th frame (should remove oldest)
        landmarks4 = create_foot_landmarks(heel_y=0.35, timestamp=0.099)
        analyzer.update(landmarks4)
        
        assert len(analyzer._history) == 3
        assert analyzer._history[0].heel.y == 0.45  # First frame removed
        assert analyzer._history[-1].heel.y == 0.35  # New frame added
        
        # Add 5th frame
        landmarks5 = create_foot_landmarks(heel_y=0.30, timestamp=0.132)
        analyzer.update(landmarks5)
        
        assert len(analyzer._history) == 3
        assert analyzer._history[0].heel.y == 0.40  # Second frame now oldest
        assert analyzer._history[-1].heel.y == 0.30  # New frame added
    
    def test_displacement_calculation_upward_movement(self):
        """Test displacement calculation for upward foot movement."""
        analyzer = MovementAnalyzer()
        
        # Simulate upward heel movement (y decreases)
        # Frame 1: heel at y=0.5
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        analyzer.update(landmarks1)
        
        # Frame 2: heel at y=0.45 (moved up by 0.05)
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.1)
        analyzer.update(landmarks2)
        
        # Displacement should be positive for upward movement
        displacement = analyzer.get_vertical_displacement("heel", time_window_ms=200)
        assert displacement == pytest.approx(0.05, abs=0.001)
    
    def test_displacement_calculation_downward_movement(self):
        """Test displacement calculation for downward foot movement."""
        analyzer = MovementAnalyzer()
        
        # Simulate downward heel movement (y increases)
        # Frame 1: heel at y=0.4
        landmarks1 = create_foot_landmarks(heel_y=0.4, timestamp=0.0)
        analyzer.update(landmarks1)
        
        # Frame 2: heel at y=0.5 (moved down by 0.1)
        landmarks2 = create_foot_landmarks(heel_y=0.5, timestamp=0.1)
        analyzer.update(landmarks2)
        
        # Displacement should be negative for downward movement
        displacement = analyzer.get_vertical_displacement("heel", time_window_ms=200)
        assert displacement == pytest.approx(-0.1, abs=0.001)
    
    def test_displacement_with_time_window(self):
        """Test displacement calculation respects time window parameter."""
        analyzer = MovementAnalyzer()
        
        # Add frames over 300ms
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.1)
        landmarks3 = create_foot_landmarks(heel_y=0.40, timestamp=0.2)
        landmarks4 = create_foot_landmarks(heel_y=0.35, timestamp=0.3)
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        analyzer.update(landmarks3)
        analyzer.update(landmarks4)
        
        # 200ms window from frame 4 (0.3s) includes frames from 0.1s onwards (frames 2, 3, 4)
        # Displacement from frame 2 (0.45) to frame 4 (0.35) = 0.45 - 0.35 = 0.10
        displacement_200 = analyzer.get_vertical_displacement("heel", time_window_ms=200)
        assert displacement_200 == pytest.approx(0.10, abs=0.001)
        
        # 300ms window from frame 4 (0.3s) includes all frames from 0.0s onwards
        # Displacement from frame 1 (0.5) to frame 4 (0.35) = 0.5 - 0.35 = 0.15
        displacement_300 = analyzer.get_vertical_displacement("heel", time_window_ms=300)
        assert displacement_300 == pytest.approx(0.15, abs=0.001)
    
    def test_displacement_different_landmarks(self):
        """Test displacement calculation for different landmark types."""
        analyzer = MovementAnalyzer()
        
        # Frame 1: all landmarks at different positions
        landmarks1 = create_foot_landmarks(
            heel_y=0.5, ankle_y=0.4, toe_y=0.5, timestamp=0.0
        )
        analyzer.update(landmarks1)
        
        # Frame 2: heel moves up, ankle moves up, toe stays
        landmarks2 = create_foot_landmarks(
            heel_y=0.45, ankle_y=0.37, toe_y=0.5, timestamp=0.1
        )
        analyzer.update(landmarks2)
        
        # Check each landmark type
        heel_disp = analyzer.get_vertical_displacement("heel", time_window_ms=200)
        ankle_disp = analyzer.get_vertical_displacement("ankle", time_window_ms=200)
        toe_disp = analyzer.get_vertical_displacement("toe", time_window_ms=200)
        
        assert heel_disp == pytest.approx(0.05, abs=0.001)
        assert ankle_disp == pytest.approx(0.03, abs=0.001)
        assert toe_disp == pytest.approx(0.0, abs=0.001)
    
    def test_displacement_insufficient_history(self):
        """Test displacement returns 0.0 when insufficient history."""
        analyzer = MovementAnalyzer()
        
        # No frames
        displacement = analyzer.get_vertical_displacement("heel", time_window_ms=200)
        assert displacement == 0.0
        
        # Only one frame
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        analyzer.update(landmarks1)
        displacement = analyzer.get_vertical_displacement("heel", time_window_ms=200)
        assert displacement == 0.0
    
    def test_velocity_calculation(self):
        """Test velocity calculation between consecutive frames."""
        analyzer = MovementAnalyzer()
        
        # Frame 1: heel at y=0.5
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        analyzer.update(landmarks1)
        
        # Frame 2: heel at y=0.45, 0.1 seconds later
        # Displacement = 0.05, time = 0.1s, velocity = 0.5 units/s
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.1)
        analyzer.update(landmarks2)
        
        velocity = analyzer.get_vertical_velocity("heel")
        assert velocity == pytest.approx(0.5, abs=0.001)
    
    def test_velocity_downward_movement(self):
        """Test velocity calculation for downward movement."""
        analyzer = MovementAnalyzer()
        
        # Frame 1: heel at y=0.4
        landmarks1 = create_foot_landmarks(heel_y=0.4, timestamp=0.0)
        analyzer.update(landmarks1)
        
        # Frame 2: heel at y=0.5, 0.2 seconds later
        # Displacement = -0.1, time = 0.2s, velocity = -0.5 units/s
        landmarks2 = create_foot_landmarks(heel_y=0.5, timestamp=0.2)
        analyzer.update(landmarks2)
        
        velocity = analyzer.get_vertical_velocity("heel")
        assert velocity == pytest.approx(-0.5, abs=0.001)
    
    def test_velocity_different_landmarks(self):
        """Test velocity calculation for different landmark types."""
        analyzer = MovementAnalyzer()
        
        # Frame 1
        landmarks1 = create_foot_landmarks(
            heel_y=0.5, ankle_y=0.4, toe_y=0.5, timestamp=0.0
        )
        analyzer.update(landmarks1)
        
        # Frame 2: 0.1s later, different movements
        landmarks2 = create_foot_landmarks(
            heel_y=0.45, ankle_y=0.37, toe_y=0.5, timestamp=0.1
        )
        analyzer.update(landmarks2)
        
        heel_vel = analyzer.get_vertical_velocity("heel")
        ankle_vel = analyzer.get_vertical_velocity("ankle")
        toe_vel = analyzer.get_vertical_velocity("toe")
        
        assert heel_vel == pytest.approx(0.5, abs=0.001)  # 0.05 / 0.1
        assert ankle_vel == pytest.approx(0.3, abs=0.001)  # 0.03 / 0.1
        assert toe_vel == pytest.approx(0.0, abs=0.001)  # 0.0 / 0.1
    
    def test_velocity_insufficient_history(self):
        """Test velocity returns 0.0 when insufficient history."""
        analyzer = MovementAnalyzer()
        
        # No frames
        velocity = analyzer.get_vertical_velocity("heel")
        assert velocity == 0.0
        
        # Only one frame
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        analyzer.update(landmarks1)
        velocity = analyzer.get_vertical_velocity("heel")
        assert velocity == 0.0
    
    def test_velocity_zero_time_delta(self):
        """Test velocity returns 0.0 when time delta is zero."""
        analyzer = MovementAnalyzer()
        
        # Two frames with same timestamp
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.0)
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        
        velocity = analyzer.get_vertical_velocity("heel")
        assert velocity == 0.0
    
    def test_stationary_detection_true(self):
        """Test stationary detection when landmark is not moving."""
        analyzer = MovementAnalyzer()
        
        # Add multiple frames with heel staying at y=0.5
        for i in range(5):
            landmarks = create_foot_landmarks(heel_y=0.5, timestamp=i * 0.033)
            analyzer.update(landmarks)
        
        # Heel should be stationary (no movement)
        is_stationary = analyzer.is_stationary("heel", threshold=0.02)
        assert is_stationary is True
    
    def test_stationary_detection_small_movement(self):
        """Test stationary detection with small movements below threshold."""
        analyzer = MovementAnalyzer()
        
        # Add frames with small heel movements (total range 0.01)
        landmarks1 = create_foot_landmarks(heel_y=0.500, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.505, timestamp=0.033)
        landmarks3 = create_foot_landmarks(heel_y=0.510, timestamp=0.066)
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        analyzer.update(landmarks3)
        
        # Total movement is 0.01, below threshold of 0.02
        is_stationary = analyzer.is_stationary("heel", threshold=0.02)
        assert is_stationary is True
    
    def test_stationary_detection_false(self):
        """Test stationary detection when landmark is moving significantly."""
        analyzer = MovementAnalyzer()
        
        # Add frames with significant heel movement
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.033)
        landmarks3 = create_foot_landmarks(heel_y=0.40, timestamp=0.066)
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        analyzer.update(landmarks3)
        
        # Total movement is 0.1, above threshold of 0.02
        is_stationary = analyzer.is_stationary("heel", threshold=0.02)
        assert is_stationary is False
    
    def test_stationary_detection_at_threshold(self):
        """Test stationary detection at exact threshold boundary."""
        analyzer = MovementAnalyzer()
        
        # Add frames with movement exactly at threshold
        landmarks1 = create_foot_landmarks(heel_y=0.50, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.52, timestamp=0.033)
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        
        # Total movement is exactly 0.02
        is_stationary = analyzer.is_stationary("heel", threshold=0.02)
        assert is_stationary is False  # Should be False (not less than threshold)
    
    def test_stationary_detection_different_landmarks(self):
        """Test stationary detection for different landmark types."""
        analyzer = MovementAnalyzer()
        
        # Heel moves, toe stays stationary
        landmarks1 = create_foot_landmarks(
            heel_y=0.5, ankle_y=0.4, toe_y=0.5, timestamp=0.0
        )
        landmarks2 = create_foot_landmarks(
            heel_y=0.45, ankle_y=0.37, toe_y=0.5, timestamp=0.033)
        landmarks3 = create_foot_landmarks(
            heel_y=0.40, ankle_y=0.34, toe_y=0.5, timestamp=0.066
        )
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        analyzer.update(landmarks3)
        
        heel_stationary = analyzer.is_stationary("heel", threshold=0.02)
        ankle_stationary = analyzer.is_stationary("ankle", threshold=0.02)
        toe_stationary = analyzer.is_stationary("toe", threshold=0.02)
        
        assert heel_stationary is False  # Moved 0.1
        assert ankle_stationary is False  # Moved 0.06
        assert toe_stationary is True  # No movement
    
    def test_stationary_insufficient_history(self):
        """Test stationary returns True when insufficient history."""
        analyzer = MovementAnalyzer()
        
        # No frames - assume stationary
        is_stationary = analyzer.is_stationary("heel", threshold=0.02)
        assert is_stationary is True
        
        # Only one frame - assume stationary
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        analyzer.update(landmarks1)
        is_stationary = analyzer.is_stationary("heel", threshold=0.02)
        assert is_stationary is True
    
    def test_reset_clears_history(self):
        """Test that reset() clears all movement history."""
        analyzer = MovementAnalyzer()
        
        # Add some frames
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.033)
        landmarks3 = create_foot_landmarks(heel_y=0.40, timestamp=0.066)
        
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        analyzer.update(landmarks3)
        
        assert len(analyzer._history) == 3
        
        # Reset
        analyzer.reset()
        
        assert len(analyzer._history) == 0
        
        # Verify calculations return default values after reset
        displacement = analyzer.get_vertical_displacement("heel", time_window_ms=200)
        velocity = analyzer.get_vertical_velocity("heel")
        is_stationary = analyzer.is_stationary("heel", threshold=0.02)
        
        assert displacement == 0.0
        assert velocity == 0.0
        assert is_stationary is True
    
    def test_invalid_landmark_type_raises_error(self):
        """Test that invalid landmark type raises ValueError."""
        analyzer = MovementAnalyzer()
        
        # Add two frames so methods don't return early
        landmarks1 = create_foot_landmarks(heel_y=0.5, timestamp=0.0)
        landmarks2 = create_foot_landmarks(heel_y=0.45, timestamp=0.1)
        analyzer.update(landmarks1)
        analyzer.update(landmarks2)
        
        # Test with invalid landmark type
        with pytest.raises(ValueError, match="Unknown landmark type"):
            analyzer.get_vertical_displacement("invalid_type", time_window_ms=200)
        
        with pytest.raises(ValueError, match="Unknown landmark type"):
            analyzer.get_vertical_velocity("invalid_type")
        
        with pytest.raises(ValueError, match="Unknown landmark type"):
            analyzer.is_stationary("invalid_type", threshold=0.02)
