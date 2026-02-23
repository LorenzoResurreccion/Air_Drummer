"""
Property-based tests for landmark data validity.

Feature: full-body-tracking
Property 5: Landmark data validity

Validates: Requirements 5.1, 5.2, 2.4, 3.3, 4.4

Tests that all landmark coordinates are in [0.0, 1.0] range and
all confidence scores are in [0.0, 1.0] range.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hypothesis import given, strategies as st, settings
from models import LandmarkData, TrackingData


# Strategy for generating valid landmark data
@st.composite
def valid_landmark(draw):
    """Generate a valid LandmarkData instance with coordinates and visibility in [0.0, 1.0]."""
    return LandmarkData(
        x=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        y=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        z=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),  # z can be negative
        visibility=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        landmark_type=draw(st.sampled_from([
            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
        ]))
    )


@st.composite
def valid_landmark_list(draw, min_size=1, max_size=22):
    """Generate a list of valid landmarks."""
    return draw(st.lists(valid_landmark(), min_size=min_size, max_size=max_size))


@st.composite
def valid_tracking_data(draw):
    """Generate valid TrackingData with optional landmark collections."""
    return TrackingData(
        timestamp=draw(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        frame_number=draw(st.integers(min_value=0, max_value=10000)),
        pose_landmarks=draw(st.one_of(st.none(), valid_landmark_list(min_size=1, max_size=22))),
        left_hand_landmarks=draw(st.one_of(st.none(), valid_landmark_list(min_size=1, max_size=21))),
        right_hand_landmarks=draw(st.one_of(st.none(), valid_landmark_list(min_size=1, max_size=21)))
    )


class TestLandmarkDataValidity:
    """
    Property 5: Landmark data validity
    
    For any detected landmark in the tracking data, the position coordinates (x, y, z)
    should be in normalized format within the range [0.0, 1.0], and the confidence
    score should be within the range [0.0, 1.0].
    """
    
    @settings(max_examples=100)
    @given(landmark=valid_landmark())
    def test_landmark_coordinates_in_valid_range(self, landmark):
        """
        Test that landmark x and y coordinates are in [0.0, 1.0] range.
        
        **Feature: full-body-tracking, Property 5: Landmark data validity**
        **Validates: Requirements 5.1, 5.2**
        """
        assert 0.0 <= landmark.x <= 1.0, f"x coordinate {landmark.x} is out of range [0.0, 1.0]"
        assert 0.0 <= landmark.y <= 1.0, f"y coordinate {landmark.y} is out of range [0.0, 1.0]"
    
    @settings(max_examples=100)
    @given(landmark=valid_landmark())
    def test_landmark_confidence_in_valid_range(self, landmark):
        """
        Test that landmark confidence/visibility score is in [0.0, 1.0] range.
        
        **Feature: full-body-tracking, Property 5: Landmark data validity**
        **Validates: Requirements 5.2, 2.4, 3.3, 4.4**
        """
        assert 0.0 <= landmark.visibility <= 1.0, \
            f"Confidence score {landmark.visibility} is out of range [0.0, 1.0]"
    
    @settings(max_examples=100)
    @given(landmark=valid_landmark())
    def test_landmark_is_valid_method(self, landmark):
        """
        Test that the is_valid() method correctly validates landmark data.
        
        **Feature: full-body-tracking, Property 5: Landmark data validity**
        **Validates: Requirements 5.1, 5.2**
        """
        # Since we're generating valid landmarks, is_valid() should always return True
        assert landmark.is_valid(), \
            f"Valid landmark failed is_valid() check: x={landmark.x}, y={landmark.y}, visibility={landmark.visibility}"
    
    @settings(max_examples=100)
    @given(tracking_data=valid_tracking_data())
    def test_all_landmarks_in_tracking_data_are_valid(self, tracking_data):
        """
        Test that all landmarks in TrackingData have valid coordinates and confidence scores.
        
        **Feature: full-body-tracking, Property 5: Landmark data validity**
        **Validates: Requirements 5.1, 5.2, 2.4, 3.3, 4.4**
        """
        # Check pose landmarks
        if tracking_data.pose_landmarks:
            for landmark in tracking_data.pose_landmarks:
                assert 0.0 <= landmark.x <= 1.0, \
                    f"Pose landmark x={landmark.x} out of range"
                assert 0.0 <= landmark.y <= 1.0, \
                    f"Pose landmark y={landmark.y} out of range"
                assert 0.0 <= landmark.visibility <= 1.0, \
                    f"Pose landmark visibility={landmark.visibility} out of range"
        
        # Check left hand landmarks
        if tracking_data.left_hand_landmarks:
            for landmark in tracking_data.left_hand_landmarks:
                assert 0.0 <= landmark.x <= 1.0, \
                    f"Left hand landmark x={landmark.x} out of range"
                assert 0.0 <= landmark.y <= 1.0, \
                    f"Left hand landmark y={landmark.y} out of range"
                assert 0.0 <= landmark.visibility <= 1.0, \
                    f"Left hand landmark visibility={landmark.visibility} out of range"
        
        # Check right hand landmarks
        if tracking_data.right_hand_landmarks:
            for landmark in tracking_data.right_hand_landmarks:
                assert 0.0 <= landmark.x <= 1.0, \
                    f"Right hand landmark x={landmark.x} out of range"
                assert 0.0 <= landmark.y <= 1.0, \
                    f"Right hand landmark y={landmark.y} out of range"
                assert 0.0 <= landmark.visibility <= 1.0, \
                    f"Right hand landmark visibility={landmark.visibility} out of range"
    
    @settings(max_examples=100)
    @given(landmark_list=valid_landmark_list(min_size=5, max_size=22))
    def test_multiple_landmarks_all_valid(self, landmark_list):
        """
        Test that collections of landmarks all maintain validity.
        
        **Feature: full-body-tracking, Property 5: Landmark data validity**
        **Validates: Requirements 5.1, 5.2**
        """
        for landmark in landmark_list:
            assert landmark.is_valid(), \
                f"Landmark {landmark.landmark_type} failed validity check"
            assert 0.0 <= landmark.x <= 1.0
            assert 0.0 <= landmark.y <= 1.0
            assert 0.0 <= landmark.visibility <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
