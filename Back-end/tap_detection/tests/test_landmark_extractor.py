"""
Unit tests for FootLandmarkExtractor.

Tests landmark extraction, validation, and handling of missing data.

Feature: right-foot-bass-drum
"""

import pytest
from hypothesis import given, strategies as st

from comp_vision.models import TrackingData, LandmarkData
from tap_detection.landmark_extractor import FootLandmarkExtractor, RIGHT_ANKLE_INDEX, RIGHT_HEEL_INDEX, RIGHT_FOOT_INDEX


def create_test_landmark(x: float, y: float, z: float, visibility: float, landmark_type: str) -> LandmarkData:
    """Helper to create test landmark data."""
    return LandmarkData(
        x=x,
        y=y,
        z=z,
        visibility=visibility,
        landmark_type=landmark_type
    )


def create_full_pose_landmarks(heel_vis: float = 0.9, ankle_vis: float = 0.9, toe_vis: float = 0.9):
    """Helper to create a full set of 33 pose landmarks with specified foot visibilities."""
    landmarks = []
    for i in range(33):
        if i == 28:  # RIGHT_ANKLE
            landmarks.append(create_test_landmark(0.5, 0.8, 0.0, ankle_vis, "RIGHT_ANKLE"))
        elif i == 30:  # RIGHT_HEEL
            landmarks.append(create_test_landmark(0.5, 0.9, 0.0, heel_vis, "RIGHT_HEEL"))
        elif i == 32:  # RIGHT_FOOT_INDEX
            landmarks.append(create_test_landmark(0.6, 0.9, 0.0, toe_vis, "RIGHT_FOOT_INDEX"))
        else:
            landmarks.append(create_test_landmark(0.5, 0.5, 0.0, 0.9, f"LANDMARK_{i}"))
    return landmarks


class TestFootLandmarkExtractor:
    """Test suite for FootLandmarkExtractor class."""
    
    def test_extract_with_valid_pose_landmarks(self):
        """Test extraction with complete pose landmarks."""
        extractor = FootLandmarkExtractor()
        
        pose_landmarks = create_full_pose_landmarks()
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        result = extractor.extract(tracking_data)
        
        assert result is not None
        assert result.heel.landmark_type == "RIGHT_HEEL"
        assert result.ankle.landmark_type == "RIGHT_ANKLE"
        assert result.toe.landmark_type == "RIGHT_FOOT_INDEX"
        assert result.timestamp == 1.0
    
    def test_extract_with_none_pose_landmarks(self):
        """Test extraction returns None when pose_landmarks is None."""
        extractor = FootLandmarkExtractor()
        
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        result = extractor.extract(tracking_data)
        
        assert result is None
    
    def test_extract_with_insufficient_landmarks(self):
        """Test extraction returns None when pose_landmarks has fewer than 33 landmarks."""
        extractor = FootLandmarkExtractor()
        
        # Only 10 landmarks instead of 33
        pose_landmarks = [create_test_landmark(0.5, 0.5, 0.0, 0.9, f"LANDMARK_{i}") for i in range(10)]
        
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        result = extractor.extract(tracking_data)
        
        assert result is None
    
    def test_validate_landmarks_with_high_confidence(self):
        """Test validation passes with high confidence landmarks."""
        extractor = FootLandmarkExtractor()
        
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.9, ankle_vis=0.9, toe_vis=0.9)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        
        assert result is True
    
    def test_validate_landmarks_with_low_confidence(self):
        """Test validation fails with low confidence landmarks."""
        extractor = FootLandmarkExtractor()
        
        # Heel has low visibility
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.3, ankle_vis=0.9, toe_vis=0.9)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        
        assert result is False
    
    def test_validate_landmarks_at_threshold_boundary(self):
        """Test validation at exact threshold boundary."""
        extractor = FootLandmarkExtractor()
        
        # All landmarks exactly at threshold
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.5, ankle_vis=0.5, toe_vis=0.5)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # At threshold should pass (>=)
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is True
        
        # Just below threshold should fail
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.51)
        assert result is False


class TestConfidenceValidation:
    """Unit tests for confidence validation - Requirements 1.4, 2.4, 3.4."""
    
    def test_validate_all_landmarks_above_threshold(self):
        """Test validation passes when all landmarks are above threshold."""
        extractor = FootLandmarkExtractor()
        
        # All landmarks well above default threshold (0.5)
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.8, ankle_vis=0.9, toe_vis=0.85)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is True
    
    def test_validate_heel_below_threshold(self):
        """Test validation fails when heel confidence is below threshold."""
        extractor = FootLandmarkExtractor()
        
        # Heel below threshold, others above
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.3, ankle_vis=0.9, toe_vis=0.9)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is False
    
    def test_validate_ankle_below_threshold(self):
        """Test validation fails when ankle confidence is below threshold."""
        extractor = FootLandmarkExtractor()
        
        # Ankle below threshold, others above
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.9, ankle_vis=0.2, toe_vis=0.9)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is False
    
    def test_validate_toe_below_threshold(self):
        """Test validation fails when toe confidence is below threshold."""
        extractor = FootLandmarkExtractor()
        
        # Toe below threshold, others above
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.9, ankle_vis=0.9, toe_vis=0.4)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is False
    
    def test_validate_multiple_landmarks_below_threshold(self):
        """Test validation fails when multiple landmarks are below threshold."""
        extractor = FootLandmarkExtractor()
        
        # Multiple landmarks below threshold
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.3, ankle_vis=0.2, toe_vis=0.9)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is False
    
    def test_validate_all_landmarks_below_threshold(self):
        """Test validation fails when all landmarks are below threshold."""
        extractor = FootLandmarkExtractor()
        
        # All landmarks below threshold
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.1, ankle_vis=0.2, toe_vis=0.3)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is False
    
    def test_validate_with_custom_threshold(self):
        """Test validation with custom confidence threshold."""
        extractor = FootLandmarkExtractor()
        
        # Landmarks at 0.7 visibility
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.7, ankle_vis=0.7, toe_vis=0.7)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # Should pass with threshold 0.6
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.6)
        assert result is True
        
        # Should fail with threshold 0.8
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.8)
        assert result is False
    
    def test_validate_exact_threshold_boundary_all_landmarks(self):
        """Test validation at exact threshold boundary for all landmarks."""
        extractor = FootLandmarkExtractor()
        
        # All landmarks exactly at 0.5
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.5, ankle_vis=0.5, toe_vis=0.5)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # Exactly at threshold should pass (>= comparison)
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is True
    
    def test_validate_just_below_threshold_boundary(self):
        """Test validation just below threshold boundary."""
        extractor = FootLandmarkExtractor()
        
        # Landmarks just below threshold (0.49)
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.49, ankle_vis=0.49, toe_vis=0.49)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # Just below threshold should fail
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is False
    
    def test_validate_just_above_threshold_boundary(self):
        """Test validation just above threshold boundary."""
        extractor = FootLandmarkExtractor()
        
        # Landmarks just above threshold (0.51)
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.51, ankle_vis=0.51, toe_vis=0.51)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # Just above threshold should pass
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is True
    
    def test_validate_mixed_confidence_one_at_boundary(self):
        """Test validation with mixed confidence scores, one at boundary."""
        extractor = FootLandmarkExtractor()
        
        # Heel at boundary, others above
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.5, ankle_vis=0.9, toe_vis=0.8)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # Should pass since all are >= threshold
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is True
    
    def test_validate_very_low_confidence(self):
        """Test validation with very low confidence scores."""
        extractor = FootLandmarkExtractor()
        
        # Very low confidence (near 0)
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.01, ankle_vis=0.02, toe_vis=0.03)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is False
    
    def test_validate_very_high_confidence(self):
        """Test validation with very high confidence scores."""
        extractor = FootLandmarkExtractor()
        
        # Very high confidence (near 1.0)
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.99, ankle_vis=0.98, toe_vis=1.0)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.5)
        assert result is True
    
    def test_validate_with_zero_threshold(self):
        """Test validation with zero threshold (should always pass)."""
        extractor = FootLandmarkExtractor()
        
        # Very low confidence
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.01, ankle_vis=0.02, toe_vis=0.03)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # With threshold 0.0, even very low confidence should pass
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=0.0)
        assert result is True
    
    def test_validate_with_maximum_threshold(self):
        """Test validation with maximum threshold (1.0)."""
        extractor = FootLandmarkExtractor()
        
        # High confidence but not perfect
        pose_landmarks = create_full_pose_landmarks(heel_vis=0.99, ankle_vis=0.98, toe_vis=0.97)
        tracking_data = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks = extractor.extract(tracking_data)
        assert foot_landmarks is not None
        
        # With threshold 1.0, should fail unless all are exactly 1.0
        result = extractor.validate_landmarks(foot_landmarks, confidence_threshold=1.0)
        assert result is False
        
        # Test with perfect confidence
        pose_landmarks_perfect = create_full_pose_landmarks(heel_vis=1.0, ankle_vis=1.0, toe_vis=1.0)
        tracking_data_perfect = TrackingData(
            timestamp=1.0,
            frame_number=1,
            pose_landmarks=pose_landmarks_perfect,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        foot_landmarks_perfect = extractor.extract(tracking_data_perfect)
        assert foot_landmarks_perfect is not None
        
        result = extractor.validate_landmarks(foot_landmarks_perfect, confidence_threshold=1.0)
        assert result is True



class TestLandmarkExtractionProperty:
    """Property-based tests for landmark extraction."""
    
    @given(
        # Generate random coordinates and visibility for landmarks
        heel_x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        heel_y=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        heel_z=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        heel_vis=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ankle_x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ankle_y=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ankle_z=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ankle_vis=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        toe_x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        toe_y=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        toe_z=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        toe_vis=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        timestamp=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        frame_number=st.integers(min_value=0, max_value=100000)
    )
    def test_property_correct_landmark_extraction(
        self,
        heel_x: float, heel_y: float, heel_z: float, heel_vis: float,
        ankle_x: float, ankle_y: float, ankle_z: float, ankle_vis: float,
        toe_x: float, toe_y: float, toe_z: float, toe_vis: float,
        timestamp: float,
        frame_number: int
    ):
        """
        Property 14: Correct landmark extraction
        
        For any TrackingData with pose_landmarks, the FootLandmarkExtractor should
        extract landmarks at the correct indices (RIGHT_HEEL at 30, RIGHT_ANKLE at 28,
        RIGHT_FOOT_INDEX at 32) and return them in the FootLandmarks structure with
        the correct timestamp.
        
        The extracted landmarks should:
        1. Be at the correct indices (28, 30, 32)
        2. Preserve all coordinate values (x, y, z, visibility)
        3. Include the correct timestamp from TrackingData
        4. Have the correct landmark types
        
        Validates: Requirements 9.2
        """
        extractor = FootLandmarkExtractor()
        
        # Create a full set of 33 pose landmarks with random data for non-foot landmarks
        pose_landmarks = []
        for i in range(33):
            if i == RIGHT_ANKLE_INDEX:  # 28
                landmark = LandmarkData(
                    x=ankle_x,
                    y=ankle_y,
                    z=ankle_z,
                    visibility=ankle_vis,
                    landmark_type="RIGHT_ANKLE"
                )
            elif i == RIGHT_HEEL_INDEX:  # 30
                landmark = LandmarkData(
                    x=heel_x,
                    y=heel_y,
                    z=heel_z,
                    visibility=heel_vis,
                    landmark_type="RIGHT_HEEL"
                )
            elif i == RIGHT_FOOT_INDEX:  # 32
                landmark = LandmarkData(
                    x=toe_x,
                    y=toe_y,
                    z=toe_z,
                    visibility=toe_vis,
                    landmark_type="RIGHT_FOOT_INDEX"
                )
            else:
                # Random data for other landmarks
                landmark = LandmarkData(
                    x=0.5,
                    y=0.5,
                    z=0.0,
                    visibility=0.9,
                    landmark_type=f"LANDMARK_{i}"
                )
            pose_landmarks.append(landmark)
        
        # Create TrackingData with the pose landmarks
        tracking_data = TrackingData(
            timestamp=timestamp,
            frame_number=frame_number,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        # Extract foot landmarks
        result = extractor.extract(tracking_data)
        
        # Property assertions: Correct extraction
        assert result is not None, "Extraction should succeed with valid pose landmarks"
        
        # Verify correct landmarks were extracted (by checking landmark_type)
        assert result.heel.landmark_type == "RIGHT_HEEL", \
            "Heel should be extracted from RIGHT_HEEL landmark"
        assert result.ankle.landmark_type == "RIGHT_ANKLE", \
            "Ankle should be extracted from RIGHT_ANKLE landmark"
        assert result.toe.landmark_type == "RIGHT_FOOT_INDEX", \
            "Toe should be extracted from RIGHT_FOOT_INDEX landmark"
        
        # Verify all coordinate values are preserved exactly
        assert result.heel.x == heel_x, "Heel x coordinate should be preserved"
        assert result.heel.y == heel_y, "Heel y coordinate should be preserved"
        assert result.heel.z == heel_z, "Heel z coordinate should be preserved"
        assert result.heel.visibility == heel_vis, "Heel visibility should be preserved"
        
        assert result.ankle.x == ankle_x, "Ankle x coordinate should be preserved"
        assert result.ankle.y == ankle_y, "Ankle y coordinate should be preserved"
        assert result.ankle.z == ankle_z, "Ankle z coordinate should be preserved"
        assert result.ankle.visibility == ankle_vis, "Ankle visibility should be preserved"
        
        assert result.toe.x == toe_x, "Toe x coordinate should be preserved"
        assert result.toe.y == toe_y, "Toe y coordinate should be preserved"
        assert result.toe.z == toe_z, "Toe z coordinate should be preserved"
        assert result.toe.visibility == toe_vis, "Toe visibility should be preserved"
        
        # Verify timestamp is preserved
        assert result.timestamp == timestamp, \
            "Timestamp should be preserved from TrackingData"


class TestMissingPoseDataProperty:
    """Property-based tests for missing pose data handling."""
    
    @given(
        timestamp=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        frame_number=st.integers(min_value=0, max_value=100000),
        # Generate different scenarios for missing/invalid pose data
        pose_data_scenario=st.sampled_from([
            'none',           # pose_landmarks is None
            'empty_list',     # pose_landmarks is empty list
            'insufficient',   # pose_landmarks has < 33 landmarks
        ]),
        # For insufficient scenario, generate number of landmarks
        num_insufficient_landmarks=st.integers(min_value=0, max_value=32)
    )
    def test_property_graceful_missing_pose_data(
        self,
        timestamp: float,
        frame_number: int,
        pose_data_scenario: str,
        num_insufficient_landmarks: int
    ):
        """
        Property 15: Graceful handling of missing pose data
        
        For any TrackingData where pose_landmarks is None, empty, or insufficient,
        the FootLandmarkExtractor should handle this gracefully without raising
        exceptions and return None to indicate that foot landmarks cannot be extracted.
        
        The extractor should:
        1. Not raise any exceptions when pose_landmarks is None
        2. Not raise any exceptions when pose_landmarks is empty
        3. Not raise any exceptions when pose_landmarks has insufficient landmarks
        4. Return None in all these cases to signal missing data
        5. Allow the system to continue processing subsequent frames
        
        Validates: Requirements 9.3
        """
        extractor = FootLandmarkExtractor()
        
        # Create pose_landmarks based on scenario
        if pose_data_scenario == 'none':
            pose_landmarks = None
        elif pose_data_scenario == 'empty_list':
            pose_landmarks = []
        else:  # 'insufficient'
            # Create a list with random number of landmarks < 33
            pose_landmarks = [
                LandmarkData(
                    x=0.5,
                    y=0.5,
                    z=0.0,
                    visibility=0.9,
                    landmark_type=f"LANDMARK_{i}"
                )
                for i in range(num_insufficient_landmarks)
            ]
        
        # Create TrackingData with the pose landmarks scenario
        tracking_data = TrackingData(
            timestamp=timestamp,
            frame_number=frame_number,
            pose_landmarks=pose_landmarks,
            left_hand_landmarks=None,
            right_hand_landmarks=None
        )
        
        # Property assertion: Graceful handling without exceptions
        try:
            result = extractor.extract(tracking_data)
            
            # Should return None for all missing/invalid pose data scenarios
            assert result is None, \
                f"Extraction should return None for scenario '{pose_data_scenario}', got {result}"
            
            # Verify no exception was raised (implicit by reaching this point)
            exception_raised = False
        except Exception as e:
            exception_raised = True
            pytest.fail(
                f"Extraction should not raise exceptions for missing pose data. "
                f"Scenario: '{pose_data_scenario}', Exception: {type(e).__name__}: {e}"
            )
        
        # Explicit assertion that no exception occurred
        assert not exception_raised, \
            "Extractor should handle missing pose data gracefully without exceptions"
