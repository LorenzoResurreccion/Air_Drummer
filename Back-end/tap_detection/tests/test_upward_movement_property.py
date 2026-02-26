"""
Property-based tests for upward movement detection across tap types.

**Feature: right-foot-bass-drum, Property 1: Upward movement detection across tap types**

This module validates that the TapDetector correctly detects the upward phase
of tap gestures across all three tap types (heel, toe, full foot) when movement
exceeds displacement and velocity thresholds within the time window.

**Validates: Requirements 1.1, 2.1, 3.1**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from tap_detection.tap_detector import TapDetector
from tap_detection.tap_models import (
    FootLandmarks,
    TapDetectionConfig,
    DetectionState,
    TapType
)
from comp_vision.models import LandmarkData


def create_foot_landmarks(
    heel_x: float = 0.5,
    heel_y: float = 0.5,
    ankle_x: float = 0.5,
    ankle_y: float = 0.4,
    toe_x: float = 0.5,
    toe_y: float = 0.5,
    timestamp: float = 0.0,
    visibility: float = 0.9
) -> FootLandmarks:
    """Helper to create FootLandmarks for testing."""
    heel = LandmarkData(
        x=heel_x, y=heel_y, z=0.0,
        visibility=visibility,
        landmark_type="RIGHT_HEEL"
    )
    ankle = LandmarkData(
        x=ankle_x, y=ankle_y, z=0.0,
        visibility=visibility,
        landmark_type="RIGHT_ANKLE"
    )
    toe = LandmarkData(
        x=toe_x, y=toe_y, z=0.0,
        visibility=visibility,
        landmark_type="RIGHT_FOOT_INDEX"
    )
    
    return FootLandmarks(
        heel=heel,
        ankle=ankle,
        toe=toe,
        timestamp=timestamp
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold (0.05 to 0.15)
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate fast enough duration to ensure velocity > 0.25
    # velocity = displacement / duration, so duration = displacement / velocity
    # For velocity > 0.25, duration < displacement / 0.25
    duration=st.floats(min_value=0.05, max_value=0.2),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_heel_lift_upward_detection(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any heel movement where the heel moves upward by more than
    min_displacement (0.05) within the time window (200ms) with velocity above
    min_velocity (0.25), and the toe remains stationary, the TapDetector should
    detect the upward phase and transition to DETECTING_UPWARD state.
    
    **Validates: Requirement 1.1 - Detect heel lift taps**
    """
    # Ensure velocity will be above threshold
    velocity = displacement / duration
    assume(velocity > 0.3)  # Give some margin above 0.25 threshold
    
    # Ensure vertical displacement is significant (> 0.03) and greater than horizontal
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate frames with heel moving upward, toe stationary
    num_frames = max(5, int(duration * 30))  # ~30 FPS
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        # Move heel upward (decreasing y)
        current_heel_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames until upward motion is detected
    upward_detected = False
    for frame in frames:
        detector.process_frame(frame)
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            upward_detected = True
            break
    
    # Property: Upward heel movement should be detected
    assert upward_detected, (
        f"Heel lift upward motion should be detected. "
        f"Displacement={displacement:.3f}, Velocity={velocity:.3f}, Duration={duration:.3f}s"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate fast enough duration
    duration=st.floats(min_value=0.05, max_value=0.2),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_toe_lift_upward_detection(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any toe movement where the toe moves upward by more than
    min_displacement (0.05) within the time window with velocity above
    min_velocity (0.25), and the heel remains stationary, the TapDetector should
    detect the upward phase and transition to DETECTING_UPWARD state.
    
    **Validates: Requirement 2.1 - Detect toe lift taps**
    """
    # Ensure velocity will be above threshold
    velocity = displacement / duration
    assume(velocity > 0.3)  # Give some margin above 0.25 threshold
    
    # Ensure vertical displacement is significant
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate frames with toe moving upward, heel stationary
    num_frames = max(5, int(duration * 30))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        # Move toe upward (decreasing y)
        current_toe_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=start_y,  # Heel stays stationary
            toe_y=current_toe_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames until upward motion is detected
    upward_detected = False
    for frame in frames:
        detector.process_frame(frame)
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            upward_detected = True
            break
    
    # Property: Upward toe movement should be detected
    assert upward_detected, (
        f"Toe lift upward motion should be detected. "
        f"Displacement={displacement:.3f}, Velocity={velocity:.3f}, Duration={duration:.3f}s"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold for both heel and toe
    heel_displacement=st.floats(min_value=0.06, max_value=0.15),
    toe_displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate fast enough duration
    duration=st.floats(min_value=0.05, max_value=0.2),
    # Starting positions
    start_heel_y=st.floats(min_value=0.3, max_value=0.7),
    start_toe_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_full_foot_lift_upward_detection(
    heel_displacement: float,
    toe_displacement: float,
    duration: float,
    start_heel_y: float,
    start_toe_y: float,
    visibility: float
):
    """
    Property: For any full foot movement where both heel and toe move upward
    by more than min_displacement (0.05) within the time window with velocity
    above min_velocity (0.25), the TapDetector should detect the upward phase
    and transition to DETECTING_UPWARD state.
    
    **Validates: Requirement 3.1 - Detect full foot lift taps**
    """
    # Ensure velocities will be above threshold
    heel_velocity = heel_displacement / duration
    toe_velocity = toe_displacement / duration
    assume(heel_velocity > 0.3 and toe_velocity > 0.3)
    
    # Ensure vertical displacements are significant
    assume(heel_displacement >= 0.03 and toe_displacement >= 0.03)
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate frames with both heel and toe moving upward
    num_frames = max(5, int(duration * 30))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        # Move both heel and toe upward (decreasing y)
        current_heel_y = start_heel_y - (t * heel_displacement)
        current_toe_y = start_toe_y - (t * toe_displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=current_toe_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames until upward motion is detected
    upward_detected = False
    for frame in frames:
        detector.process_frame(frame)
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            upward_detected = True
            break
    
    # Property: Upward full foot movement should be detected
    assert upward_detected, (
        f"Full foot lift upward motion should be detected. "
        f"Heel displacement={heel_displacement:.3f}, Toe displacement={toe_displacement:.3f}, "
        f"Duration={duration:.3f}s"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement below threshold
    displacement=st.floats(min_value=0.01, max_value=0.049),
    # Generate any duration
    duration=st.floats(min_value=0.05, max_value=0.3),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_insufficient_displacement_not_detected(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement where displacement is below min_displacement
    threshold (0.05), the TapDetector should NOT detect upward motion, regardless
    of velocity.
    
    This validates the displacement threshold requirement.
    
    **Validates: Requirements 1.1, 2.1, 3.1 (threshold enforcement)**
    """
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400
    )
    detector = TapDetector(config)
    
    # Generate frames with small displacement
    num_frames = max(5, int(duration * 30))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process all frames
    upward_detected = False
    for frame in frames:
        detector.process_frame(frame)
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            upward_detected = True
            break
    
    # Property: Insufficient displacement should NOT be detected
    assert not upward_detected, (
        f"Movement with displacement={displacement:.3f} < 0.05 should not be detected"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate duration that results in velocity below threshold
    # velocity = displacement / duration, so for velocity < 0.25, duration > displacement / 0.25
    duration=st.floats(min_value=0.3, max_value=0.8),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_insufficient_velocity_not_detected(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement where velocity is below min_velocity
    threshold (0.25), the TapDetector should NOT detect upward motion, even
    if displacement is sufficient.
    
    This validates the velocity threshold requirement.
    
    **Validates: Requirements 1.1, 2.1, 3.1 (threshold enforcement)**
    """
    # Ensure velocity is below threshold
    velocity = displacement / duration
    assume(velocity < 0.24)
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400
    )
    detector = TapDetector(config)
    
    # Generate frames with slow movement
    num_frames = max(5, int(duration * 30))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process all frames
    upward_detected = False
    for frame in frames:
        detector.process_frame(frame)
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            upward_detected = True
            break
    
    # Property: Insufficient velocity should NOT be detected
    assert not upward_detected, (
        f"Movement with velocity={velocity:.3f} < 0.25 should not be detected. "
        f"Displacement={displacement:.3f}, Duration={duration:.3f}s"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate fast enough duration
    duration=st.floats(min_value=0.05, max_value=0.2),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score BELOW threshold
    visibility=st.floats(min_value=0.1, max_value=0.49)
)
def test_property_low_confidence_not_detected(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement where landmark confidence is below the
    confidence threshold (0.5), the TapDetector should NOT detect upward motion,
    even if displacement and velocity are sufficient.
    
    This validates the confidence filtering requirement.
    
    **Validates: Requirements 1.1, 2.1, 3.1 (confidence filtering)**
    """
    # Ensure velocity would be above threshold if confidence was sufficient
    velocity = displacement / duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5
    )
    detector = TapDetector(config)
    
    # Generate frames with low confidence
    num_frames = max(5, int(duration * 30))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility  # Low confidence
        ))
    
    # Process all frames
    upward_detected = False
    for frame in frames:
        detector.process_frame(frame)
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            upward_detected = True
            break
    
    # Property: Low confidence landmarks should NOT trigger detection
    assert not upward_detected, (
        f"Movement with confidence={visibility:.3f} < 0.5 should not be detected. "
        f"Displacement={displacement:.3f}, Velocity={velocity:.3f}"
    )
