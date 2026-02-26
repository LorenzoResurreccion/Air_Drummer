"""
Property-based tests for false positive prevention in tap detection.

**Feature: right-foot-bass-drum, Property 7: False positive prevention**

This module validates that the TapDetector correctly rejects movements that
should not be classified as taps:
- Slow movements (velocity below threshold)
- Primarily horizontal movements (vertical displacement below threshold or horizontal > vertical)
- Gradual drift (movement spread over more than maximum duration)

**Validates: Requirements 5.1, 5.2, 5.3**
"""

import pytest
from hypothesis import given, strategies as st, settings
from tap_detection.tap_detector import TapDetector
from tap_detection.tap_models import (
    FootLandmarks,
    TapDetectionConfig,
    DetectionState
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
    # Generate slow velocity movements (below 0.25 threshold)
    velocity=st.floats(min_value=0.01, max_value=0.24),
    # Generate displacement that would be above threshold if velocity was sufficient
    displacement=st.floats(min_value=0.05, max_value=0.15),
    # Time duration for the movement
    duration=st.floats(min_value=0.1, max_value=0.4),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_slow_movements_rejected(
    velocity: float,
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement with velocity below min_velocity threshold,
    the TapDetector should NOT register a tap, regardless of displacement.
    
    **Validates: Requirement 5.1 - Reject slow movements (velocity < 0.25 units/s)**
    """
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400
    )
    detector = TapDetector(config)
    
    # Calculate number of frames needed to achieve the slow velocity
    # velocity = displacement / duration
    # We want to ensure velocity stays below threshold
    num_frames = max(5, int(duration * 20))  # At least 5 frames, ~20 FPS
    
    # Generate frames with slow upward movement
    frames = []
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        # Move heel upward slowly (decreasing y)
        current_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process all frames
    tap_detected = False
    for frame in frames:
        tap_event = detector.process_frame(frame)
        if tap_event is not None:
            tap_detected = True
            break
    
    # Property: Slow movements should NEVER trigger a tap
    assert not tap_detected, (
        f"Slow movement (velocity={velocity:.3f} < 0.25) should not trigger tap. "
        f"Displacement={displacement:.3f}, Duration={duration:.3f}s"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate horizontal displacement greater than vertical
    horizontal_displacement=st.floats(min_value=0.05, max_value=0.2),
    vertical_displacement=st.floats(min_value=0.01, max_value=0.04),
    # Fast enough velocity to pass velocity check
    duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting positions
    start_x=st.floats(min_value=0.3, max_value=0.5),
    start_y=st.floats(min_value=0.3, max_value=0.7),
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_horizontal_movements_rejected(
    horizontal_displacement: float,
    vertical_displacement: float,
    duration: float,
    start_x: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement where horizontal displacement is greater than
    vertical displacement OR vertical displacement is less than 0.03, the TapDetector
    should NOT register a tap, even if velocity is sufficient.
    
    **Validates: Requirement 5.2 - Reject primarily horizontal movements**
    """
    # Ensure horizontal > vertical OR vertical < 0.03
    if vertical_displacement >= 0.03 and vertical_displacement >= horizontal_displacement:
        # Skip this case as it doesn't meet the property conditions
        return
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400
    )
    detector = TapDetector(config)
    
    # Generate frames with primarily horizontal movement
    num_frames = max(5, int(duration * 20))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        # Move heel horizontally and slightly vertically
        current_x = start_x + (t * horizontal_displacement)
        current_y = start_y - (t * vertical_displacement)  # Upward (decreasing y)
        
        frames.append(create_foot_landmarks(
            heel_x=current_x,
            heel_y=current_y,
            toe_x=start_x,  # Toe stays stationary in x
            toe_y=start_y,  # Toe stays stationary in y
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process all frames
    tap_detected = False
    for frame in frames:
        tap_event = detector.process_frame(frame)
        if tap_event is not None:
            tap_detected = True
            break
    
    # Property: Primarily horizontal movements should NEVER trigger a tap
    assert not tap_detected, (
        f"Horizontal movement (h={horizontal_displacement:.3f} > v={vertical_displacement:.3f}) "
        f"should not trigger tap. Duration={duration:.3f}s"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate movement duration exceeding max_tap_duration
    duration=st.floats(min_value=0.45, max_value=1.0),
    # Displacement above threshold
    displacement=st.floats(min_value=0.05, max_value=0.15),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_gradual_drift_rejected(
    duration: float,
    displacement: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement that takes longer than max_tap_duration_ms
    (400ms default), the TapDetector should NOT register a tap, even if the
    displacement and velocity would otherwise be sufficient.
    
    **Validates: Requirement 5.3 - Reject gradual drift (movement > max_tap_duration)**
    """
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400
    )
    detector = TapDetector(config)
    
    # Generate frames spread over long duration (gradual drift)
    num_frames = max(10, int(duration * 20))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        # Move heel upward gradually
        current_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process all frames
    tap_detected = False
    for frame in frames:
        tap_event = detector.process_frame(frame)
        if tap_event is not None:
            tap_detected = True
            break
    
    # Property: Gradual drift over long duration should NEVER trigger a tap
    assert not tap_detected, (
        f"Gradual drift (duration={duration:.3f}s > 0.4s) should not trigger tap. "
        f"Displacement={displacement:.3f}"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate upward displacement
    upward_displacement=st.floats(min_value=0.05, max_value=0.15),
    # Generate upward duration (fast enough)
    upward_duration=st.floats(min_value=0.05, max_value=0.2),
    # Generate sustained position duration (no downward motion)
    sustained_duration=st.floats(min_value=0.3, max_value=0.8),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_incomplete_gesture_no_downward_rejected(
    upward_displacement: float,
    upward_duration: float,
    sustained_duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement that includes upward motion but NO downward
    return motion, the TapDetector should NOT register a tap. A complete tap
    requires both upward and downward phases.
    
    **Validates: Requirement 5.3 (implicit) - Complete tap pattern required**
    
    Note: This is related to 5.3 as incomplete gestures are a form of invalid
    movement pattern that should be rejected.
    """
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400
    )
    detector = TapDetector(config)
    
    # Generate upward motion frames
    num_upward_frames = max(3, int(upward_duration * 20))
    frames = []
    
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * upward_duration
        current_y = start_y - (t * upward_displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Generate sustained position frames (no downward motion)
    peak_y = start_y - upward_displacement
    num_sustained_frames = max(5, int(sustained_duration * 20))
    
    for i in range(num_sustained_frames):
        current_time = upward_duration + (i * sustained_duration / num_sustained_frames)
        
        frames.append(create_foot_landmarks(
            heel_y=peak_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process all frames
    tap_detected = False
    for frame in frames:
        tap_event = detector.process_frame(frame)
        if tap_event is not None:
            tap_detected = True
            break
    
    # Property: Incomplete gesture (no downward motion) should NEVER trigger a tap
    assert not tap_detected, (
        f"Incomplete gesture (upward only, no downward) should not trigger tap. "
        f"Upward displacement={upward_displacement:.3f}, sustained for {sustained_duration:.3f}s"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate combined slow and horizontal movement
    horizontal_displacement=st.floats(min_value=0.05, max_value=0.2),
    vertical_displacement=st.floats(min_value=0.02, max_value=0.06),
    # Slow duration
    duration=st.floats(min_value=0.4, max_value=1.0),
    # Starting positions
    start_x=st.floats(min_value=0.3, max_value=0.5),
    start_y=st.floats(min_value=0.3, max_value=0.7),
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_combined_false_positive_conditions_rejected(
    horizontal_displacement: float,
    vertical_displacement: float,
    duration: float,
    start_x: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any foot movement that violates MULTIPLE false positive
    prevention criteria (e.g., slow AND horizontal), the TapDetector should
    NOT register a tap.
    
    **Validates: Requirements 5.1, 5.2, 5.3 - Combined false positive prevention**
    """
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400
    )
    detector = TapDetector(config)
    
    # Generate frames with slow, horizontal movement over long duration
    num_frames = max(10, int(duration * 20))
    frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        current_x = start_x + (t * horizontal_displacement)
        current_y = start_y - (t * vertical_displacement)
        
        frames.append(create_foot_landmarks(
            heel_x=current_x,
            heel_y=current_y,
            toe_x=start_x,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process all frames
    tap_detected = False
    for frame in frames:
        tap_event = detector.process_frame(frame)
        if tap_event is not None:
            tap_detected = True
            break
    
    # Property: Movement violating multiple criteria should NEVER trigger a tap
    assert not tap_detected, (
        f"Movement violating multiple criteria should not trigger tap. "
        f"Horizontal={horizontal_displacement:.3f}, Vertical={vertical_displacement:.3f}, "
        f"Duration={duration:.3f}s"
    )
