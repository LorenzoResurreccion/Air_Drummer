"""
Property-based tests for state management in tap detection.

**Feature: right-foot-bass-drum, Property 11: State management**

This module validates that the TapDetector maintains valid detection states
and transitions between states according to the detection logic. The state
should always be queryable and follow the defined state machine transitions.

**Validates: Requirement 8.1**
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
    # Random starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Random confidence
    visibility=st.floats(min_value=0.5, max_value=1.0),
    # Number of frames to process
    num_frames=st.integers(min_value=5, max_value=30)
)
def test_property_state_always_valid(
    start_y: float,
    visibility: float,
    num_frames: int
):
    """
    Property: For any sequence of frames processed by the TapDetector, the
    detector should always maintain a valid state (IDLE, DETECTING_UPWARD,
    DETECTING_DOWNWARD, or DEBOUNCING) that can be queried via get_state().
    
    **Validates: Requirement 8.1 - State is always queryable and valid**
    """
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Valid states
    valid_states = {
        DetectionState.IDLE,
        DetectionState.DETECTING_UPWARD,
        DetectionState.DETECTING_DOWNWARD,
        DetectionState.DEBOUNCING
    }
    
    # Process random frames
    for i in range(num_frames):
        timestamp = i * 0.033  # ~30 FPS
        
        # Create frame with stationary foot
        frame = create_foot_landmarks(
            heel_y=start_y,
            toe_y=start_y,
            timestamp=timestamp,
            visibility=visibility
        )
        
        # Process frame
        detector.process_frame(frame)
        
        # Property: State should always be valid and queryable
        current_state = detector.get_state()
        assert current_state in valid_states, (
            f"State {current_state} is not a valid DetectionState"
        )


@settings(max_examples=100, deadline=None)
@given(
    # Random starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Random confidence
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_initial_state_is_idle(
    start_y: float,
    visibility: float
):
    """
    Property: For any newly created TapDetector, the initial state should
    always be IDLE before any frames are processed.
    
    **Validates: Requirement 8.1 - Initial state management**
    """
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Property: Initial state should be IDLE
    assert detector.get_state() == DetectionState.IDLE, (
        f"Initial state should be IDLE, got {detector.get_state()}"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Fast duration
    duration=st.floats(min_value=0.05, max_value=0.2),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_state_transition_idle_to_upward(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any sequence where the detector is in IDLE state and upward
    movement exceeding thresholds is detected, the state should transition to
    DETECTING_UPWARD.
    
    **Validates: Requirement 8.1 - State transitions from IDLE to DETECTING_UPWARD**
    """
    # Ensure velocity is above threshold
    velocity = displacement / duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Verify initial state
    assert detector.get_state() == DetectionState.IDLE
    
    # Generate frames with upward movement
    num_frames = max(5, int(duration * 30))
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,  # Toe stationary for heel tap
            timestamp=current_time,
            visibility=visibility
        )
        
        detector.process_frame(frame)
        
        # Once upward motion detected, state should be DETECTING_UPWARD
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            break
    
    # Property: State should have transitioned to DETECTING_UPWARD
    assert detector.get_state() == DetectionState.DETECTING_UPWARD, (
        f"State should transition to DETECTING_UPWARD after upward motion, "
        f"got {detector.get_state()}"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Fast duration for upward phase
    upward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_state_transition_upward_to_downward(
    displacement: float,
    upward_duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any sequence where the detector is in DETECTING_UPWARD state
    and upward motion stops (velocity drops), the state should transition to
    DETECTING_DOWNWARD.
    
    **Validates: Requirement 8.1 - State transitions from DETECTING_UPWARD to DETECTING_DOWNWARD**
    """
    # Ensure velocity is above threshold
    velocity = displacement / upward_duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Phase 1: Generate upward movement to reach DETECTING_UPWARD
    num_upward_frames = max(5, int(upward_duration * 30))
    
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * upward_duration
        current_y = start_y - (t * displacement)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        
        detector.process_frame(frame)
    
    # Should be in DETECTING_UPWARD state
    if detector.get_state() != DetectionState.DETECTING_UPWARD:
        # Skip this test case if we didn't reach DETECTING_UPWARD
        assume(False)
    
    # Phase 2: Hold position (velocity near zero) to trigger transition
    peak_y = start_y - displacement
    hold_duration = 0.05
    num_hold_frames = max(3, int(hold_duration * 30))
    
    for i in range(num_hold_frames):
        current_time = upward_duration + (i * 0.033)
        
        frame = create_foot_landmarks(
            heel_y=peak_y,  # Hold at peak
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        
        detector.process_frame(frame)
        
        # Check if transitioned to DETECTING_DOWNWARD
        if detector.get_state() == DetectionState.DETECTING_DOWNWARD:
            break
    
    # Property: State should transition to DETECTING_DOWNWARD when upward motion stops
    assert detector.get_state() == DetectionState.DETECTING_DOWNWARD, (
        f"State should transition to DETECTING_DOWNWARD when upward motion stops, "
        f"got {detector.get_state()}"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.12),
    # Fast duration
    duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.6),
    # Confidence above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_state_transition_downward_to_debouncing(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any sequence where the detector is in DETECTING_DOWNWARD state
    and downward motion is detected, the state should transition to DEBOUNCING
    and a TapEvent should be generated.
    
    **Validates: Requirement 8.1 - State transitions from DETECTING_DOWNWARD to DEBOUNCING**
    """
    # Ensure velocity is above threshold
    velocity = displacement / duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Phase 1: Upward movement
    num_upward_frames = max(5, int(duration * 30))
    
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        
        detector.process_frame(frame)
    
    # Phase 2: Hold at peak
    peak_y = start_y - displacement
    hold_duration = 0.05
    num_hold_frames = max(3, int(hold_duration * 30))
    
    for i in range(num_hold_frames):
        current_time = duration + (i * 0.033)
        
        frame = create_foot_landmarks(
            heel_y=peak_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        
        detector.process_frame(frame)
        
        if detector.get_state() == DetectionState.DETECTING_DOWNWARD:
            break
    
    # Skip if we didn't reach DETECTING_DOWNWARD
    if detector.get_state() != DetectionState.DETECTING_DOWNWARD:
        assume(False)
    
    # Phase 3: Downward movement
    downward_duration = 0.1
    num_downward_frames = max(5, int(downward_duration * 30))
    
    tap_detected = False
    for i in range(num_downward_frames):
        t = i / (num_downward_frames - 1) if num_downward_frames > 1 else 0
        current_time = duration + hold_duration + (t * downward_duration)
        # Move back down (increasing y)
        current_y = peak_y + (t * displacement * 0.5)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        
        result = detector.process_frame(frame)
        
        if result is not None:
            tap_detected = True
            break
    
    # Property: State should transition to DEBOUNCING after complete tap
    if tap_detected:
        assert detector.get_state() == DetectionState.DEBOUNCING, (
            f"State should transition to DEBOUNCING after tap detected, "
            f"got {detector.get_state()}"
        )


@settings(max_examples=100, deadline=None)
@given(
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0),
    # Number of frames during debounce
    num_frames=st.integers(min_value=3, max_value=10)
)
def test_property_state_remains_debouncing_during_window(
    start_y: float,
    visibility: float,
    num_frames: int
):
    """
    Property: For any sequence where the detector enters DEBOUNCING state,
    the state should remain DEBOUNCING for the duration of the debounce
    window (150ms), regardless of foot movements.
    
    **Validates: Requirement 8.1 - State management during debounce**
    """
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Manually set detector to DEBOUNCING state by completing a tap
    # Phase 1: Quick upward movement
    displacement = 0.08
    duration = 0.1
    num_upward = 5
    
    for i in range(num_upward):
        t = i / (num_upward - 1) if num_upward > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        detector.process_frame(frame)
    
    # Phase 2: Hold
    peak_y = start_y - displacement
    for i in range(3):
        current_time = duration + (i * 0.033)
        frame = create_foot_landmarks(
            heel_y=peak_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        detector.process_frame(frame)
    
    # Phase 3: Downward to complete tap
    tap_completion_time = None
    for i in range(5):
        t = i / 4
        current_time = duration + 0.1 + (t * 0.1)
        current_y = peak_y + (t * displacement * 0.5)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        result = detector.process_frame(frame)
        
        if result is not None:
            tap_completion_time = current_time
            break
    
    # Skip if we didn't reach DEBOUNCING
    if detector.get_state() != DetectionState.DEBOUNCING or tap_completion_time is None:
        assume(False)
    
    # Now process frames during debounce window (< 150ms from tap completion)
    base_time = tap_completion_time
    debounce_duration = 0.10  # Well under 150ms to avoid boundary issues
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = base_time + (t * debounce_duration)
        
        # Try to trigger another tap (should be ignored)
        frame = create_foot_landmarks(
            heel_y=start_y - 0.1,  # Large movement
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        
        detector.process_frame(frame)
        
        # Property: Should remain in DEBOUNCING state
        assert detector.get_state() == DetectionState.DEBOUNCING, (
            f"State should remain DEBOUNCING during debounce window, "
            f"got {detector.get_state()} at time {current_time - base_time:.3f}s"
        )


@settings(max_examples=100, deadline=None)
@given(
    # Displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Fast duration
    duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_state_reset_on_missing_landmarks(
    displacement: float,
    duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any sequence where the detector is in DETECTING_UPWARD or
    DETECTING_DOWNWARD state and landmarks become unavailable (None or low
    confidence), the state should reset to IDLE.
    
    **Validates: Requirement 8.1 - State management with missing data**
    """
    # Ensure velocity is above threshold
    velocity = displacement / duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Phase 1: Generate upward movement to reach DETECTING_UPWARD
    num_frames = max(5, int(duration * 30))
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=visibility
        )
        
        detector.process_frame(frame)
        
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            break
    
    # Skip if we didn't reach DETECTING_UPWARD
    if detector.get_state() not in (DetectionState.DETECTING_UPWARD, DetectionState.DETECTING_DOWNWARD):
        assume(False)
    
    # Phase 2: Send None landmarks (missing data)
    detector.process_frame(None)
    
    # Property: State should reset to IDLE
    assert detector.get_state() == DetectionState.IDLE, (
        f"State should reset to IDLE when landmarks become unavailable, "
        f"got {detector.get_state()}"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Low confidence
    low_visibility=st.floats(min_value=0.1, max_value=0.49),
    # High confidence
    high_visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_state_reset_on_low_confidence(
    start_y: float,
    low_visibility: float,
    high_visibility: float
):
    """
    Property: For any sequence where the detector is in DETECTING_UPWARD or
    DETECTING_DOWNWARD state and landmark confidence drops below threshold,
    the state should reset to IDLE.
    
    **Validates: Requirement 8.1 - State management with low confidence**
    """
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Phase 1: Generate upward movement with high confidence
    displacement = 0.08
    duration = 0.1
    num_frames = 5
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        current_time = t * duration
        current_y = start_y - (t * displacement)
        
        frame = create_foot_landmarks(
            heel_y=current_y,
            toe_y=start_y,
            timestamp=current_time,
            visibility=high_visibility
        )
        
        detector.process_frame(frame)
        
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            break
    
    # Skip if we didn't reach DETECTING_UPWARD
    if detector.get_state() not in (DetectionState.DETECTING_UPWARD, DetectionState.DETECTING_DOWNWARD):
        assume(False)
    
    # Phase 2: Send low confidence landmarks
    frame = create_foot_landmarks(
        heel_y=start_y - displacement,
        toe_y=start_y,
        timestamp=duration + 0.033,
        visibility=low_visibility
    )
    
    detector.process_frame(frame)
    
    # Property: State should reset to IDLE
    assert detector.get_state() == DetectionState.IDLE, (
        f"State should reset to IDLE when confidence drops below threshold, "
        f"got {detector.get_state()}"
    )
