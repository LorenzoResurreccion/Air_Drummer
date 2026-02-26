"""
Property-based tests for complete tap pattern recognition.

**Feature: right-foot-bass-drum, Property 3: Complete tap pattern recognition**

This module validates that the TapDetector correctly recognizes complete tap
patterns that include both an upward phase (displacement exceeding threshold)
followed by a downward phase (return motion) within the maximum tap duration.

**Validates: Requirements 1.3, 2.3, 3.3, 5.4**
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
    # Generate displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate upward duration (fast enough for velocity threshold)
    upward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Generate downward duration
    downward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_complete_heel_tap_pattern(
    displacement: float,
    upward_duration: float,
    downward_duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any heel tap gesture where the heel moves upward by more than
    the displacement threshold, then returns downward, all within the maximum
    tap duration, the TapDetector should mark the tap as complete and return
    a TapEvent.
    
    **Validates: Requirement 1.3 - Complete heel tap pattern**
    """
    # Ensure velocity will be above threshold
    velocity = displacement / upward_duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    # Ensure total duration is within max_tap_duration
    total_duration = upward_duration + downward_duration
    assume(total_duration < 0.4)  # 400ms
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate complete tap sequence: upward then downward
    frames = []
    
    # Phase 1: Upward motion
    num_upward_frames = max(5, int(upward_duration * 30))
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * upward_duration
        # Move heel upward (decreasing y)
        current_heel_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Phase 2: Downward motion
    num_downward_frames = max(5, int(downward_duration * 30))
    peak_y = start_y - displacement
    for i in range(num_downward_frames):
        t = i / (num_downward_frames - 1) if num_downward_frames > 1 else 0
        current_time = upward_duration + (t * downward_duration)
        # Move heel downward (increasing y back to start)
        current_heel_y = peak_y + (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames and check for complete tap
    tap_detected = False
    tap_event = None
    
    for frame in frames:
        result = detector.process_frame(frame)
        if result is not None:
            tap_detected = True
            tap_event = result
            break
    
    # Property: Complete tap pattern should be detected
    assert tap_detected, (
        f"Complete heel tap pattern should be detected. "
        f"Displacement={displacement:.3f}, Upward duration={upward_duration:.3f}s, "
        f"Downward duration={downward_duration:.3f}s, Total={total_duration:.3f}s"
    )
    
    # Verify tap event properties
    assert tap_event.tap_type == TapType.HEEL, "Should be detected as HEEL tap"
    assert tap_event.duration_ms <= 400, "Duration should be within max_tap_duration"
    assert tap_event.peak_displacement > 0, "Should have positive peak displacement"


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate upward duration
    upward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Generate downward duration
    downward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_complete_toe_tap_pattern(
    displacement: float,
    upward_duration: float,
    downward_duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any toe tap gesture where the toe moves upward by more than
    the displacement threshold, then returns downward, all within the maximum
    tap duration, the TapDetector should mark the tap as complete and return
    a TapEvent.
    
    **Validates: Requirement 2.3 - Complete toe tap pattern**
    """
    # Ensure velocity will be above threshold
    velocity = displacement / upward_duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    # Ensure total duration is within max_tap_duration
    total_duration = upward_duration + downward_duration
    assume(total_duration < 0.4)  # 400ms
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate complete tap sequence: upward then downward
    frames = []
    
    # Phase 1: Upward motion
    num_upward_frames = max(5, int(upward_duration * 30))
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * upward_duration
        # Move toe upward (decreasing y)
        current_toe_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=start_y,  # Heel stays stationary
            toe_y=current_toe_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Phase 2: Downward motion
    num_downward_frames = max(5, int(downward_duration * 30))
    peak_y = start_y - displacement
    for i in range(num_downward_frames):
        t = i / (num_downward_frames - 1) if num_downward_frames > 1 else 0
        current_time = upward_duration + (t * downward_duration)
        # Move toe downward (increasing y back to start)
        current_toe_y = peak_y + (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=start_y,  # Heel stays stationary
            toe_y=current_toe_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames and check for complete tap
    tap_detected = False
    tap_event = None
    
    for frame in frames:
        result = detector.process_frame(frame)
        if result is not None:
            tap_detected = True
            tap_event = result
            break
    
    # Property: Complete tap pattern should be detected
    assert tap_detected, (
        f"Complete toe tap pattern should be detected. "
        f"Displacement={displacement:.3f}, Upward duration={upward_duration:.3f}s, "
        f"Downward duration={downward_duration:.3f}s, Total={total_duration:.3f}s"
    )
    
    # Verify tap event properties
    assert tap_event.tap_type == TapType.TOE, "Should be detected as TOE tap"
    assert tap_event.duration_ms <= 400, "Duration should be within max_tap_duration"
    assert tap_event.peak_displacement > 0, "Should have positive peak displacement"


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold for both heel and toe
    heel_displacement=st.floats(min_value=0.06, max_value=0.15),
    toe_displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate upward duration
    upward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Generate downward duration
    downward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting positions
    start_heel_y=st.floats(min_value=0.3, max_value=0.7),
    start_toe_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_complete_full_foot_tap_pattern(
    heel_displacement: float,
    toe_displacement: float,
    upward_duration: float,
    downward_duration: float,
    start_heel_y: float,
    start_toe_y: float,
    visibility: float
):
    """
    Property: For any full foot tap gesture where both heel and toe move upward
    by more than the displacement threshold, then return downward, all within
    the maximum tap duration, the TapDetector should mark the tap as complete
    and return a TapEvent.
    
    **Validates: Requirement 3.3 - Complete full foot tap pattern**
    """
    # Ensure velocities will be above threshold
    heel_velocity = heel_displacement / upward_duration
    toe_velocity = toe_displacement / upward_duration
    assume(heel_velocity > 0.3 and toe_velocity > 0.3)
    assume(heel_displacement >= 0.03 and toe_displacement >= 0.03)
    
    # Ensure total duration is within max_tap_duration
    total_duration = upward_duration + downward_duration
    assume(total_duration < 0.4)  # 400ms
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate complete tap sequence: upward then downward
    frames = []
    
    # Phase 1: Upward motion
    num_upward_frames = max(5, int(upward_duration * 30))
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * upward_duration
        # Move both heel and toe upward (decreasing y)
        current_heel_y = start_heel_y - (t * heel_displacement)
        current_toe_y = start_toe_y - (t * toe_displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=current_toe_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Phase 2: Downward motion
    num_downward_frames = max(5, int(downward_duration * 30))
    peak_heel_y = start_heel_y - heel_displacement
    peak_toe_y = start_toe_y - toe_displacement
    for i in range(num_downward_frames):
        t = i / (num_downward_frames - 1) if num_downward_frames > 1 else 0
        current_time = upward_duration + (t * downward_duration)
        # Move both heel and toe downward (increasing y back to start)
        current_heel_y = peak_heel_y + (t * heel_displacement)
        current_toe_y = peak_toe_y + (t * toe_displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=current_toe_y,
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames and check for complete tap
    tap_detected = False
    tap_event = None
    
    for frame in frames:
        result = detector.process_frame(frame)
        if result is not None:
            tap_detected = True
            tap_event = result
            break
    
    # Property: Complete tap pattern should be detected
    assert tap_detected, (
        f"Complete full foot tap pattern should be detected. "
        f"Heel displacement={heel_displacement:.3f}, Toe displacement={toe_displacement:.3f}, "
        f"Upward duration={upward_duration:.3f}s, Downward duration={downward_duration:.3f}s, "
        f"Total={total_duration:.3f}s"
    )
    
    # Verify tap event properties
    assert tap_event.tap_type == TapType.FULL_FOOT, "Should be detected as FULL_FOOT tap"
    assert tap_event.duration_ms <= 400, "Duration should be within max_tap_duration"
    assert tap_event.peak_displacement > 0, "Should have positive peak displacement"


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate upward duration
    upward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Generate long pause before downward motion (exceeds max_tap_duration)
    pause_duration=st.floats(min_value=0.3, max_value=0.5),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_incomplete_tap_timeout(
    displacement: float,
    upward_duration: float,
    pause_duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any tap gesture where the upward motion is detected but the
    downward motion does not occur within max_tap_duration (400ms), the
    TapDetector should NOT mark the tap as complete and should reset to IDLE.
    
    This validates the timeout requirement for incomplete gestures.
    
    **Validates: Requirement 5.4 - Complete tap pattern within time limit**
    """
    # Ensure velocity will be above threshold
    velocity = displacement / upward_duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    # Ensure total duration exceeds max_tap_duration
    # Add extra buffer to account for detection delay
    total_duration = upward_duration + pause_duration
    assume(total_duration > 0.5)  # Exceeds 400ms with buffer for detection delay
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate incomplete tap sequence: upward then long pause
    frames = []
    
    # Phase 1: Upward motion
    num_upward_frames = max(5, int(upward_duration * 30))
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * upward_duration
        # Move heel upward (decreasing y)
        current_heel_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Phase 2: Long pause at peak (no downward motion)
    # Ensure we have enough pause frames to exceed timeout from gesture start
    num_pause_frames = max(10, int(pause_duration * 30))
    peak_y = start_y - displacement
    for i in range(num_pause_frames):
        t = i / (num_pause_frames - 1) if num_pause_frames > 1 else 0
        current_time = upward_duration + (t * pause_duration)
        # Stay at peak position
        
        frames.append(create_foot_landmarks(
            heel_y=peak_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames and check that tap is NOT completed
    tap_detected = False
    final_state = None
    gesture_start_time = None
    
    for frame in frames:
        result = detector.process_frame(frame)
        if result is not None:
            tap_detected = True
            break
        
        # Track when gesture starts
        if gesture_start_time is None and detector._gesture_start_time is not None:
            gesture_start_time = detector._gesture_start_time
        
        final_state = detector.get_state()
    
    # If gesture started but hasn't timed out yet, generate more frames
    if gesture_start_time is not None and final_state != DetectionState.IDLE:
        last_timestamp = frames[-1].timestamp
        # Generate frames until we're well past the timeout
        while True:
            last_timestamp += 0.033  # ~30 FPS
            extra_frame = create_foot_landmarks(
                heel_y=peak_y,
                toe_y=start_y,
                timestamp=last_timestamp,
                visibility=visibility
            )
            
            result = detector.process_frame(extra_frame)
            if result is not None:
                tap_detected = True
                break
            
            final_state = detector.get_state()
            elapsed_ms = (last_timestamp - gesture_start_time) * 1000
            
            # Stop once we've exceeded timeout and state has been updated
            if elapsed_ms > 450:
                break
    
    # Property: Incomplete tap (timeout) should NOT be detected
    assert not tap_detected, (
        f"Incomplete tap with timeout should NOT be detected. "
        f"Displacement={displacement:.3f}, Upward duration={upward_duration:.3f}s, "
        f"Pause duration={pause_duration:.3f}s, Total={total_duration:.3f}s"
    )
    
    # Verify detector reset to IDLE after timeout
    assert final_state == DetectionState.IDLE, (
        f"Detector should reset to IDLE after timeout, but is in {final_state}"
    )


@settings(max_examples=100, deadline=None)
@given(
    # Generate displacement above threshold
    displacement=st.floats(min_value=0.06, max_value=0.15),
    # Generate upward duration
    upward_duration=st.floats(min_value=0.05, max_value=0.15),
    # Starting position
    start_y=st.floats(min_value=0.3, max_value=0.7),
    # Confidence score above threshold
    visibility=st.floats(min_value=0.5, max_value=1.0)
)
def test_property_incomplete_tap_no_downward_motion(
    displacement: float,
    upward_duration: float,
    start_y: float,
    visibility: float
):
    """
    Property: For any tap gesture where only upward motion is detected without
    any downward motion, the TapDetector should NOT mark the tap as complete.
    
    This validates that both upward and downward phases are required.
    
    **Validates: Requirement 5.4 - Complete tap pattern requires both phases**
    """
    # Ensure velocity will be above threshold
    velocity = displacement / upward_duration
    assume(velocity > 0.3)
    assume(displacement >= 0.03)
    
    config = TapDetectionConfig(
        min_displacement=0.05,
        min_velocity=0.25,
        max_tap_duration_ms=400,
        confidence_threshold=0.5,
        stationary_threshold=0.02
    )
    detector = TapDetector(config)
    
    # Generate only upward motion (no downward)
    frames = []
    
    num_upward_frames = max(5, int(upward_duration * 30))
    for i in range(num_upward_frames):
        t = i / (num_upward_frames - 1) if num_upward_frames > 1 else 0
        current_time = t * upward_duration
        # Move heel upward (decreasing y)
        current_heel_y = start_y - (t * displacement)
        
        frames.append(create_foot_landmarks(
            heel_y=current_heel_y,
            toe_y=start_y,  # Toe stays stationary
            timestamp=current_time,
            visibility=visibility
        ))
    
    # Process frames and check that tap is NOT completed
    tap_detected = False
    upward_state_reached = False
    
    for frame in frames:
        result = detector.process_frame(frame)
        if result is not None:
            tap_detected = True
            break
        if detector.get_state() == DetectionState.DETECTING_UPWARD:
            upward_state_reached = True
    
    # Property: Incomplete tap (no downward) should NOT be detected
    assert not tap_detected, (
        f"Incomplete tap without downward motion should NOT be detected. "
        f"Displacement={displacement:.3f}, Duration={upward_duration:.3f}s"
    )
    
    # Verify upward motion was detected (but not completed)
    assert upward_state_reached, (
        "Upward motion should be detected even if tap not completed"
    )
