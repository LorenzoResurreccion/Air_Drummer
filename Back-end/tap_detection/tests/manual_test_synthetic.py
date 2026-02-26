"""
Manual test script for tap detection with synthetic FootLandmarks sequences.
This script demonstrates state transitions and tap classification.
"""

import time
from tap_detection.tap_models import FootLandmarks, TapDetectionConfig, TapType, DetectionState
from tap_detection.tap_detector import TapDetector
from comp_vision.models import LandmarkData


def create_landmark(x, y, z, confidence, landmark_type):
    """Helper to create a LandmarkData object."""
    return LandmarkData(x=x, y=y, z=z, visibility=confidence, landmark_type=landmark_type)


def create_foot_landmarks(heel_y, ankle_y, toe_y, timestamp, confidence=0.9):
    """Helper to create FootLandmarks with specified y positions."""
    return FootLandmarks(
        heel=create_landmark(0.5, heel_y, 0.0, confidence, "RIGHT_HEEL"),
        ankle=create_landmark(0.5, ankle_y, 0.0, confidence, "RIGHT_ANKLE"),
        toe=create_landmark(0.6, toe_y, 0.0, confidence, "RIGHT_FOOT_INDEX"),
        timestamp=timestamp
    )


def test_heel_tap_sequence():
    """Test heel lift tap with stationary toe."""
    print("\n=== Testing Heel Tap Sequence ===")
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    # Simulate heel tap: heel moves up, toe stays down
    base_time = time.time()
    
    # Initial position
    landmarks = create_foot_landmarks(heel_y=0.8, ankle_y=0.7, toe_y=0.85, timestamp=base_time)
    result = detector.process_frame(landmarks)
    print(f"Frame 1 - State: {detector.get_state()}, Result: {result}")
    
    # Heel starts moving up (y decreases)
    for i in range(1, 5):
        landmarks = create_foot_landmarks(
            heel_y=0.8 - (i * 0.02),  # Heel moves up
            ankle_y=0.7 - (i * 0.01),  # Ankle moves slightly
            toe_y=0.85,  # Toe stays stationary
            timestamp=base_time + (i * 0.033)  # ~30 FPS
        )
        result = detector.process_frame(landmarks)
        print(f"Frame {i+1} - State: {detector.get_state()}, Result: {result}")
    
    # Heel returns down
    for i in range(5, 8):
        landmarks = create_foot_landmarks(
            heel_y=0.8 - 0.08 + ((i-4) * 0.03),  # Heel moves down
            ankle_y=0.7 - 0.04 + ((i-4) * 0.015),
            toe_y=0.85,
            timestamp=base_time + (i * 0.033)
        )
        result = detector.process_frame(landmarks)
        print(f"Frame {i+1} - State: {detector.get_state()}, Result: {result}")
        if result:
            print(f"  ✓ TAP DETECTED: Type={result.tap_type}, Displacement={result.peak_displacement:.3f}, Duration={result.duration_ms:.1f}ms")
    
    # Check tap history
    history = detector.get_tap_history()
    print(f"\nTap History: {len(history)} taps")
    for tap in history:
        print(f"  - {tap.tap_type.value}: {tap.peak_displacement:.3f} displacement, {tap.duration_ms:.1f}ms")


def test_toe_tap_sequence():
    """Test toe lift tap with stationary heel."""
    print("\n=== Testing Toe Tap Sequence ===")
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    base_time = time.time()
    
    # Initial position
    landmarks = create_foot_landmarks(heel_y=0.8, ankle_y=0.7, toe_y=0.85, timestamp=base_time)
    result = detector.process_frame(landmarks)
    print(f"Frame 1 - State: {detector.get_state()}, Result: {result}")
    
    # Toe moves up, heel stays down
    for i in range(1, 5):
        landmarks = create_foot_landmarks(
            heel_y=0.8,  # Heel stays stationary
            ankle_y=0.7 - (i * 0.005),  # Ankle moves slightly
            toe_y=0.85 - (i * 0.02),  # Toe moves up
            timestamp=base_time + (i * 0.033)
        )
        result = detector.process_frame(landmarks)
        print(f"Frame {i+1} - State: {detector.get_state()}, Result: {result}")
    
    # Toe returns down
    for i in range(5, 8):
        landmarks = create_foot_landmarks(
            heel_y=0.8,
            ankle_y=0.7 - 0.02 + ((i-4) * 0.008),
            toe_y=0.85 - 0.08 + ((i-4) * 0.03),
            timestamp=base_time + (i * 0.033)
        )
        result = detector.process_frame(landmarks)
        print(f"Frame {i+1} - State: {detector.get_state()}, Result: {result}")
        if result:
            print(f"  ✓ TAP DETECTED: Type={result.tap_type}, Displacement={result.peak_displacement:.3f}, Duration={result.duration_ms:.1f}ms")
    
    history = detector.get_tap_history()
    print(f"\nTap History: {len(history)} taps")
    for tap in history:
        print(f"  - {tap.tap_type.value}: {tap.peak_displacement:.3f} displacement, {tap.duration_ms:.1f}ms")


def test_full_foot_tap_sequence():
    """Test full foot lift tap."""
    print("\n=== Testing Full Foot Tap Sequence ===")
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    base_time = time.time()
    
    # Initial position
    landmarks = create_foot_landmarks(heel_y=0.8, ankle_y=0.7, toe_y=0.85, timestamp=base_time)
    result = detector.process_frame(landmarks)
    print(f"Frame 1 - State: {detector.get_state()}, Result: {result}")
    
    # Both heel and toe move up
    for i in range(1, 5):
        landmarks = create_foot_landmarks(
            heel_y=0.8 - (i * 0.02),  # Heel moves up
            ankle_y=0.7 - (i * 0.015),  # Ankle moves up
            toe_y=0.85 - (i * 0.02),  # Toe moves up
            timestamp=base_time + (i * 0.033)
        )
        result = detector.process_frame(landmarks)
        print(f"Frame {i+1} - State: {detector.get_state()}, Result: {result}")
    
    # Both return down
    for i in range(5, 8):
        landmarks = create_foot_landmarks(
            heel_y=0.8 - 0.08 + ((i-4) * 0.03),
            ankle_y=0.7 - 0.06 + ((i-4) * 0.025),
            toe_y=0.85 - 0.08 + ((i-4) * 0.03),
            timestamp=base_time + (i * 0.033)
        )
        result = detector.process_frame(landmarks)
        print(f"Frame {i+1} - State: {detector.get_state()}, Result: {result}")
        if result:
            print(f"  ✓ TAP DETECTED: Type={result.tap_type}, Displacement={result.peak_displacement:.3f}, Duration={result.duration_ms:.1f}ms")
    
    history = detector.get_tap_history()
    print(f"\nTap History: {len(history)} taps")
    for tap in history:
        print(f"  - {tap.tap_type.value}: {tap.peak_displacement:.3f} displacement, {tap.duration_ms:.1f}ms")


def test_false_positive_rejection():
    """Test that slow movements are rejected."""
    print("\n=== Testing False Positive Rejection ===")
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    base_time = time.time()
    
    # Slow gradual movement (should be rejected)
    for i in range(20):
        landmarks = create_foot_landmarks(
            heel_y=0.8 - (i * 0.005),  # Very slow movement
            ankle_y=0.7 - (i * 0.003),
            toe_y=0.85,
            timestamp=base_time + (i * 0.1)  # Slow frame rate
        )
        result = detector.process_frame(landmarks)
        if result:
            print(f"Frame {i+1} - UNEXPECTED TAP: {result}")
    
    history = detector.get_tap_history()
    if len(history) == 0:
        print("✓ Correctly rejected slow movement (no taps detected)")
    else:
        print(f"✗ False positive: {len(history)} taps detected")


def test_state_transitions():
    """Test state machine transitions."""
    print("\n=== Testing State Transitions ===")
    
    config = TapDetectionConfig()
    detector = TapDetector(config)
    
    base_time = time.time()
    
    print(f"Initial state: {detector.get_state()}")
    assert detector.get_state() == DetectionState.IDLE
    
    # Build up history
    for i in range(3):
        landmarks = create_foot_landmarks(heel_y=0.8, ankle_y=0.7, toe_y=0.85, timestamp=base_time + i*0.033)
        detector.process_frame(landmarks)
    
    # Trigger upward movement
    landmarks = create_foot_landmarks(heel_y=0.7, ankle_y=0.65, toe_y=0.85, timestamp=base_time + 0.1)
    detector.process_frame(landmarks)
    print(f"After upward movement: {detector.get_state()}")
    
    # Continue upward
    landmarks = create_foot_landmarks(heel_y=0.65, ankle_y=0.6, toe_y=0.85, timestamp=base_time + 0.133)
    detector.process_frame(landmarks)
    print(f"Continuing upward: {detector.get_state()}")
    
    # Downward movement
    landmarks = create_foot_landmarks(heel_y=0.75, ankle_y=0.68, toe_y=0.85, timestamp=base_time + 0.166)
    result = detector.process_frame(landmarks)
    print(f"After downward movement: {detector.get_state()}, Tap: {result is not None}")
    
    if result:
        print(f"State after tap: {detector.get_state()}")


if __name__ == "__main__":
    print("=" * 60)
    print("Manual Test: Tap Detection with Synthetic Data")
    print("=" * 60)
    
    test_heel_tap_sequence()
    test_toe_tap_sequence()
    test_full_foot_tap_sequence()
    test_false_positive_rejection()
    test_state_transitions()
    
    print("\n" + "=" * 60)
    print("Manual testing complete!")
    print("=" * 60)
