"""
Backend integration test for tap detection system.

This test verifies that:
1. All components initialize correctly
2. TapDetectionManager processes tracking data
3. WebSocket server can be initialized
4. Tap events are emitted correctly
5. The complete pipeline works end-to-end
"""

import time
import socketio
from comp_vision.models import TrackingData, LandmarkData
from tap_detection.tap_detection_manager import TapDetectionManager
from tap_detection.tap_models import TapDetectionConfig, TapType


def create_tracking_data_with_foot(heel_y, ankle_y, toe_y, timestamp, confidence=0.9):
    """Create TrackingData with right foot landmarks."""
    # Create 33 pose landmarks (MediaPipe Holistic format)
    pose_landmarks = []
    for i in range(33):
        if i == 28:  # RIGHT_ANKLE
            landmark = LandmarkData(x=0.5, y=ankle_y, z=0.0, visibility=confidence, landmark_type="RIGHT_ANKLE")
        elif i == 30:  # RIGHT_HEEL
            landmark = LandmarkData(x=0.5, y=heel_y, z=0.0, visibility=confidence, landmark_type="RIGHT_HEEL")
        elif i == 32:  # RIGHT_FOOT_INDEX
            landmark = LandmarkData(x=0.6, y=toe_y, z=0.0, visibility=confidence, landmark_type="RIGHT_FOOT_INDEX")
        else:
            # Dummy landmarks for other body parts
            landmark = LandmarkData(x=0.5, y=0.5, z=0.0, visibility=0.9, landmark_type=f"LANDMARK_{i}")
        pose_landmarks.append(landmark)
    
    return TrackingData(
        timestamp=timestamp,
        frame_number=0,
        pose_landmarks=pose_landmarks,
        left_hand_landmarks=None,
        right_hand_landmarks=None
    )


def test_manager_initialization():
    """Test that TapDetectionManager initializes correctly."""
    print("\n=== Test 1: Manager Initialization ===")
    
    config = TapDetectionConfig()
    manager = TapDetectionManager(config=config, debounce_ms=150)
    
    assert manager.get_state().value == "idle"
    assert len(manager.get_tap_history()) == 0
    assert manager.get_trigger_count() == 0
    assert not manager.is_debouncing()
    
    print("✓ TapDetectionManager initialized successfully")
    print(f"  - Initial state: {manager.get_state()}")
    print(f"  - Trigger count: {manager.get_trigger_count()}")


def test_heel_tap_detection():
    """Test heel tap detection with synthetic tracking data."""
    print("\n=== Test 2: Heel Tap Detection ===")
    
    config = TapDetectionConfig()
    manager = TapDetectionManager(config=config, debounce_ms=150)
    
    base_time = time.time()
    
    # Build up history with stationary foot
    for i in range(3):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8, ankle_y=0.7, toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Heel moves up (tap starts)
    for i in range(3, 7):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - ((i-2) * 0.02),  # Heel moves up
            ankle_y=0.7 - ((i-2) * 0.01),
            toe_y=0.85,  # Toe stays stationary
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Heel returns down (tap completes)
    for i in range(7, 10):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - 0.08 + ((i-6) * 0.03),  # Heel moves down
            ankle_y=0.7 - 0.04 + ((i-6) * 0.015),
            toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Check results
    history = manager.get_tap_history()
    assert len(history) > 0, "No tap detected"
    assert history[0].tap_type == TapType.HEEL, f"Expected HEEL tap, got {history[0].tap_type}"
    assert manager.get_trigger_count() > 0, "No trigger event emitted"
    
    print("✓ Heel tap detected successfully")
    print(f"  - Tap type: {history[0].tap_type.value}")
    print(f"  - Displacement: {history[0].peak_displacement:.3f}")
    print(f"  - Duration: {history[0].duration_ms:.1f}ms")
    print(f"  - Trigger count: {manager.get_trigger_count()}")


def test_toe_tap_detection():
    """Test toe tap detection with synthetic tracking data."""
    print("\n=== Test 3: Toe Tap Detection ===")
    
    config = TapDetectionConfig()
    manager = TapDetectionManager(config=config, debounce_ms=150)
    
    base_time = time.time()
    
    # Build up history
    for i in range(3):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8, ankle_y=0.7, toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Toe moves up
    for i in range(3, 7):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8,  # Heel stays stationary
            ankle_y=0.7 - ((i-2) * 0.005),
            toe_y=0.85 - ((i-2) * 0.02),  # Toe moves up
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Toe returns down
    for i in range(7, 10):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8,
            ankle_y=0.7 - 0.02 + ((i-6) * 0.008),
            toe_y=0.85 - 0.08 + ((i-6) * 0.03),
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Check results
    history = manager.get_tap_history()
    assert len(history) > 0, "No tap detected"
    assert history[0].tap_type == TapType.TOE, f"Expected TOE tap, got {history[0].tap_type}"
    
    print("✓ Toe tap detected successfully")
    print(f"  - Tap type: {history[0].tap_type.value}")
    print(f"  - Displacement: {history[0].peak_displacement:.3f}")
    print(f"  - Duration: {history[0].duration_ms:.1f}ms")


def test_full_foot_tap_detection():
    """Test full foot tap detection with synthetic tracking data."""
    print("\n=== Test 4: Full Foot Tap Detection ===")
    
    config = TapDetectionConfig()
    manager = TapDetectionManager(config=config, debounce_ms=150)
    
    base_time = time.time()
    
    # Build up history
    for i in range(3):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8, ankle_y=0.7, toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Both heel and toe move up
    for i in range(3, 7):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - ((i-2) * 0.02),  # Heel moves up
            ankle_y=0.7 - ((i-2) * 0.015),  # Ankle moves up
            toe_y=0.85 - ((i-2) * 0.02),  # Toe moves up
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Both return down
    for i in range(7, 10):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - 0.08 + ((i-6) * 0.03),
            ankle_y=0.7 - 0.06 + ((i-6) * 0.025),
            toe_y=0.85 - 0.08 + ((i-6) * 0.03),
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    # Check results
    history = manager.get_tap_history()
    assert len(history) > 0, "No tap detected"
    assert history[0].tap_type == TapType.FULL_FOOT, f"Expected FULL_FOOT tap, got {history[0].tap_type}"
    
    print("✓ Full foot tap detected successfully")
    print(f"  - Tap type: {history[0].tap_type.value}")
    print(f"  - Displacement: {history[0].peak_displacement:.3f}")
    print(f"  - Duration: {history[0].duration_ms:.1f}ms")


def test_debouncing():
    """Test that debouncing prevents duplicate triggers."""
    print("\n=== Test 5: Debouncing ===")
    
    config = TapDetectionConfig()
    manager = TapDetectionManager(config=config, debounce_ms=150)
    
    base_time = time.time()
    
    # First tap
    for i in range(3):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8, ankle_y=0.7, toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    for i in range(3, 7):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - ((i-2) * 0.02),
            ankle_y=0.7 - ((i-2) * 0.01),
            toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    for i in range(7, 10):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - 0.08 + ((i-6) * 0.03),
            ankle_y=0.7 - 0.04 + ((i-6) * 0.015),
            toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    first_trigger_count = manager.get_trigger_count()
    
    # Try second tap immediately (should be debounced)
    for i in range(10, 13):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8, ankle_y=0.7, toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    for i in range(13, 17):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - ((i-12) * 0.02),
            ankle_y=0.7 - ((i-12) * 0.01),
            toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    for i in range(17, 20):
        tracking_data = create_tracking_data_with_foot(
            heel_y=0.8 - 0.08 + ((i-16) * 0.03),
            ankle_y=0.7 - 0.04 + ((i-16) * 0.015),
            toe_y=0.85,
            timestamp=base_time + i * 0.033
        )
        manager.process_tracking_data(tracking_data)
    
    second_trigger_count = manager.get_trigger_count()
    
    # Second tap should be filtered by debounce
    assert second_trigger_count == first_trigger_count, "Debouncing failed - second tap was not filtered"
    
    print("✓ Debouncing works correctly")
    print(f"  - First tap triggered: count = {first_trigger_count}")
    print(f"  - Second tap filtered: count = {second_trigger_count}")


def test_missing_landmarks_handling():
    """Test that missing landmarks are handled gracefully."""
    print("\n=== Test 6: Missing Landmarks Handling ===")
    
    config = TapDetectionConfig()
    manager = TapDetectionManager(config=config, debounce_ms=150)
    
    base_time = time.time()
    
    # Process with missing pose landmarks
    tracking_data = TrackingData(
        timestamp=base_time,
        frame_number=0,
        pose_landmarks=None,
        left_hand_landmarks=None,
        right_hand_landmarks=None
    )
    
    # Should not crash
    manager.process_tracking_data(tracking_data)
    
    assert manager.get_state().value == "idle"
    assert len(manager.get_tap_history()) == 0
    
    print("✓ Missing landmarks handled gracefully")
    print(f"  - State remains: {manager.get_state()}")


def test_websocket_server_initialization():
    """Test that WebSocket server can be initialized."""
    print("\n=== Test 7: WebSocket Server Initialization ===")
    
    # Create Socket.IO server
    sio = socketio.Server(cors_allowed_origins='*', async_mode='threading')
    
    # Initialize manager with custom server
    config = TapDetectionConfig()
    manager = TapDetectionManager(config=config, debounce_ms=150, sio=sio)
    
    # Verify server is accessible
    server = manager.get_socketio_server()
    assert server is not None
    assert server == sio
    
    print("✓ WebSocket server initialized successfully")
    print(f"  - Server type: {type(server).__name__}")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Backend Integration Tests")
    print("=" * 60)
    
    try:
        test_manager_initialization()
        test_heel_tap_detection()
        test_toe_tap_detection()
        test_full_foot_tap_detection()
        test_debouncing()
        test_missing_landmarks_handling()
        test_websocket_server_initialization()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
