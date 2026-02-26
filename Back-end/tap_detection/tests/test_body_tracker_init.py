"""
Test script to verify Body-Tracker initialization with tap detection.

This script tests that:
1. Body-Tracker can be imported successfully
2. All components initialize correctly
3. Tap detection is integrated properly
4. WebSocket server is configured

Note: This test does NOT require a camera to be connected.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required modules can be imported."""
    print("\n=== Test 1: Module Imports ===")
    
    try:
        from tap_detection.tap_detection_manager import TapDetectionManager
        from tap_detection.tap_models import TapDetectionConfig
        from tap_detection.bass_drum_trigger import BassDrumTrigger
        import socketio
        
        print("✓ All tap detection modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_tap_detection_config():
    """Test tap detection configuration."""
    print("\n=== Test 2: Tap Detection Configuration ===")
    
    try:
        from tap_detection.tap_models import TapDetectionConfig
        
        # Test default configuration
        config = TapDetectionConfig()
        config.validate()
        
        print("✓ Default configuration is valid")
        print(f"  - min_displacement: {config.min_displacement}")
        print(f"  - min_velocity: {config.min_velocity}")
        print(f"  - max_tap_duration_ms: {config.max_tap_duration_ms}")
        print(f"  - confidence_threshold: {config.confidence_threshold}")
        
        # Test custom configuration
        custom_config = TapDetectionConfig(
            min_displacement=0.06,
            min_velocity=0.3,
            max_tap_duration_ms=350,
            confidence_threshold=0.6
        )
        custom_config.validate()
        
        print("✓ Custom configuration is valid")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_websocket_server():
    """Test WebSocket server initialization."""
    print("\n=== Test 3: WebSocket Server ===")
    
    try:
        import socketio
        
        # Create Socket.IO server
        sio = socketio.Server(cors_allowed_origins='*', async_mode='threading')
        
        print("✓ WebSocket server created successfully")
        print(f"  - Server type: {type(sio).__name__}")
        print(f"  - CORS enabled: *")
        print(f"  - Async mode: threading")
        
        return True
        
    except Exception as e:
        print(f"✗ WebSocket server test failed: {e}")
        return False


def test_tap_detection_manager():
    """Test TapDetectionManager initialization."""
    print("\n=== Test 4: TapDetectionManager ===")
    
    try:
        from tap_detection.tap_detection_manager import TapDetectionManager
        from tap_detection.tap_models import TapDetectionConfig
        import socketio
        
        # Create Socket.IO server
        sio = socketio.Server(cors_allowed_origins='*', async_mode='threading')
        
        # Create manager with custom configuration
        config = TapDetectionConfig(
            min_displacement=0.05,
            min_velocity=0.25
        )
        
        manager = TapDetectionManager(
            config=config,
            debounce_ms=150,
            sio=sio
        )
        
        print("✓ TapDetectionManager initialized successfully")
        print(f"  - Initial state: {manager.get_state()}")
        print(f"  - Trigger count: {manager.get_trigger_count()}")
        print(f"  - Debouncing: {manager.is_debouncing()}")
        print(f"  - Tap history: {len(manager.get_tap_history())} taps")
        
        return True
        
    except Exception as e:
        print(f"✗ TapDetectionManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_body_tracker_class():
    """Test that BodyTracker class can be imported (without camera)."""
    print("\n=== Test 5: BodyTracker Class ===")
    
    try:
        # Import the module (don't instantiate - requires camera)
        sys.path.insert(0, '.')
        import importlib.util
        spec = importlib.util.spec_from_file_location("Body_Tracker", "Body-Tracker.py")
        Body_Tracker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Body_Tracker)
        
        # Check that BodyTracker class exists
        assert hasattr(Body_Tracker, 'BodyTracker'), "BodyTracker class not found"
        
        print("✓ BodyTracker class imported successfully")
        print("  Note: Not instantiating (requires camera)")
        
        # Check that parse_arguments function exists
        assert hasattr(Body_Tracker, 'parse_arguments'), "parse_arguments function not found"
        
        print("✓ Command-line argument parser available")
        
        return True
        
    except Exception as e:
        print(f"✗ BodyTracker class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_command_line_args():
    """Test command-line argument parsing."""
    print("\n=== Test 6: Command-Line Arguments ===")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Body_Tracker", "Body-Tracker.py")
        Body_Tracker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Body_Tracker)
        
        # Parse default arguments
        sys.argv = ['Body-Tracker.py']
        args = Body_Tracker.parse_arguments()
        
        print("✓ Default arguments parsed successfully")
        print(f"  - camera: {args.camera}")
        print(f"  - tap_detection: {'enabled' if not args.disable_tap_detection else 'disabled'}")
        print(f"  - tap_min_displacement: {args.tap_min_displacement}")
        print(f"  - tap_min_velocity: {args.tap_min_velocity}")
        print(f"  - tap_debounce: {args.tap_debounce}ms")
        print(f"  - websocket_port: {args.websocket_port}")
        
        # Test custom arguments
        sys.argv = [
            'Body-Tracker.py',
            '--tap-min-displacement', '0.06',
            '--tap-min-velocity', '0.3',
            '--tap-debounce', '200',
            '--websocket-port', '5001'
        ]
        args = Body_Tracker.parse_arguments()
        
        assert args.tap_min_displacement == 0.06
        assert args.tap_min_velocity == 0.3
        assert args.tap_debounce == 200
        assert args.websocket_port == 5001
        
        print("✓ Custom arguments parsed correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Command-line arguments test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all initialization tests."""
    print("=" * 60)
    print("Body-Tracker Initialization Tests")
    print("=" * 60)
    print("\nThese tests verify the backend integration without")
    print("requiring a camera to be connected.")
    
    results = []
    
    results.append(("Module Imports", test_imports()))
    results.append(("Tap Detection Config", test_tap_detection_config()))
    results.append(("WebSocket Server", test_websocket_server()))
    results.append(("TapDetectionManager", test_tap_detection_manager()))
    results.append(("BodyTracker Class", test_body_tracker_class()))
    results.append(("Command-Line Args", test_command_line_args()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe backend is properly integrated and ready to use.")
        print("\nTo run with a camera:")
        print("  python Back-end/Body-Tracker.py")
        print("\nTo run with custom tap detection settings:")
        print("  python Back-end/Body-Tracker.py --tap-min-displacement 0.06 --tap-debounce 200")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
