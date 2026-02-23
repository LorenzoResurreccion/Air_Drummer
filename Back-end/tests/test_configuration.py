"""
Unit tests for configuration and optimization features.

Tests command-line argument parsing, FPS counter, and configuration options.
"""

import pytest
import numpy as np
import sys
import os

# Add Back-end directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from video_capture import VideoCaptureManager
from holistic_detector import HolisticDetector


class TestVideoCaptureConfiguration:
    """Test VideoCaptureManager configuration features."""
    
    def test_set_resolution_method_exists(self):
        """Test that set_resolution method exists."""
        vcm = VideoCaptureManager.__new__(VideoCaptureManager)
        assert hasattr(vcm, 'set_resolution')
        assert callable(vcm.set_resolution)


class TestHolisticDetectorConfiguration:
    """Test HolisticDetector configuration features."""
    
    def test_model_complexity_lite(self):
        """Test initialization with lite model (complexity 0)."""
        detector = HolisticDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0
        )
        assert detector is not None
        detector.close()
    
    def test_model_complexity_full(self):
        """Test initialization with full model (complexity 1)."""
        detector = HolisticDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        assert detector is not None
        detector.close()
    
    def test_model_complexity_heavy(self):
        """Test initialization with heavy model (complexity 2)."""
        detector = HolisticDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        assert detector is not None
        detector.close()
    
    def test_configurable_confidence_thresholds(self):
        """Test that confidence thresholds can be configured."""
        detector = HolisticDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        assert detector is not None
        detector.close()


class TestFPSCounter:
    """Test FPS counter functionality."""
    
    def test_fps_counter_methods_exist(self):
        """Test that FPS counter methods exist in BodyTracker."""
        # Import using importlib to handle hyphenated filename
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "Body_Tracker", 
            os.path.join(os.path.dirname(__file__), '..', 'Body-Tracker.py')
        )
        Body_Tracker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Body_Tracker)
        
        # Check that the methods exist in the module
        assert hasattr(Body_Tracker.BodyTracker, '_update_fps')
        assert hasattr(Body_Tracker.BodyTracker, '_add_fps_counter')


class TestCommandLineArguments:
    """Test command-line argument parsing."""
    
    def test_parse_arguments_function_exists(self):
        """Test that parse_arguments function exists."""
        # Import using importlib to handle hyphenated filename
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "Body_Tracker", 
            os.path.join(os.path.dirname(__file__), '..', 'Body-Tracker.py')
        )
        Body_Tracker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(Body_Tracker)
        
        assert hasattr(Body_Tracker, 'parse_arguments')
        assert callable(Body_Tracker.parse_arguments)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
