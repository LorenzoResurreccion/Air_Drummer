"""
Computer vision components for AIR DRUMMER body tracking.

This package contains all the core computer vision functionality including
video capture, frame processing, landmark detection, data extraction, and
visual rendering.
"""

from comp_vision.models import LandmarkData, TrackingData
from comp_vision.video_capture import VideoCaptureManager, CameraAccessError
from comp_vision.frame_processor import FramePreprocessor
from comp_vision.holistic_detector import HolisticDetector
from comp_vision.data_extractor import TrackingDataExtractor
from comp_vision.visual_renderer import VisualRenderer

__all__ = [
    'LandmarkData',
    'TrackingData',
    'VideoCaptureManager',
    'CameraAccessError',
    'FramePreprocessor',
    'HolisticDetector',
    'TrackingDataExtractor',
    'VisualRenderer',
]
