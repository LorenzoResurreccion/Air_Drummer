"""
Tap detection components for right foot bass drum triggering.

This package contains all the tap detection functionality including
data models, landmark extraction, movement analysis, tap detection,
debouncing, and trigger mechanisms.
"""

from tap_detection.tap_models import (
    TapType,
    DetectionState,
    FootLandmarks,
    TapEvent,
    TapDetectionConfig,
)

__all__ = [
    'TapType',
    'DetectionState',
    'FootLandmarks',
    'TapEvent',
    'TapDetectionConfig',
]
