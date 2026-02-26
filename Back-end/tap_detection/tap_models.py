"""
Data models for tap detection system.

This module defines the core data structures used for right foot tap detection,
including foot landmarks, tap events, tap types, detection states, and configuration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from comp_vision.models import LandmarkData


class TapType(Enum):
    """
    Enumeration of tap gesture types.
    
    Attributes:
        HEEL: Heel lift with toe stationary
        TOE: Toe lift with heel stationary
        FULL_FOOT: Both heel and toe lift together
    """
    HEEL = "heel"
    TOE = "toe"
    FULL_FOOT = "full_foot"


class DetectionState(Enum):
    """
    Current state of tap detection state machine.
    
    Attributes:
        IDLE: Waiting for tap to start
        DETECTING_UPWARD: Upward motion detected, waiting for completion
        DETECTING_DOWNWARD: Upward complete, waiting for downward motion
        DEBOUNCING: Cooldown period after tap detected
    """
    IDLE = "idle"
    DETECTING_UPWARD = "upward"
    DETECTING_DOWNWARD = "downward"
    DEBOUNCING = "debouncing"


@dataclass
class FootLandmarks:
    """
    Right foot landmark data for a single frame.
    
    Attributes:
        heel: RIGHT_HEEL landmark (index 30)
        ankle: RIGHT_ANKLE landmark (index 28)
        toe: RIGHT_FOOT_INDEX landmark (index 32)
        timestamp: Frame timestamp in seconds
    """
    heel: LandmarkData
    ankle: LandmarkData
    toe: LandmarkData
    timestamp: float
    
    def all_reliable(self, threshold: float = 0.5) -> bool:
        """
        Check if all foot landmarks meet confidence threshold.
        
        Args:
            threshold: Minimum confidence threshold (default 0.5)
            
        Returns:
            True if all landmarks have visibility above threshold, False otherwise
        """
        return (
            self.heel.is_reliable(threshold) and
            self.ankle.is_reliable(threshold) and
            self.toe.is_reliable(threshold)
        )


@dataclass
class TapEvent:
    """
    Information about a detected tap gesture.
    
    Attributes:
        tap_type: Type of tap (HEEL, TOE, or FULL_FOOT)
        timestamp: When tap completed (seconds)
        peak_displacement: Maximum displacement during tap (normalized units)
        duration_ms: Total tap duration (milliseconds)
    """
    tap_type: TapType
    timestamp: float
    peak_displacement: float
    duration_ms: float
    
    def to_dict(self) -> dict:
        """
        Convert tap event to dictionary for serialization.
        
        Returns:
            Dictionary representation of tap event
        """
        return {
            'tap_type': self.tap_type.value,
            'timestamp': self.timestamp,
            'peak_displacement': self.peak_displacement,
            'duration_ms': self.duration_ms
        }


@dataclass
class TapDetectionConfig:
    """
    Configuration parameters for tap detection.
    
    Attributes:
        min_displacement: Minimum vertical displacement to register tap (normalized units)
        min_velocity: Minimum vertical velocity to register tap (normalized units per second)
        max_tap_duration_ms: Maximum time for complete tap gesture (milliseconds)
        confidence_threshold: Minimum landmark confidence score
        stationary_threshold: Maximum movement for anchor point validation (normalized units)
    """
    min_displacement: float = 0.05
    min_velocity: float = 0.25
    max_tap_duration_ms: int = 400
    confidence_threshold: float = 0.5
    stationary_threshold: float = 0.02
    
    def validate(self) -> bool:
        """
        Validate configuration parameters are in reasonable ranges.
        
        Returns:
            True if all parameters are within valid ranges, False otherwise
            
        Raises:
            ValueError: If any parameter is outside valid range with descriptive message
        """
        errors = []
        
        if not (0.01 <= self.min_displacement <= 0.15):
            errors.append(
                f"min_displacement must be between 0.01 and 0.15, got {self.min_displacement}"
            )
        
        if not (0.1 <= self.min_velocity <= 1.0):
            errors.append(
                f"min_velocity must be between 0.1 and 1.0, got {self.min_velocity}"
            )
        
        if not (50 <= self.max_tap_duration_ms <= 1000):
            errors.append(
                f"max_tap_duration_ms must be between 50 and 1000, got {self.max_tap_duration_ms}"
            )
        
        if not (0.3 <= self.confidence_threshold <= 0.9):
            errors.append(
                f"confidence_threshold must be between 0.3 and 0.9, got {self.confidence_threshold}"
            )
        
        if not (0.01 <= self.stationary_threshold <= 0.05):
            errors.append(
                f"stationary_threshold must be between 0.01 and 0.05, got {self.stationary_threshold}"
            )
        
        if errors:
            raise ValueError(
                "Invalid configuration parameters:\n" + "\n".join(f"  - {err}" for err in errors)
            )
        
        return True
