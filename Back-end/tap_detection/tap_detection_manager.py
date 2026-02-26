"""
Tap detection manager - main controller for tap detection pipeline.

This module implements the TapDetectionManager class which orchestrates the
complete tap detection pipeline. It coordinates data flow between all components:
landmark extraction, movement analysis, tap detection, debouncing, and triggering.
"""

import logging
from typing import Optional
import socketio

from comp_vision.models import TrackingData
from tap_detection.landmark_extractor import FootLandmarkExtractor
from tap_detection.movement_analyzer import MovementAnalyzer
from tap_detection.tap_detector import TapDetector
from tap_detection.debounce_filter import DebounceFilter
from tap_detection.bass_drum_trigger import BassDrumTrigger
from tap_detection.tap_models import (
    TapDetectionConfig,
    TapEvent,
    DetectionState
)


# Configure logging
logger = logging.getLogger(__name__)


class TapDetectionManager:
    """
    Main controller for tap detection system.
    
    Orchestrates the complete tap detection pipeline by coordinating all
    components: landmark extraction, movement analysis, tap detection,
    debouncing, and audio triggering. Provides a simple interface for
    processing tracking data and accessing detection state.
    
    Pipeline flow:
    1. Extract foot landmarks from TrackingData
    2. Validate landmark confidence
    3. Analyze movement patterns
    4. Detect tap gestures
    5. Filter with debouncing
    6. Trigger bass drum sound
    
    Attributes:
        config: Tap detection configuration
        extractor: Extracts foot landmarks from tracking data
        detector: Detects tap gestures from movement patterns
        debounce_filter: Prevents duplicate triggers
        trigger: Emits trigger events to audio player
    """
    
    def __init__(
        self,
        config: Optional[TapDetectionConfig] = None,
        debounce_ms: int = 150,
        sio: Optional[socketio.Server] = None
    ):
        """
        Initialize tap detection system with all components.
        
        Args:
            config: Tap detection configuration. If None, uses default values.
            debounce_ms: Debounce window duration in milliseconds (default 150)
            sio: Socket.IO server instance for event emission. If None, creates new server.
        """
        # Use default configuration if none provided
        self.config = config if config is not None else TapDetectionConfig()
        
        # Validate configuration
        self.config.validate()
        
        # Initialize all pipeline components
        self.extractor = FootLandmarkExtractor()
        self.detector = TapDetector(self.config)
        self.debounce_filter = DebounceFilter(debounce_ms)
        self.trigger = BassDrumTrigger(sio)
        
        logger.info(
            f"TapDetectionManager initialized with config: "
            f"min_displacement={self.config.min_displacement}, "
            f"min_velocity={self.config.min_velocity}, "
            f"debounce={debounce_ms}ms"
        )
    
    def process_tracking_data(self, tracking_data: TrackingData) -> None:
        """
        Process frame from body tracking system.
        
        This is the main entry point for the tap detection pipeline. It
        orchestrates data flow through all components and handles the
        complete detection and triggering process.
        
        Handles missing or invalid tracking data gracefully without raising
        exceptions, allowing the tracking system to continue operating.
        
        Args:
            tracking_data: TrackingData object from body tracking system
        """
        try:
            # Step 1: Extract foot landmarks
            foot_landmarks = self.extractor.extract(tracking_data)
            
            # Step 2: Validate landmarks (None if missing or low confidence)
            if foot_landmarks is not None:
                if not self.extractor.validate_landmarks(
                    foot_landmarks, 
                    self.config.confidence_threshold
                ):
                    foot_landmarks = None
            
            # Step 3: Process frame with tap detector
            tap_event = self.detector.process_frame(foot_landmarks)
            
            # Step 4: If tap detected, apply debouncing and trigger
            if tap_event is not None:
                if self.debounce_filter.should_trigger(tap_event):
                    # Step 5: Trigger bass drum sound
                    self.trigger.trigger(tap_event)
                    
                    logger.info(
                        f"Tap triggered: type={tap_event.tap_type.value}, "
                        f"displacement={tap_event.peak_displacement:.3f}, "
                        f"duration={tap_event.duration_ms:.1f}ms"
                    )
                else:
                    logger.debug(
                        f"Tap filtered by debounce: type={tap_event.tap_type.value}"
                    )
        
        except Exception as e:
            # Log error but don't crash - tracking should continue
            logger.error(f"Error processing tracking data: {e}", exc_info=True)
    
    def get_state(self) -> DetectionState:
        """
        Get current tap detection state.
        
        Delegates to the TapDetector to retrieve the current state of the
        detection state machine (IDLE, DETECTING_UPWARD, DETECTING_DOWNWARD,
        or DEBOUNCING).
        
        Returns:
            Current DetectionState
        """
        return self.detector.get_state()
    
    def get_tap_history(self) -> list[TapEvent]:
        """
        Get recent tap history.
        
        Delegates to the TapDetector to retrieve the history of recent
        tap events (up to 10 most recent taps).
        
        Returns:
            List of recent TapEvent objects
        """
        return self.detector.get_tap_history()
    
    def update_config(self, config: TapDetectionConfig) -> None:
        """
        Update detection configuration at runtime.
        
        Validates the new configuration and updates the detector with new
        parameters. This allows for runtime tuning of detection sensitivity
        without restarting the system.
        
        Args:
            config: New TapDetectionConfig to apply
            
        Raises:
            ValueError: If configuration validation fails
        """
        # Validate new configuration
        config.validate()
        
        # Update configuration
        self.config = config
        
        # Create new detector with updated config
        # Preserve tap history from old detector
        old_history = self.detector.get_tap_history()
        self.detector = TapDetector(config)
        
        # Restore tap history (manually add to new detector's history)
        for tap_event in old_history:
            self.detector._tap_history.append(tap_event)
        
        logger.info(
            f"Configuration updated: "
            f"min_displacement={config.min_displacement}, "
            f"min_velocity={config.min_velocity}, "
            f"max_duration={config.max_tap_duration_ms}ms"
        )
    
    def get_trigger_count(self) -> int:
        """
        Get total number of triggers sent.
        
        Useful for debugging and monitoring tap detection performance.
        
        Returns:
            Total number of trigger events emitted
        """
        return self.trigger.get_trigger_count()
    
    def is_debouncing(self) -> bool:
        """
        Check if currently in debounce state.
        
        Returns:
            True if within debounce window, False otherwise
        """
        return self.debounce_filter.is_debouncing()
    
    def get_socketio_server(self) -> socketio.Server:
        """
        Get the Socket.IO server instance for WebSocket communication.
        
        This is useful for integrating the server with the main application
        (e.g., wrapping with WSGI app for serving).
        
        Returns:
            Socket.IO server instance
        """
        return self.trigger.get_server()
