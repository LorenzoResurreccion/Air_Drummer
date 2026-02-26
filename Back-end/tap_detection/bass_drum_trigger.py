"""
Bass drum trigger component for WebSocket-based audio triggering.

This module implements the BassDrumTrigger class which emits tap events
to the frontend audio player via WebSocket communication. It handles
connection errors gracefully and maintains trigger statistics for debugging.
"""

import logging
from typing import Optional
import socketio

from tap_detection.tap_models import TapEvent


# Configure logging
logger = logging.getLogger(__name__)


class BassDrumTrigger:
    """
    Trigger bass drum sounds via WebSocket events.
    
    This class emits tap events to connected frontend clients through
    a Socket.IO server. It includes timestamp and tap metadata in events
    for latency monitoring and potential future features like velocity-sensitive
    volume control.
    
    Attributes:
        sio: Socket.IO server instance for event emission
        trigger_count: Total number of triggers sent (for debugging)
    """
    
    def __init__(self, sio: Optional[socketio.Server] = None):
        """
        Initialize bass drum trigger.
        
        Args:
            sio: Socket.IO server instance. If None, creates a new server.
        """
        self.sio = sio if sio is not None else socketio.Server(
            cors_allowed_origins='*',
            async_mode='threading'
        )
        self.trigger_count = 0
        logger.info("BassDrumTrigger initialized")
    
    def trigger(self, tap_event: TapEvent) -> None:
        """
        Send trigger event to audio player via WebSocket.
        
        Emits a 'bass_drum_trigger' event containing tap metadata including
        timestamp, tap type, peak displacement (can be used for volume control),
        and duration.
        
        Connection errors are logged but do not raise exceptions to avoid
        crashing the tracking pipeline.
        
        Args:
            tap_event: The detected tap event to trigger
        """
        try:
            # Prepare event data with all tap metadata
            event_data = {
                'event_type': 'bass_drum_trigger',
                'timestamp': tap_event.timestamp,
                'tap_type': tap_event.tap_type.value,
                'peak_displacement': tap_event.peak_displacement,
                'duration_ms': tap_event.duration_ms,
                # Velocity can be derived from displacement/duration for future volume control
                'velocity': tap_event.peak_displacement / (tap_event.duration_ms / 1000.0) if tap_event.duration_ms > 0 else 0.0
            }
            
            # Emit event to all connected clients
            self.sio.emit('bass_drum_trigger', event_data)
            
            # Increment trigger counter
            self.trigger_count += 1
            
            logger.debug(
                f"Triggered bass drum: type={tap_event.tap_type.value}, "
                f"displacement={tap_event.peak_displacement:.3f}, "
                f"duration={tap_event.duration_ms:.1f}ms, "
                f"count={self.trigger_count}"
            )
            
        except Exception as e:
            # Log error but don't crash - tracking should continue
            logger.error(f"Failed to emit bass drum trigger event: {e}")
    
    def get_trigger_count(self) -> int:
        """
        Get total number of triggers sent.
        
        Useful for debugging and monitoring tap detection performance.
        
        Returns:
            Total number of trigger events emitted since initialization
        """
        return self.trigger_count
    
    def reset_trigger_count(self) -> None:
        """
        Reset trigger counter to zero.
        
        Useful for testing and debugging scenarios.
        """
        self.trigger_count = 0
        logger.debug("Trigger count reset to 0")
    
    def get_server(self) -> socketio.Server:
        """
        Get the Socket.IO server instance.
        
        Returns:
            The Socket.IO server instance used for event emission
        """
        return self.sio
