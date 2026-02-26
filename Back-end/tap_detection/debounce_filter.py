"""
Debounce filter for tap detection system.

This module implements debouncing logic to prevent multiple triggers from a single
physical tap. It ensures each tap gesture triggers exactly one sound by filtering
out subsequent detections within a configurable time window.
"""

import time
from typing import Optional

from tap_detection.tap_models import TapEvent


class DebounceFilter:
    """
    Filter to prevent multiple triggers from single physical tap.
    
    The debounce filter tracks the timestamp of the last triggered tap and
    filters out any taps that occur within the debounce window. This ensures
    that rapid successive detections from a single physical tap only result
    in one trigger event.
    
    Attributes:
        debounce_window_ms: Duration of debounce window in milliseconds
        _last_trigger_time: Timestamp of last triggered tap (None if no taps yet)
    """
    
    def __init__(self, debounce_window_ms: int = 150):
        """
        Initialize debounce filter with configurable window.
        
        Args:
            debounce_window_ms: Duration of debounce window in milliseconds (default 150)
        """
        self.debounce_window_ms = debounce_window_ms
        self._last_trigger_time: Optional[float] = None
    
    def should_trigger(self, tap_event: TapEvent) -> bool:
        """
        Check if tap should trigger sound or be filtered.
        
        Determines whether the given tap event should be allowed to trigger
        a sound based on the time elapsed since the last triggered tap.
        If the tap is allowed, updates the last trigger time.
        
        Args:
            tap_event: The tap event to evaluate
            
        Returns:
            True if tap should trigger (outside debounce window), False if filtered
        """
        # If no previous tap, allow this one
        if self._last_trigger_time is None:
            self._last_trigger_time = tap_event.timestamp
            return True
        
        # Calculate time since last trigger in milliseconds
        time_since_last_ms = (tap_event.timestamp - self._last_trigger_time) * 1000
        
        # Check if outside debounce window
        if time_since_last_ms >= self.debounce_window_ms:
            self._last_trigger_time = tap_event.timestamp
            return True
        
        # Within debounce window, filter this tap
        return False
    
    def is_debouncing(self) -> bool:
        """
        Check if currently in debounce state.
        
        Returns:
            True if within debounce window of last trigger, False otherwise
        """
        if self._last_trigger_time is None:
            return False
        
        # Get current time
        current_time = time.time()
        
        # Calculate time since last trigger in milliseconds
        time_since_last_ms = (current_time - self._last_trigger_time) * 1000
        
        # Check if still within debounce window
        return time_since_last_ms < self.debounce_window_ms
