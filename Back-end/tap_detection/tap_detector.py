"""
Tap detector with state machine for gesture recognition.

This module implements the TapDetector class which uses a state machine to
detect tap gestures from foot landmark movements. It identifies heel lifts,
toe lifts, and full foot lifts by analyzing movement patterns, validating
anchor points, and classifying tap types.
"""

from collections import deque
from typing import Optional
import time

from tap_detection.tap_models import (
    FootLandmarks,
    TapEvent,
    TapType,
    DetectionState,
    TapDetectionConfig
)
from tap_detection.movement_analyzer import MovementAnalyzer


class TapDetector:
    """
    Detects tap gestures using state machine and movement analysis.
    
    The detector uses a four-state machine:
    - IDLE: Waiting for tap to start
    - DETECTING_UPWARD: Upward motion detected, tracking gesture
    - DETECTING_DOWNWARD: Upward complete, waiting for downward motion
    - DEBOUNCING: Cooldown after tap detected
    
    Attributes:
        config: Configuration parameters for detection
        _state: Current detection state
        _movement_analyzer: Analyzer for calculating movement metrics
        _tap_history: Recent tap events (max 10)
        _gesture_start_time: Timestamp when upward motion began
        _peak_displacement: Maximum displacement during current gesture
        _detected_tap_type: Type of tap being tracked
    """
    
    def __init__(self, config: TapDetectionConfig):
        """
        Initialize TapDetector with configuration.
        
        Args:
            config: TapDetectionConfig with detection parameters
        """
        # Validate configuration
        config.validate()
        
        self.config = config
        self._state = DetectionState.IDLE
        self._movement_analyzer = MovementAnalyzer(history_size=10)
        self._tap_history: deque[TapEvent] = deque(maxlen=10)
        
        # Gesture tracking state
        self._gesture_start_time: Optional[float] = None
        self._peak_displacement: float = 0.0
        self._detected_tap_type: Optional[TapType] = None
        self._debounce_end_time: Optional[float] = None
    
    def process_frame(
        self, 
        landmarks: Optional[FootLandmarks]
    ) -> Optional[TapEvent]:
        """
        Process frame and return tap event if detected.
        
        This is the main entry point for tap detection. It handles missing
        landmarks, updates movement history, manages state transitions, and
        returns TapEvent when a complete tap gesture is detected.
        
        Args:
            landmarks: FootLandmarks for current frame, or None if unavailable
            
        Returns:
            TapEvent if complete tap detected, None otherwise
        """
        # Handle missing landmarks
        if landmarks is None:
            return self._handle_missing_landmarks()
        
        # Check confidence threshold
        if not landmarks.all_reliable(self.config.confidence_threshold):
            return self._handle_missing_landmarks()
        
        # Update movement history
        self._movement_analyzer.update(landmarks)
        
        # Check debounce state
        if self._state == DetectionState.DEBOUNCING:
            if landmarks.timestamp >= self._debounce_end_time:
                self._state = DetectionState.IDLE
            return None
        
        # Process based on current state
        if self._state == DetectionState.IDLE:
            return self._process_idle_state(landmarks)
        elif self._state == DetectionState.DETECTING_UPWARD:
            return self._process_upward_state(landmarks)
        elif self._state == DetectionState.DETECTING_DOWNWARD:
            return self._process_downward_state(landmarks)
        
        return None

    
    def _handle_missing_landmarks(self) -> None:
        """
        Handle missing or low confidence landmarks.
        
        If a gesture is in progress, cancel it and reset to IDLE.
        This prevents false positives from incomplete gestures.
        
        Returns:
            None (no tap event)
        """
        if self._state in (DetectionState.DETECTING_UPWARD, DetectionState.DETECTING_DOWNWARD):
            # Cancel in-progress gesture
            self._reset_gesture_state()
            self._state = DetectionState.IDLE
        
        return None
    
    def _process_idle_state(self, landmarks: FootLandmarks) -> None:
        """
        Process frame in IDLE state - look for upward motion.
        
        Checks for upward movement that exceeds displacement and velocity
        thresholds. If detected, transitions to DETECTING_UPWARD state.
        
        Applies false positive prevention:
        - Rejects slow movements (velocity < threshold)
        - Rejects primarily horizontal movements (vertical < 0.03 or horizontal > vertical)
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            None (no tap event in IDLE state)
        """
        # Check for upward movement in heel, toe, or both
        heel_displacement = self._movement_analyzer.get_vertical_displacement(
            "heel", self.config.max_tap_duration_ms
        )
        toe_displacement = self._movement_analyzer.get_vertical_displacement(
            "toe", self.config.max_tap_duration_ms
        )
        ankle_displacement = self._movement_analyzer.get_vertical_displacement(
            "ankle", self.config.max_tap_duration_ms
        )
        
        heel_velocity = self._movement_analyzer.get_vertical_velocity("heel")
        toe_velocity = self._movement_analyzer.get_vertical_velocity("toe")
        
        # Get horizontal displacements for false positive prevention
        heel_horizontal = self._movement_analyzer.get_horizontal_displacement(
            "heel", self.config.max_tap_duration_ms
        )
        toe_horizontal = self._movement_analyzer.get_horizontal_displacement(
            "toe", self.config.max_tap_duration_ms
        )
        
        # Determine if upward motion detected and classify tap type
        tap_type = None
        peak_displacement = 0.0
        
        # Check for full foot lift (both heel and toe moving up)
        if (heel_displacement > self.config.min_displacement and
            toe_displacement > self.config.min_displacement and
            heel_velocity > self.config.min_velocity and
            toe_velocity > self.config.min_velocity):
            
            # False positive prevention: check for primarily horizontal movement
            # Vertical displacement must be at least 0.03 and greater than horizontal
            if (heel_displacement >= 0.03 and toe_displacement >= 0.03 and
                heel_displacement > heel_horizontal and 
                toe_displacement > toe_horizontal):
                tap_type = TapType.FULL_FOOT
                peak_displacement = max(heel_displacement, toe_displacement)
        
        # Check for heel lift (heel up, toe stationary)
        elif (heel_displacement > self.config.min_displacement and
              heel_velocity > self.config.min_velocity and
              self._movement_analyzer.is_stationary("toe", self.config.stationary_threshold)):
            
            # False positive prevention: check for primarily horizontal movement
            if heel_displacement >= 0.03 and heel_displacement > heel_horizontal:
                tap_type = TapType.HEEL
                peak_displacement = heel_displacement
        
        # Check for toe lift (toe up, heel stationary)
        elif (toe_displacement > self.config.min_displacement and
              toe_velocity > self.config.min_velocity and
              self._movement_analyzer.is_stationary("heel", self.config.stationary_threshold)):
            
            # False positive prevention: check for primarily horizontal movement
            if toe_displacement >= 0.03 and toe_displacement > toe_horizontal:
                tap_type = TapType.TOE
                peak_displacement = toe_displacement
        
        # If upward motion detected, transition to DETECTING_UPWARD
        if tap_type is not None:
            self._state = DetectionState.DETECTING_UPWARD
            self._gesture_start_time = landmarks.timestamp
            self._peak_displacement = peak_displacement
            self._detected_tap_type = tap_type
        
        return None
    
    def _process_upward_state(self, landmarks: FootLandmarks) -> Optional[TapEvent]:
        """
        Process frame in DETECTING_UPWARD state.
        
        Continues tracking upward motion and updates peak displacement.
        Validates anchor points remain stationary. Checks for timeout.
        Transitions to DETECTING_DOWNWARD when upward motion stops.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            None (no complete tap yet)
        """
        # Check for timeout - use >= to handle boundary cases
        elapsed_ms = (landmarks.timestamp - self._gesture_start_time) * 1000
        if elapsed_ms >= self.config.max_tap_duration_ms:
            # Gesture took too long, reset
            self._reset_gesture_state()
            self._state = DetectionState.IDLE
            return None
        
        # Update peak displacement
        if self._detected_tap_type == TapType.HEEL:
            current_displacement = self._movement_analyzer.get_vertical_displacement(
                "heel", int(elapsed_ms)
            )
            self._peak_displacement = max(self._peak_displacement, current_displacement)
            
            # Validate toe remains stationary
            if not self._movement_analyzer.is_stationary("toe", self.config.stationary_threshold):
                # Anchor point moved, cancel gesture
                self._reset_gesture_state()
                self._state = DetectionState.IDLE
                return None
        
        elif self._detected_tap_type == TapType.TOE:
            current_displacement = self._movement_analyzer.get_vertical_displacement(
                "toe", int(elapsed_ms)
            )
            self._peak_displacement = max(self._peak_displacement, current_displacement)
            
            # Validate heel remains stationary
            if not self._movement_analyzer.is_stationary("heel", self.config.stationary_threshold):
                # Anchor point moved, cancel gesture
                self._reset_gesture_state()
                self._state = DetectionState.IDLE
                return None
        
        elif self._detected_tap_type == TapType.FULL_FOOT:
            heel_displacement = self._movement_analyzer.get_vertical_displacement(
                "heel", int(elapsed_ms)
            )
            toe_displacement = self._movement_analyzer.get_vertical_displacement(
                "toe", int(elapsed_ms)
            )
            current_displacement = max(heel_displacement, toe_displacement)
            self._peak_displacement = max(self._peak_displacement, current_displacement)
        
        # Check if upward motion has stopped (velocity near zero or negative)
        if self._detected_tap_type == TapType.HEEL:
            velocity = self._movement_analyzer.get_vertical_velocity("heel")
            if velocity < self.config.min_velocity * 0.3:  # Threshold for stopping
                self._state = DetectionState.DETECTING_DOWNWARD
        
        elif self._detected_tap_type == TapType.TOE:
            velocity = self._movement_analyzer.get_vertical_velocity("toe")
            if velocity < self.config.min_velocity * 0.3:
                self._state = DetectionState.DETECTING_DOWNWARD
        
        elif self._detected_tap_type == TapType.FULL_FOOT:
            heel_velocity = self._movement_analyzer.get_vertical_velocity("heel")
            toe_velocity = self._movement_analyzer.get_vertical_velocity("toe")
            if (heel_velocity < self.config.min_velocity * 0.3 and
                toe_velocity < self.config.min_velocity * 0.3):
                self._state = DetectionState.DETECTING_DOWNWARD
        
        return None

    
    def _process_downward_state(self, landmarks: FootLandmarks) -> Optional[TapEvent]:
        """
        Process frame in DETECTING_DOWNWARD state.
        
        Waits for downward motion to complete the tap pattern. When detected,
        creates TapEvent, records it in history, and transitions to DEBOUNCING.
        
        Args:
            landmarks: Current frame landmarks
            
        Returns:
            TapEvent if downward motion detected, None otherwise
        """
        # Check for timeout - use >= to handle boundary cases
        elapsed_ms = (landmarks.timestamp - self._gesture_start_time) * 1000
        if elapsed_ms >= self.config.max_tap_duration_ms:
            # Gesture took too long, reset
            self._reset_gesture_state()
            self._state = DetectionState.IDLE
            return None
        
        # Check for downward motion (negative displacement = downward)
        downward_detected = False
        
        if self._detected_tap_type == TapType.HEEL:
            # Check if heel is moving back down
            heel_velocity = self._movement_analyzer.get_vertical_velocity("heel")
            if heel_velocity < -self.config.min_velocity * 0.3:  # Negative = downward
                downward_detected = True
        
        elif self._detected_tap_type == TapType.TOE:
            # Check if toe is moving back down
            toe_velocity = self._movement_analyzer.get_vertical_velocity("toe")
            if toe_velocity < -self.config.min_velocity * 0.3:
                downward_detected = True
        
        elif self._detected_tap_type == TapType.FULL_FOOT:
            # Check if both heel and toe are moving back down
            heel_velocity = self._movement_analyzer.get_vertical_velocity("heel")
            toe_velocity = self._movement_analyzer.get_vertical_velocity("toe")
            if (heel_velocity < -self.config.min_velocity * 0.3 or
                toe_velocity < -self.config.min_velocity * 0.3):
                downward_detected = True
        
        if downward_detected:
            # Complete tap detected! Create TapEvent
            tap_event = TapEvent(
                tap_type=self._detected_tap_type,
                timestamp=landmarks.timestamp,
                peak_displacement=self._peak_displacement,
                duration_ms=elapsed_ms
            )
            
            # Add to history
            self._tap_history.append(tap_event)
            
            # Transition to DEBOUNCING state
            self._state = DetectionState.DEBOUNCING
            self._debounce_end_time = landmarks.timestamp + 0.15  # 150ms debounce
            
            # Reset gesture state
            self._reset_gesture_state()
            
            return tap_event
        
        return None
    
    def _reset_gesture_state(self) -> None:
        """
        Reset gesture tracking state variables.
        
        Called when gesture is cancelled or completed.
        """
        self._gesture_start_time = None
        self._peak_displacement = 0.0
        self._detected_tap_type = None
    
    def get_state(self) -> DetectionState:
        """
        Get current detection state.
        
        Returns:
            Current DetectionState (IDLE, DETECTING_UPWARD, DETECTING_DOWNWARD, or DEBOUNCING)
        """
        return self._state
    
    def get_tap_history(self) -> list[TapEvent]:
        """
        Get history of recent taps.
        
        Returns:
            List of up to 10 most recent TapEvent objects
        """
        return list(self._tap_history)
