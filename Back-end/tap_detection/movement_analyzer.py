"""
Movement analyzer for tap detection.

This module provides the MovementAnalyzer class which tracks foot landmark
movements over time and calculates displacement, velocity, and stationary
status for tap gesture detection.
"""

from collections import deque
from typing import Optional

from tap_detection.tap_models import FootLandmarks


class MovementAnalyzer:
    """
    Analyzes foot landmark movements over time for tap detection.
    
    Maintains a circular buffer of recent FootLandmarks and provides methods
    to calculate vertical displacement, velocity, and stationary status.
    
    Coordinate system: MediaPipe uses normalized coordinates where y increases
    downward (0.0 at top, 1.0 at bottom). Therefore:
    - Upward movement = decreasing y value
    - Downward movement = increasing y value
    - Displacement for upward motion = y_start - y_end (positive value)
    
    Attributes:
        history_size: Maximum number of frames to keep in history
        _history: Circular buffer of FootLandmarks
    """
    
    def __init__(self, history_size: int = 10):
        """
        Initialize MovementAnalyzer with configurable history buffer.
        
        Args:
            history_size: Maximum number of frames to keep in history (default 10)
        """
        self.history_size = history_size
        self._history: deque[FootLandmarks] = deque(maxlen=history_size)
    
    def update(self, landmarks: FootLandmarks) -> None:
        """
        Add new landmark data to movement history.
        
        Args:
            landmarks: FootLandmarks for current frame
        """
        self._history.append(landmarks)
    
    def get_vertical_displacement(
        self, 
        landmark_type: str, 
        time_window_ms: int = 200
    ) -> float:
        """
        Calculate vertical displacement over specified time window.
        
        Displacement is calculated as the difference in y-coordinate between
        the oldest and newest landmarks within the time window. Positive values
        indicate upward movement (decreasing y), negative values indicate
        downward movement (increasing y).
        
        Args:
            landmark_type: Type of landmark to analyze ("heel", "ankle", or "toe")
            time_window_ms: Time window in milliseconds (default 200)
            
        Returns:
            Vertical displacement in normalized units (positive = upward)
            Returns 0.0 if insufficient history
        """
        if len(self._history) < 2:
            return 0.0
        
        # Convert time window to seconds
        time_window_sec = time_window_ms / 1000.0
        
        # Get the most recent landmark
        newest = self._history[-1]
        newest_time = newest.timestamp
        
        # Find the oldest landmark within the time window
        oldest = None
        for frame in self._history:
            if newest_time - frame.timestamp <= time_window_sec:
                oldest = frame
                break
        
        if oldest is None:
            # All frames are outside the time window, use the oldest available
            oldest = self._history[0]
        
        # Get y-coordinates for the specified landmark type
        oldest_y = self._get_landmark_y(oldest, landmark_type)
        newest_y = self._get_landmark_y(newest, landmark_type)
        
        # Calculate displacement (positive for upward movement)
        # Since y increases downward, upward movement means y decreases
        displacement = oldest_y - newest_y
        
        return displacement
    
    def get_vertical_velocity(self, landmark_type: str) -> float:
        """
        Calculate current vertical velocity.
        
        Velocity is calculated as displacement divided by time delta between
        the two most recent frames. Positive values indicate upward movement.
        
        Args:
            landmark_type: Type of landmark to analyze ("heel", "ankle", or "toe")
            
        Returns:
            Vertical velocity in normalized units per second (positive = upward)
            Returns 0.0 if insufficient history
        """
        if len(self._history) < 2:
            return 0.0
        
        # Get the two most recent frames
        newest = self._history[-1]
        previous = self._history[-2]
        
        # Calculate time delta
        time_delta = newest.timestamp - previous.timestamp
        
        if time_delta <= 0:
            return 0.0
        
        # Get y-coordinates
        previous_y = self._get_landmark_y(previous, landmark_type)
        newest_y = self._get_landmark_y(newest, landmark_type)
        
        # Calculate displacement (positive for upward)
        displacement = previous_y - newest_y
        
        # Calculate velocity
        velocity = displacement / time_delta
        
        return velocity
    
    def get_horizontal_displacement(
        self, 
        landmark_type: str, 
        time_window_ms: int = 200
    ) -> float:
        """
        Calculate horizontal displacement over specified time window.
        
        Displacement is calculated as the absolute difference in x-coordinate
        between the oldest and newest landmarks within the time window.
        
        Args:
            landmark_type: Type of landmark to analyze ("heel", "ankle", or "toe")
            time_window_ms: Time window in milliseconds (default 200)
            
        Returns:
            Horizontal displacement in normalized units (absolute value)
            Returns 0.0 if insufficient history
        """
        if len(self._history) < 2:
            return 0.0
        
        # Convert time window to seconds
        time_window_sec = time_window_ms / 1000.0
        
        # Get the most recent landmark
        newest = self._history[-1]
        newest_time = newest.timestamp
        
        # Find the oldest landmark within the time window
        oldest = None
        for frame in self._history:
            if newest_time - frame.timestamp <= time_window_sec:
                oldest = frame
                break
        
        if oldest is None:
            # All frames are outside the time window, use the oldest available
            oldest = self._history[0]
        
        # Get x-coordinates for the specified landmark type
        oldest_x = self._get_landmark_x(oldest, landmark_type)
        newest_x = self._get_landmark_x(newest, landmark_type)
        
        # Calculate absolute horizontal displacement
        displacement = abs(newest_x - oldest_x)
        
        return displacement
    
    def is_stationary(
        self, 
        landmark_type: str, 
        threshold: float = 0.02
    ) -> bool:
        """
        Check if landmark is relatively stationary (anchor point validation).
        
        A landmark is considered stationary if its total vertical movement
        over the entire history is less than the threshold.
        
        Args:
            landmark_type: Type of landmark to analyze ("heel", "ankle", or "toe")
            threshold: Maximum movement to be considered stationary (default 0.02)
            
        Returns:
            True if landmark movement is below threshold, False otherwise
            Returns True if insufficient history (assume stationary)
        """
        if len(self._history) < 2:
            return True
        
        # Find min and max y-coordinates across all history
        y_values = [
            self._get_landmark_y(frame, landmark_type) 
            for frame in self._history
        ]
        
        min_y = min(y_values)
        max_y = max(y_values)
        
        # Calculate total range of movement
        movement_range = max_y - min_y
        
        return movement_range < threshold
    
    def reset(self) -> None:
        """
        Clear movement history.
        
        Useful when landmarks become unavailable or when resetting detection state.
        """
        self._history.clear()
    
    def _get_landmark_y(self, landmarks: FootLandmarks, landmark_type: str) -> float:
        """
        Get y-coordinate for specified landmark type.
        
        Args:
            landmarks: FootLandmarks containing heel, ankle, and toe
            landmark_type: Type of landmark ("heel", "ankle", or "toe")
            
        Returns:
            Y-coordinate of the specified landmark
            
        Raises:
            ValueError: If landmark_type is not recognized
        """
        if landmark_type == "heel":
            return landmarks.heel.y
        elif landmark_type == "ankle":
            return landmarks.ankle.y
        elif landmark_type == "toe":
            return landmarks.toe.y
        else:
            raise ValueError(
                f"Unknown landmark type: {landmark_type}. "
                f"Must be 'heel', 'ankle', or 'toe'"
            )
    
    def _get_landmark_x(self, landmarks: FootLandmarks, landmark_type: str) -> float:
        """
        Get x-coordinate for specified landmark type.
        
        Args:
            landmarks: FootLandmarks containing heel, ankle, and toe
            landmark_type: Type of landmark ("heel", "ankle", or "toe")
            
        Returns:
            X-coordinate of the specified landmark
            
        Raises:
            ValueError: If landmark_type is not recognized
        """
        if landmark_type == "heel":
            return landmarks.heel.x
        elif landmark_type == "ankle":
            return landmarks.ankle.x
        elif landmark_type == "toe":
            return landmarks.toe.x
        else:
            raise ValueError(
                f"Unknown landmark type: {landmark_type}. "
                f"Must be 'heel', 'ankle', or 'toe'"
            )
