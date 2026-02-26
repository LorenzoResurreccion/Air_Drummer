"""
Video Capture Manager for AIR DRUMMER body tracking system.

This module provides camera initialization, frame reading, and resource management
for the full body tracking feature.
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraAccessError(Exception):
    """Exception raised when camera cannot be accessed or initialized."""
    pass


class VideoCaptureManager:
    """
    Manages video capture from device camera with error handling and resource cleanup.
    
    Attributes:
        camera_index: Index of the camera device to use (default 0)
        cap: OpenCV VideoCapture object
    """
    
    def __init__(self, camera_index: int = 0, max_reconnect_attempts: int = 3):
        """
        Initialize camera capture.
        
        Args:
            camera_index: Index of the camera device (default 0 for primary camera)
            max_reconnect_attempts: Maximum number of reconnection attempts (default 3)
            
        Raises:
            CameraAccessError: If camera cannot be accessed or opened
        """
        self.camera_index = camera_index
        self.max_reconnect_attempts = max_reconnect_attempts
        self.consecutive_read_failures = 0
        self.read_failure_threshold = 30  # Trigger reconnect after 30 consecutive failures
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """
        Internal method to initialize camera capture.
        
        Raises:
            CameraAccessError: If camera cannot be accessed or opened
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise CameraAccessError(
                f"Failed to access camera at index {self.camera_index}. "
                f"Please check:\n"
                f"  - Camera is connected and not in use by another application\n"
                f"  - Camera permissions are granted\n"
                f"  - Camera index is correct (try 0, 1, or 2)"
            )
        
        # Give camera time to initialize (especially important on macOS)
        time.sleep(0.5)
        
        # Verify camera can actually read frames
        for attempt in range(3):
            success, _ = self.cap.read()
            if success:
                logger.info(f"Camera at index {self.camera_index} initialized successfully")
                break
            time.sleep(0.2)
        else:
            self.cap.release()
            raise CameraAccessError(
                f"Camera at index {self.camera_index} opened but cannot read frames. "
                f"This may indicate:\n"
                f"  - Camera is being used by another application\n"
                f"  - Camera permissions are not fully granted\n"
                f"  - Camera hardware issue"
            )
    
    def is_opened(self) -> bool:
        """
        Check if camera is successfully opened and ready to capture.
        
        Returns:
            True if camera is opened, False otherwise
        """
        return self.cap is not None and self.cap.isOpened()
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set the capture resolution for the camera.
        
        Args:
            width: Desired frame width
            height: Desired frame height
            
        Returns:
            True if resolution was set successfully, False otherwise
        """
        if not self.is_opened():
            logger.error("Cannot set resolution: camera not opened")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify the resolution was set (some cameras may not support requested resolution)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != width or actual_height != height:
            logger.warning(
                f"Requested resolution {width}x{height} not supported. "
                f"Using {actual_width}x{actual_height} instead."
            )
        
        return True
    
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Read a single frame from the camera with automatic reconnection on failure.
        
        Returns:
            Tuple containing:
                - success (bool): True if frame was read successfully, False otherwise
                - frame (np.ndarray): The captured frame, or None if read failed
        """
        if not self.is_opened():
            logger.warning("Camera not opened, attempting to reconnect...")
            if not self._attempt_reconnect():
                return False, None
        
        success, frame = self.cap.read()
        
        if not success:
            self.consecutive_read_failures += 1
            logger.warning(f"Frame read failed (failure #{self.consecutive_read_failures})")
            
            # Attempt reconnection if threshold exceeded
            if self.consecutive_read_failures >= self.read_failure_threshold:
                logger.warning(
                    f"Camera read failures exceeded threshold ({self.read_failure_threshold}). "
                    f"Attempting to reconnect..."
                )
                if self._attempt_reconnect():
                    # Try reading again after reconnection
                    success, frame = self.cap.read()
                    if success:
                        logger.info("Successfully read frame after reconnection")
                        self.consecutive_read_failures = 0
                else:
                    logger.error("Camera reconnection failed")
        else:
            # Reset failure counter on successful read
            if self.consecutive_read_failures > 0:
                logger.info(f"Camera recovered after {self.consecutive_read_failures} failures")
            self.consecutive_read_failures = 0
        
        return success, frame
    
    def _attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect to the camera.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info(f"Attempting camera reconnection (max {self.max_reconnect_attempts} attempts)...")
        
        for attempt in range(self.max_reconnect_attempts):
            try:
                # Release existing capture
                if self.cap is not None:
                    self.cap.release()
                    time.sleep(0.5)
                
                # Try to reinitialize
                self._initialize_camera()
                
                # Verify we can read a frame
                success, _ = self.cap.read()
                if success:
                    logger.info(f"Camera reconnected successfully on attempt {attempt + 1}")
                    self.consecutive_read_failures = 0
                    return True
                    
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                time.sleep(1.0)  # Wait before next attempt
        
        logger.error(f"Failed to reconnect camera after {self.max_reconnect_attempts} attempts")
        return False
    
    def release(self):
        """
        Release camera resources and cleanup.
        
        This method should be called when done with the camera to ensure
        proper resource cleanup. It's safe to call multiple times.
        """
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera resources released successfully")
            except Exception as e:
                # Log error but don't raise - cleanup should be best-effort
                logger.error(f"Error during camera release: {e}")
            finally:
                self.cap = None
    
    def __enter__(self):
        """Context manager entry - returns self for use in 'with' statements."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resources are released."""
        self.release()
        return False
    
    def __del__(self):
        """Destructor - ensures resources are released when object is garbage collected."""
        self.release()
