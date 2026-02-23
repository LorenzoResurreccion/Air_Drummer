"""
Video Capture Manager for AIR DRUMMER body tracking system.

This module provides camera initialization, frame reading, and resource management
for the full body tracking feature.
"""

import cv2
import numpy as np
from typing import Tuple


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
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize camera capture.
        
        Args:
            camera_index: Index of the camera device (default 0 for primary camera)
            
        Raises:
            CameraAccessError: If camera cannot be accessed or opened
        """
        import time
        
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise CameraAccessError(
                f"Failed to access camera at index {camera_index}. "
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
                break
            time.sleep(0.2)
        else:
            self.cap.release()
            raise CameraAccessError(
                f"Camera at index {camera_index} opened but cannot read frames. "
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
    
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Read a single frame from the camera.
        
        Returns:
            Tuple containing:
                - success (bool): True if frame was read successfully, False otherwise
                - frame (np.ndarray): The captured frame, or None if read failed
        """
        if not self.is_opened():
            return False, None
        
        success, frame = self.cap.read()
        return success, frame
    
    def release(self):
        """
        Release camera resources and cleanup.
        
        This method should be called when done with the camera to ensure
        proper resource cleanup. It's safe to call multiple times.
        """
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                # Log error but don't raise - cleanup should be best-effort
                print(f"Warning: Error during camera release: {e}")
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
