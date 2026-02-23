"""
Frame preprocessing module for AIR DRUMMER body tracking.

This module handles frame format conversions required for MediaPipe processing
and OpenCV display, including color space conversions and optional mirroring.
"""

import cv2
import numpy as np


class FramePreprocessor:
    """
    Preprocesses video frames for MediaPipe processing and OpenCV display.
    
    Handles BGR to RGB conversion (MediaPipe requirement), optional horizontal
    flipping for mirror effect, and RGB to BGR conversion for display.
    """
    
    def preprocess(self, frame: np.ndarray, flip: bool = True) -> np.ndarray:
        """
        Convert frame from BGR to RGB and optionally flip horizontally.
        
        MediaPipe requires RGB format, while OpenCV captures in BGR format.
        The optional flip creates a mirror effect for more intuitive user experience.
        
        Args:
            frame: Input frame in BGR format (OpenCV default)
            flip: If True, flip frame horizontally for mirror effect
            
        Returns:
            Frame in RGB format, optionally flipped
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Optionally flip horizontally for mirror effect
        if flip:
            rgb_frame = cv2.flip(rgb_frame, 1)
            
        return rgb_frame
    
    def postprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame back from RGB to BGR for OpenCV display.
        
        After MediaPipe processing and annotation, frames need to be converted
        back to BGR format for proper display with cv2.imshow().
        
        Args:
            frame: Input frame in RGB format
            
        Returns:
            Frame in BGR format for OpenCV display
        """
        # Convert RGB back to BGR for OpenCV display
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        return bgr_frame
