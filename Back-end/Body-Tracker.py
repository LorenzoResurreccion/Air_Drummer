# Import needed Libraries/Objects
import cv2
import time
from typing import Optional, Tuple
import numpy as np

# Import all components
from video_capture import VideoCaptureManager
from frame_processor import FramePreprocessor
from holistic_detector import HolisticDetector
from data_extractor import TrackingDataExtractor
from visual_renderer import VisualRenderer
from models import TrackingData


class BodyTracker:
    """Main controller that orchestrates the full body tracking pipeline."""
    
    def __init__(self, 
                 camera_index: int = 0,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 flip_frame: bool = True):
        """
        Initialize body tracking system.
        
        Args:
            camera_index: Camera device index (default 0)
            min_detection_confidence: Minimum confidence for initial detection
            min_tracking_confidence: Minimum confidence for tracking
            flip_frame: Whether to flip frame horizontally for mirror effect
        """
        self.camera_index = camera_index
        self.flip_frame = flip_frame
        self.frame_number = 0
        
        # Initialize all components
        self.capture_manager = VideoCaptureManager(camera_index)
        self.frame_processor = FramePreprocessor()
        self.detector = HolisticDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.data_extractor = TrackingDataExtractor()
        self.renderer = VisualRenderer()
        
        # Verify camera is opened
        if not self.capture_manager.is_opened():
            raise RuntimeError(f"Failed to open camera with index {camera_index}")
    
    def process_frame(self) -> Tuple[Optional[TrackingData], Optional[np.ndarray]]:
        """
        Process a single frame through the complete pipeline.
        
        Returns:
            Tuple of (tracking_data, annotated_frame)
            Returns (None, None) if frame capture fails
        """
        # Capture frame
        success, frame = self.capture_manager.read_frame()
        if not success:
            return None, None
        
        # Preprocess frame for MediaPipe
        rgb_frame = self.frame_processor.preprocess(frame, flip=self.flip_frame)
        
        # Detect landmarks
        results = self.detector.detect(rgb_frame)
        
        # Extract tracking data
        timestamp = time.time()
        tracking_data = self.data_extractor.extract(results, timestamp)
        
        # Render visual feedback
        annotated_frame = self.renderer.render(rgb_frame, results)
        
        # Convert back to BGR for display
        display_frame = self.frame_processor.postprocess(annotated_frame)
        
        self.frame_number += 1
        
        return tracking_data, display_frame
    
    def start(self):
        """
        Start the main tracking loop.
        
        Continuously processes frames and displays visual feedback.
        Press 'q' to quit.
        """
        print("Body Tracker started. Press 'q' to quit.")
        
        try:
            while self.capture_manager.is_opened():
                tracking_data, display_frame = self.process_frame()
                
                """
                if display_frame is None:
                    print("Failed to capture frame")
                    break
                """
                
                # Display the annotated frame
                cv2.imshow('Full Body Tracking - AIR DRUMMER', display_frame)
                
                # Check for quit command
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("Quit command received")
                    break
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop tracking and release all resources."""
        print("Stopping Body Tracker...")
        
        # Release all resources
        self.capture_manager.release()
        self.detector.close()
        cv2.destroyAllWindows()
        
        print("Body Tracker stopped successfully")


def main():
    """Main entry point for the body tracking application."""
    try:
        # Initialize and start body tracker
        tracker = BodyTracker(
            camera_index=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            flip_frame=True
        )
        tracker.start()
    
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please check that your camera is connected and accessible.")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()



