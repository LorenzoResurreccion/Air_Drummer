# Import needed Libraries/Objects
import cv2
import time
import logging
import argparse
from typing import Optional, Tuple
import numpy as np

# Import all components
from video_capture import VideoCaptureManager, CameraAccessError
from frame_processor import FramePreprocessor
from holistic_detector import HolisticDetector
from data_extractor import TrackingDataExtractor
from visual_renderer import VisualRenderer
from models import TrackingData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BodyTracker:
    """Main controller that orchestrates the full body tracking pipeline."""
    
    def __init__(self, 
                 camera_index: int = 0,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 flip_frame: bool = True,
                 frame_width: Optional[int] = None,
                 frame_height: Optional[int] = None,
                 show_fps: bool = False):
        """
        Initialize body tracking system.
        
        Args:
            camera_index: Camera device index (default 0)
            min_detection_confidence: Minimum confidence for initial detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for tracking (0.0-1.0)
            model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy)
            flip_frame: Whether to flip frame horizontally for mirror effect
            frame_width: Optional frame width for capture resolution
            frame_height: Optional frame height for capture resolution
            show_fps: Whether to display FPS counter on screen
        """
        self.camera_index = camera_index
        self.flip_frame = flip_frame
        self.frame_number = 0
        self.detection_failures = 0
        self.frame_failures = 0
        self.show_fps = show_fps
        
        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        try:
            # Initialize all components
            logger.info("Initializing Body Tracker components...")
            
            self.capture_manager = VideoCaptureManager(camera_index)
            
            # Set frame resolution if specified
            if frame_width is not None and frame_height is not None:
                self.capture_manager.set_resolution(frame_width, frame_height)
                logger.info(f"Set capture resolution to {frame_width}x{frame_height}")
            
            self.frame_processor = FramePreprocessor()
            self.detector = HolisticDetector(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity
            )
            self.data_extractor = TrackingDataExtractor()
            self.renderer = VisualRenderer()
            
            # Verify camera is opened
            if not self.capture_manager.is_opened():
                raise RuntimeError(f"Failed to open camera with index {camera_index}")
            
            logger.info("Body Tracker initialized successfully")
            
        except CameraAccessError as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Body Tracker: {e}")
            # Attempt cleanup of any partially initialized components
            self._cleanup_resources()
            raise
    
    def process_frame(self) -> Tuple[Optional[TrackingData], Optional[np.ndarray]]:
        """
        Process a single frame through the complete pipeline with error handling.
        
        Returns:
            Tuple of (tracking_data, annotated_frame)
            Returns (None, None) if frame capture fails
        """
        try:
            # Capture frame
            success, frame = self.capture_manager.read_frame()
            if not success or frame is None:
                self.frame_failures += 1
                if self.frame_failures % 30 == 0:  # Log every 30 failures
                    logger.warning(f"Frame capture failed {self.frame_failures} times")
                return None, None
            
            # Reset frame failure counter on success
            if self.frame_failures > 0:
                logger.info(f"Frame capture recovered after {self.frame_failures} failures")
                self.frame_failures = 0
            
            # Preprocess frame for MediaPipe
            rgb_frame = self.frame_processor.preprocess(frame, flip=self.flip_frame)
            
            # Detect landmarks with error handling
            results = self.detector.detect(rgb_frame)
            
            # Handle detection failure gracefully
            if results is None:
                self.detection_failures += 1
                # Continue processing with empty results for graceful degradation
                logger.debug(f"Detection failed, continuing with empty results")
            else:
                # Reset detection failure counter
                if self.detection_failures > 0:
                    logger.info(f"Detection recovered after {self.detection_failures} failures")
                    self.detection_failures = 0
            
            # Extract tracking data (handles None results gracefully)
            timestamp = time.time()
            tracking_data = self.data_extractor.extract(results, timestamp)
            
            # Render visual feedback (handles None results gracefully)
            annotated_frame = self.renderer.render(rgb_frame, results)
            
            # Convert back to BGR for display
            display_frame = self.frame_processor.postprocess(annotated_frame)
            
            # Add FPS counter if enabled
            if self.show_fps:
                display_frame = self._add_fps_counter(display_frame)
            
            self.frame_number += 1
            
            return tracking_data, display_frame
            
        except Exception as e:
            logger.error(f"Error processing frame {self.frame_number}: {e}")
            # Return None to indicate processing failure but don't crash
            return None, None
    
    def start(self):
        """
        Start the main tracking loop with error handling and recovery.
        
        Continuously processes frames and displays visual feedback.
        Press 'q' to quit.
        """
        logger.info("Body Tracker started. Press 'q' to quit.")
        
        try:
            while self.capture_manager.is_opened():
                tracking_data, display_frame = self.process_frame()
                
                # Update FPS counter
                self._update_fps()
                
                # Skip display if frame processing failed
                if display_frame is None:
                    # Don't break immediately - allow for recovery
                    time.sleep(0.01)  # Small delay to prevent tight loop
                    continue
                
                # Display the annotated frame
                try:
                    cv2.imshow('Full Body Tracking - AIR DRUMMER', display_frame)
                except Exception as e:
                    logger.error(f"Error displaying frame: {e}")
                    # Continue without display rather than crashing
                
                # Check for quit command
                try:
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        logger.info("Quit command received")
                        break
                except Exception as e:
                    logger.error(f"Error checking keyboard input: {e}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C)")
        
        except Exception as e:
            logger.error(f"Unexpected error in tracking loop: {e}")
        
        finally:
            self.stop()
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_frame_count += 1
        elapsed_time = time.time() - self.fps_start_time
        
        # Update FPS every second
        if elapsed_time >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed_time
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
    
    def _add_fps_counter(self, frame: np.ndarray) -> np.ndarray:
        """
        Add FPS counter overlay to frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with FPS counter overlay
        """
        fps_text = f"FPS: {self.current_fps:.1f}"
        
        # Add semi-transparent background for better readability
        text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (10, 10), (20 + text_size[0], 40 + text_size[1]), (0, 0, 0), -1)
        
        # Add FPS text
        cv2.putText(frame, fps_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        return frame
    
    def _cleanup_resources(self):
        """
        Internal method to cleanup resources with error handling.
        
        This is a best-effort cleanup that logs errors but doesn't raise exceptions.
        """
        # Release camera
        try:
            if hasattr(self, 'capture_manager') and self.capture_manager:
                self.capture_manager.release()
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")
        
        # Close detector
        try:
            if hasattr(self, 'detector') and self.detector:
                self.detector.close()
        except Exception as e:
            logger.error(f"Error closing detector: {e}")
        
        # Destroy OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error destroying windows: {e}")
    
    def stop(self):
        """Stop tracking and release all resources with comprehensive error handling."""
        logger.info("Stopping Body Tracker...")
        
        self._cleanup_resources()
        
        logger.info("Body Tracker stopped successfully")


def parse_arguments():
    """
    Parse command-line arguments for configuration.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='AIR DRUMMER - Full Body Tracking System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Camera configuration
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index')
    
    # Detection confidence thresholds
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum confidence for initial detection (0.0-1.0)')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5,
                       help='Minimum confidence for tracking (0.0-1.0)')
    
    # Model configuration
    parser.add_argument('--model-complexity', type=int, default=1, choices=[0, 1, 2],
                       help='MediaPipe model complexity (0=lite, 1=full, 2=heavy)')
    
    # Frame configuration
    parser.add_argument('--no-flip', action='store_true',
                       help='Disable horizontal flip (mirror effect)')
    parser.add_argument('--width', type=int, default=None,
                       help='Frame width for capture resolution')
    parser.add_argument('--height', type=int, default=None,
                       help='Frame height for capture resolution')
    
    # Performance monitoring
    parser.add_argument('--show-fps', action='store_true',
                       help='Display FPS counter on screen')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    """Main entry point for the body tracking application with comprehensive error handling."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Configure logging level
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    try:
        logger.info("Starting AIR DRUMMER Body Tracker...")
        logger.info(f"Configuration: camera={args.camera}, "
                   f"detection_conf={args.min_detection_confidence}, "
                   f"tracking_conf={args.min_tracking_confidence}, "
                   f"model_complexity={args.model_complexity}")
        
        # Initialize and start body tracker with parsed arguments
        tracker = BodyTracker(
            camera_index=args.camera,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            model_complexity=args.model_complexity,
            flip_frame=not args.no_flip,
            frame_width=args.width,
            frame_height=args.height,
            show_fps=args.show_fps
        )
        tracker.start()
    
    except CameraAccessError as e:
        logger.error(f"Camera access error: {e}")
        print("\nCamera Error:")
        print(str(e))
        print("\nPlease resolve the camera issue and try again.")
    
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"\nError: {e}")
        print("Please check that your camera is connected and accessible.")
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\nInterrupted by user")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUnexpected error occurred: {e}")
        print("Please check the logs for more details.")
    
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()



