"""
Unit tests for VideoCaptureManager component.

Tests cover:
- Successful camera initialization
- Camera initialization failure with invalid camera index
- Proper resource release
- Frame reading functionality
- Context manager behavior

Requirements: 1.1, 1.2, 1.4
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_capture import VideoCaptureManager, CameraAccessError


class TestVideoCaptureManagerInitialization:
    """Test camera initialization scenarios."""
    
    def test_successful_camera_initialization(self):
        """
        Test that VideoCaptureManager successfully initializes with a valid camera.
        
        Validates: Requirement 1.1 - System shall initialize device camera
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            # Mock successful camera opening
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            # Initialize manager
            manager = VideoCaptureManager(camera_index=0)
            
            # Verify camera was initialized with correct index
            mock_video_capture.assert_called_once_with(0)
            
            # Verify manager reports camera as opened
            assert manager.is_opened() is True
            
            # Cleanup
            manager.release()
    
    def test_camera_initialization_with_custom_index(self):
        """
        Test that VideoCaptureManager can initialize with a custom camera index.
        
        Validates: Requirement 1.1 - System shall initialize device camera
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            # Initialize with custom camera index
            manager = VideoCaptureManager(camera_index=1)
            
            # Verify correct camera index was used
            mock_video_capture.assert_called_once_with(1)
            assert manager.camera_index == 1
            
            manager.release()
    
    def test_camera_initialization_failure_invalid_index(self):
        """
        Test that VideoCaptureManager raises CameraAccessError with invalid camera index.
        
        Validates: Requirement 1.2 - System shall return error when camera unavailable
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            # Mock failed camera opening
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_video_capture.return_value = mock_cap
            
            # Attempt to initialize with invalid index should raise error
            with pytest.raises(CameraAccessError) as exc_info:
                VideoCaptureManager(camera_index=99)
            
            # Verify error message contains helpful information
            error_message = str(exc_info.value)
            assert "Failed to access camera at index 99" in error_message
            assert "Camera is connected" in error_message
            assert "Camera permissions" in error_message
    
    def test_camera_initialization_failure_access_denied(self):
        """
        Test that VideoCaptureManager handles camera access denied scenario.
        
        Validates: Requirement 1.2 - System shall return error when access denied
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            # Mock camera access denied (camera exists but can't be opened)
            mock_cap = Mock()
            mock_cap.isOpened.return_value = False
            mock_video_capture.return_value = mock_cap
            
            # Should raise CameraAccessError
            with pytest.raises(CameraAccessError):
                VideoCaptureManager(camera_index=0)


class TestVideoCaptureManagerFrameReading:
    """Test frame reading functionality."""
    
    def test_read_frame_success(self):
        """
        Test successful frame reading from camera.
        
        Validates: Requirement 1.1 - System shall capture video frames
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            
            # Mock successful frame read
            fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cap.read.return_value = (True, fake_frame)
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            # Read frame
            success, frame = manager.read_frame()
            
            # Verify frame was read successfully
            assert success is True
            assert frame is not None
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)
            
            manager.release()
    
    def test_read_frame_failure(self):
        """
        Test frame reading failure handling.
        
        Validates: Requirement 1.1 - System shall handle frame read failures
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            
            # Mock failed frame read
            mock_cap.read.return_value = (False, None)
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            # Read frame
            success, frame = manager.read_frame()
            
            # Verify failure is reported correctly
            assert success is False
            assert frame is None
            
            manager.release()
    
    def test_read_frame_when_camera_not_opened(self):
        """
        Test that read_frame handles closed camera gracefully.
        
        Validates: Requirement 1.4 - System shall handle resource states properly
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            # Simulate camera being closed
            mock_cap.isOpened.return_value = False
            
            # Attempt to read frame
            success, frame = manager.read_frame()
            
            # Should return failure without crashing
            assert success is False
            assert frame is None
            
            manager.release()


class TestVideoCaptureManagerResourceRelease:
    """Test proper resource cleanup and release."""
    
    def test_proper_resource_release(self):
        """
        Test that release() properly cleans up camera resources.
        
        Validates: Requirement 1.4 - System shall release camera resources properly
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            # Release resources
            manager.release()
            
            # Verify cap.release() was called
            mock_cap.release.assert_called_once()
            
            # Verify manager reports camera as not opened
            assert manager.is_opened() is False
    
    def test_multiple_release_calls_safe(self):
        """
        Test that calling release() multiple times is safe.
        
        Validates: Requirement 1.4 - System shall handle cleanup safely
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            # Call release multiple times
            manager.release()
            manager.release()
            manager.release()
            
            # Should not raise any exceptions
            # First call releases, subsequent calls are no-ops
            assert manager.cap is None
    
    def test_release_with_error_handling(self):
        """
        Test that release() handles errors during cleanup gracefully.
        
        Validates: Requirement 1.4 - System shall handle cleanup errors
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            
            # Mock release to raise an exception
            mock_cap.release.side_effect = Exception("Release error")
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            # Release should not raise exception even if cap.release() fails
            manager.release()
            
            # Verify cap is set to None despite error
            assert manager.cap is None
    
    def test_context_manager_releases_resources(self):
        """
        Test that using VideoCaptureManager as context manager releases resources.
        
        Validates: Requirement 1.4 - System shall release resources on exit
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            # Use as context manager
            with VideoCaptureManager(camera_index=0) as manager:
                assert manager.is_opened() is True
            
            # Verify release was called on exit
            mock_cap.release.assert_called_once()
    
    def test_destructor_releases_resources(self):
        """
        Test that destructor releases resources when object is garbage collected.
        
        Validates: Requirement 1.4 - System shall cleanup on object destruction
        """
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            # Explicitly delete the object
            del manager
            
            # Verify release was called
            mock_cap.release.assert_called()


class TestVideoCaptureManagerIsOpened:
    """Test is_opened() method behavior."""
    
    def test_is_opened_returns_true_when_camera_active(self):
        """Test is_opened() returns True when camera is active."""
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            
            assert manager.is_opened() is True
            
            manager.release()
    
    def test_is_opened_returns_false_after_release(self):
        """Test is_opened() returns False after release."""
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            manager.release()
            
            assert manager.is_opened() is False
    
    def test_is_opened_handles_none_cap(self):
        """Test is_opened() handles None cap gracefully."""
        with patch('cv2.VideoCapture') as mock_video_capture:
            mock_cap = Mock()
            mock_cap.isOpened.return_value = True
            mock_video_capture.return_value = mock_cap
            
            manager = VideoCaptureManager(camera_index=0)
            manager.cap = None
            
            assert manager.is_opened() is False
