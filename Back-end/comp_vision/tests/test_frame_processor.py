"""
Unit tests for FramePreprocessor component.

Tests cover:
- BGR to RGB conversion correctness
- Horizontal flip functionality
- RGB to BGR conversion for display
- Frame dimension preservation

Requirements: 1.3
"""

import pytest
import numpy as np

from comp_vision.frame_processor import FramePreprocessor


class TestFramePreprocessorBGRtoRGB:
    """Test BGR to RGB conversion functionality."""
    
    def test_bgr_to_rgb_conversion_correctness(self):
        """
        Test that preprocess() correctly converts BGR to RGB format.
        
        Validates: Requirement 1.3 - Frame preprocessing for MediaPipe
        """
        preprocessor = FramePreprocessor()
        
        # Create a test frame with distinct BGR values
        # Blue channel = 100, Green channel = 150, Red channel = 200
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 100  # Blue channel
        bgr_frame[:, :, 1] = 150  # Green channel
        bgr_frame[:, :, 2] = 200  # Red channel
        
        # Process without flip to test only color conversion
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=False)
        
        # Verify RGB conversion: R should be 200, G should be 150, B should be 100
        assert rgb_frame[0, 0, 0] == 200  # Red channel (was BGR's red)
        assert rgb_frame[0, 0, 1] == 150  # Green channel (unchanged)
        assert rgb_frame[0, 0, 2] == 100  # Blue channel (was BGR's blue)
    
    def test_bgr_to_rgb_preserves_dimensions(self):
        """
        Test that BGR to RGB conversion preserves frame dimensions.
        
        Validates: Requirement 1.3 - Frame preprocessing maintains frame structure
        """
        preprocessor = FramePreprocessor()
        
        # Test with various frame sizes
        test_sizes = [(480, 640, 3), (720, 1280, 3), (100, 100, 3)]
        
        for size in test_sizes:
            bgr_frame = np.random.randint(0, 256, size, dtype=np.uint8)
            rgb_frame = preprocessor.preprocess(bgr_frame, flip=False)
            
            # Verify dimensions are preserved
            assert rgb_frame.shape == bgr_frame.shape
    
    def test_bgr_to_rgb_preserves_dtype(self):
        """
        Test that BGR to RGB conversion preserves data type.
        
        Validates: Requirement 1.3 - Frame preprocessing maintains data type
        """
        preprocessor = FramePreprocessor()
        
        bgr_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=False)
        
        # Verify data type is preserved
        assert rgb_frame.dtype == np.uint8


class TestFramePreprocessorHorizontalFlip:
    """Test horizontal flip functionality."""
    
    def test_horizontal_flip_enabled_by_default(self):
        """
        Test that horizontal flip is applied by default for mirror effect.
        
        Validates: Requirement 1.3 - Optional horizontal flip for mirror effect
        """
        preprocessor = FramePreprocessor()
        
        # Create a frame with distinct left and right sides
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_frame[:, :50, :] = 100  # Left half is darker
        bgr_frame[:, 50:, :] = 200  # Right half is brighter
        
        # Process with default flip=True
        rgb_frame = preprocessor.preprocess(bgr_frame)
        
        # After flip, left and right should be swapped
        # Original left (100) should now be on right
        # Original right (200) should now be on left
        assert rgb_frame[50, 10, 0] == 200  # Left side should be brighter (was right)
        assert rgb_frame[50, 90, 0] == 100  # Right side should be darker (was left)
    
    def test_horizontal_flip_can_be_disabled(self):
        """
        Test that horizontal flip can be disabled when flip=False.
        
        Validates: Requirement 1.3 - Optional horizontal flip control
        """
        preprocessor = FramePreprocessor()
        
        # Create a frame with distinct left and right sides
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_frame[:, :50, :] = 100  # Left half
        bgr_frame[:, 50:, :] = 200  # Right half
        
        # Process with flip=False
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=False)
        
        # Without flip, left and right should remain the same
        # (accounting for BGR to RGB conversion)
        assert rgb_frame[50, 10, 0] == 100  # Left side should remain darker
        assert rgb_frame[50, 90, 0] == 200  # Right side should remain brighter
    
    def test_horizontal_flip_preserves_dimensions(self):
        """
        Test that horizontal flip preserves frame dimensions.
        
        Validates: Requirement 1.3 - Frame preprocessing maintains dimensions
        """
        preprocessor = FramePreprocessor()
        
        bgr_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=True)
        
        # Verify dimensions are preserved after flip
        assert rgb_frame.shape == bgr_frame.shape
    
    def test_horizontal_flip_correctness(self):
        """
        Test that horizontal flip correctly mirrors the frame.
        
        Validates: Requirement 1.3 - Horizontal flip implementation correctness
        """
        preprocessor = FramePreprocessor()
        
        # Create a frame with a vertical gradient
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for col in range(100):
            bgr_frame[:, col, :] = col * 2  # Gradient from 0 to 198
        
        # Process with flip
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=True)
        
        # Verify flip: column 0 should have value from original column 99
        # Original column 99 had value 198, after BGR->RGB conversion
        assert rgb_frame[50, 0, 0] == 198  # Leftmost should be brightest
        assert rgb_frame[50, 99, 0] == 0   # Rightmost should be darkest


class TestFramePreprocessorRGBtoBGR:
    """Test RGB to BGR conversion for display."""
    
    def test_rgb_to_bgr_conversion_correctness(self):
        """
        Test that postprocess() correctly converts RGB back to BGR format.
        
        Validates: Requirement 1.3 - RGB to BGR conversion for OpenCV display
        """
        preprocessor = FramePreprocessor()
        
        # Create a test frame with distinct RGB values
        # Red channel = 200, Green channel = 150, Blue channel = 100
        rgb_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_frame[:, :, 0] = 200  # Red channel
        rgb_frame[:, :, 1] = 150  # Green channel
        rgb_frame[:, :, 2] = 100  # Blue channel
        
        # Convert back to BGR
        bgr_frame = preprocessor.postprocess(rgb_frame)
        
        # Verify BGR conversion: B should be 100, G should be 150, R should be 200
        assert bgr_frame[0, 0, 0] == 100  # Blue channel (was RGB's blue)
        assert bgr_frame[0, 0, 1] == 150  # Green channel (unchanged)
        assert bgr_frame[0, 0, 2] == 200  # Red channel (was RGB's red)
    
    def test_rgb_to_bgr_preserves_dimensions(self):
        """
        Test that RGB to BGR conversion preserves frame dimensions.
        
        Validates: Requirement 1.3 - Postprocessing maintains frame structure
        """
        preprocessor = FramePreprocessor()
        
        # Test with various frame sizes
        test_sizes = [(480, 640, 3), (720, 1280, 3), (100, 100, 3)]
        
        for size in test_sizes:
            rgb_frame = np.random.randint(0, 256, size, dtype=np.uint8)
            bgr_frame = preprocessor.postprocess(rgb_frame)
            
            # Verify dimensions are preserved
            assert bgr_frame.shape == rgb_frame.shape
    
    def test_rgb_to_bgr_preserves_dtype(self):
        """
        Test that RGB to BGR conversion preserves data type.
        
        Validates: Requirement 1.3 - Postprocessing maintains data type
        """
        preprocessor = FramePreprocessor()
        
        rgb_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        bgr_frame = preprocessor.postprocess(rgb_frame)
        
        # Verify data type is preserved
        assert bgr_frame.dtype == np.uint8
    
    def test_roundtrip_conversion_without_flip(self):
        """
        Test that BGR->RGB->BGR roundtrip preserves original frame (without flip).
        
        Validates: Requirement 1.3 - Color conversion correctness
        """
        preprocessor = FramePreprocessor()
        
        # Create original BGR frame
        original_bgr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Convert to RGB and back to BGR (without flip)
        rgb_frame = preprocessor.preprocess(original_bgr, flip=False)
        final_bgr = preprocessor.postprocess(rgb_frame)
        
        # Verify roundtrip preserves original values
        np.testing.assert_array_equal(final_bgr, original_bgr)


class TestFramePreprocessorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_preprocess_with_single_pixel_frame(self):
        """
        Test preprocessing with minimal 1x1 frame.
        
        Validates: Requirement 1.3 - Handle edge case frame sizes
        """
        preprocessor = FramePreprocessor()
        
        bgr_frame = np.array([[[100, 150, 200]]], dtype=np.uint8)
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=False)
        
        # Verify conversion works for minimal frame
        assert rgb_frame.shape == (1, 1, 3)
        assert rgb_frame[0, 0, 0] == 200  # Red
        assert rgb_frame[0, 0, 1] == 150  # Green
        assert rgb_frame[0, 0, 2] == 100  # Blue
    
    def test_postprocess_with_single_pixel_frame(self):
        """
        Test postprocessing with minimal 1x1 frame.
        
        Validates: Requirement 1.3 - Handle edge case frame sizes
        """
        preprocessor = FramePreprocessor()
        
        rgb_frame = np.array([[[200, 150, 100]]], dtype=np.uint8)
        bgr_frame = preprocessor.postprocess(rgb_frame)
        
        # Verify conversion works for minimal frame
        assert bgr_frame.shape == (1, 1, 3)
        assert bgr_frame[0, 0, 0] == 100  # Blue
        assert bgr_frame[0, 0, 1] == 150  # Green
        assert bgr_frame[0, 0, 2] == 200  # Red
    
    def test_preprocess_with_all_black_frame(self):
        """
        Test preprocessing with all-black frame.
        
        Validates: Requirement 1.3 - Handle uniform color frames
        """
        preprocessor = FramePreprocessor()
        
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=True)
        
        # Verify all-black frame remains all-black
        assert np.all(rgb_frame == 0)
    
    def test_preprocess_with_all_white_frame(self):
        """
        Test preprocessing with all-white frame.
        
        Validates: Requirement 1.3 - Handle uniform color frames
        """
        preprocessor = FramePreprocessor()
        
        bgr_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        rgb_frame = preprocessor.preprocess(bgr_frame, flip=True)
        
        # Verify all-white frame remains all-white
        assert np.all(rgb_frame == 255)
