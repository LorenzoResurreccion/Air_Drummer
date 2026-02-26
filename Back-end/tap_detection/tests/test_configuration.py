"""
Property-based tests for TapDetectionConfig validation.

This module tests the configuration validation logic using property-based testing
to ensure that all parameter ranges are correctly enforced.

Feature: right-foot-bass-drum
"""

import pytest
from hypothesis import given, strategies as st

from tap_detection.tap_models import TapDetectionConfig


class TestConfigurationValidation:
    """Property-based tests for configuration validation."""
    
    @given(
        min_displacement=st.floats(min_value=0.01, max_value=0.15),
        min_velocity=st.floats(min_value=0.1, max_value=1.0),
        max_tap_duration_ms=st.integers(min_value=50, max_value=1000),
        confidence_threshold=st.floats(min_value=0.3, max_value=0.9),
        stationary_threshold=st.floats(min_value=0.01, max_value=0.05)
    )
    def test_property_valid_configuration_passes_validation(
        self,
        min_displacement: float,
        min_velocity: float,
        max_tap_duration_ms: int,
        confidence_threshold: float,
        stationary_threshold: float
    ):
        """
        Property 16: Configuration validation
        
        For any provided TapDetectionConfig with parameters within valid ranges,
        the validation should pass and return True.
        
        Valid ranges:
        - min_displacement: 0.01-0.15
        - min_velocity: 0.1-1.0
        - max_tap_duration_ms: 50-1000
        - confidence_threshold: 0.3-0.9
        - stationary_threshold: 0.01-0.05
        
        Validates: Requirements 10.5
        """
        config = TapDetectionConfig(
            min_displacement=min_displacement,
            min_velocity=min_velocity,
            max_tap_duration_ms=max_tap_duration_ms,
            confidence_threshold=confidence_threshold,
            stationary_threshold=stationary_threshold
        )
        
        # Should not raise any exception and return True
        assert config.validate() is True
    
    @given(
        min_displacement=st.one_of(
            st.floats(max_value=0.009, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.151, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_property_invalid_min_displacement_raises_error(self, min_displacement: float):
        """
        Property 16: Configuration validation
        
        For any TapDetectionConfig where min_displacement is outside the valid range
        (0.01-0.15), the validation should fail and raise ValueError with a
        descriptive message.
        
        Validates: Requirements 10.5
        """
        config = TapDetectionConfig(min_displacement=min_displacement)
        
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        
        # Verify error message mentions min_displacement
        assert "min_displacement" in str(exc_info.value)
        assert str(min_displacement) in str(exc_info.value)
    
    @given(
        min_velocity=st.one_of(
            st.floats(max_value=0.09, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_property_invalid_min_velocity_raises_error(self, min_velocity: float):
        """
        Property 16: Configuration validation
        
        For any TapDetectionConfig where min_velocity is outside the valid range
        (0.1-1.0), the validation should fail and raise ValueError with a
        descriptive message.
        
        Validates: Requirements 10.5
        """
        config = TapDetectionConfig(min_velocity=min_velocity)
        
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        
        # Verify error message mentions min_velocity
        assert "min_velocity" in str(exc_info.value)
        assert str(min_velocity) in str(exc_info.value)
    
    @given(
        max_tap_duration_ms=st.one_of(
            st.integers(max_value=49),
            st.integers(min_value=1001, max_value=10000)
        )
    )
    def test_property_invalid_max_tap_duration_raises_error(self, max_tap_duration_ms: int):
        """
        Property 16: Configuration validation
        
        For any TapDetectionConfig where max_tap_duration_ms is outside the valid range
        (50-1000), the validation should fail and raise ValueError with a
        descriptive message.
        
        Validates: Requirements 10.5
        """
        config = TapDetectionConfig(max_tap_duration_ms=max_tap_duration_ms)
        
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        
        # Verify error message mentions max_tap_duration_ms
        assert "max_tap_duration_ms" in str(exc_info.value)
        assert str(max_tap_duration_ms) in str(exc_info.value)
    
    @given(
        confidence_threshold=st.one_of(
            st.floats(max_value=0.29, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.91, max_value=1.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_property_invalid_confidence_threshold_raises_error(self, confidence_threshold: float):
        """
        Property 16: Configuration validation
        
        For any TapDetectionConfig where confidence_threshold is outside the valid range
        (0.3-0.9), the validation should fail and raise ValueError with a
        descriptive message.
        
        Validates: Requirements 10.5
        """
        config = TapDetectionConfig(confidence_threshold=confidence_threshold)
        
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        
        # Verify error message mentions confidence_threshold
        assert "confidence_threshold" in str(exc_info.value)
        assert str(confidence_threshold) in str(exc_info.value)
    
    @given(
        stationary_threshold=st.one_of(
            st.floats(max_value=0.009, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.051, max_value=1.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_property_invalid_stationary_threshold_raises_error(self, stationary_threshold: float):
        """
        Property 16: Configuration validation
        
        For any TapDetectionConfig where stationary_threshold is outside the valid range
        (0.01-0.05), the validation should fail and raise ValueError with a
        descriptive message.
        
        Validates: Requirements 10.5
        """
        config = TapDetectionConfig(stationary_threshold=stationary_threshold)
        
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        
        # Verify error message mentions stationary_threshold
        assert "stationary_threshold" in str(exc_info.value)
        assert str(stationary_threshold) in str(exc_info.value)
    
    @given(
        min_displacement=st.one_of(
            st.floats(max_value=0.009, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.151, max_value=10.0, allow_nan=False, allow_infinity=False)
        ),
        min_velocity=st.one_of(
            st.floats(max_value=0.09, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.01, max_value=10.0, allow_nan=False, allow_infinity=False)
        )
    )
    def test_property_multiple_invalid_parameters_raises_error(
        self,
        min_displacement: float,
        min_velocity: float
    ):
        """
        Property 16: Configuration validation
        
        For any TapDetectionConfig where multiple parameters are outside valid ranges,
        the validation should fail and raise ValueError with a descriptive message
        that includes all invalid parameters.
        
        Validates: Requirements 10.5
        """
        config = TapDetectionConfig(
            min_displacement=min_displacement,
            min_velocity=min_velocity
        )
        
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        
        error_message = str(exc_info.value)
        
        # Verify error message mentions both invalid parameters
        assert "min_displacement" in error_message
        assert "min_velocity" in error_message
        assert str(min_displacement) in error_message
        assert str(min_velocity) in error_message
    
    def test_default_configuration_is_valid(self):
        """
        Test that the default TapDetectionConfig values are valid.
        
        This ensures that users can create a config without parameters
        and it will pass validation.
        """
        config = TapDetectionConfig()
        
        # Default config should be valid
        assert config.validate() is True
        
        # Verify default values are within expected ranges
        assert 0.01 <= config.min_displacement <= 0.15
        assert 0.1 <= config.min_velocity <= 1.0
        assert 50 <= config.max_tap_duration_ms <= 1000
        assert 0.3 <= config.confidence_threshold <= 0.9
        assert 0.01 <= config.stationary_threshold <= 0.05
    
    def test_boundary_values_are_valid(self):
        """
        Test that boundary values for all parameters are considered valid.
        
        This ensures that the validation uses inclusive ranges.
        """
        # Test minimum boundary values
        config_min = TapDetectionConfig(
            min_displacement=0.01,
            min_velocity=0.1,
            max_tap_duration_ms=50,
            confidence_threshold=0.3,
            stationary_threshold=0.01
        )
        assert config_min.validate() is True
        
        # Test maximum boundary values
        config_max = TapDetectionConfig(
            min_displacement=0.15,
            min_velocity=1.0,
            max_tap_duration_ms=1000,
            confidence_threshold=0.9,
            stationary_threshold=0.05
        )
        assert config_max.validate() is True
    
    def test_just_outside_boundaries_are_invalid(self):
        """
        Test that values just outside the boundaries are correctly rejected.
        """
        # Just below minimum displacement
        config = TapDetectionConfig(min_displacement=0.00999)
        with pytest.raises(ValueError, match="min_displacement"):
            config.validate()
        
        # Just above maximum displacement
        config = TapDetectionConfig(min_displacement=0.15001)
        with pytest.raises(ValueError, match="min_displacement"):
            config.validate()
        
        # Just below minimum velocity
        config = TapDetectionConfig(min_velocity=0.0999)
        with pytest.raises(ValueError, match="min_velocity"):
            config.validate()
        
        # Just above maximum velocity
        config = TapDetectionConfig(min_velocity=1.0001)
        with pytest.raises(ValueError, match="min_velocity"):
            config.validate()
        
        # Just below minimum duration
        config = TapDetectionConfig(max_tap_duration_ms=49)
        with pytest.raises(ValueError, match="max_tap_duration_ms"):
            config.validate()
        
        # Just above maximum duration
        config = TapDetectionConfig(max_tap_duration_ms=1001)
        with pytest.raises(ValueError, match="max_tap_duration_ms"):
            config.validate()
        
        # Just below minimum confidence
        config = TapDetectionConfig(confidence_threshold=0.2999)
        with pytest.raises(ValueError, match="confidence_threshold"):
            config.validate()
        
        # Just above maximum confidence
        config = TapDetectionConfig(confidence_threshold=0.9001)
        with pytest.raises(ValueError, match="confidence_threshold"):
            config.validate()
        
        # Just below minimum stationary threshold
        config = TapDetectionConfig(stationary_threshold=0.00999)
        with pytest.raises(ValueError, match="stationary_threshold"):
            config.validate()
        
        # Just above maximum stationary threshold
        config = TapDetectionConfig(stationary_threshold=0.05001)
        with pytest.raises(ValueError, match="stationary_threshold"):
            config.validate()
