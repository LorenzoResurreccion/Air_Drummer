"""
Unit tests for DebounceFilter.

Tests the debounce filtering logic to ensure taps within the debounce window
are filtered and taps outside the window are allowed.
"""

import time
import pytest

from tap_detection.debounce_filter import DebounceFilter
from tap_detection.tap_models import TapEvent, TapType


def test_first_tap_always_triggers():
    """First tap should always be allowed through."""
    filter = DebounceFilter(debounce_window_ms=150)
    
    tap = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.0,
        peak_displacement=0.08,
        duration_ms=200
    )
    
    assert filter.should_trigger(tap) is True


def test_tap_within_debounce_window_filtered():
    """Tap within debounce window should be filtered."""
    filter = DebounceFilter(debounce_window_ms=150)
    
    tap1 = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.0,
        peak_displacement=0.08,
        duration_ms=200
    )
    
    tap2 = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.1,  # 100ms later
        peak_displacement=0.07,
        duration_ms=180
    )
    
    assert filter.should_trigger(tap1) is True
    assert filter.should_trigger(tap2) is False


def test_tap_after_debounce_window_allowed():
    """Tap after debounce window expires should be allowed."""
    filter = DebounceFilter(debounce_window_ms=150)
    
    tap1 = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.0,
        peak_displacement=0.08,
        duration_ms=200
    )
    
    tap2 = TapEvent(
        tap_type=TapType.TOE,
        timestamp=1.2,  # 200ms later
        peak_displacement=0.09,
        duration_ms=190
    )
    
    assert filter.should_trigger(tap1) is True
    assert filter.should_trigger(tap2) is True


def test_boundary_condition_at_window_edge():
    """Tap at or just after debounce window boundary should be allowed."""
    filter = DebounceFilter(debounce_window_ms=150)
    
    tap1 = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.0,
        peak_displacement=0.08,
        duration_ms=200
    )
    
    # Test just before boundary (should be filtered)
    tap2_before = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.149,  # 149ms later
        peak_displacement=0.07,
        duration_ms=180
    )
    
    # Test just after boundary (should be allowed)
    tap2_after = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.151,  # 151ms later
        peak_displacement=0.07,
        duration_ms=180
    )
    
    assert filter.should_trigger(tap1) is True
    assert filter.should_trigger(tap2_before) is False
    
    # Reset for second test
    filter2 = DebounceFilter(debounce_window_ms=150)
    filter2.should_trigger(tap1)
    assert filter2.should_trigger(tap2_after) is True


def test_multiple_taps_in_sequence():
    """Multiple taps with varying intervals should be filtered correctly."""
    filter = DebounceFilter(debounce_window_ms=150)
    
    taps = [
        TapEvent(TapType.HEEL, 1.0, 0.08, 200),    # Allowed (first)
        TapEvent(TapType.HEEL, 1.05, 0.07, 180),   # Filtered (50ms)
        TapEvent(TapType.HEEL, 1.1, 0.09, 190),    # Filtered (100ms)
        TapEvent(TapType.TOE, 1.2, 0.08, 200),     # Allowed (200ms from first)
        TapEvent(TapType.TOE, 1.25, 0.07, 180),    # Filtered (50ms from last allowed)
        TapEvent(TapType.FULL_FOOT, 1.4, 0.1, 210) # Allowed (200ms from last allowed)
    ]
    
    expected = [True, False, False, True, False, True]
    
    for tap, expected_result in zip(taps, expected):
        assert filter.should_trigger(tap) == expected_result


def test_is_debouncing_initially_false():
    """is_debouncing should return False when no taps have been triggered."""
    filter = DebounceFilter(debounce_window_ms=150)
    assert filter.is_debouncing() is False


def test_is_debouncing_true_within_window():
    """is_debouncing should return True within debounce window."""
    filter = DebounceFilter(debounce_window_ms=200)
    
    tap = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=time.time(),
        peak_displacement=0.08,
        duration_ms=200
    )
    
    filter.should_trigger(tap)
    
    # Should be debouncing immediately after trigger
    assert filter.is_debouncing() is True


def test_is_debouncing_false_after_window():
    """is_debouncing should return False after debounce window expires."""
    filter = DebounceFilter(debounce_window_ms=100)
    
    tap = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=time.time() - 0.15,  # 150ms ago
        peak_displacement=0.08,
        duration_ms=200
    )
    
    filter.should_trigger(tap)
    
    # Should not be debouncing after window expires
    assert filter.is_debouncing() is False


def test_custom_debounce_window():
    """DebounceFilter should respect custom debounce window duration."""
    filter = DebounceFilter(debounce_window_ms=300)
    
    tap1 = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.0,
        peak_displacement=0.08,
        duration_ms=200
    )
    
    tap2 = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.25,  # 250ms later
        peak_displacement=0.07,
        duration_ms=180
    )
    
    tap3 = TapEvent(
        tap_type=TapType.HEEL,
        timestamp=1.35,  # 350ms from first
        peak_displacement=0.09,
        duration_ms=190
    )
    
    assert filter.should_trigger(tap1) is True
    assert filter.should_trigger(tap2) is False  # Within 300ms window
    assert filter.should_trigger(tap3) is True   # Outside 300ms window from tap1
