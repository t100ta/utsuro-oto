"""Tests for utsuro_oto.utils — geometry helpers."""

import math

import numpy as np
import pytest

from utsuro_oto.utils import allow_multiturn, angle_diff, finger_orientation_deg

# ── finger_orientation_deg ────────────────────────────────────────────


class TestFingerOrientationDeg:
    def test_pointing_straight_up(self):
        """MCP directly below TIP → angle ≈ 0°."""
        mcp = np.array([0.0, 0.5])
        tip = np.array([0.0, 0.0])  # tip is above (smaller y in image coords)
        angle = finger_orientation_deg(mcp, tip)
        assert angle == pytest.approx(0.0, abs=1e-6)

    def test_pointing_right(self):
        """MCP left of TIP → angle ≈ +90° (tilted to the right)."""
        mcp = np.array([0.0, 0.0])
        tip = np.array([1.0, 0.0])
        angle = finger_orientation_deg(mcp, tip)
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_pointing_left(self):
        """MCP right of TIP → angle ≈ -90° (tilted to the left)."""
        mcp = np.array([1.0, 0.0])
        tip = np.array([0.0, 0.0])
        angle = finger_orientation_deg(mcp, tip)
        assert angle == pytest.approx(-90.0, abs=1e-6)

    def test_pointing_straight_down(self):
        """MCP above TIP → 180°."""
        mcp = np.array([0.0, 0.0])
        tip = np.array([0.0, 1.0])  # tip below mcp in image coords
        angle = finger_orientation_deg(mcp, tip)
        assert abs(angle) == pytest.approx(180.0, abs=1e-6)

    def test_diagonal_45_degrees(self):
        """Tip to the upper-right at 45° from vertical."""
        mcp = np.array([0.0, 1.0])
        tip = np.array([1.0, 0.0])
        angle = finger_orientation_deg(mcp, tip)
        assert angle == pytest.approx(45.0, abs=1e-6)


# ── angle_diff ────────────────────────────────────────────────────────


class TestAngleDiff:
    def test_zero_diff(self):
        assert angle_diff(0.0, 0.0) == pytest.approx(0.0)

    def test_positive_diff(self):
        assert angle_diff(1.0, 0.5) == pytest.approx(0.5)

    def test_negative_diff(self):
        assert angle_diff(0.5, 1.0) == pytest.approx(-0.5)

    def test_wraparound_near_pi(self):
        """0.1 → (2π - 0.1) is a clockwise step of 0.2, not 2π - 0.2."""
        a = 0.1
        b = 2 * math.pi - 0.1
        result = angle_diff(a, b)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_wraparound_negative(self):
        """(2π - 0.1) → 0.1 is a counter-clockwise step of -0.2."""
        a = 2 * math.pi - 0.1
        b = 0.1
        result = angle_diff(a, b)
        assert result == pytest.approx(-0.2, abs=1e-9)

    def test_result_always_in_minus_pi_to_pi(self):
        for _ in range(50):
            import random

            a = random.uniform(-10, 10)
            b = random.uniform(-10, 10)
            result = angle_diff(a, b)
            assert -math.pi <= result <= math.pi, f"angle_diff({a}, {b}) = {result}"


# ── allow_multiturn ────────────────────────────────────────────────────


class TestAllowMultiturn:
    def test_no_motion_when_already_at_target(self):
        result = allow_multiturn([1.0], [1.0], max_delta=0.5)
        assert result[0] == pytest.approx(1.0)

    def test_small_step_allowed_fully(self):
        result = allow_multiturn([1.3], [1.0], max_delta=0.5)
        assert result[0] == pytest.approx(1.3)

    def test_large_step_clamped_to_max_delta(self):
        """A step of 2.0 rad (< π, so unambiguously forward) is clamped to max_delta."""
        result = allow_multiturn([2.0], [0.0], max_delta=0.5)
        assert result[0] == pytest.approx(0.5)

    def test_negative_step_clamped(self):
        """A step of -2.0 rad (< π backward) is clamped to -max_delta."""
        result = allow_multiturn([-2.0], [0.0], max_delta=0.5)
        assert result[0] == pytest.approx(-0.5)

    def test_shortest_path_chosen_near_pi(self):
        """Going from π-0.1 to -(π-0.1) should wrap via +0.2, not travel 2π-0.2."""
        prev = [math.pi - 0.1]
        new_target = [-(math.pi - 0.1)]
        result = allow_multiturn(new_target, prev, max_delta=math.pi)
        # Shortest diff is +0.2 (the short way around)
        assert result[0] == pytest.approx(math.pi + 0.1, abs=1e-9)

    def test_values_clipped_to_three_pi(self):
        """Values that exceed ±3π should be wrapped back."""
        # Start at 3π - 0.1 and step +0.5 → exceeds 3π → gets wrapped
        prev = [3 * math.pi - 0.1]
        new_target = [3 * math.pi + 0.5]  # would be 3π + 0.4 after step
        result = allow_multiturn(new_target, prev, max_delta=1.0)
        assert -3 * math.pi <= result[0] <= 3 * math.pi

    def test_multi_joint(self):
        """Works with multiple joints; each is clamped independently."""
        result = allow_multiturn([10.0, -10.0], [1.0, -1.0], max_delta=0.5)
        assert result[0] == pytest.approx(1.5)
        assert result[1] == pytest.approx(-1.5)
