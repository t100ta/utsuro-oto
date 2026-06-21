"""Geometry utilities shared across control loops.

Adapted from pollen-robotics/hand_tracker_v2.
"""

from __future__ import annotations

import copy
import math

import numpy as np


def finger_orientation_deg(mcp: np.ndarray, tip: np.ndarray) -> float:
    """Return the orientation angle (degrees) of a finger from MCP to TIP.

    0° = pointing straight up; positive = tilted right; negative = tilted left.
    Used to map finger tilt to vibrato depth or instrument selection.
    """
    v = np.array([tip[0] - mcp[0], tip[1] - mcp[1]], dtype=float)
    v[1] = -v[1]  # flip Y so +Y means "up" in standard orientation
    return math.degrees(math.atan2(v[0], v[1]))


def angle_diff(a: float, b: float) -> float:
    """Smallest signed angular difference between two angles (radians)."""
    d = a - b
    return ((d + math.pi) % (2 * math.pi)) - math.pi


def allow_multiturn(
    new_joints: list[float],
    prev_joints: list[float],
    max_delta: float,
) -> list[float]:
    """Guarantee shortest-path rotation, allowing >2π if needed.

    Clamps each joint's step to ``max_delta`` per control tick so motors
    don't snap to a new target discontinuously.
    """
    new_joints = copy.deepcopy(new_joints)
    for i in range(len(new_joints)):
        diff = angle_diff(new_joints[i], prev_joints[i])
        if abs(diff) > max_delta:
            diff = math.copysign(max_delta, diff)
        new_joints[i] = prev_joints[i] + diff
        # Keep within ±3π to prevent integer overflow over very long sessions
        if new_joints[i] > 3 * math.pi:
            new_joints[i] -= 2 * math.pi
        elif new_joints[i] < -3 * math.pi:
            new_joints[i] += 2 * math.pi
    return new_joints
