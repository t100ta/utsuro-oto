"""Hand Tracker using MediaPipe to detect hand positions in images.

Adapted from pollen-robotics/hand_tracker_v2.
Returns normalized coordinates in [-1, 1] for each landmark.

Environment variables::

    UTSURO_OTO_FRAME_IS_RGB      Set to "1" if the camera backend already
                                  delivers RGB frames (skips BGR→RGB conversion
                                  to avoid double-inversion of colours).
    UTSURO_OTO_MODEL_COMPLEXITY  MediaPipe model complexity 0 (fast, default)
                                  or 1 (more accurate, higher CPU).
    UTSURO_OTO_MIN_DET_CONF      MediaPipe min_detection_confidence (0.0-1.0,
                                  default 0.5).  Lower values detect more in
                                  poor lighting at the cost of more false positives.
"""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np

# ── Environment-variable configuration ──────────────────────────────────────
_FRAME_IS_RGB = os.environ.get("UTSURO_OTO_FRAME_IS_RGB", "0") == "1"
_MODEL_COMPLEXITY = int(os.environ.get("UTSURO_OTO_MODEL_COMPLEXITY", "0"))
_MIN_DET_CONF = float(os.environ.get("UTSURO_OTO_MIN_DET_CONF", "0.5"))

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class HandTracker:
    """Hand Tracker using MediaPipe Hands to detect hand positions."""

    def __init__(self, nb_hands: int = 2, model_complexity: int = _MODEL_COMPLEXITY) -> None:
        """Initialize the Hand Tracker.

        Args:
            nb_hands: Maximum number of hands to detect.
            model_complexity: 0 = fast/lightweight (recommended for Wireless CM4),
                              1 = more accurate.  Overridden by
                              ``UTSURO_OTO_MODEL_COMPLEXITY`` env var.

        """
        complexity = model_complexity  # may be overridden by env via default arg
        det_conf = _MIN_DET_CONF
        print(f"[Vision] HandTracker: complexity={complexity}, det_conf={det_conf}, frame_is_rgb={_FRAME_IS_RGB}")
        self.hands = mp_hands.Hands(
            static_image_mode=False,  # video mode: tracking between frames (faster, more robust)
            max_num_hands=nb_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=0.5,
            model_complexity=complexity,
        )

    def _norm(self, xy: tuple[float, float]) -> np.ndarray:
        """Normalise image coords [0,1] → [-1,1] with x-flip for mirror-like feel."""
        return np.array([-(xy[0] - 0.5) * 2, (xy[1] - 0.5) * 2])

    def get_hands_positions(self, img: np.ndarray) -> list[dict[str, np.ndarray]] | None:
        """Detect hands and return normalised landmark positions.

        Returns:
            List of dicts with keys ``palm``, ``index_tip``, ``index_mcp``,
            ``middle``; each value is an (x, y) ndarray in [-1, 1].
            Returns None if no hands found.

        """
        img = cv2.flip(img, 1)
        # GStreamer local backend may deliver RGB directly; avoid double-conversion.
        rgb = img if _FRAME_IS_RGB else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        if results.multi_hand_landmarks is None:
            return None

        hand_positions = []
        for landmarks in results.multi_hand_landmarks:
            palm_center = self._norm(
                (
                    landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                    landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                )
            )
            index_tip = self._norm(
                (
                    landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                )
            )
            index_mcp = self._norm(
                (
                    landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                )
            )
            middle_tip = self._norm(
                (
                    landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                    landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                )
            )
            hand_positions.append(
                {
                    "palm": palm_center,
                    "index_tip": index_tip,
                    "index_mcp": index_mcp,
                    "middle": middle_tip,
                }
            )

        return hand_positions
