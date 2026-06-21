"""Tests for utsuro_oto.hand_tracker — MediaPipe coordinate normalization.

MediaPipe is mocked so these tests run without loading the full model.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from utsuro_oto.hand_tracker import HandTracker

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def tracker():
    """HandTracker with mp_hands.Hands fully mocked (no model loading)."""
    with patch("utsuro_oto.hand_tracker.mp_hands") as mock_mp_hands:
        mock_mp_hands.Hands.return_value = MagicMock(name="hands_model")
        # HandLandmark enums — provide the index values MediaPipe uses
        mock_mp_hands.HandLandmark.MIDDLE_FINGER_PIP = 10
        mock_mp_hands.HandLandmark.INDEX_FINGER_TIP = 8
        mock_mp_hands.HandLandmark.INDEX_FINGER_MCP = 5
        mock_mp_hands.HandLandmark.MIDDLE_FINGER_TIP = 12
        yield HandTracker(nb_hands=2, model_complexity=0)


def _make_landmark(x: float, y: float) -> MagicMock:
    lm = MagicMock()
    lm.x = x
    lm.y = y
    return lm


def _make_landmarks_list(mapping: dict[int, tuple[float, float]]) -> MagicMock:
    """Build a MediaPipe-style landmark container for a single hand."""
    lm_list = [_make_landmark(0.5, 0.5)] * 21  # default all to center
    for idx, (x, y) in mapping.items():
        lm_list[idx] = _make_landmark(x, y)
    container = MagicMock()
    container.landmark = lm_list
    return container


# ── _norm ──────────────────────────────────────────────────────────────


class TestNorm:
    def test_center_maps_to_origin(self, tracker):
        result = tracker._norm((0.5, 0.5))
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)

    def test_right_edge_maps_to_negative_x(self, tracker):
        """Image x=1.0 (right edge) should map to -1 after flip."""
        result = tracker._norm((1.0, 0.5))
        assert result[0] == pytest.approx(-1.0)
        assert result[1] == pytest.approx(0.0)

    def test_left_edge_maps_to_positive_x(self, tracker):
        """Image x=0.0 (left edge) should map to +1."""
        result = tracker._norm((0.0, 0.5))
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)

    def test_top_edge_maps_to_negative_y(self, tracker):
        """Image y=0.0 (top) maps to -1."""
        result = tracker._norm((0.5, 0.0))
        assert result[1] == pytest.approx(-1.0)

    def test_bottom_edge_maps_to_positive_y(self, tracker):
        """Image y=1.0 (bottom) maps to +1."""
        result = tracker._norm((0.5, 1.0))
        assert result[1] == pytest.approx(1.0)

    def test_output_is_ndarray(self, tracker):
        result = tracker._norm((0.5, 0.5))
        assert isinstance(result, np.ndarray)


# ── get_hands_positions ────────────────────────────────────────────────


class TestGetHandsPositions:
    def _fake_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_none_when_no_hands(self, tracker):
        fake_results = MagicMock()
        fake_results.multi_hand_landmarks = None
        tracker.hands.process.return_value = fake_results

        result = tracker.get_hands_positions(self._fake_frame())
        assert result is None

    def test_returns_list_with_correct_keys(self, tracker):
        """Each hand dict must have palm, index_tip, index_mcp, middle keys."""
        hand_lm = _make_landmarks_list(
            {
                10: (0.5, 0.3),  # MIDDLE_FINGER_PIP → palm
                8: (0.6, 0.2),  # INDEX_FINGER_TIP
                5: (0.55, 0.35),  # INDEX_FINGER_MCP
                12: (0.5, 0.15),  # MIDDLE_FINGER_TIP
            }
        )
        fake_results = MagicMock()
        fake_results.multi_hand_landmarks = [hand_lm]
        tracker.hands.process.return_value = fake_results

        result = tracker.get_hands_positions(self._fake_frame())

        assert result is not None
        assert len(result) == 1
        hand = result[0]
        assert set(hand.keys()) == {"palm", "index_tip", "index_mcp", "middle"}

    def test_all_values_are_ndarrays_in_range(self, tracker):
        hand_lm = _make_landmarks_list(
            {
                10: (0.5, 0.3),
                8: (0.6, 0.2),
                5: (0.55, 0.35),
                12: (0.5, 0.15),
            }
        )
        fake_results = MagicMock()
        fake_results.multi_hand_landmarks = [hand_lm]
        tracker.hands.process.return_value = fake_results

        result = tracker.get_hands_positions(self._fake_frame())
        assert result is not None
        for key, val in result[0].items():
            assert isinstance(val, np.ndarray), f"{key} should be ndarray"
            assert val.shape == (2,), f"{key} should have shape (2,)"
            assert -1.0 <= val[0] <= 1.0, f"{key}[0] out of range: {val[0]}"
            assert -1.0 <= val[1] <= 1.0, f"{key}[1] out of range: {val[1]}"

    def test_multiple_hands_returned(self, tracker):
        hand_lm = _make_landmarks_list({10: (0.5, 0.3), 8: (0.6, 0.2), 5: (0.55, 0.35), 12: (0.5, 0.15)})
        fake_results = MagicMock()
        fake_results.multi_hand_landmarks = [hand_lm, hand_lm]
        tracker.hands.process.return_value = fake_results

        result = tracker.get_hands_positions(self._fake_frame())
        assert result is not None
        assert len(result) == 2

    def test_palm_coordinate_from_middle_finger_pip(self, tracker):
        """Palm is derived from MIDDLE_FINGER_PIP (idx=10) — verify the mapping."""
        # Put MIDDLE_FINGER_PIP at image x=0.0, y=0.0 → norm gives (+1, -1)
        hand_lm = _make_landmarks_list(
            {
                10: (0.0, 0.0),
                8: (0.5, 0.5),
                5: (0.5, 0.5),
                12: (0.5, 0.5),
            }
        )
        fake_results = MagicMock()
        fake_results.multi_hand_landmarks = [hand_lm]
        tracker.hands.process.return_value = fake_results

        result = tracker.get_hands_positions(self._fake_frame())
        assert result is not None
        palm = result[0]["palm"]
        assert palm[0] == pytest.approx(1.0)  # x flipped
        assert palm[1] == pytest.approx(-1.0)  # y top
