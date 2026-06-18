"""Tests for the pure helper functions in thereminvox.main.

Only hardware-free helpers are tested here.  The threaded control loops
(_vision_loop, _audio_motion_loop, run) require a live Reachy Mini and
are excluded from unit tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Patch heavy optional dependencies before importing the app module
# so that tests don't fail on machines without FluidSynth or mediapipe.
_PROBE_FAIL = MagicMock()
_PROBE_FAIL.ok = False
_PROBE_FAIL.error = "test: FluidSynth mocked away"


@pytest.fixture(scope="module", autouse=True)
def _mock_fluidsynth_at_import():
    """Ensure FluidSynth probe fails (no system lib needed) when importing main."""
    with patch("thereminvox.fluidsynth_check.probe_fluidsynth", return_value=_PROBE_FAIL):
        yield


# ── _annotate_frame ────────────────────────────────────────────────────

class TestAnnotateFrame:
    @pytest.fixture(autouse=True)
    def _import_helper(self):
        from thereminvox.main import _annotate_frame
        self._annotate = _annotate_frame

    def _blank(self, h: int = 360, w: int = 640) -> np.ndarray:
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_no_hands_returns_frame_unchanged(self):
        frame = self._blank()
        result = self._annotate(frame, None)
        assert result is frame  # same object, not a copy

    def test_empty_hand_list_returns_frame_unchanged(self):
        frame = self._blank()
        result = self._annotate(frame, [])
        assert result is frame

    def test_frame_shape_preserved_with_hands(self):
        frame = self._blank()
        hands = [{"palm": np.array([0.0, 0.0]), "index_tip": np.array([0.1, -0.1])}]
        result = self._annotate(frame, hands)
        assert result.shape == frame.shape

    def test_annotation_does_not_raise_on_out_of_bounds_coords(self):
        """Coordinates slightly outside [-1,1] must not raise exceptions."""
        frame = self._blank()
        hands = [{"palm": np.array([1.5, -1.5])}]  # beyond image bounds
        # cv2.circle clips automatically — must not raise
        self._annotate(frame, hands)

    def test_missing_key_in_hand_dict_is_skipped(self):
        """Hand dict with no 'palm' or 'index_tip' keys must not raise."""
        frame = self._blank()
        hands = [{"middle": np.array([0.0, 0.0])}]  # only 'middle', no known draw keys
        result = self._annotate(frame, hands)
        assert result.shape == frame.shape

    def test_multiple_hands_all_drawn(self):
        frame = self._blank()
        hands = [
            {"palm": np.array([0.0, 0.0])},
            {"palm": np.array([0.5, 0.5])},
        ]
        result = self._annotate(frame, hands)
        assert result.shape == frame.shape
        # Frame should no longer be all zeros (circles were drawn)
        assert result.max() > 0


# ── POST /config — content-type agnostic ──────────────────────────────

class TestConfigEndpoint:
    """The POST /config handler must accept JSON regardless of Content-Type.

    The Reachy Mini hub proxy sometimes strips or rewrites the Content-Type
    header before forwarding to the settings_app FastAPI instance, which
    caused 422 errors when using a Pydantic body model.  The handler now
    uses ``Request.json()`` directly to avoid this dependency.
    """

    @pytest.fixture
    def config_app(self):
        """Return a minimal FastAPI that reproduces the /config POST handler."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient

        import thereminvox.config as cfg_mod

        app = FastAPI()

        @app.post("/config")
        async def update_config(request: Request) -> dict:
            try:
                data = await request.json()
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}
            scale = data.get("scale")
            if isinstance(scale, str):
                cfg_mod.set_scale(scale)
            idx = data.get("instrument_idx")
            if isinstance(idx, int):
                cfg_mod.set_instrument_idx(idx)
            return {"scale": cfg_mod.get_scale()}

        return TestClient(app)

    def test_json_header_works(self, config_app):
        r = config_app.post("/config", json={"scale": "blues"})
        assert r.status_code == 200
        assert r.json()["scale"] == "blues"

    def test_no_content_type_still_works(self, config_app):
        """Key regression: missing Content-Type must not return 422."""
        import json
        r = config_app.post("/config", content=json.dumps({"scale": "chromatic"}))
        assert r.status_code == 200
        assert r.json()["scale"] == "chromatic"

    def test_text_plain_content_type_works(self, config_app):
        import json
        r = config_app.post(
            "/config",
            content=json.dumps({"scale": "pentatonic_minor"}),
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 200
        assert r.json()["scale"] == "pentatonic_minor"

    def test_empty_body_returns_200(self, config_app):
        r = config_app.post("/config")
        assert r.status_code == 200

    def test_instrument_idx_zero_is_accepted(self, config_app):
        """instrument_idx=0 must not be silently dropped (it is falsy in Python)."""
        r = config_app.post("/config", json={"instrument_idx": 0})
        assert r.status_code == 200
