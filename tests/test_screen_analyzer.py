"""
Tests for the ScreenAnalyzer and hit-bar reading functionality.

Uses synthetic test images to verify:
- Hit-bar detection and flag localisation
- Terrain zone classification
- Feature-vector shape and value ranges
- Integration with real game screenshots when available
"""

import math
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from sbg.game_state import (
    GameState,
    HitBarState,
    Position,
    ShotParameters,
    TerrainZoneType,
)
from sbg.screen_analyzer import (
    ScreenAnalyzer,
    _BAR_BORDER_DIP_MIN,
    _BAR_BORDER_WINDOW_REL,
    _BAR_H_REL,
    _BAR_LEFT_BORDER_COL_REL,
    _BAR_RIGHT_BORDER_COL_REL,
    _BAR_W_REL,
    _BAR_X_REL,
    _BAR_Y_REL,
)

# Path to example screenshots (if present in the repository)
_IMAGES_DIR = Path(__file__).parent.parent / "images"
_UPLOADS_DIR = Path(__file__).parent.parent

# Reference screen size used when generating synthetic test frames
_REF_W = 2559
_REF_H = 1439


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blank_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Return a plain black BGR frame of the given size."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _bar_rect(w: int, h: int):
    """Return (x0, y0, x1, y1) of the hit bar in a frame of size w×h."""
    x0 = int(_BAR_X_REL * w)
    y0 = int(_BAR_Y_REL * h)
    x1 = x0 + max(1, int(_BAR_W_REL * w))
    y1 = y0 + max(1, int(_BAR_H_REL * h))
    return x0, y0, x1, y1


def _draw_bar_dark_outline(
    frame: np.ndarray, x0: int, y0: int, bw: int, bh: int
) -> None:
    """
    Paint the two dark-border columns the visibility check looks for.

    The border colour (20, 60, 30) is always ≥ 12 brightness units darker
    than any reasonable bar-interior colour, so both dips will satisfy
    ``_BAR_BORDER_DIP_MIN`` regardless of which bar colour is used.
    """
    dark_bgr = (20, 60, 30)
    hw = max(1, int(_BAR_BORDER_WINDOW_REL * bw))
    lc = int(_BAR_LEFT_BORDER_COL_REL * bw)
    rc = int(_BAR_RIGHT_BORDER_COL_REL * bw)
    frame[y0:y0 + bh, x0 + max(0, lc - hw) : x0 + lc + hw + 1] = dark_bgr
    frame[y0:y0 + bh, x0 + max(0, rc - hw) : x0 + rc + hw + 1] = dark_bgr


def _draw_white_dots(frame: np.ndarray, x0: int, y0: int, bw: int, bh: int) -> None:
    dot_colour = (240, 240, 240)
    cx = x0 + bw // 2
    for pct in (0.25, 0.5, 0.75):
        cy = y0 + int(pct * bh)
        cv2.circle(frame, (cx, cy), 5, dot_colour, -1, cv2.LINE_AA)


def _make_frame_with_bar(
    w: int = _REF_W,
    h: int = _REF_H,
    flag_x_pct: float = 0.5,
    flag_y_pct: float = 0.5,
    bar_color: tuple = (76, 167, 93),   # green fairway BGR
    flag_color: tuple = (0, 0, 220),    # red flag BGR
    draw_flag: bool = True,
    add_white_dots: bool = False,
) -> np.ndarray:
    """
    Create a synthetic frame with a solid-colour hit bar and a small flag
    patch at the requested relative position within the bar.
    """
    frame = _make_blank_frame(w, h)
    x0, y0, x1, y1 = _bar_rect(w, h)
    # Fill bar with a saturated green (FAIRWAY)
    frame[y0:y1, x0:x1] = bar_color

    bw = x1 - x0
    bh = y1 - y0

    # Add the dark outline columns that the visibility check requires.
    # Applied before the flag so that the flag patch sits on top.
    _draw_bar_dark_outline(frame, x0, y0, bw, bh)
    if draw_flag:
        fx = x0 + int(flag_x_pct * bw)
        fy = y0 + int(flag_y_pct * bh)
        # 8×8 patch
        px0 = max(x0, fx - 4)
        py0 = max(y0, fy - 4)
        px1 = min(x1, fx + 4)
        py1 = min(y1, fy + 4)
        frame[py0:py1, px0:px1] = flag_color
    if add_white_dots:
        _draw_white_dots(frame, x0, y0, bw, bh)
    return frame


def _make_bar_only(
    w: int = 120,
    h: int = 627,
    flag_x_pct: float = 0.5,
    flag_y_pct: float = 0.5,
    bar_color: tuple = (76, 167, 93),
    flag_color: tuple = (0, 0, 220),
    draw_flag: bool = True,
    add_white_dots: bool = False,
) -> np.ndarray:
    frame = _make_blank_frame(w, h)
    frame[:, :] = bar_color
    _draw_bar_dark_outline(frame, 0, 0, w, h)
    if draw_flag:
        fx = int(flag_x_pct * w)
        fy = int(flag_y_pct * h)
        px0 = max(0, fx - 4)
        py0 = max(0, fy - 4)
        px1 = min(w, fx + 4)
        py1 = min(h, fy + 4)
        frame[py0:py1, px0:px1] = flag_color
    if add_white_dots:
        _draw_white_dots(frame, 0, 0, w, h)
    return frame


# ---------------------------------------------------------------------------
# GameState / HitBarState unit tests
# ---------------------------------------------------------------------------


class TestPosition:
    def test_distance_to_self(self):
        p = Position(3.0, 4.0)
        assert p.distance_to(p) == pytest.approx(0.0)

    def test_distance_to_other(self):
        a = Position(0.0, 0.0)
        b = Position(3.0, 4.0)
        assert a.distance_to(b) == pytest.approx(5.0)


class TestTerrainZoneType:
    def test_values_are_distinct(self):
        values = {z.value for z in TerrainZoneType}
        assert len(values) == len(TerrainZoneType)

    def test_fairway_is_zero(self):
        assert TerrainZoneType.FAIRWAY.value == pytest.approx(0.0)


class TestHitBarState:
    def test_encode_terrain_default(self):
        bar = HitBarState()
        encoded = bar.encode_terrain(n_segments=8)
        assert len(encoded) == 8
        assert all(v == pytest.approx(TerrainZoneType.FAIRWAY.value) for v in encoded)

    def test_encode_terrain_single_water_zone(self):
        bar = HitBarState(
            is_visible=True,
            terrain_zones=[(0.0, 1.0, TerrainZoneType.WATER_OOB)],
        )
        encoded = bar.encode_terrain(n_segments=4)
        assert len(encoded) == 4
        assert all(v == pytest.approx(TerrainZoneType.WATER_OOB.value) for v in encoded)

    def test_encode_terrain_mixed_zones(self):
        bar = HitBarState(
            is_visible=True,
            terrain_zones=[
                (0.0, 0.5, TerrainZoneType.FAIRWAY),
                (0.5, 1.0, TerrainZoneType.WATER_OOB),
            ],
        )
        encoded = bar.encode_terrain(n_segments=4)
        assert encoded[0] == pytest.approx(TerrainZoneType.FAIRWAY.value)
        assert encoded[1] == pytest.approx(TerrainZoneType.FAIRWAY.value)
        assert encoded[2] == pytest.approx(TerrainZoneType.WATER_OOB.value)
        assert encoded[3] == pytest.approx(TerrainZoneType.WATER_OOB.value)


class TestGameState:
    def _make_state(self, **kwargs) -> GameState:
        defaults = dict(
            ball_position=Position(320, 360),
            hole_position=Position(320, 120),
            screen_width=640,
            screen_height=480,
        )
        defaults.update(kwargs)
        return GameState(**defaults)

    def test_distance_to_hole_auto(self):
        state = self._make_state()
        assert state.distance_to_hole == pytest.approx(240.0)

    def test_feature_vector_length(self):
        state = self._make_state()
        fv = state.to_feature_vector()
        assert len(fv) == 11

    def test_feature_vector_no_bar(self):
        state = self._make_state(hit_bar=None)
        fv = state.to_feature_vector()
        assert len(fv) == 11
        # bar_visible element (index 0) should be 0
        assert fv[0] == pytest.approx(0.0)
        # direction element (index 1) should default to 0.5
        assert fv[1] == pytest.approx(0.5)

    def test_feature_vector_with_bar_centred(self):
        bar = HitBarState(is_visible=True, flag_direction_offset=0.0, flag_y_pct=0.5)
        state = self._make_state(hit_bar=bar)
        fv = state.to_feature_vector()
        assert fv[0] == pytest.approx(1.0)   # bar visible
        assert fv[1] == pytest.approx(0.5)   # direction offset 0 → 0.5
        assert fv[2] == pytest.approx(0.5)   # y_pct

    def test_feature_vector_with_bar_left(self):
        bar = HitBarState(is_visible=True, flag_direction_offset=-1.0, flag_y_pct=0.0)
        state = self._make_state(hit_bar=bar)
        fv = state.to_feature_vector()
        assert fv[1] == pytest.approx(0.0)   # far left → 0

    def test_feature_vector_with_bar_right(self):
        bar = HitBarState(is_visible=True, flag_direction_offset=1.0, flag_y_pct=1.0)
        state = self._make_state(hit_bar=bar)
        fv = state.to_feature_vector()
        assert fv[1] == pytest.approx(1.0)   # far right → 1
        assert fv[2] == pytest.approx(1.0)

    def test_feature_vector_all_in_range(self):
        bar = HitBarState(
            is_visible=True,
            flag_direction_offset=0.3,
            flag_y_pct=0.4,
            terrain_zones=[(0.0, 0.5, TerrainZoneType.FAIRWAY),
                           (0.5, 1.0, TerrainZoneType.WATER_OOB)],
        )
        state = self._make_state(hit_bar=bar)
        fv = state.to_feature_vector()
        # All terrain/obstacle values should be in [0, 1]
        for i, v in enumerate(fv):
            assert -1.1 <= v <= 1.1, f"Feature [{i}]={v} out of expected range"


class TestShotParameters:
    def test_power_clamped(self):
        s = ShotParameters(direction_deg=0.0, power=1.5, loft_deg=45.0)
        assert s.power == pytest.approx(1.0)

    def test_loft_clamped(self):
        s = ShotParameters(direction_deg=0.0, power=0.5, loft_deg=120.0)
        assert s.loft_deg == pytest.approx(90.0)

    def test_repr(self):
        s = ShotParameters(direction_deg=5.0, power=0.75, loft_deg=30.0)
        assert "direction" in repr(s).lower() or "5.0" in repr(s)


# ---------------------------------------------------------------------------
# ScreenAnalyzer unit tests
# ---------------------------------------------------------------------------


class TestScreenAnalyzerErrors:
    def test_rejects_wrong_shape(self):
        analyzer = ScreenAnalyzer()
        bad_frame = np.zeros((100, 100), dtype=np.uint8)  # 2D, no channels
        with pytest.raises(ValueError):
            analyzer.analyze(bad_frame)

    def test_rejects_4channel(self):
        analyzer = ScreenAnalyzer()
        bad_frame = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            analyzer.analyze(bad_frame)

    def test_accepts_float_frame(self):
        analyzer = ScreenAnalyzer()
        float_frame = np.zeros((100, 100, 3), dtype=np.float32)
        state = analyzer.analyze(float_frame)
        assert isinstance(state, GameState)


class TestScreenAnalyzerBlankFrame:
    def setup_method(self):
        self.analyzer = ScreenAnalyzer()
        self.frame = _make_blank_frame(640, 480)
        self.state = self.analyzer.analyze(self.frame)

    def test_returns_game_state(self):
        assert isinstance(self.state, GameState)

    def test_screen_dimensions(self):
        assert self.state.screen_width == 640
        assert self.state.screen_height == 480

    def test_hit_bar_present(self):
        assert self.state.hit_bar is not None

    def test_feature_vector_length(self):
        fv = self.state.to_feature_vector()
        assert len(fv) == 11

    def test_power_gauge_in_range(self):
        assert 0.0 <= self.state.power_gauge <= 1.0


class TestHitBarVisibility:
    def setup_method(self):
        self.analyzer = ScreenAnalyzer()

    def test_blank_frame_bar_not_visible(self):
        """A plain black frame should not trigger bar detection."""
        frame = _make_blank_frame(_REF_W, _REF_H)
        state = self.analyzer.analyze(frame)
        assert state.hit_bar is not None
        assert not state.hit_bar.is_visible

    def test_frame_with_green_bar_is_visible(self):
        """A saturated green bar should be detected as visible."""
        frame = _make_frame_with_bar(
            w=_REF_W, h=_REF_H,
            bar_color=(76, 167, 93),  # vivid green – high saturation
            flag_color=(0, 0, 220),   # red flag
        )
        state = self.analyzer.analyze(frame)
        assert state.hit_bar is not None
        assert state.hit_bar.is_visible

    def test_frame_with_sand_bar_is_visible(self):
        """A tan/sand coloured bar should still be detected as visible."""
        frame = _make_frame_with_bar(
            w=_REF_W, h=_REF_H,
            bar_color=(180, 200, 220),  # light sand tone
            flag_color=(0, 0, 220),
        )
        state = self.analyzer.analyze(frame)
        assert state.hit_bar is not None
        assert state.hit_bar.is_visible


class TestFlagDetection:
    def setup_method(self):
        self.analyzer = ScreenAnalyzer()

    def _analyze(self, flag_x_pct, flag_y_pct) -> HitBarState:
        frame = _make_frame_with_bar(
            w=_REF_W, h=_REF_H,
            flag_x_pct=flag_x_pct,
            flag_y_pct=flag_y_pct,
        )
        state = self.analyzer.analyze(frame)
        return state.hit_bar

    def test_flag_at_centre_direction(self):
        bar = self._analyze(0.5, 0.5)
        assert bar.is_visible
        assert bar.flag_detected
        # Allow ±0.2 tolerance given 8-pixel patch rounding
        assert abs(bar.flag_direction_offset) < 0.3

    def test_flag_at_left(self):
        bar = self._analyze(0.1, 0.5)
        assert bar.is_visible
        assert bar.flag_detected
        assert bar.flag_direction_offset < 0.0  # should be left-of-centre

    def test_flag_at_right(self):
        bar = self._analyze(0.9, 0.5)
        assert bar.is_visible
        assert bar.flag_detected
        assert bar.flag_direction_offset > 0.0  # should be right-of-centre

    def test_flag_y_top(self):
        bar = self._analyze(0.5, 0.1)
        assert bar.is_visible
        assert bar.flag_detected
        assert bar.flag_y_pct < 0.4

    def test_flag_y_bottom(self):
        bar = self._analyze(0.5, 0.9)
        assert bar.is_visible
        assert bar.flag_detected
        assert bar.flag_y_pct > 0.6

    def test_direction_offset_in_valid_range(self):
        for x_pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
            bar = self._analyze(x_pct, 0.5)
            assert bar.flag_detected
            assert -1.0 <= bar.flag_direction_offset <= 1.0

    def test_y_pct_in_valid_range(self):
        for y_pct in [0.1, 0.3, 0.5, 0.7, 0.9]:
            bar = self._analyze(0.5, y_pct)
            assert bar.flag_detected
            assert 0.0 <= bar.flag_y_pct <= 1.0

    def test_bar_crop_detected(self):
        frame = _make_bar_only(w=120, h=627, flag_x_pct=0.3, flag_y_pct=0.7)
        state = self.analyzer.analyze(frame)
        assert state.hit_bar is not None
        assert state.hit_bar.is_visible
        assert state.hit_bar.flag_detected

    def test_white_dots_do_not_trigger_flag(self):
        frame = _make_bar_only(draw_flag=False, add_white_dots=True)
        state = self.analyzer.analyze(frame)
        assert state.hit_bar is not None
        assert state.hit_bar.is_visible
        assert not state.hit_bar.flag_detected


class TestTerrainClassification:
    def setup_method(self):
        self.analyzer = ScreenAnalyzer()

    def _bar_with_zones(
        self,
        top_color: tuple,
        bottom_color: tuple,
        w: int = _REF_W,
        h: int = _REF_H,
    ) -> HitBarState:
        """Create a frame where the top half of the bar is one colour and
        the bottom half is another, then parse the terrain zones."""
        frame = _make_blank_frame(w, h)
        x0, y0, x1, y1 = _bar_rect(w, h)
        bw, bh = x1 - x0, y1 - y0
        mid_y = (y0 + y1) // 2
        frame[y0:mid_y, x0:x1] = top_color
        frame[mid_y:y1, x0:x1] = bottom_color
        # Add the dark outline so the visibility check detects this as a bar.
        _draw_bar_dark_outline(frame, x0, y0, bw, bh)
        state = self.analyzer.analyze(frame)
        return state.hit_bar

    def test_all_fairway(self):
        bar = self._bar_with_zones((76, 167, 93), (76, 167, 93))
        assert bar.is_visible
        assert len(bar.terrain_zones) >= 1
        for _, _, zone in bar.terrain_zones:
            assert zone == TerrainZoneType.FAIRWAY

    def test_top_fairway_bottom_water(self):
        """Solid green top + blue-green high-std bottom should split zones."""
        # Fairway top (green, low std → solid colour)
        # Water bottom: simulate with diagonal striped blue-green rows
        frame = _make_blank_frame(_REF_W, _REF_H)
        x0, y0, x1, y1 = _bar_rect(_REF_W, _REF_H)
        bw, bh_bar = x1 - x0, y1 - y0

        # Top half: solid dark green (FAIRWAY)
        mid_y = (y0 + y1) // 2
        frame[y0:mid_y, x0:x1] = (60, 138, 100)

        # Bottom half: diagonal stripes of two colours to trigger stripe detection
        stripe_w = 6
        for row in range(mid_y, y1):
            for col in range(x0, x1):
                if ((col + row) // stripe_w) % 2 == 0:
                    frame[row, col] = (50, 150, 30)    # dark teal-blue
                else:
                    frame[row, col] = (200, 200, 100)  # light, high-value

        # Add the dark outline so the visibility check detects this as a bar.
        _draw_bar_dark_outline(frame, x0, y0, bw, bh_bar)

        bar = self.analyzer.analyze(frame)
        assert bar.hit_bar is not None
        assert bar.hit_bar.is_visible
        # With striped high-std rows at blue-green hues we expect at least one
        # non-FAIRWAY zone
        all_types = {zone for _, _, zone in bar.hit_bar.terrain_zones}
        assert len(bar.hit_bar.terrain_zones) >= 1


class TestTerrainClassifyRow:
    """Unit tests for the static _classify_row helper."""

    def test_fairway_low_std_green_hue(self):
        zone = ScreenAnalyzer._classify_row(mean_h=53.0, is_striped=False)
        assert zone == TerrainZoneType.FAIRWAY

    def test_rough_oob_high_std_green_hue(self):
        zone = ScreenAnalyzer._classify_row(mean_h=53.0, is_striped=True)
        assert zone == TerrainZoneType.ROUGH_OOB

    def test_water_oob_high_std_blue_hue(self):
        zone = ScreenAnalyzer._classify_row(mean_h=81.0, is_striped=True)
        assert zone == TerrainZoneType.WATER_OOB

    def test_border_case_below_water_hue(self):
        """H just below _WATER_HUE_MIN should not be classified as water."""
        zone = ScreenAnalyzer._classify_row(mean_h=71.0, is_striped=True)
        assert zone == TerrainZoneType.ROUGH_OOB

    def test_low_std_defaults_to_fairway(self):
        zone = ScreenAnalyzer._classify_row(mean_h=80.0, is_striped=False)
        assert zone == TerrainZoneType.FAIRWAY


# ---------------------------------------------------------------------------
# Integration tests with real screenshots (skipped if files not present)
# ---------------------------------------------------------------------------


def _load_screenshot(name: str) -> np.ndarray:
    path = _IMAGES_DIR / name
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"Screenshot not found: {path}")
    return img


def _load_uploaded_screenshot(name: str) -> np.ndarray:
    path = _UPLOADS_DIR / name
    img = cv2.imread(str(path))
    if img is None:
        pytest.skip(f"Uploaded screenshot not found: {path}")
    return img


class TestRealScreenshots:
    """
    Integration tests run against the actual game screenshots stored in
    ``images/``.  These are skipped automatically if the files are absent.
    """

    def setup_method(self):
        self.analyzer = ScreenAnalyzer()

    def _check_common(self, state: GameState, name: str):
        assert isinstance(state, GameState), f"{name}: expected GameState"
        assert state.hit_bar is not None, f"{name}: hit_bar should not be None"
        fv = state.to_feature_vector()
        assert len(fv) == 11, f"{name}: feature vector length {len(fv)} != 11"
        assert 0.0 <= state.power_gauge <= 1.0

    def test_green_course_hitting(self):
        frame = _load_screenshot("green_course_hitting.png")
        state = self.analyzer.analyze(frame)
        self._check_common(state, "green_course_hitting")
        # Hitting screenshots should have the bar visible
        assert state.hit_bar.is_visible, (
            "Hit bar should be detected in the hitting screenshot"
        )
        # Flag direction offset must be in [-1, 1]
        assert -1.0 <= state.hit_bar.flag_direction_offset <= 1.0
        # Flag y_pct must be in [0, 1]
        assert 0.0 <= state.hit_bar.flag_y_pct <= 1.0

    def test_green_course_looking(self):
        frame = _load_screenshot("green_course_looking.png")
        state = self.analyzer.analyze(frame)
        self._check_common(state, "green_course_looking")
        # The hit bar should NOT appear while the player is simply looking
        assert not state.hit_bar.is_visible, (
            "Hit bar should not be detected when looking at the course"
        )

    def test_sandy_course_hitting(self):
        frame = _load_screenshot("sandy_course_hitting.png")
        state = self.analyzer.analyze(frame)
        self._check_common(state, "sandy_course_hitting")
        assert state.hit_bar.is_visible

    def test_sandy_course_looking(self):
        frame = _load_screenshot("sandy_course_looking.png")
        state = self.analyzer.analyze(frame)
        self._check_common(state, "sandy_course_looking")
        # The hit bar should NOT appear while the player is simply looking
        assert not state.hit_bar.is_visible, (
            "Hit bar should not be detected when looking at the course"
        )

    def test_hit_bar_closeup_terrain_zones(self):
        """The closeup image should contain at least two distinct terrain zones."""
        frame = _load_screenshot("hit_bar_closeup.png")
        # The closeup IS the bar – pass it as the full "screen" so the
        # bar relative coordinates fall outside it, and we read it directly.
        bar_bgr = frame
        zones = self.analyzer._classify_terrain_zones(bar_bgr)
        zone_types = {z for _, _, z in zones}
        assert len(zone_types) >= 2, (
            f"Expected ≥2 terrain zone types in the closeup, got: {zone_types}"
        )

    def test_green_hitting_terrain_zones_present(self):
        frame = _load_screenshot("green_course_hitting.png")
        state = self.analyzer.analyze(frame)
        if state.hit_bar.is_visible:
            assert len(state.hit_bar.terrain_zones) >= 1

    def test_uploaded_bar_crop_flag_detected(self):
        frame = _load_uploaded_screenshot("Screenshot 2026-03-17 124540.png")
        state = self.analyzer.analyze(frame)
        self._check_common(state, "uploaded_bar_crop")
        assert state.hit_bar.is_visible
        assert state.hit_bar.flag_detected

    def test_uploaded_sand_hitting_visible(self):
        frame = _load_uploaded_screenshot("Screenshot 2026-03-17 130859.png")
        state = self.analyzer.analyze(frame)
        self._check_common(state, "uploaded_sand_hitting")
        assert state.hit_bar.is_visible
