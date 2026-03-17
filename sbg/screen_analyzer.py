"""
Screen analyzer for Super Battle Golf.

Reads a captured game-screen image with OpenCV and extracts the information
needed to predict the best chip shot:

* The **hit bar** – a narrow vertical UI strip that appears while the player
  is aiming.  It visualises the terrain along the shot path and places a flag
  icon whose horizontal position encodes aim direction and whose vertical
  position encodes distance to the hole.
* Generic heuristics for ball / hole positions, wind, and power gauge (used
  as fallback features when the hit bar is not visible).

Hit-bar coordinate reference
-----------------------------
All bar coordinates were determined from 2559 × 1439 gameplay screenshots and
are stored as *relative* fractions of the screen dimensions so that the
analyzer works at any resolution.

  +-----------+-------------------------------------------+
  | Parameter | Value (relative to screen W × H)          |
  +===========+===========================================+
  | x start   | 649 / 2559 ≈ 0.2536                       |
  | y start   | 672 / 1439 ≈ 0.4670                       |
  | width     |  86 / 2559 ≈ 0.0336                       |
  | height    | 574 / 1439 ≈ 0.3989                       |
  +-----------+-------------------------------------------+

Flag icon
---------
A small red/orange triangular icon inside the bar.  Detected via an HSV
colour mask.  Its centre position gives:

* **direction** – horizontal offset from bar centre, normalised to [−1, 1].
* **distance**  – vertical fraction (0 = top, 1 = bottom).

Terrain zone classification
---------------------------
Each row of the bar is classified by its mean HSV hue and horizontal colour
variance:

* **FAIRWAY**   – green hue (H ≈ 40–72), low-to-moderate variance.
* **ROUGH_OOB** – green hue, high variance (prominent diagonal stripes).
* **WATER_OOB** – blue-green hue (H ≈ 72–100), high variance.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from sbg.game_state import (
    GameState,
    HitBarState,
    Position,
    TerrainZoneType,
)


# ---------------------------------------------------------------------------
# Hit-bar layout constants (relative to screen dimensions)
# Reference resolution: 2559 × 1439
# ---------------------------------------------------------------------------

_BAR_X_REL: float = 649 / 2559   # left edge of bar
_BAR_Y_REL: float = 672 / 1439   # top edge of bar
_BAR_W_REL: float = 86  / 2559   # bar width
_BAR_H_REL: float = 574 / 1439   # bar height

# ---------------------------------------------------------------------------
# Dark-outline visibility detection
# The hit bar has a semi-transparent dark border that creates characteristic
# brightness dips at ~17 % and ~81 % of the bar width.  Requiring *both* dips
# to exceed the threshold discriminates the bar from the game world, which may
# share similar colours but lacks this symmetric dark outline.
# Column positions calibrated from 2559 × 1439 reference screenshots.
# ---------------------------------------------------------------------------

# Relative column centre of the left / right dark-border band
_BAR_LEFT_BORDER_COL_REL: float  = 15 / 86   # ≈ 0.174
_BAR_RIGHT_BORDER_COL_REL: float = 70 / 86   # ≈ 0.814
# Half-width (in relative columns) of the search window around each centre
_BAR_BORDER_WINDOW_REL: float    =  5 / 86   # ≈ 0.058
# Interior column range (used as the brightness reference)
_BAR_INTERIOR_START_REL: float   = 22 / 86   # ≈ 0.256
_BAR_INTERIOR_END_REL: float     = 65 / 86   # ≈ 0.756
# Minimum brightness drop (interior mean − border min) required on each side
_BAR_BORDER_DIP_MIN: float       = 12.0

# ---------------------------------------------------------------------------
# HSV colour masks for flag detection (OpenCV H range 0–180)
# The flag is a small red/orange icon.
# ---------------------------------------------------------------------------

_FLAG_LOWER1 = np.array([0,   40,  50], dtype=np.uint8)
_FLAG_UPPER1 = np.array([12, 255, 255], dtype=np.uint8)
_FLAG_LOWER2 = np.array([168,  40,  50], dtype=np.uint8)
_FLAG_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

# Minimum pixel area for a candidate flag contour
_FLAG_MIN_AREA: int = 8

# ---------------------------------------------------------------------------
# Terrain-zone thresholds (derived from hit-bar close-up analysis)
# ---------------------------------------------------------------------------

# Hue boundary (OpenCV H, 0–180) separating FAIRWAY (green) from
# WATER_OOB (blue-green) zones.
_WATER_HUE_MIN: int = 72

# Standard deviation of BGR values across a bar row above which the row is
# considered "striped" (out-of-bounds).
_STRIPE_STD_THRESHOLD: float = 42.0

# Hue lower/upper bounds for green-family zones (FAIRWAY and ROUGH_OOB)
_GREEN_HUE_MIN: int = 40
_GREEN_HUE_MAX: int = 100

# ---------------------------------------------------------------------------
# Generic detection constants (fallback when hit bar is absent)
# ---------------------------------------------------------------------------

_BALL_LOWER = np.array([20, 100, 180], dtype=np.uint8)
_BALL_UPPER = np.array([40, 255, 255], dtype=np.uint8)
_HOLE_LOWER = np.array([0,   0,   0], dtype=np.uint8)
_HOLE_UPPER = np.array([180, 255, 60], dtype=np.uint8)
_GREEN_LOWER = np.array([35,  40,  40], dtype=np.uint8)
_GREEN_UPPER = np.array([90, 255, 200], dtype=np.uint8)
_OBSTACLE_LOWER = np.array([90,  50,  50], dtype=np.uint8)
_OBSTACLE_UPPER = np.array([140, 255, 200], dtype=np.uint8)

_MIN_BALL_AREA: int = 20
_MIN_HOLE_AREA: int = 30
_TERRAIN_SAMPLES: int = 16


class ScreenAnalyzer:
    """
    Analyzes a Super Battle Golf screen image to extract game-state features.

    The primary source of information is the **hit bar**: a narrow vertical
    UI strip visible while the player is aiming.  When the bar is visible,
    its flag icon directly encodes aim direction and distance to the hole.
    Generic colour-based heuristics for ball/hole positions serve as
    supplementary features.

    Usage::

        analyzer = ScreenAnalyzer()
        state = analyzer.analyze(frame)   # frame is a BGR numpy array

    Parameters:
        terrain_samples: Number of samples along the ball→hole path used by
            the fallback terrain/obstacle scan.
    """

    def __init__(self, terrain_samples: int = _TERRAIN_SAMPLES) -> None:
        self._terrain_samples = terrain_samples

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(self, frame: np.ndarray) -> GameState:
        """
        Process a game-screen frame and return the corresponding GameState.

        Parameters:
            frame: An H×W×3 BGR image as a numpy uint8 array (as returned
                by ``cv2.imread`` or ``cv2.VideoCapture``).

        Returns:
            GameState populated with detected positions, hit-bar data, and
            supplementary terrain/wind information.

        Raises:
            ValueError: If *frame* has an unexpected shape or dtype.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected an H×W×3 image, got shape {frame.shape}"
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hit_bar = self._read_hit_bar(frame, w, h)
        ball_pos = self._detect_ball(hsv, w, h)
        hole_pos = self._detect_hole(hsv, w, h)
        terrain, obstacles = self._sample_path(hsv, ball_pos, hole_pos, w, h)
        wind_speed, wind_dir = self._detect_wind(frame, w, h)
        power = self._detect_power_gauge(frame, w, h)

        return GameState(
            ball_position=ball_pos,
            hole_position=hole_pos,
            screen_width=w,
            screen_height=h,
            wind_speed=wind_speed,
            wind_direction_deg=wind_dir,
            terrain_elevation=terrain,
            obstacle_map=obstacles,
            power_gauge=power,
            hit_bar=hit_bar,
        )

    # ------------------------------------------------------------------
    # Hit-bar reading
    # ------------------------------------------------------------------

    def _read_hit_bar(
        self, frame: np.ndarray, w: int, h: int
    ) -> HitBarState:
        """
        Extract and parse the hit-bar UI element from the frame.

        The bar occupies a fixed relative region of the screen.  This method:

        1. Crops the bar region.
        2. Checks whether the bar is currently visible (player is aiming).
        3. Detects the flag icon and reads direction + distance.
        4. Classifies terrain zones row-by-row.

        Parameters:
            frame: Full BGR screen frame.
            w: Screen width in pixels.
            h: Screen height in pixels.

        Returns:
            HitBarState (``is_visible=False`` when the bar is not present).
        """
        x0 = int(_BAR_X_REL * w)
        y0 = int(_BAR_Y_REL * h)
        bw = max(1, int(_BAR_W_REL * w))
        bh = max(1, int(_BAR_H_REL * h))

        x1 = min(x0 + bw, w)
        y1 = min(y0 + bh, h)
        bar_crop = frame[y0:y1, x0:x1]

        if bar_crop.size == 0:
            return HitBarState(is_visible=False)

        if not self._is_bar_visible(bar_crop):
            return HitBarState(is_visible=False)

        bar_hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
        flag_dir, flag_y = self._detect_flag(bar_crop, bar_hsv)
        zones = self._classify_terrain_zones(bar_crop)

        return HitBarState(
            is_visible=True,
            flag_direction_offset=flag_dir,
            flag_y_pct=flag_y,
            terrain_zones=zones,
        )

    def _is_bar_visible(self, bar_bgr: np.ndarray) -> bool:
        """
        Return True if the bar crop shows the hit bar's characteristic dark outline.

        The hit bar is surrounded by a semi-transparent dark border.  When the
        bar is displayed, the per-column mean brightness of the crop has two
        distinct dips — one on the left side (~17 % of bar width) and one on
        the right (~81 %) — that are notably darker than the bright terrain
        content in between.  Requiring *both* dips to be present prevents false
        positives from the game world, which may have a dark region on one side
        but not the other (e.g. a gradient across the frame).

        Parameters:
            bar_bgr: BGR crop of the expected bar region.

        Returns:
            bool
        """
        bh, bw = bar_bgr.shape[:2]
        if bh == 0 or bw == 0:
            return False

        gray = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Mean brightness per column (averaged over the full bar height)
        col_means = gray.mean(axis=0)

        # Interior brightness reference
        int_start = max(0, int(_BAR_INTERIOR_START_REL * bw))
        int_end = min(bw, int(_BAR_INTERIOR_END_REL * bw))
        if int_start >= int_end:
            return False
        interior = float(col_means[int_start:int_end].mean())

        # Minimum brightness within each border search window
        hw = max(1, int(_BAR_BORDER_WINDOW_REL * bw))
        left_centre = int(_BAR_LEFT_BORDER_COL_REL * bw)
        right_centre = int(_BAR_RIGHT_BORDER_COL_REL * bw)

        left_win = col_means[max(0, left_centre - hw) : left_centre + hw + 1]
        right_win = col_means[max(0, right_centre - hw) : right_centre + hw + 1]
        if left_win.size == 0 or right_win.size == 0:
            return False

        left_dip = interior - float(left_win.min())
        right_dip = interior - float(right_win.min())
        return left_dip >= _BAR_BORDER_DIP_MIN and right_dip >= _BAR_BORDER_DIP_MIN

    def _detect_flag(
        self,
        bar_bgr: np.ndarray,
        bar_hsv: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Locate the red flag icon inside the bar crop and return its normalised
        position.

        The flag's horizontal centre encodes the **aim direction**:
          * 0 → exactly centred (straight shot)
          * negative → aim left
          * positive → aim right

        The flag's vertical centre encodes **distance to the hole** as a
        fraction of bar height (0 = top, 1 = bottom).

        Parameters:
            bar_bgr: BGR crop of the bar.
            bar_hsv: HSV crop of the bar (same shape).

        Returns:
            Tuple ``(direction_offset, y_pct)``.  Falls back to ``(0.0, 0.5)``
            if the flag cannot be detected.
        """
        bh, bw = bar_bgr.shape[:2]

        # Red HSV mask (hue wraps around 0)
        mask1 = cv2.inRange(bar_hsv, _FLAG_LOWER1, _FLAG_UPPER1)
        mask2 = cv2.inRange(bar_hsv, _FLAG_LOWER2, _FLAG_UPPER2)
        flag_mask = cv2.bitwise_or(mask1, mask2)

        # Small morphological close to connect nearby flag pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        flag_mask = cv2.morphologyEx(flag_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            flag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, 0.5

        # Use the largest qualifying contour
        valid = [c for c in contours if cv2.contourArea(c) >= _FLAG_MIN_AREA]
        if not valid:
            return 0.0, 0.5

        largest = max(valid, key=cv2.contourArea)
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return 0.0, 0.5

        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]

        # direction_offset: centre = 0, left = -1, right = +1
        direction_offset = (cx / bw - 0.5) * 2.0
        y_pct = cy / bh

        return float(np.clip(direction_offset, -1.0, 1.0)), float(np.clip(y_pct, 0.0, 1.0))

    def _classify_terrain_zones(
        self, bar_bgr: np.ndarray
    ) -> List[Tuple[float, float, TerrainZoneType]]:
        """
        Classify terrain zones in the bar by analysing each row's colour.

        Each row is assigned one of:
        * FAIRWAY   – green hue, low stripe variance
        * ROUGH_OOB – green hue, high stripe variance
        * WATER_OOB – blue-green hue, high stripe variance

        Consecutive rows of the same type are merged into a single zone
        entry ``(start_pct, end_pct, TerrainZoneType)``.

        Parameters:
            bar_bgr: BGR crop of the bar (H × W × 3).

        Returns:
            List of ``(start_pct, end_pct, TerrainZoneType)`` tuples.
        """
        bh, bw = bar_bgr.shape[:2]
        if bh == 0:
            return []

        # Trim border columns so the dark outline does not inflate per-row std.
        # Only the interior terrain content (between the two dark border bands)
        # is needed for zone classification.
        int_col_start = max(0, int(_BAR_INTERIOR_START_REL * bw))
        int_col_end = min(bw, int(_BAR_INTERIOR_END_REL * bw))
        interior_bgr = bar_bgr[:, int_col_start:int_col_end]

        bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)
        interior_hsv = bar_hsv[:, int_col_start:int_col_end]

        # Per-row classification
        row_zones: List[TerrainZoneType] = []
        for y in range(bh):
            row_bgr = interior_bgr[y].astype(np.float32)
            row_h = interior_hsv[y, :, 0].astype(np.float32)
            mean_h = float(row_h.mean())
            bgr_std = float(row_bgr.std())

            zone = self._classify_row(mean_h, bgr_std)
            row_zones.append(zone)

        # Merge consecutive equal zones
        zones: List[Tuple[float, float, TerrainZoneType]] = []
        if not row_zones:
            return zones

        current = row_zones[0]
        run_start = 0
        for y in range(1, bh):
            if row_zones[y] != current:
                zones.append((run_start / bh, y / bh, current))
                current = row_zones[y]
                run_start = y
        zones.append((run_start / bh, 1.0, current))
        return zones

    @staticmethod
    def _classify_row(mean_h: float, bgr_std: float) -> TerrainZoneType:
        """
        Classify a single bar row from its mean HSV hue and BGR std-dev.

        Parameters:
            mean_h: Mean OpenCV hue value of the row (0–180).
            bgr_std: Standard deviation of BGR values across the row.

        Returns:
            TerrainZoneType
        """
        # Water / hazard: notably more blue-green hue AND highly striped
        if mean_h >= _WATER_HUE_MIN and bgr_std >= _STRIPE_STD_THRESHOLD:
            return TerrainZoneType.WATER_OOB

        # Rough OOB: green hue but strongly striped
        if (
            _GREEN_HUE_MIN <= mean_h <= _GREEN_HUE_MAX
            and bgr_std >= _STRIPE_STD_THRESHOLD
        ):
            return TerrainZoneType.ROUGH_OOB

        # Everything else: fairway / green
        return TerrainZoneType.FAIRWAY

    # ------------------------------------------------------------------
    # Generic detection helpers (supplement hit-bar when needed)
    # ------------------------------------------------------------------

    def _detect_ball(self, hsv: np.ndarray, w: int, h: int) -> Position:
        """Return the detected ball position, falling back to screen centre."""
        mask = cv2.inRange(hsv, _BALL_LOWER, _BALL_UPPER)
        pos = self._largest_contour_centre(mask, _MIN_BALL_AREA)
        if pos is not None:
            return Position(x=float(pos[0]), y=float(pos[1]))
        return Position(x=w / 2.0, y=h * 0.75)

    def _detect_hole(self, hsv: np.ndarray, w: int, h: int) -> Position:
        """Return the detected hole/flag position, falling back to top-centre."""
        mask = cv2.inRange(hsv, _HOLE_LOWER, _HOLE_UPPER)
        pos = self._largest_contour_centre(mask, _MIN_HOLE_AREA)
        if pos is not None:
            return Position(x=float(pos[0]), y=float(pos[1]))
        return Position(x=w / 2.0, y=h * 0.25)

    def _sample_path(
        self,
        hsv: np.ndarray,
        ball: Position,
        hole: Position,
        w: int,
        h: int,
    ) -> Tuple[list, list]:
        """
        Sample terrain elevation and obstacle presence along the straight-line
        path between the ball and the hole.

        Returns:
            Tuple of (terrain_elevation, obstacle_map), each a list of
            ``_terrain_samples`` floats in [0, 1].
        """
        n = self._terrain_samples
        terrain: List[float] = []
        obstacles: List[float] = []

        obstacle_mask = cv2.inRange(hsv, _OBSTACLE_LOWER, _OBSTACLE_UPPER)

        for i in range(n):
            t = i / max(n - 1, 1)
            px = int(ball.x + t * (hole.x - ball.x))
            py = int(ball.y + t * (hole.y - ball.y))
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))

            elevation = 1.0 - py / h
            terrain.append(float(elevation))

            obs_val = float(obstacle_mask[py, px]) / 255.0
            obstacles.append(obs_val)

        return terrain, obstacles

    def _detect_wind(
        self, frame: np.ndarray, w: int, h: int
    ) -> Tuple[float, float]:
        """
        Estimate wind speed and direction from the top-right corner.

        Returns:
            ``(wind_speed, wind_direction_deg)``
        """
        roi = frame[0:int(h * 0.15), int(w * 0.80):]
        if roi.size == 0:
            return 0.0, 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, math.pi / 180, threshold=10,
            minLineLength=5, maxLineGap=3,
        )
        if lines is None or len(lines) == 0:
            return 0.0, 0.0

        best = max(lines, key=lambda l: math.hypot(
            l[0][2] - l[0][0], l[0][3] - l[0][1]
        ))
        x1, y1, x2, y2 = best[0]
        angle_deg = math.degrees(math.atan2(-(y2 - y1), x2 - x1))
        length = math.hypot(x2 - x1, y2 - y1)
        roi_w = roi.shape[1]
        wind_speed = min(length / roi_w * 20.0, 20.0) if roi_w > 0 else 0.0
        return float(wind_speed), float(angle_deg % 360)

    def _detect_power_gauge(
        self, frame: np.ndarray, w: int, h: int
    ) -> float:
        """
        Estimate the power-gauge fill level from the bottom-left corner.

        Returns:
            A float in [0, 1].
        """
        roi = frame[int(h * 0.90):, :int(w * 0.20)]
        if roi.size == 0:
            return 0.5

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_lower1 = np.array([0,   100, 100], dtype=np.uint8)
        red_upper1 = np.array([10,  255, 255], dtype=np.uint8)
        red_lower2 = np.array([160, 100, 100], dtype=np.uint8)
        red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv_roi, red_lower1, red_upper1),
            cv2.inRange(hsv_roi, red_lower2, red_upper2),
        )
        total = roi.shape[0] * roi.shape[1]
        if total == 0:
            return 0.5
        return min(1.0, float(np.sum(mask > 0)) / total)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _largest_contour_centre(
        mask: np.ndarray, min_area: float
    ) -> Optional[Tuple[float, float]]:
        """
        Return the centroid of the largest contour in *mask*, or None.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return None
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return None
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
