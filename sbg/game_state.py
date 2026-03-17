"""
Game state representation for Super Battle Golf.

Defines dataclasses for the game state and shot parameters
used by the screen analyzer and ML model.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


@dataclass
class Position:
    """A 2D position in screen or game coordinates."""

    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        """Return the Euclidean distance to another position."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class TerrainZoneType(Enum):
    """
    Classification of a terrain zone visible in the hit bar.

    Values map to the encoded float used in the feature vector:
    FAIRWAY = 0.0, ROUGH_OOB = 0.5, WATER_OOB = 1.0.
    """

    FAIRWAY = 0.0     # Safe landing area (green or fairway)
    ROUGH_OOB = 0.5   # Out-of-bounds rough (striped green)
    WATER_OOB = 1.0   # Water hazard (striped blue-green)


@dataclass
class HitBarState:
    """
    Parsed state of the hit-bar UI element.

    The hit bar is a narrow vertical strip that appears on-screen during the
    player's shot.  It visualises the terrain along the shot path and shows a
    flag icon whose position encodes the aim direction and distance to the
    hole.

    Attributes:
        is_visible: Whether the hit bar was detected in the current frame.
        flag_direction_offset: Horizontal offset of the flag within the bar,
            normalised to ``[-1, 1]``.  A value of ``0`` means the flag is
            centred (shoot straight); ``-1`` = far left; ``+1`` = far right.
        flag_y_pct: Vertical position of the flag within the bar as a
            fraction of bar height in ``[0, 1]``.  ``0`` is the top of the
            bar, ``1`` is the bottom.  This encodes distance to the hole
            relative to the maximum shot range shown by the bar.
        flag_detected: Whether the flag icon was confidently detected in the
            current bar crop.
        terrain_zones: List of ``(start_pct, end_pct, TerrainZoneType)``
            tuples describing the terrain zones from top to bottom of the
            bar (each ``*_pct`` value is in ``[0, 1]``).
    """

    is_visible: bool = False
    flag_direction_offset: float = 0.0
    flag_y_pct: float = 0.5
    flag_detected: bool = False
    terrain_zones: List[Tuple[float, float, "TerrainZoneType"]] = field(
        default_factory=list
    )

    def encode_terrain(self, n_segments: int = 8) -> List[float]:
        """
        Encode the terrain zones as a fixed-length numeric vector.

        The bar is divided into *n_segments* equal vertical slices.  Each
        slice is assigned the ``TerrainZoneType.value`` of the zone that
        covers its midpoint.  Slices with no matching zone default to
        ``TerrainZoneType.FAIRWAY.value``.

        Parameters:
            n_segments: Number of equal-height segments to produce.

        Returns:
            A list of *n_segments* floats in ``{0.0, 0.5, 1.0}``.
        """
        result = [TerrainZoneType.FAIRWAY.value] * n_segments
        for i in range(n_segments):
            mid = (i + 0.5) / n_segments
            for start, end, zone in self.terrain_zones:
                if start <= mid < end:
                    result[i] = zone.value
                    break
        return result


@dataclass
class GameState:
    """
    Represents the current state of a Super Battle Golf game, as detected
    from the game screen by the ScreenAnalyzer.

    Attributes:
        ball_position: Position of the golf ball on screen (pixels).
        hole_position: Position of the hole on screen (pixels).
        screen_width: Width of the game screen in pixels.
        screen_height: Height of the game screen in pixels.
        wind_speed: Detected wind speed (arbitrary units, positive = right).
        wind_direction_deg: Wind direction in degrees (0 = right, 90 = up).
        terrain_elevation: Normalised elevation map as a 1D array sampled
            along the horizontal axis between ball and hole (values 0–1).
        obstacle_map: Binary array of the same length as terrain_elevation
            indicating the presence of obstacles (1 = obstacle present).
        power_gauge: Current power-gauge value in the range [0, 1].
        distance_to_hole: Distance from ball to hole in game units.
        hit_bar: Parsed state of the hit-bar UI element, or ``None`` if the
            bar was not visible in the current frame.
    """

    ball_position: Position
    hole_position: Position
    screen_width: int = 640
    screen_height: int = 480
    wind_speed: float = 0.0
    wind_direction_deg: float = 0.0
    terrain_elevation: list = field(default_factory=list)
    obstacle_map: list = field(default_factory=list)
    power_gauge: float = 0.5
    distance_to_hole: Optional[float] = None
    hit_bar: Optional[HitBarState] = None

    def __post_init__(self) -> None:
        if self.distance_to_hole is None:
            self.distance_to_hole = self.ball_position.distance_to(
                self.hole_position
            )

    def to_feature_vector(self) -> list:
        """
        Convert the game state into a flat feature vector suitable for the ML
        model.  All values are normalised to approximately [0, 1].

        The vector is structured as follows (11 values total):

        * [0]    Hit-bar visibility flag (0 or 1).
        * [1]    Flag direction offset, mapped to ``[0, 1]``
                 (0 = far left, 0.5 = centre, 1 = far right).
        * [2]    Flag Y position within the bar (0 = top, 1 = bottom).
        * [3–10] Terrain-zone encoding (8 segments, each 0/0.5/1).

        Returns:
            A list of 11 floats.
        """
        if self.hit_bar is not None and self.hit_bar.is_visible:
            bar_visible = 1.0
            # Map direction offset [-1, 1] → [0, 1] for the feature vector
            bar_dir = (self.hit_bar.flag_direction_offset + 1.0) / 2.0
            bar_y = self.hit_bar.flag_y_pct
            bar_terrain = self.hit_bar.encode_terrain(n_segments=8)
        else:
            bar_visible = 0.0
            bar_dir = 0.5
            bar_y = 0.5
            bar_terrain = [TerrainZoneType.FAIRWAY.value] * 8

        return [bar_visible, bar_dir, bar_y] + bar_terrain


@dataclass
class ShotParameters:
    """
    Recommended shot parameters output by the ML model.

    Attributes:
        direction_deg: Horizontal aim direction in degrees.
            0° means straight at the hole; positive values aim right.
        power: Shot power in the range [0, 1] (0 = minimum, 1 = maximum).
        loft_deg: Loft angle in degrees (0 = flat, 90 = straight up).
            A chip shot typically uses 30–60°.
    """

    direction_deg: float
    power: float
    loft_deg: float

    def __post_init__(self) -> None:
        self.power = max(0.0, min(1.0, self.power))
        self.loft_deg = max(0.0, min(90.0, self.loft_deg))

    def __repr__(self) -> str:
        return (
            f"ShotParameters("
            f"direction={self.direction_deg:.1f}°, "
            f"power={self.power:.2f}, "
            f"loft={self.loft_deg:.1f}°)"
        )
