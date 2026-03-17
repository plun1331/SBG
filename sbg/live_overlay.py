"""
Live gameplay overlay for Super Battle Golf.

Captures the screen in real time, runs the ScreenAnalyzer on every frame, and
displays an annotated window showing what the analyzer has detected.  Optionally
records the annotated output to a video file.

Usage (command line)::

    python -m sbg.live_overlay

Usage (API)::

    from sbg.live_overlay import LiveOverlay

    overlay = LiveOverlay()
    overlay.run()                          # display only (press Q to quit)

    overlay = LiveOverlay(output="out.mp4")
    overlay.run()                          # display + save to out.mp4

Keyboard shortcuts while the window is open:

    Q / Esc  – quit
    R        – toggle recording on / off (only meaningful without a fixed
               output path)
    Space    – pause / resume

Overlay elements drawn on each frame:

* **Ball marker** – white circle at the detected ball location.
* **Hole marker** – magenta ring at the detected hole location.
* **Ball→hole path samples** – dots along the shot line with red points
  indicating detected obstacles.
* **Wind vector** – arrow showing detected wind direction and speed.
* **Power gauge** – small bar showing the detected power level.
* **Bar region rectangle** – green when the hit bar is detected, red when
  not detected.  Always drawn so you can confirm the region is correct.
* **Terrain-zone strips** – semi-transparent colour bands along the bar:
  green = FAIRWAY, orange = ROUGH_OOB, blue = WATER_OOB.
* **Flag marker** – a cross-hair at the detected flag position (yellow).
* **Status text** – direction offset, y-pct, bar visibility, fps.

Dependencies:

    mss>=9.0.0      – cross-platform screen capture
    opencv-python   – drawing and display (already required by the project)
    numpy           – array ops (already required by the project)
"""

from __future__ import annotations

import time
from pathlib import Path
import math
from typing import Optional

import cv2
import numpy as np

try:
    import mss
    import mss.tools
    _MSS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MSS_AVAILABLE = False

from sbg.game_state import HitBarState, TerrainZoneType
from sbg.screen_analyzer import (
    ScreenAnalyzer,
    _BAR_H_REL,
    _BAR_W_REL,
    _BAR_X_REL,
    _BAR_Y_REL,
)

# ---------------------------------------------------------------------------
# Overlay drawing constants
# ---------------------------------------------------------------------------

# BGR colours used for the bar-region border
_COLOUR_BAR_VISIBLE   = (0, 220, 0)    # green – bar on screen
_COLOUR_BAR_INVISIBLE = (0, 0, 200)    # red   – bar not detected

# BGR colours for terrain-zone strips (semi-transparent blends)
_ZONE_COLOURS = {
    TerrainZoneType.FAIRWAY:   (30, 160, 30),   # green
    TerrainZoneType.ROUGH_OOB: (30, 110, 200),  # orange-ish
    TerrainZoneType.WATER_OOB: (180, 80, 30),   # blue
}
_ZONE_ALPHA = 0.35   # opacity of the terrain-zone overlay

# Flag marker size in pixels (cross-hair arm length)
_FLAG_ARM = 10

# HUD text style
_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE  = 0.55
_FONT_THICK  = 1
_TEXT_COLOUR = (220, 220, 220)  # light grey
_TEXT_SHADOW = (0, 0, 0)        # black drop-shadow

# Scene markers
_COLOUR_BALL = (245, 245, 245)
_COLOUR_BALL_OUTLINE = (0, 0, 0)
_COLOUR_HOLE = (200, 0, 200)
_COLOUR_PATH = (60, 160, 60)
_COLOUR_OBSTACLE = (0, 0, 255)
_COLOUR_WIND = (0, 220, 255)
_COLOUR_POWER = (0, 80, 220)

_MARKER_RADIUS = 6
_PATH_RADIUS = 3
_POWER_BAR_W = 130
_POWER_BAR_H = 10
_MAX_WIND_SPEED = 20.0       # Matches ScreenAnalyzer wind scaling (0–20 units)
_MIN_WIND_SPEED = 0.1        # Below this, wind is treated as negligible
_OBSTACLE_THRESHOLD = 0.5    # Normalised obstacle map cutoff (0–1)

# Display window name
_WINDOW_NAME = "SBG Live Overlay  [Q = quit | R = rec | Space = pause]"

# Default target frame rate for the capture loop (fps)
_TARGET_FPS: int = 30

# Default output video codec and extension
_FOURCC = "mp4v"
_OUTPUT_EXT = ".mp4"


class LiveOverlay:
    """
    Live screen-capture overlay for Super Battle Golf.

    Captures the primary monitor, analyses each frame with :class:`ScreenAnalyzer`,
    and renders an annotated window.  Optionally saves the annotated stream to a
    video file.

    Parameters:
        monitor: mss monitor index.  ``1`` = primary monitor (default).
        output: Path for the recorded video.  ``None`` = no recording unless
            the user presses **R** during playback.
        fps: Capture / display target frame rate.
        analyzer: A pre-built :class:`ScreenAnalyzer` instance.  If ``None``
            one is created automatically.
    """

    def __init__(
        self,
        monitor: int = 1,
        output: Optional[str] = None,
        fps: int = _TARGET_FPS,
        analyzer: Optional[ScreenAnalyzer] = None,
    ) -> None:
        if not _MSS_AVAILABLE:
            raise ImportError(
                "The 'mss' package is required for live screen capture.  "
                "Install it with:  pip install mss"
            )
        self._monitor_idx = monitor
        self._output_path = output
        self._fps = max(1, fps)
        self._analyzer = analyzer or ScreenAnalyzer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the capture-and-display loop.

        Blocks until the user presses **Q** or **Esc** in the display window.
        If *output* was specified at construction the annotated video is written
        there.  If the user presses **R** and no fixed *output* path was given,
        recording toggles on/off with an auto-generated filename.
        """
        with mss.mss() as sct:
            monitor = sct.monitors[self._monitor_idx]
            writer: Optional[cv2.VideoWriter] = None
            writer_path: Optional[str] = self._output_path
            recording = self._output_path is not None
            paused = False

            # Open a fixed output writer right away if a path was given.
            if self._output_path is not None:
                writer = self._open_writer(
                    self._output_path, monitor["width"], monitor["height"]
                )

            frame_interval = 1.0 / self._fps
            fps_display = self._fps
            t_last = time.perf_counter()
            fps_counter = _FpsCounter()

            cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)

            try:
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):          # Q or Esc → quit
                        break
                    if key == ord("r") and self._output_path is None:
                        # Toggle ad-hoc recording
                        recording = not recording
                        if recording:
                            writer_path = _auto_output_name()
                            writer = self._open_writer(
                                writer_path, monitor["width"], monitor["height"]
                            )
                        else:
                            if writer is not None:
                                writer.release()
                                writer = None
                    if key == ord(" "):                # Space → pause/resume
                        paused = not paused

                    if paused:
                        continue

                    # Throttle to target fps
                    now = time.perf_counter()
                    elapsed = now - t_last
                    if elapsed < frame_interval:
                        continue
                    t_last = now

                    # Capture
                    raw = sct.grab(monitor)
                    # mss returns BGRA; convert to BGR
                    frame = cv2.cvtColor(
                        np.frombuffer(raw.raw, dtype=np.uint8).reshape(
                            raw.height, raw.width, 4
                        ),
                        cv2.COLOR_BGRA2BGR,
                    )

                    # Analyse
                    try:
                        state = self._analyzer.analyze(frame)
                    except Exception:
                        state = None

                    # Draw overlay
                    fps_display = fps_counter.tick()
                    annotated = self._draw_overlay(
                        frame, state, fps_display, recording
                    )

                    # Write to file if recording
                    if recording and writer is not None:
                        writer.write(annotated)

                    cv2.imshow(_WINDOW_NAME, annotated)

            finally:
                if writer is not None:
                    writer.release()
                cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Overlay drawing
    # ------------------------------------------------------------------

    def _draw_overlay(
        self,
        frame: np.ndarray,
        state,
        fps: float,
        recording: bool,
    ) -> np.ndarray:
        """
        Draw analysis results onto *frame* and return the annotated copy.

        Parameters:
            frame:     Original BGR screen capture.
            state:     :class:`~sbg.game_state.GameState` or ``None`` if
                       analysis failed.
            fps:       Current display frame rate for the HUD.
            recording: Whether recording is currently active.

        Returns:
            Annotated BGR image (same size as *frame*).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # Bar bounding box in pixel coordinates
        bx0 = int(_BAR_X_REL * w)
        by0 = int(_BAR_Y_REL * h)
        bw  = max(1, int(_BAR_W_REL * w))
        bh  = max(1, int(_BAR_H_REL * h))
        bx1 = bx0 + bw
        by1 = by0 + bh

        if state is not None:
            _draw_ball_marker(out, state.ball_position)
            _draw_hole_marker(out, state.hole_position)
            _draw_path_samples(
                out,
                state.ball_position,
                state.hole_position,
                state.terrain_elevation,
                state.obstacle_map,
            )
            _draw_wind_vector(out, state.wind_speed, state.wind_direction_deg)
            _draw_power_gauge(out, state.power_gauge)

        if state is not None and state.hit_bar is not None:
            bar: HitBarState = state.hit_bar

            if bar.is_visible:
                # Semi-transparent terrain-zone bands inside the bar
                overlay = out.copy()
                for start_pct, end_pct, zone in bar.terrain_zones:
                    zy0 = by0 + int(start_pct * bh)
                    zy1 = by0 + int(end_pct   * bh)
                    colour = _ZONE_COLOURS.get(zone, (128, 128, 128))
                    overlay[zy0:zy1, bx0:bx1] = colour
                cv2.addWeighted(overlay, _ZONE_ALPHA, out, 1 - _ZONE_ALPHA, 0, out)

                # Bar rectangle (green = detected)
                cv2.rectangle(out, (bx0, by0), (bx1, by1), _COLOUR_BAR_VISIBLE, 2)

                # Status text
                lines = ["BAR VISIBLE"]
                if bar.flag_detected:
                    flag_cx = bx0 + int((bar.flag_direction_offset * 0.5 + 0.5) * bw)
                    flag_cy = by0 + int(bar.flag_y_pct * bh)
                    _draw_crosshair(out, flag_cx, flag_cy, _FLAG_ARM, (0, 230, 230))
                    lines.extend([
                        f"dir  {bar.flag_direction_offset:+.2f}",
                        f"y    {bar.flag_y_pct:.2f}",
                    ])
                else:
                    lines.append("FLAG NOT DETECTED")
                text_colour = _COLOUR_BAR_VISIBLE
            else:
                # Bar rectangle (red = not detected)
                cv2.rectangle(out, (bx0, by0), (bx1, by1), _COLOUR_BAR_INVISIBLE, 2)
                lines = ["BAR NOT DETECTED"]
                text_colour = _COLOUR_BAR_INVISIBLE

            _draw_text_block(out, lines, bx0, by0 - 10, text_colour)
        else:
            # Draw grey rectangle when analysis failed
            cv2.rectangle(out, (bx0, by0), (bx1, by1), (128, 128, 128), 1)

        # HUD: fps + recording indicator in the top-left corner
        hud = [f"FPS {fps:.0f}"]
        if recording:
            hud.append("● REC")
        _draw_text_block(out, hud, 12, 24, _TEXT_COLOUR)

        if state is not None:
            info_lines = [
                f"BALL {_format_coord(state.ball_position)}",
                f"HOLE {_format_coord(state.hole_position)}",
                f"DIST {state.distance_to_hole:.0f}",
                f"WIND {state.wind_speed:.1f} @ {state.wind_direction_deg:.0f}°",
                f"POWER {state.power_gauge:.2f}",
            ]
            _draw_text_block(out, info_lines, 12, 150, _TEXT_COLOUR)

        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_writer(
        path: str, width: int, height: int
    ) -> cv2.VideoWriter:
        """Open a VideoWriter for *path* at the monitor resolution."""
        fourcc = cv2.VideoWriter_fourcc(*_FOURCC)
        writer = cv2.VideoWriter(path, fourcc, _TARGET_FPS, (width, height))
        if not writer.isOpened():
            raise OSError(
                f"Could not open video writer at '{path}'. "
                "Ensure the directory exists and the codec is available."
            )
        return writer


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------


def _draw_crosshair(
    img: np.ndarray, cx: int, cy: int, arm: int, colour: tuple
) -> None:
    """Draw a cross-hair centred on (cx, cy)."""
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), colour, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), colour, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, colour, -1, cv2.LINE_AA)


def _draw_text_block(
    img: np.ndarray,
    lines: list,
    x: int,
    y: int,
    colour: tuple,
) -> None:
    """Render a stack of text lines with a drop-shadow for readability."""
    line_h = int(20 * _FONT_SCALE / 0.55)
    for i, line in enumerate(lines):
        ty = y - (len(lines) - 1 - i) * line_h
        # shadow
        cv2.putText(
            img, line, (x + 1, ty + 1),
            _FONT, _FONT_SCALE, _TEXT_SHADOW, _FONT_THICK + 1, cv2.LINE_AA,
        )
        # text
        cv2.putText(
            img, line, (x, ty),
            _FONT, _FONT_SCALE, colour, _FONT_THICK, cv2.LINE_AA,
        )


def _format_coord(pos) -> str:
    return f"{pos.x:.0f}, {pos.y:.0f}"


def _clamp_point(img: np.ndarray, x: float, y: float) -> tuple[int, int]:
    h, w = img.shape[:2]
    cx = int(max(0, min(w - 1, round(x))))
    cy = int(max(0, min(h - 1, round(y))))
    return cx, cy


def _terrain_colour(value: float) -> tuple[int, int, int]:
    v = max(0.0, min(1.0, value))
    green = int(80 + 150 * v)
    return (40, green, 40)


def _draw_ball_marker(img: np.ndarray, pos) -> None:
    cx, cy = _clamp_point(img, pos.x, pos.y)
    cv2.circle(img, (cx, cy), _MARKER_RADIUS, _COLOUR_BALL_OUTLINE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), _MARKER_RADIUS - 2, _COLOUR_BALL, -1, cv2.LINE_AA)


def _draw_hole_marker(img: np.ndarray, pos) -> None:
    cx, cy = _clamp_point(img, pos.x, pos.y)
    cv2.circle(img, (cx, cy), _MARKER_RADIUS + 2, _COLOUR_HOLE, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, _COLOUR_HOLE, -1, cv2.LINE_AA)


def _draw_path_samples(
    img: np.ndarray,
    ball,
    hole,
    terrain: list,
    obstacles: list,
) -> None:
    if not terrain:
        return
    obs = list(obstacles) if obstacles else []
    if len(obs) < len(terrain):
        obs.extend([0.0] * (len(terrain) - len(obs)))
    elif len(obs) > len(terrain):
        obs = obs[:len(terrain)]
    n = len(terrain)
    if n <= 0:
        return
    bx, by = ball.x, ball.y
    hx, hy = hole.x, hole.y
    for i in range(n):
        t = i / max(n - 1, 1)
        px = bx + t * (hx - bx)
        py = by + t * (hy - by)
        cx, cy = _clamp_point(img, px, py)
        if obs[i] >= _OBSTACLE_THRESHOLD:
            colour = _COLOUR_OBSTACLE
            radius = _PATH_RADIUS + 1
        else:
            colour = _terrain_colour(terrain[i])
            radius = _PATH_RADIUS
        cv2.circle(img, (cx, cy), radius, colour, -1, cv2.LINE_AA)


def _draw_wind_vector(
    img: np.ndarray,
    wind_speed: float,
    wind_direction_deg: float,
) -> None:
    h, w = img.shape[:2]
    origin = (w - 70, 40)
    if wind_speed <= _MIN_WIND_SPEED:
        cv2.circle(img, origin, 3, _COLOUR_WIND, -1, cv2.LINE_AA)
        return
    length = max(8, int(40 * min(wind_speed / _MAX_WIND_SPEED, 1.0)))
    angle = math.radians(wind_direction_deg)
    dx = math.cos(angle) * length
    dy = -math.sin(angle) * length
    end = (int(origin[0] + dx), int(origin[1] + dy))
    cv2.arrowedLine(img, origin, end, _COLOUR_WIND, 2, cv2.LINE_AA, tipLength=0.3)


def _draw_power_gauge(img: np.ndarray, power: float) -> None:
    h, w = img.shape[:2]
    x0 = 12
    y0 = max(12, h - 28)
    x1 = x0 + _POWER_BAR_W
    y1 = y0 + _POWER_BAR_H
    fill = int(x0 + _POWER_BAR_W * max(0.0, min(1.0, power)))
    cv2.rectangle(img, (x0, y0), (x1, y1), _TEXT_SHADOW, 1)
    cv2.rectangle(img, (x0, y0), (fill, y1), _COLOUR_POWER, -1)
    cv2.putText(
        img, "POWER", (x0, y0 - 6),
        _FONT, _FONT_SCALE * 0.8, _TEXT_COLOUR, _FONT_THICK, cv2.LINE_AA,
    )


def _auto_output_name() -> str:
    """Return a timestamped output filename in the current directory."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(Path(f"sbg_recording_{ts}{_OUTPUT_EXT}"))


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------


class _FpsCounter:
    """Rolling average FPS counter."""

    def __init__(self, window: int = 30) -> None:
        self._times: list = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ---------------------------------------------------------------------------
# Module entry-point  (python -m sbg.live_overlay)
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Live Super Battle Golf screen overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Keyboard shortcuts in the overlay window:\n"
            "  Q / Esc  – quit\n"
            "  R        – toggle recording (auto-named .mp4 in current dir)\n"
            "  Space    – pause / resume\n"
        ),
    )
    parser.add_argument(
        "--monitor", type=int, default=1,
        help="mss monitor index to capture (default: 1 = primary)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save annotated recording to this file (e.g. out.mp4).  "
             "If omitted you can toggle recording with R.",
    )
    parser.add_argument(
        "--fps", type=int, default=_TARGET_FPS,
        help=f"Target capture frame rate (default: {_TARGET_FPS})",
    )
    args = parser.parse_args()

    overlay = LiveOverlay(
        monitor=args.monitor,
        output=args.output,
        fps=args.fps,
    )
    overlay.run()


if __name__ == "__main__":
    _main()
