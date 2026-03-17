"""
Game controller for Super Battle Golf.

Translates shot parameters into OS-level mouse inputs:

* **Direction** – horizontal mouse movement (mouse left / mouse right).
* **Loft**      – scroll-wheel steps (scroll up = more loft, down = less).
* **Power**     – hold mouse button 1 for a duration proportional to power.

Usage::

    from sbg.controller import GameController, ControllerConfig
    from sbg.game_state import ShotParameters

    cfg = ControllerConfig(pixels_per_degree=8.0, max_power_hold_s=2.0)
    ctrl = GameController(cfg)

    # Execute a full shot sequence
    ctrl.execute_shot(direction_deg=5.0, power=0.7, loft_deg=45.0)

    # Or call each step individually
    ctrl.look(direction_deg=5.0)
    ctrl.adjust_loft(loft_deg=45.0)
    ctrl.shoot(power=0.7)

Use ``ControllerConfig(dry_run=True)`` in tests or when *pyautogui* is not
installed — inputs are printed to stdout instead of sent to the OS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

try:
    # Check whether pyautogui is importable by inspecting the module spec;
    # we avoid executing the module here because it probes the X display on
    # Linux which fails in headless environments.
    import importlib.util as _ilu

    _PYAUTOGUI_AVAILABLE = _ilu.find_spec("pyautogui") is not None
except Exception:  # pragma: no cover
    _PYAUTOGUI_AVAILABLE = False

# Deferred: the actual `pyautogui` object is imported lazily inside each
# helper so that headless / dry-run usage never triggers the X-display probe.


@dataclass
class ControllerConfig:
    """
    Tuning parameters for the :class:`GameController`.

    Attributes
    ----------
    pixels_per_degree:
        Horizontal mouse pixels moved per degree of aim direction.
        Calibrate this to match the game's mouse sensitivity.
    max_power_hold_s:
        Seconds to hold mouse button 1 for full power (``power=1.0``).
        Shorter values produce weaker shots.
    scroll_steps_per_degree:
        Mouse-scroll steps per degree of loft.  Positive = scroll up
        (increases loft); negative values would invert the axis.
    move_duration:
        Seconds over which each mouse movement is interpolated (smooths
        the motion so the game registers it correctly).
    post_move_delay:
        Pause (seconds) after the direction move before applying loft.
    post_scroll_delay:
        Pause (seconds) after loft is set before pressing the fire button.
    post_shot_delay:
        Pause (seconds) after releasing the fire button so the shot
        animation can complete before the next frame is captured.
    dry_run:
        When ``True`` inputs are printed to stdout but **not** sent to the
        OS.  Useful for unit tests and usage without an active game window.
    """

    pixels_per_degree: float = 8.0
    max_power_hold_s: float = 2.0
    scroll_steps_per_degree: float = 0.5
    move_duration: float = 0.05
    post_move_delay: float = 0.1
    post_scroll_delay: float = 0.1
    post_shot_delay: float = 2.0
    dry_run: bool = False


class GameController:
    """
    Sends mouse and scroll-wheel inputs to Super Battle Golf.

    The three controllable axes map directly to game mechanics:

    * **Mouse left / right** – rotates the character's view horizontally,
      changing the aim direction.
    * **Scroll wheel up / down** – adjusts the club loft angle.
    * **Hold mouse button 1** – charges the shot; longer hold = more power.

    Parameters
    ----------
    config:
        :class:`ControllerConfig` instance.  A default config is used when
        ``None``.

    Raises
    ------
    ImportError
        If *pyautogui* is not installed and ``config.dry_run`` is ``False``.
    """

    def __init__(self, config: Optional[ControllerConfig] = None) -> None:
        self.config = config or ControllerConfig()
        if not _PYAUTOGUI_AVAILABLE and not self.config.dry_run:
            raise ImportError(
                "pyautogui is required for GameController.  "
                "Install it with:  pip install pyautogui\n"
                "Or use ControllerConfig(dry_run=True) to suppress inputs."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def look(self, direction_deg: float) -> None:
        """
        Rotate the camera horizontally by *direction_deg* degrees.

        Positive values aim right; negative values aim left.  The pixel
        offset is computed as ``direction_deg * config.pixels_per_degree``
        and sent as a relative mouse movement.

        Parameters
        ----------
        direction_deg:
            Aim offset in degrees (typically −45 … +45).
        """
        pixels = int(direction_deg * self.config.pixels_per_degree)
        self._mouse_move_relative(pixels, 0)

    def adjust_loft(self, loft_deg: float) -> None:
        """
        Set the loft angle by scrolling the mouse wheel.

        The number of scroll steps is ``int(loft_deg *
        config.scroll_steps_per_degree)``.  Positive values scroll up
        (increasing loft); negative values scroll down.

        Parameters
        ----------
        loft_deg:
            Loft angle in degrees (0 = flat, 90 = straight up).
        """
        steps = int(loft_deg * self.config.scroll_steps_per_degree)
        if steps:
            self._scroll(steps)

    def shoot(self, power: float) -> None:
        """
        Fire the shot by holding mouse button 1 for a duration
        proportional to *power*.

        Parameters
        ----------
        power:
            Shot power in ``[0, 1]``.  ``1.0`` holds the button for
            ``config.max_power_hold_s`` seconds.
        """
        hold_s = max(0.0, min(1.0, power)) * self.config.max_power_hold_s
        self._mouse_hold(hold_s)

    def execute_shot(
        self,
        direction_deg: float,
        power: float,
        loft_deg: float,
    ) -> None:
        """
        Perform a complete shot sequence in three steps:

        1. **Look** – rotate the camera to the aim direction.
        2. **Loft** – apply scroll-wheel inputs for the loft angle.
        3. **Shoot** – hold mouse button 1 for the required power duration.

        Parameters
        ----------
        direction_deg:
            Aim offset in degrees (−45 … +45).
        power:
            Shot power in ``[0, 1]``.
        loft_deg:
            Loft angle in degrees (0–90).
        """
        self.look(direction_deg)
        time.sleep(self.config.post_move_delay)
        self.adjust_loft(loft_deg)
        time.sleep(self.config.post_scroll_delay)
        self.shoot(power)
        time.sleep(self.config.post_shot_delay)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mouse_move_relative(self, dx: int, dy: int) -> None:
        if self.config.dry_run:
            print(f"[DRY RUN] mouse.moveRel({dx}, {dy})")
            return
        import pyautogui as _pag  # noqa: PLC0415  (lazy – avoids X-display probe)
        _pag.FAILSAFE = True
        _pag.moveRel(dx, dy, duration=self.config.move_duration)

    def _scroll(self, steps: int) -> None:
        if self.config.dry_run:
            print(f"[DRY RUN] scroll({steps})")
            return
        import pyautogui as _pag  # noqa: PLC0415
        _pag.FAILSAFE = True
        _pag.scroll(steps)

    def _mouse_hold(self, duration: float) -> None:
        if self.config.dry_run:
            print(f"[DRY RUN] mouseDown() / sleep({duration:.3f}s) / mouseUp()")
            return
        import pyautogui as _pag  # noqa: PLC0415
        _pag.FAILSAFE = True
        _pag.mouseDown(button="left")
        time.sleep(duration)
        _pag.mouseUp(button="left")
