"""
Interactive training session for Super Battle Golf.

Provides two complementary training modes:

**AI mode** (default)
    The model predicts shot parameters; the :class:`~sbg.controller.GameController`
    executes the shot; you score the outcome (1–10).  High-scoring shots
    reinforce the model's behaviour through an online gradient step.

**Manual mode**
    You take each shot yourself in the game.  Shot parameters are detected
    automatically:

    * **Direction** – read from the hit-bar flag position in the pre-shot
      frame (same source the AI uses).
    * **Loft** – scroll-wheel steps counted by a passive mouse listener
      (*pynput* required; falls back to a typed prompt when unavailable).
    * **Power** – duration of the left-mouse hold measured by the same
      listener, normalised to ``[0, 1]``.

    When *pynput* is installed the user never needs to type parameters.

In both modes, frames where the hit-bar flag is **not** detected are silently
skipped — there is nothing useful to learn from a shot without aiming
information.

Scored samples are saved as ``.npz`` files compatible with
:class:`~sbg.train.ShotDataset`.  The ``score`` field (1–10) stored in each
file is used as a per-sample loss weight during offline retraining.

Usage (command line)::

    python -m sbg.interactive_trainer               # AI mode
    python -m sbg.interactive_trainer --manual      # manual mode

Usage (API)::

    from sbg.interactive_trainer import InteractiveTrainer

    trainer = InteractiveTrainer(model_path="shot_model.pt", data_dir="data/")
    trainer.run()               # AI mode
    trainer.run(manual=True)    # manual mode
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import mss

    _MSS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MSS_AVAILABLE = False

try:
    # Use find_spec to check availability without importing the module, which
    # would attempt an X-display connection in headless environments.
    import importlib.util as _ilu

    _PYNPUT_AVAILABLE = _ilu.find_spec("pynput") is not None
except Exception:  # pragma: no cover
    _PYNPUT_AVAILABLE = False

# pynput.mouse is imported lazily inside _ShotMonitor.arm() to avoid the
# X-display probe at module import time on headless systems.

from sbg.controller import GameController, ControllerConfig
from sbg.game_state import GameState, ShotParameters
from sbg.model import DEFAULT_IMG_H, DEFAULT_IMG_W
from sbg.screen_analyzer import ScreenAnalyzer
from sbg.shot_predictor import ShotPredictor

# Default loft degrees per scroll click, derived from the default ControllerConfig
# so that detection and execution share the same calibration out of the box.
_DEFAULT_DEGREES_PER_SCROLL: float = 1.0 / ControllerConfig().scroll_steps_per_degree


# ---------------------------------------------------------------------------
# Shot monitor
# ---------------------------------------------------------------------------

class _ShotMonitor:
    """
    Passively observes mouse events to capture the parameters the user
    applied while taking a manual shot.

    * **Scroll-wheel steps** are counted from the moment :meth:`arm` is
      called until the left mouse button is released.  The step count is
      converted to a loft angle using *degrees_per_scroll*.
    * **Left-button hold duration** is measured from press to release and
      normalised to ``[0, 1]`` using *max_power_hold_s*.

    Requires *pynput*; check :data:`_PYNPUT_AVAILABLE` before using.

    Parameters
    ----------
    max_power_hold_s:
        Duration (seconds) corresponding to full power (``power = 1.0``).
        Should match :attr:`~sbg.controller.ControllerConfig.max_power_hold_s`.
    degrees_per_scroll:
        Loft degrees represented by a single scroll-wheel click.  Defaults
        to ``1 / ControllerConfig().scroll_steps_per_degree`` so that
        detection and execution use the same calibration.
    shot_timeout_s:
        Maximum seconds to wait for the user to take a shot before giving up.
    """

    def __init__(
        self,
        max_power_hold_s: float = 2.0,
        degrees_per_scroll: float = _DEFAULT_DEGREES_PER_SCROLL,
        shot_timeout_s: float = 60.0,
    ) -> None:
        self._max_hold = max_power_hold_s
        self._deg_per_scroll = degrees_per_scroll
        self._timeout = shot_timeout_s

        self._scroll_steps: float = 0.0
        self._press_time: Optional[float] = None
        self._hold_s: Optional[float] = None
        self._done: bool = False
        self._listener: Optional[object] = None

    # ------------------------------------------------------------------

    def arm(self) -> None:
        """
        Start listening for mouse events.

        Call this just before the user begins their shot sequence.  The
        listener runs in a background thread until the left mouse button
        is released (shot fired) or :meth:`disarm` is called.
        """
        self._scroll_steps = 0.0
        self._press_time = None
        self._hold_s = None
        self._done = False

        import pynput.mouse as _pynput_mouse  # noqa: PLC0415  (lazy – avoids X-display probe)

        def _on_scroll(x: int, y: int, dx: float, dy: float) -> None:
            if not self._done:
                self._scroll_steps += dy  # positive = scroll up = more loft

        def _on_click(
            x: int, y: int, button: object, pressed: bool
        ) -> Optional[bool]:
            if self._done:
                return False  # stop listener
            if button == _pynput_mouse.Button.left:
                if pressed:
                    self._press_time = time.monotonic()
                elif self._press_time is not None:
                    self._hold_s = time.monotonic() - self._press_time
                    self._done = True
                    return False  # stop listener
            return None

        self._listener = _pynput_mouse.Listener(
            on_scroll=_on_scroll,
            on_click=_on_click,
        )
        self._listener.start()  # type: ignore[union-attr]

    def disarm(self) -> None:
        """Stop the listener without waiting for a shot."""
        if self._listener is not None:
            self._listener.stop()  # type: ignore[union-attr]
        self._done = True

    def wait_for_shot(self) -> bool:
        """
        Block until the user fires (releases left mouse) or the timeout
        elapses.

        Returns
        -------
        bool
            ``True`` if a shot was detected; ``False`` if timed out.
        """
        deadline = time.monotonic() + self._timeout
        while not self._done and time.monotonic() < deadline:
            time.sleep(0.05)
        self.disarm()
        return self._hold_s is not None

    # ------------------------------------------------------------------
    # Decoded parameters

    @property
    def loft_deg(self) -> float:
        """Loft angle decoded from scroll-wheel steps (degrees, ≥ 0)."""
        return max(0.0, min(90.0, self._scroll_steps * self._deg_per_scroll))

    @property
    def power(self) -> float:
        """Power decoded from hold duration, normalised to ``[0, 1]``."""
        if self._hold_s is None:
            return 0.0
        return max(0.0, min(1.0, self._hold_s / self._max_hold))


# ---------------------------------------------------------------------------
# InteractiveTrainer
# ---------------------------------------------------------------------------

class InteractiveTrainer:
    """
    Human-in-the-loop training session for SBG.

    After each shot (AI-executed or manually played) the user rates the
    outcome on a **1–10** scale.  Rated samples are saved as ``.npz`` files
    and optionally used for real-time online fine-tuning.

    Shots are only recorded when the hit-bar flag is detected; frames
    without the flag are silently skipped.

    Parameters
    ----------
    model_path:
        Path to saved model weights.  A freshly-initialised model is used
        when ``None``.
    data_dir:
        Directory where scored ``.npz`` samples are written.
    controller_config:
        Configuration for :class:`~sbg.controller.GameController`.  A
        default config is used when ``None``.
    online_learning:
        Whether to fine-tune the model in real time after each scored shot.
    online_min_score:
        Minimum score (inclusive, 1–10) required to trigger an online
        gradient update in **AI mode**.  Manual-mode updates are always
        performed regardless of score.
    learning_rate:
        Learning rate for the online Adam optimiser.
    img_h / img_w:
        Image dimensions expected by the model.
    monitor:
        mss monitor index (1-based).
    save_path:
        Where to persist the model after each online update.  The model is
        kept in memory only when ``None``.
    shot_timeout:
        Seconds to wait for the user to fire in manual mode before skipping.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        data_dir: Union[str, Path] = "data",
        controller_config: Optional[ControllerConfig] = None,
        online_learning: bool = True,
        online_min_score: int = 6,
        learning_rate: float = 1e-4,
        img_h: int = DEFAULT_IMG_H,
        img_w: int = DEFAULT_IMG_W,
        monitor: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        shot_timeout: float = 60.0,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._predictor = ShotPredictor(
            model_path=model_path, img_h=img_h, img_w=img_w
        )
        self._analyzer = ScreenAnalyzer()
        self._ctrl_cfg = controller_config or ControllerConfig()
        self._controller = GameController(self._ctrl_cfg)
        self._online = online_learning
        self._min_score = max(1, min(10, online_min_score))
        self._monitor = monitor
        self._save_path = Path(save_path) if save_path else None
        self._shot_timeout = shot_timeout

        if online_learning:
            import torch

            self._optimizer: torch.optim.Optimizer = torch.optim.Adam(
                self._predictor._model.parameters(), lr=learning_rate
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, manual: bool = False) -> None:
        """
        Start the interactive training loop.

        Press **Ctrl+C** (or enter ``q`` at any prompt) to stop; all
        samples collected so far are already saved to *data_dir*.

        Parameters
        ----------
        manual:
            When ``True`` the user takes each shot manually.  Direction is
            read from the hit-bar flag; loft and power are captured from
            mouse events (*pynput* required) or prompted as a fallback.
            When ``False`` (default) the model predicts and the
            :class:`~sbg.controller.GameController` executes the shot.
        """
        mode_label = "MANUAL" if manual else "AI"
        print(f"=== SBG Interactive Trainer  [{mode_label} mode] ===")
        print(f"Data directory : {self._data_dir.resolve()}")
        if manual and _PYNPUT_AVAILABLE:
            print("Auto-detection : direction from screen | loft+power from mouse events")
        elif manual:
            print("Auto-detection : direction from screen  (pynput not found — loft/power will be prompted)")
        print("Shots without a visible flag are skipped automatically.")
        print("Press Ctrl+C to stop.\n")

        sample_count = 0
        try:
            while True:
                frame = self._capture_frame()
                if frame is None:
                    print("[WARN] Screen capture unavailable – retrying in 1 s…")
                    time.sleep(1.0)
                    continue

                state = self._analyzer.analyze(frame)

                if not self._flag_visible(state):
                    # No flag in hit bar — nothing to learn from; poll again.
                    time.sleep(0.1)
                    continue

                if manual:
                    result = self._run_manual_shot(frame, state)
                else:
                    result = self._run_ai_shot(frame, state)

                if result is None:
                    continue

                params, score = result
                sample_count += 1
                features = np.array(state.to_feature_vector(), dtype=np.float32)
                self._save_sample(frame, features, params, score, sample_count)
                print(f"  Sample #{sample_count} saved (score={score}/10).")

                if self._online:
                    do_update = manual or (score >= self._min_score)
                    if do_update:
                        self._online_update(frame, features, params, score)
                    if self._save_path:
                        self._predictor._model.save(str(self._save_path))
                        print(f"  Model checkpoint saved → {self._save_path}")

        except KeyboardInterrupt:
            print(f"\nSession ended. {sample_count} samples saved.")

    # ------------------------------------------------------------------
    # Shot modes
    # ------------------------------------------------------------------

    def _run_ai_shot(
        self,
        frame: np.ndarray,
        state: GameState,
    ) -> Optional[Tuple[ShotParameters, int]]:
        """Predict, execute, then score.  Returns ``(params, score)`` or ``None``."""
        params = self._predictor.predict_from_state(state, frame)
        print(
            f"\n  Predicted → direction={params.direction_deg:+.1f}°  "
            f"power={params.power:.2f}  loft={params.loft_deg:.1f}°"
        )

        if not self._prompt_yes_no("Execute this shot?"):
            return None

        print("  Executing shot…")
        self._controller.execute_shot(
            params.direction_deg, params.power, params.loft_deg
        )

        score = self._prompt_score()
        if score is None:
            return None

        return params, score

    def _run_manual_shot(
        self,
        frame: np.ndarray,
        state: GameState,
    ) -> Optional[Tuple[ShotParameters, int]]:
        """
        User takes the shot manually.

        Direction is read from the pre-shot hit-bar frame.  Loft and power
        are captured automatically via *pynput* mouse monitoring when
        available, otherwise the user is prompted to enter them.

        Returns ``(params, score)`` or ``None`` to skip.
        """
        # Direction is encoded in the hit-bar flag position.
        assert state.hit_bar is not None
        direction_deg = state.hit_bar.flag_direction_offset * 45.0

        if _PYNPUT_AVAILABLE:
            params = self._detect_manual_params(direction_deg)
        else:
            params = self._prompt_manual_params(direction_deg)

        if params is None:
            return None

        score = self._prompt_score()
        if score is None:
            return None

        return params, score

    # ------------------------------------------------------------------
    # Automatic parameter detection (pynput path)
    # ------------------------------------------------------------------

    def _detect_manual_params(self, direction_deg: float) -> Optional[ShotParameters]:
        """
        Wait for the user to take a shot while a background mouse listener
        records scroll steps (loft) and left-button hold duration (power).

        Parameters
        ----------
        direction_deg:
            Aim direction already decoded from the hit-bar flag.

        Returns
        -------
        ShotParameters or None
            ``None`` if the shot timed out or the user cancelled.
        """
        monitor = _ShotMonitor(
            max_power_hold_s=self._ctrl_cfg.max_power_hold_s,
            degrees_per_scroll=1.0 / self._ctrl_cfg.scroll_steps_per_degree,
            shot_timeout_s=self._shot_timeout,
        )

        print(
            f"\n  Flag detected  direction={direction_deg:+.1f}° (from screen).\n"
            "  Take your shot now – scroll to set loft, hold left-click for power.\n"
            f"  (Timeout: {self._shot_timeout:.0f} s)"
        )
        monitor.arm()
        fired = monitor.wait_for_shot()

        if not fired:
            print("  Timed out waiting for shot – skipping.")
            return None

        loft_deg = monitor.loft_deg
        power = monitor.power
        print(
            f"  Detected   → direction={direction_deg:+.1f}°  "
            f"loft={loft_deg:.1f}°  power={power:.2f}"
        )

        # Let the user confirm or override detected values.
        if not self._prompt_yes_no("Accept these parameters?"):
            return self._prompt_manual_params(direction_deg)

        return ShotParameters(
            direction_deg=direction_deg,
            power=power,
            loft_deg=loft_deg,
        )

    # ------------------------------------------------------------------
    # Fallback: typed prompts
    # ------------------------------------------------------------------

    def _prompt_yes_no(self, question: str) -> bool:
        """Ask a yes/no question; ``q`` / ``quit`` raises KeyboardInterrupt."""
        while True:
            try:
                ans = input(f"  {question} [y/n/q]: ").strip().lower()
            except EOFError:
                return False
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False
            if ans in ("q", "quit"):
                raise KeyboardInterrupt
            print("  Please enter y, n, or q.")

    def _prompt_score(self) -> Optional[int]:
        """Prompt for a shot quality score in 1–10; returns ``None`` on skip."""
        while True:
            try:
                raw = input("  Score this shot (1–10, or 0 to skip): ").strip()
            except EOFError:
                return None
            try:
                val = int(raw)
            except ValueError:
                print("  Please enter a whole number between 0 and 10.")
                continue
            if val == 0:
                return None
            if 1 <= val <= 10:
                return val
            print("  Score must be 1–10 (or 0 to skip).")

    def _prompt_manual_params(self, direction_deg: float) -> Optional[ShotParameters]:
        """
        Ask the user to enter loft and power (direction already known from
        the hit-bar flag).  Used as a fallback when *pynput* is unavailable
        or when the user rejects the auto-detected values.

        Parameters
        ----------
        direction_deg:
            Aim direction already decoded from the hit-bar flag.
        """
        print(
            f"  Direction detected from screen: {direction_deg:+.1f}°\n"
            "  Enter the remaining parameters (leave blank to skip this shot):"
        )
        try:
            l_raw = input("    Loft (degrees, 0–90): ").strip()
            if not l_raw:
                return None
            loft_deg = float(l_raw)

            p_raw = input("    Power (0.0–1.0): ").strip()
            if not p_raw:
                return None
            power = float(p_raw)
        except (ValueError, EOFError):
            print("  Invalid input – skipping this shot.")
            return None

        return ShotParameters(
            direction_deg=direction_deg,
            power=max(0.0, min(1.0, power)),
            loft_deg=max(0.0, min(90.0, loft_deg)),
        )

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture the current screen and return it as a BGR numpy array."""
        if not _MSS_AVAILABLE:
            print(
                "[ERROR] 'mss' is not installed.  "
                "Install it with:  pip install mss"
            )
            return None

        import cv2

        with mss.mss() as sct:
            monitors = sct.monitors
            mon = (
                monitors[self._monitor]
                if self._monitor < len(monitors)
                else monitors[1]
            )
            raw = sct.grab(mon)
            return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flag_visible(state: GameState) -> bool:
        """Return ``True`` only when the hit-bar flag is visible and detected."""
        bar = state.hit_bar
        return bar is not None and bar.is_visible and bar.flag_detected

    def _save_sample(
        self,
        frame: np.ndarray,
        features: np.ndarray,
        params: ShotParameters,
        score: int,
        idx: int,
    ) -> None:
        """Persist a scored shot sample to disk as a ``.npz`` file."""
        path = self._data_dir / f"shot_{idx:06d}.npz"
        np.savez_compressed(
            str(path),
            image=frame,
            features=features,
            direction=np.float32(params.direction_deg),
            power=np.float32(params.power),
            loft=np.float32(params.loft_deg),
            score=np.float32(score),
        )

    # ------------------------------------------------------------------
    # Online learning
    # ------------------------------------------------------------------

    def _online_update(
        self,
        frame: np.ndarray,
        features: np.ndarray,
        params: ShotParameters,
        score: int,
    ) -> None:
        """
        Perform one supervised gradient step using *params* as targets and
        ``score / 10`` as the sample weight.

        In **AI mode** this reinforces the model's own predictions for shots
        scored at or above ``online_min_score``.

        In **manual mode** this always runs, training the model toward the
        parameters the user actually chose, proportional to their quality.

        Parameters
        ----------
        frame:
            Raw BGR screenshot captured before the shot.
        features:
            Pre-computed ``GameState.to_feature_vector()`` array.
        params:
            The shot parameters that were executed (model prediction in AI
            mode; auto-detected or user-provided values in manual mode).
        score:
            User-assigned quality score in 1–10.
        """
        import torch
        import torch.nn.functional as F

        model = self._predictor._model
        model.train()

        img_tensor = self._predictor._preprocess_frame(frame)
        feat_tensor = (
            torch.tensor(features, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._predictor.device)
        )

        dir_target = torch.tensor(
            [[params.direction_deg]], dtype=torch.float32
        ).to(self._predictor.device)
        pow_target = torch.tensor(
            [[params.power]], dtype=torch.float32
        ).to(self._predictor.device)
        loft_target = torch.tensor(
            [[params.loft_deg]], dtype=torch.float32
        ).to(self._predictor.device)

        self._optimizer.zero_grad()
        dir_pred, pow_pred, loft_pred = model(img_tensor, feat_tensor)

        weight = score / 10.0
        loss = weight * (
            2.0 * F.mse_loss(dir_pred, dir_target.squeeze(1))
            + F.mse_loss(pow_pred, pow_target.squeeze(1))
            + F.mse_loss(loft_pred / 90.0, loft_target.squeeze(1) / 90.0)
        )
        loss.backward()
        self._optimizer.step()
        model.eval()

        print(f"  Online update: loss={loss.item():.4f}  weight={weight:.1f}")


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="SBG Interactive Trainer – score shots and teach the model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="Path to model weights (.pt)")
    parser.add_argument(
        "--data-dir", default="data", help="Directory to save scored samples"
    )
    parser.add_argument(
        "--save", default=None, help="Save updated model after each online step"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Manual training mode: you take the shots; "
            "direction is read from the screen, loft+power from mouse events"
        ),
    )
    parser.add_argument(
        "--no-online",
        action="store_true",
        help="Disable online learning (save samples only)",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=6,
        help="Minimum score for online update in AI mode",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Online learning rate"
    )
    parser.add_argument(
        "--monitor", type=int, default=1, help="Monitor index (1-based)"
    )
    parser.add_argument(
        "--shot-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for a manual shot before skipping",
    )
    args = parser.parse_args()

    trainer = InteractiveTrainer(
        model_path=args.model,
        data_dir=args.data_dir,
        online_learning=not args.no_online,
        online_min_score=args.min_score,
        learning_rate=args.lr,
        monitor=args.monitor,
        save_path=args.save,
        shot_timeout=args.shot_timeout,
    )
    trainer.run(manual=args.manual)
