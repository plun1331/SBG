"""
Tests for the GameController (dry-run mode).

Verifies that all three axes of control produce the expected calls
without requiring a display or pyautogui installation.
"""

import io
import time

import pytest

from sbg.controller import ControllerConfig, GameController


@pytest.fixture()
def ctrl():
    """A GameController with dry_run=True (no OS inputs are sent)."""
    return GameController(ControllerConfig(dry_run=True))


# ---------------------------------------------------------------------------
# look()
# ---------------------------------------------------------------------------


class TestLook:
    def test_positive_direction_moves_right(self, ctrl, capsys):
        ctrl.look(10.0)
        out = capsys.readouterr().out
        # 10 deg × 8 px/deg = 80 px
        assert "80" in out
        assert "moveRel" in out

    def test_negative_direction_moves_left(self, ctrl, capsys):
        ctrl.look(-5.0)
        out = capsys.readouterr().out
        assert "-40" in out

    def test_zero_direction(self, ctrl, capsys):
        ctrl.look(0.0)
        out = capsys.readouterr().out
        assert "moveRel(0, 0)" in out

    def test_fractional_degrees_truncated_to_int(self, ctrl, capsys):
        ctrl.look(1.9)
        out = capsys.readouterr().out
        # int(1.9 * 8) = int(15.2) = 15
        assert "15" in out


# ---------------------------------------------------------------------------
# adjust_loft()
# ---------------------------------------------------------------------------


class TestAdjustLoft:
    def test_positive_loft_scrolls_up(self, ctrl, capsys):
        ctrl.adjust_loft(10.0)
        out = capsys.readouterr().out
        # int(10 * 0.5) = 5 scroll steps
        assert "scroll(5)" in out

    def test_negative_loft_scrolls_down(self, ctrl, capsys):
        ctrl.adjust_loft(-20.0)
        out = capsys.readouterr().out
        assert "scroll(-10)" in out

    def test_zero_loft_no_scroll(self, ctrl, capsys):
        ctrl.adjust_loft(0.0)
        out = capsys.readouterr().out
        # int(0 * 0.5) = 0 → no scroll call
        assert "scroll" not in out

    def test_fractional_truncated(self, ctrl, capsys):
        ctrl.adjust_loft(3.0)
        # int(3 * 0.5) = int(1.5) = 1
        out = capsys.readouterr().out
        assert "scroll(1)" in out


# ---------------------------------------------------------------------------
# shoot()
# ---------------------------------------------------------------------------


class TestShoot:
    def test_full_power(self, ctrl, capsys):
        ctrl.shoot(1.0)
        out = capsys.readouterr().out
        assert "mouseDown" in out
        assert "mouseUp" in out
        # 1.0 * max_power_hold_s (2.0) = 2.000
        assert "2.000" in out

    def test_half_power(self, ctrl, capsys):
        ctrl.shoot(0.5)
        out = capsys.readouterr().out
        assert "1.000" in out

    def test_zero_power(self, ctrl, capsys):
        ctrl.shoot(0.0)
        out = capsys.readouterr().out
        assert "0.000" in out

    def test_power_clamped_above_one(self, ctrl, capsys):
        ctrl.shoot(2.0)
        out = capsys.readouterr().out
        # clamp to 1.0 → 2.000 s
        assert "2.000" in out

    def test_power_clamped_below_zero(self, ctrl, capsys):
        ctrl.shoot(-0.5)
        out = capsys.readouterr().out
        assert "0.000" in out


# ---------------------------------------------------------------------------
# execute_shot()
# ---------------------------------------------------------------------------


class TestExecuteShot:
    def test_all_three_steps_emitted(self, ctrl, capsys):
        ctrl.execute_shot(direction_deg=3.0, power=0.6, loft_deg=30.0)
        out = capsys.readouterr().out
        assert "moveRel" in out
        assert "scroll" in out
        assert "mouseDown" in out
        assert "mouseUp" in out

    def test_steps_in_order(self, ctrl, capsys):
        ctrl.execute_shot(direction_deg=1.0, power=0.5, loft_deg=20.0)
        out = capsys.readouterr().out
        # moveRel must come before scroll, scroll before mouseDown
        move_idx = out.index("moveRel")
        scroll_idx = out.index("scroll")
        hold_idx = out.index("mouseDown")
        assert move_idx < scroll_idx < hold_idx


# ---------------------------------------------------------------------------
# ControllerConfig defaults
# ---------------------------------------------------------------------------


class TestControllerConfig:
    def test_default_dry_run_false(self):
        cfg = ControllerConfig()
        assert cfg.dry_run is False

    def test_custom_pixels_per_degree(self):
        cfg = ControllerConfig(pixels_per_degree=4.0)
        ctrl = GameController(cfg)
        assert ctrl.config.pixels_per_degree == 4.0

    def test_import_error_without_pyautogui(self, monkeypatch):
        """Constructing without dry_run raises ImportError if pyautogui missing."""
        import sbg.controller as ctrl_mod

        monkeypatch.setattr(ctrl_mod, "_PYAUTOGUI_AVAILABLE", False)
        with pytest.raises(ImportError, match="pyautogui"):
            GameController(ControllerConfig(dry_run=False))

    def test_no_error_dry_run_without_pyautogui(self, monkeypatch):
        """dry_run=True never raises even without pyautogui."""
        import sbg.controller as ctrl_mod

        monkeypatch.setattr(ctrl_mod, "_PYAUTOGUI_AVAILABLE", False)
        ctrl = GameController(ControllerConfig(dry_run=True))
        assert ctrl is not None
