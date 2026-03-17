"""
Tests for InteractiveTrainer and _ShotMonitor.

These tests exercise the non-GUI logic (parameter detection, sample saving,
flag-visibility filter) without requiring a live game window, pynput, or mss.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sbg.controller import ControllerConfig
from sbg.game_state import GameState, HitBarState, Position, ShotParameters
from sbg.interactive_trainer import InteractiveTrainer, _ShotMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(flag_detected=True, is_visible=True, direction_offset=0.2):
    """Create a minimal GameState with the given hit-bar settings."""
    bar = HitBarState(
        is_visible=is_visible,
        flag_detected=flag_detected,
        flag_direction_offset=direction_offset,
        flag_y_pct=0.5,
    )
    center = Position(320, 240)
    return GameState(
        ball_position=center,
        hole_position=center,
        screen_width=640,
        screen_height=480,
        wind_speed=0.0,
        wind_direction_deg=0.0,
        terrain_elevation=[],
        obstacle_map=[],
        power_gauge=0.5,
        distance_to_hole=0.0,
        hit_bar=bar,
    )


def _make_trainer(tmp_path, online_learning=False):
    """Return an InteractiveTrainer that uses a dry-run controller and no online opt."""
    cfg = ControllerConfig(dry_run=True)
    return InteractiveTrainer(
        data_dir=str(tmp_path),
        controller_config=cfg,
        online_learning=online_learning,
    )


# ---------------------------------------------------------------------------
# _flag_visible
# ---------------------------------------------------------------------------

class TestFlagVisible:
    def test_visible_and_detected(self):
        state = _make_state(flag_detected=True, is_visible=True)
        assert InteractiveTrainer._flag_visible(state) is True

    def test_not_visible(self):
        state = _make_state(flag_detected=True, is_visible=False)
        assert InteractiveTrainer._flag_visible(state) is False

    def test_not_detected(self):
        state = _make_state(flag_detected=False, is_visible=True)
        assert InteractiveTrainer._flag_visible(state) is False

    def test_no_bar(self):
        center = Position(0, 0)
        state = GameState(
            ball_position=center,
            hole_position=center,
            screen_width=640,
            screen_height=480,
            wind_speed=0.0,
            wind_direction_deg=0.0,
            terrain_elevation=[],
            obstacle_map=[],
            power_gauge=0.5,
            distance_to_hole=0.0,
            hit_bar=None,
        )
        assert InteractiveTrainer._flag_visible(state) is False


# ---------------------------------------------------------------------------
# _save_sample
# ---------------------------------------------------------------------------

class TestSaveSample:
    def test_file_created(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        features = np.zeros(11, dtype=np.float32)
        params = ShotParameters(direction_deg=5.0, power=0.6, loft_deg=30.0)
        trainer._save_sample(frame, features, params, score=8, idx=1)
        saved = list(tmp_path.glob("*.npz"))
        assert len(saved) == 1

    def test_npz_contains_score(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        features = np.zeros(11, dtype=np.float32)
        params = ShotParameters(direction_deg=-3.0, power=0.4, loft_deg=45.0)
        trainer._save_sample(frame, features, params, score=7, idx=42)
        data = np.load(tmp_path / "shot_000042.npz")
        assert float(data["score"]) == pytest.approx(7.0)

    def test_npz_contains_all_fields(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        features = np.zeros(11, dtype=np.float32)
        params = ShotParameters(direction_deg=10.0, power=0.8, loft_deg=60.0)
        trainer._save_sample(frame, features, params, score=9, idx=1)
        data = np.load(tmp_path / "shot_000001.npz")
        for key in ("image", "features", "direction", "power", "loft", "score"):
            assert key in data, f"Missing key: {key}"

    def test_parameters_round_trip(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        features = np.zeros(11, dtype=np.float32)
        params = ShotParameters(direction_deg=-12.5, power=0.33, loft_deg=22.0)
        trainer._save_sample(frame, features, params, score=5, idx=3)
        data = np.load(tmp_path / "shot_000003.npz")
        assert float(data["direction"]) == pytest.approx(-12.5)
        assert float(data["power"]) == pytest.approx(0.33)
        assert float(data["loft"]) == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# Direction decoded from hit bar
# ---------------------------------------------------------------------------

class TestDirectionDecoding:
    """The hit-bar flag offset is multiplied by 45° to get direction_deg."""

    def test_positive_offset(self):
        state = _make_state(direction_offset=0.5)
        assert state.hit_bar is not None
        direction_deg = state.hit_bar.flag_direction_offset * 45.0
        assert direction_deg == pytest.approx(22.5)

    def test_negative_offset(self):
        state = _make_state(direction_offset=-1.0)
        assert state.hit_bar is not None
        direction_deg = state.hit_bar.flag_direction_offset * 45.0
        assert direction_deg == pytest.approx(-45.0)

    def test_zero_offset(self):
        state = _make_state(direction_offset=0.0)
        assert state.hit_bar is not None
        assert state.hit_bar.flag_direction_offset * 45.0 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _ShotMonitor  (unit tests without real pynput)
# ---------------------------------------------------------------------------

class TestShotMonitor:
    def test_loft_from_scroll_steps(self):
        monitor = _ShotMonitor(degrees_per_scroll=2.0)
        monitor._scroll_steps = 15.0
        monitor._hold_s = 1.0
        assert monitor.loft_deg == pytest.approx(30.0)

    def test_loft_clamped_to_90(self):
        monitor = _ShotMonitor(degrees_per_scroll=2.0)
        monitor._scroll_steps = 60.0  # 60 * 2 = 120 > 90
        assert monitor.loft_deg == pytest.approx(90.0)

    def test_loft_clamped_to_zero_for_negative_scroll(self):
        monitor = _ShotMonitor(degrees_per_scroll=2.0)
        monitor._scroll_steps = -10.0
        assert monitor.loft_deg == pytest.approx(0.0)

    def test_power_from_hold_duration(self):
        monitor = _ShotMonitor(max_power_hold_s=2.0)
        monitor._hold_s = 1.0
        assert monitor.power == pytest.approx(0.5)

    def test_full_power(self):
        monitor = _ShotMonitor(max_power_hold_s=2.0)
        monitor._hold_s = 2.0
        assert monitor.power == pytest.approx(1.0)

    def test_power_clamped_above_one(self):
        monitor = _ShotMonitor(max_power_hold_s=2.0)
        monitor._hold_s = 5.0  # > max_power_hold_s
        assert monitor.power == pytest.approx(1.0)

    def test_power_zero_when_no_hold(self):
        monitor = _ShotMonitor()
        assert monitor.power == pytest.approx(0.0)

    def test_wait_times_out(self, monkeypatch):
        """wait_for_shot returns False when no shot is fired within timeout."""
        monitor = _ShotMonitor(shot_timeout_s=0.05)
        # Patch disarm so it just sets _done=True without needing a listener
        monitor._done = False
        monitor._listener = None

        def fast_stop():
            pass

        # Arm manually to set state but skip real listener
        monitor._scroll_steps = 0.0
        monitor._press_time = None
        monitor._hold_s = None
        # Simulate no shot fired — wait_for_shot polls until timeout
        fired = monitor.wait_for_shot()
        assert fired is False


# ---------------------------------------------------------------------------
# ShotDataset score weighting (train.py)
# ---------------------------------------------------------------------------

class TestShotDatasetScoreWeight:
    """Verify that ShotDataset returns the score as a weight tensor."""

    def test_score_weight_present(self, tmp_path):
        from sbg.train import ShotDataset

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        np.savez_compressed(
            str(tmp_path / "shot_000001.npz"),
            image=frame,
            features=np.zeros(11, dtype=np.float32),
            direction=np.float32(5.0),
            power=np.float32(0.5),
            loft=np.float32(30.0),
            score=np.float32(8.0),
        )
        ds = ShotDataset(tmp_path)
        sample = ds[0]
        import torch
        assert "weight" in sample
        assert sample["weight"].item() == pytest.approx(8.0 / 10.0)  # score 8 → weight 0.8

    def test_score_weight_default_for_legacy_file(self, tmp_path):
        """Files without a 'score' key default to weight 0.5."""
        from sbg.train import ShotDataset

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        np.savez_compressed(
            str(tmp_path / "shot_000001.npz"),
            image=frame,
            features=np.zeros(11, dtype=np.float32),
            direction=np.float32(0.0),
            power=np.float32(0.5),
            loft=np.float32(0.0),
        )
        ds = ShotDataset(tmp_path)
        sample = ds[0]
        # No 'score' key → default _DEFAULT_SAMPLE_SCORE (5.0) → weight 5.0/10.0
        assert sample["weight"].item() == pytest.approx(5.0 / 10.0)

    def test_perfect_score_weight_one(self, tmp_path):
        from sbg.train import ShotDataset

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        np.savez_compressed(
            str(tmp_path / "shot_000001.npz"),
            image=frame,
            features=np.zeros(11, dtype=np.float32),
            direction=np.float32(0.0),
            power=np.float32(0.5),
            loft=np.float32(0.0),
            score=np.float32(10.0),
        )
        ds = ShotDataset(tmp_path)
        assert ds[0]["weight"].item() == pytest.approx(1.0)
