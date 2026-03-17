# SBG — Super Battle Golf ML Model

An ML model that analyzes screenshots from the game **Super Battle Golf** to automatically predict optimal golf shot parameters: aim direction, power, and loft.

## Overview

SBG uses computer vision to parse the game screen (detecting the hit bar only) and feeds that information into a dual-branch neural network that outputs a recommended shot.

| Input | Output |
|---|---|
| Raw game screenshot (BGR) | `direction_deg` – aim angle (−45° … +45°) |
| | `power` – shot strength (0–1) |
| | `loft_deg` – club loft angle (0°–90°) |

### Example screenshots

| Green course – looking | Green course – hitting |
|---|---|
| ![Green course looking](<Screenshot 2026-03-16 232337.png>) | ![Green course hitting](<Screenshot 2026-03-16 232406.png>) |

## Architecture

```
Game Screen Image
       │
       ▼
┌─────────────────────────┐
│    ScreenAnalyzer       │  ← OpenCV-based screen parsing
│  (hit bar, ball, hole,  │
│   wind, terrain, etc.)  │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│       GameState         │  ← Structured game-state dataclass
│  to_feature_vector()    │  → 11-dimensional hit-bar vector
└─────────────────────────┘
       │              │
 11-dim features   image tensor
       │              │
       ▼              ▼
┌─────────────────────────────────┐
│     ShotPredictorModel          │  ← Dual-branch PyTorch CNN
│  ├─ Feature branch (MLP)        │
│  └─ Image branch (4-layer CNN)  │
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│     ShotParameters      │  ← direction_deg, power, loft_deg
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│     GameController      │  ← Translates params to OS mouse inputs
│  ├─ look(direction_deg) │     mouse move left/right
│  ├─ adjust_loft(deg)    │     scroll wheel up/down
│  └─ shoot(power)        │     hold left-click
└─────────────────────────┘
```

## Project Structure

```
SBG/
├── sbg/
│   ├── __init__.py             # Package exports
│   ├── controller.py           # GameController – translates params to mouse inputs
│   ├── game_state.py           # GameState, HitBarState, ShotParameters dataclasses
│   ├── interactive_trainer.py  # Human-in-the-loop training (AI + manual modes)
│   ├── live_overlay.py         # Live screen overlay + recording
│   ├── model.py                # ShotPredictorModel (PyTorch)
│   ├── screen_analyzer.py      # OpenCV-based screen parsing
│   ├── shot_predictor.py       # End-to-end predictor (ScreenAnalyzer + model)
│   └── train.py                # Training loop, ShotDataset, loss function
├── tests/
│   ├── test_controller.py
│   ├── test_interactive_trainer.py
│   └── test_screen_analyzer.py
├── images/                     # Example annotated screenshots
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- [PyTorch](https://pytorch.org/) ≥ 2.0
- OpenCV ≥ 4.8
- NumPy ≥ 1.24
- Pillow ≥ 10.0
- torchvision ≥ 0.15
- pyautogui ≥ 0.9.54  *(AI shot execution)*
- pynput ≥ 1.7.7  *(manual-mode parameter detection)*

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Predict a shot from a screenshot

```python
import cv2
from sbg import ShotPredictor

predictor = ShotPredictor(model_path="shot_model.pt")   # omit path to use a random model
frame = cv2.imread("screenshot.png")                     # BGR image

params = predictor.predict_from_frame(frame)
print(f"Direction: {params.direction_deg:.1f}°")
print(f"Power:     {params.power:.2f}")
print(f"Loft:      {params.loft_deg:.1f}°")
```

### Execute a shot automatically

```python
from sbg import GameController, ControllerConfig

ctrl = GameController(ControllerConfig(
    pixels_per_degree=8.0,    # tune to match your game's mouse sensitivity
    max_power_hold_s=2.0,     # seconds for full-power hold
))
ctrl.execute_shot(direction_deg=3.5, power=0.75, loft_deg=20.0)
```

### Analyse a screen without a model

```python
import cv2
from sbg import ScreenAnalyzer

analyzer = ScreenAnalyzer()
frame = cv2.imread("screenshot.png")
state = analyzer.analyze(frame)

print(f"Hit bar visible: {state.hit_bar.is_visible}")
print(f"Flag direction:  {state.hit_bar.flag_direction_offset:+.2f}")
print(f"Flag y%:         {state.hit_bar.flag_y_pct:.2f}")
```

### Live overlay + recording

Record gameplay from your screen and overlay the hit-bar analysis in real time:

```bash
python -m sbg.live_overlay
```

To save the annotated output to a video file:

```bash
python -m sbg.live_overlay --output out.mp4
```

Keyboard shortcuts inside the overlay window:

- **Q / Esc** — quit
- **R** — toggle recording (auto-named file when `--output` is omitted)
- **Space** — pause / resume

Use `--monitor` to pick which display to capture if needed.

On Windows 11, you can enable a transparent click-through fullscreen overlay
that is excluded from screen capture (prevents overlay feedback/lag):

```bash
python -m sbg.live_overlay --transparent
```

To show only the UI elements (no game frame), use:

```bash
python -m sbg.live_overlay --overlay-only
```

## Interactive Training

The interactive trainer lets you teach the model by rating real shots — either let the AI play and score its shots, or play yourself and have your technique recorded automatically.

### AI mode (model executes shots)

The model predicts shot parameters, the `GameController` executes them via mouse inputs, and you score the outcome on a 1–10 scale.

```bash
python -m sbg.interactive_trainer --model shot_model.pt --save shot_model.pt
```

High-scoring shots (≥ 6 by default) trigger an immediate online gradient update.  All scored shots are also saved as `.npz` files for offline retraining.

### Manual mode (you execute shots)

You play the game yourself.  The trainer automatically detects your parameters as you play:

| Parameter | How detected |
|---|---|
| **Direction** | Read from the hit-bar flag position in the screenshot |
| **Loft** | Scroll-wheel steps counted by a background mouse listener |
| **Power** | Left-mouse hold duration measured by the same listener |

No typing required — after your shot you only enter a score (1–10).

```bash
python -m sbg.interactive_trainer --manual --model shot_model.pt --save shot_model.pt
```

> **Note:** *pynput* must be installed for automatic loft/power detection.  If it is unavailable, only direction is auto-detected and you will be prompted to enter loft and power.
>
> **Loft baseline:** manual-mode loft is measured as scroll steps during the shot, so the starting loft is whatever the game is currently set to. For consistent labels, reset to a known loft (or correct it in the confirmation prompt) before each shot.

### Common options

| Flag | Default | Description |
|---|---|---|
| `--model PATH` | *(none)* | Load existing model weights |
| `--save PATH` | *(none)* | Save model after each online update |
| `--data-dir DIR` | `data/` | Directory for scored `.npz` samples |
| `--manual` | off | Enable manual training mode |
| `--no-online` | off | Save samples only, no live gradient updates |
| `--min-score N` | `6` | Min score for online update in AI mode |
| `--lr FLOAT` | `1e-4` | Online learning rate |
| `--monitor N` | `1` | Monitor index (1-based) |
| `--shot-timeout S` | `60` | Seconds to wait for a manual shot |

### API usage

```python
from sbg import InteractiveTrainer

trainer = InteractiveTrainer(
    model_path="shot_model.pt",
    data_dir="data/",
    online_learning=True,
    online_min_score=7,
    save_path="shot_model.pt",
)
trainer.run()            # AI mode
trainer.run(manual=True) # manual mode
```

## Training

Training data is stored as `.npz` files (one per sample).  Files produced by the interactive trainer include an optional `score` field that is used as a per-sample loss weight during offline retraining — high-quality shots (score 10) contribute ten times more than minimal shots (score 1).

| Key | Shape | Description |
|---|---|---|
| `image` | `(H, W, 3)` uint8 | Raw BGR game screenshot |
| `features` | `(11,)` float32 | `GameState.to_feature_vector()` |
| `direction` | scalar float | Ground-truth aim offset (degrees) |
| `power` | scalar float | Ground-truth power in [0, 1] |
| `loft` | scalar float | Ground-truth loft angle (degrees) |
| `score` | scalar float | *Optional* shot quality score (1–10); defaults to 5 |

```python
from sbg.train import train

model = train(
    data_dir="data/",
    save_path="shot_model.pt",
    epochs=100,
    batch_size=32,
    lr=1e-3,
)
```

## Key Concepts

### Hit Bar

The hit bar is the vertical UI element on the left side of the screen that encodes:
- **Flag position** – horizontal offset maps to aim direction; vertical position encodes distance to hole.
- **Terrain zones** – colour-coded rows indicate FAIRWAY, ROUGH_OOB, or WATER_OOB along the ball's flight path.

Shots without a detected flag are automatically ignored by the interactive trainer — only frames where the flag is clearly visible are recorded.

### Feature Vector (11 dimensions)

| Segment | Dims | Description |
|---|---|---|
| Hit-bar state | 11 | Visibility, flag position (x, y), terrain zone encoding |

### Model Outputs

| Output | Activation | Range |
|---|---|---|
| `direction_deg` | tanh × 45 | −45° to +45° |
| `power` | sigmoid | 0 to 1 |
| `loft_deg` | sigmoid × 90 | 0° to 90° |

### GameController Calibration

Two `ControllerConfig` parameters need to match your in-game settings:

- **`pixels_per_degree`** – how many pixels of mouse movement rotate the camera by 1°.  Adjust until the AI aims at the correct direction.
- **`max_power_hold_s`** – seconds of holding left-click that produces a 100% power shot.  Watch the power bar to calibrate.
- **`scroll_steps_per_degree`** – scroll clicks per 1° of loft.  This also drives the loft decoder in manual mode.

## Running Tests

```bash
pip install pytest
pytest tests/
```

## License

This project is provided as-is for educational and research purposes.
