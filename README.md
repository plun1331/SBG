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
```

## Project Structure

```
SBG/
├── sbg/
│   ├── __init__.py          # Package exports
│   ├── game_state.py        # GameState, HitBarState, ShotParameters dataclasses
│   ├── live_overlay.py      # Live screen overlay + recording
│   ├── model.py             # ShotPredictorModel (PyTorch)
│   ├── screen_analyzer.py   # OpenCV-based screen parsing
│   ├── shot_predictor.py    # End-to-end predictor (ScreenAnalyzer + model)
│   └── train.py             # Training loop, ShotDataset, loss function
├── tests/
│   └── test_screen_analyzer.py
├── images/                  # Example annotated screenshots
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

### Train the model

Training data is stored as `.npz` files (one per sample) in a directory. Each file must contain:

| Key | Shape | Description |
|---|---|---|
| `image` | `(H, W, 3)` uint8 | Raw BGR game screenshot |
| `features` | `(11,)` float32 | `GameState.to_feature_vector()` |
| `direction` | scalar float | Ground-truth aim offset (degrees) |
| `power` | scalar float | Ground-truth power in [0, 1] |
| `loft` | scalar float | Ground-truth loft angle (degrees) |

```python
from sbg.train import train

model = train(
    data_dir="data/",
    output_path="shot_model.pt",
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
)
```

## Key Concepts

### Hit Bar

The hit bar is the vertical UI element on the left side of the screen that encodes:
- **Flag position** – horizontal offset maps to aim direction; vertical position encodes distance to hole.
- **Terrain zones** – colour-coded rows indicate FAIRWAY, ROUGH_OOB, or WATER_OOB along the ball's flight path.

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

## Running Tests

```bash
pip install pytest
pytest tests/
```

## License

This project is provided as-is for educational and research purposes.
