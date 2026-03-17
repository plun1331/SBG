# SBG

Screen-analysis utilities for Super Battle Golf.

## Live overlay + recording

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
