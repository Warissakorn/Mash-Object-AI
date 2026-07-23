# Mash-Object-AI — Vehicle Re-ID A/B Matcher

Verify whether a vehicle that passed **point A** is the **same vehicle** that
passed **point B** — using **visual appearance only** (re-identification
embeddings). **No license-plate reading.** The two points can be far apart and
have different camera angles.

- **Input:** two folders of still frames (one per point), each frame may contain
  **multiple vehicles**, each frame carries a timestamp (from its filename,
  EXIF, or file mtime).
- **Output:** for a chosen vehicle at A, the most visually similar vehicles at B,
  ranked by cosine similarity and filtered by a plausible travel-time window.
- **UI:** a simple Tkinter desktop app (plus a CLI for scripting/testing).

## Pipeline

```
Point A images ┐
               ├─► [1] Load frames + read timestamp
Point B images ┘        │
                        ▼
            [2] Detect every vehicle (YOLO) → crop each vehicle
                        │
                        ▼
            [3] Crop → feature embedding (Re-ID, swappable)
                        │
                        ▼
            [4] Match A×B: cosine similarity + temporal gating
                        │
                        ▼
            [5] Tkinter GUI: pick a vehicle at A → see best matches at B
```

| Stage | Module | Notes |
|-------|--------|-------|
| Frame loader | [`src/mash_reid/frame_loader.py`](src/mash_reid/frame_loader.py) | Timestamp from filename regex → EXIF → mtime |
| Detector | [`src/mash_reid/detector.py`](src/mash_reid/detector.py) | Ultralytics YOLO (`yolov8n.pt`), COCO car/motorcycle/bus/truck |
| Embedder | [`src/mash_reid/embedder.py`](src/mash_reid/embedder.py) | Swappable interface; default = ResNet50 (ImageNet), 2048-D, L2-normalized |
| Matcher | [`src/mash_reid/matcher.py`](src/mash_reid/matcher.py) | Cosine similarity + temporal gating + top-k / Hungarian |
| Pipeline | [`src/mash_reid/pipeline.py`](src/mash_reid/pipeline.py) | Folder → detections + embeddings |
| GUI | [`app/gui.py`](app/gui.py) | Tkinter desktop app (entry point) |
| CLI | [`cli.py`](cli.py) | Headless smoke test |

## Install

```bash
pip install -r requirements.txt
```

Model weights download automatically on first use and are then cached:
- YOLO `yolov8n.pt` (~6 MB)
- ResNet50 ImageNet weights (~100 MB)

> The default embedder (ImageNet ResNet50) is a solid, immediately-usable
> baseline. For maximum cross-camera accuracy, swap in a dedicated Re-ID model
> (e.g. OSNet via `torchreid`, or CLIP) by subclassing `Embedder` — no other
> module needs to change.

## Timestamp filename format

By default filenames are parsed with the regex `(?P<ts>\d{8}_\d{6})` and the
format `%Y%m%d_%H%M%S`, so these all work:

```
A_20260723_101530.jpg      → 2026-07-23 10:15:30
cam-B-20260723_101600.png  → 2026-07-23 10:16:00
```

If no timestamp is found in the name, EXIF `DateTimeOriginal` is tried, then the
file's modification time. Override the pattern via `--regex` / `--format`
(CLI) or `config.py`.

## Run the GUI

```bash
python app/gui.py
```

1. Choose the **Point A** and **Point B** folders.
2. Click **Process** (first run downloads the model weights).
3. Click a vehicle in the left gallery.
4. The right panel shows its best matches at B, with similarity score and time
   gap. Click a match thumbnail to view the full source frame with its bounding
   box drawn.
5. Drag the sliders (similarity threshold, min/max travel time) to re-rank
   live — the models are **not** re-run.

## Run the CLI

```bash
python cli.py --dir-a path/to/A --dir-b path/to/B --threshold 0.6
```

Useful options: `--min-travel` / `--max-travel` (seconds), `--top-k`,
`--one-to-one` (Hungarian one-to-one assignment), `--regex` / `--format`.

## Tests

The matcher and frame-loader tests need **no network and no ML models** — they
use synthetic embeddings and temp files:

```bash
python -m pytest tests/ -q
```

## Configuration

Defaults live in [`config.py`](config.py): vehicle classes, detection
confidence, embedding input size, timestamp regex/format, similarity threshold,
and travel-time window.

## Limitations

- ImageNet ResNet50 is a baseline; a dedicated Re-ID model will be more robust
  across large viewpoint changes. The `Embedder` interface is built for that
  upgrade.
- Matching is appearance-only by design — no plate reading.
- Temporal gating assumes vehicles pass A **before** B; set the window to match
  the real A→B travel time to cut false matches.
