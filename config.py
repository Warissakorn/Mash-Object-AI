"""Central configuration / default values for the vehicle Re-ID matcher.

Everything here is a plain default; the GUI and CLI expose the values that a
user is likely to want to change (thresholds, travel-time window, ...).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# YOLO weights file. `yolov8n.pt` (nano, ~6 MB) is downloaded automatically by
# Ultralytics on first use and then cached.
YOLO_WEIGHTS = "yolov8n.pt"

# COCO class ids that count as "a vehicle".
#   2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASS_IDS = (2, 3, 5, 7)
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Minimum detection confidence to keep a box.
DETECTION_CONFIDENCE = 0.35

# Discard tiny detections (likely noise / far-away vehicles) whose crop is
# smaller than this on the short side, in pixels.
MIN_CROP_SIZE = 24

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

# Input resolution fed to the embedder (H, W). ResNet expects 224x224-ish.
EMBED_INPUT_SIZE = (256, 128)  # (H, W) — a portrait-ish crop suits vehicles ok

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

# Regex used to pull a timestamp out of a filename. Named groups are combined
# with TIMESTAMP_FORMAT below. Default matches e.g. "A_20260723_101530.jpg".
TIMESTAMP_REGEX = r"(?P<ts>\d{8}_\d{6})"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# Cosine-similarity below this is never reported as a match.
SIMILARITY_THRESHOLD = 0.5

# Temporal gating: a B-detection may only match an A-detection when
#   MIN_TRAVEL_SECONDS <= (t_B - t_A) <= MAX_TRAVEL_SECONDS
# i.e. the vehicle must pass A before B, within a plausible travel time.
MIN_TRAVEL_SECONDS = 0
MAX_TRAVEL_SECONDS = 3600  # 1 hour

# How many candidate matches to return per query by default.
TOP_K = 5

# Supported image extensions when scanning a folder.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
