"""Vehicle detector — a thin wrapper over Ultralytics YOLO.

The heavy `ultralytics` import is deferred until a detector is actually
constructed, so importing this module (e.g. during unit tests) is cheap and
does not require torch to be installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import config


@dataclass
class Detection:
    """One detected vehicle within a frame."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    class_name: str
    crop: np.ndarray  # BGR image crop (H, W, 3), uint8

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class VehicleDetector:
    """Detect vehicles in BGR images using YOLO.

    Parameters
    ----------
    weights:
        Path/name of the YOLO weights (auto-downloaded by Ultralytics).
    confidence:
        Minimum confidence to keep a detection.
    class_ids:
        COCO class ids treated as vehicles.
    min_crop_size:
        Detections whose shorter side is below this (pixels) are dropped.
    """

    def __init__(
        self,
        weights: str = config.YOLO_WEIGHTS,
        confidence: float = config.DETECTION_CONFIDENCE,
        class_ids: tuple[int, ...] = config.VEHICLE_CLASS_IDS,
        min_crop_size: int = config.MIN_CROP_SIZE,
    ) -> None:
        from ultralytics import YOLO  # deferred heavy import

        self.model = YOLO(weights)
        self.confidence = confidence
        self.class_ids = set(class_ids)
        self.min_crop_size = min_crop_size

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Return the list of vehicle detections in ``image_bgr``."""
        results = self.model.predict(
            source=image_bgr,
            conf=self.confidence,
            verbose=False,
        )
        detections: list[Detection] = []
        h, w = image_bgr.shape[:2]

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in self.class_ids:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                # Clamp to image bounds.
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if min(x2 - x1, y2 - y1) < self.min_crop_size:
                    continue
                crop = image_bgr[y1:y2, x1:x2].copy()
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=class_id,
                        class_name=config.VEHICLE_CLASS_NAMES.get(
                            class_id, str(class_id)
                        ),
                        crop=crop,
                    )
                )
        return detections
