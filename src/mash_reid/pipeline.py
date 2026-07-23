"""Orchestration: a folder of frames -> per-vehicle detections + embeddings.

`Pipeline` ties together the frame loader, detector and embedder, and produces
a flat list of `VehicleRecord`s (one per detected vehicle across all frames of
a point). Heavy objects (detector / embedder) are created lazily so that
constructing a Pipeline is cheap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

import config
from . import frame_loader
from .detector import Detection, VehicleDetector
from .embedder import Embedder, ResNet50Embedder


@dataclass
class VehicleRecord:
    """A single detected vehicle, ready for matching."""

    point: str
    frame_path: str
    timestamp: datetime
    bbox: tuple[int, int, int, int]
    confidence: float
    class_name: str
    embedding: np.ndarray  # (dim,) L2-normalized
    crop: np.ndarray = field(repr=False, default=None)  # BGR crop (optional)

    @property
    def frame_name(self) -> str:
        return os.path.basename(self.frame_path)


class Pipeline:
    """Detect + embed every vehicle in a folder of frames."""

    def __init__(
        self,
        detector: Optional[VehicleDetector] = None,
        embedder: Optional[Embedder] = None,
        keep_crops: bool = True,
    ) -> None:
        self._detector = detector
        self._embedder = embedder
        self.keep_crops = keep_crops

    # -- lazy accessors -----------------------------------------------------
    @property
    def detector(self) -> VehicleDetector:
        if self._detector is None:
            self._detector = VehicleDetector()
        return self._detector

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = ResNet50Embedder()
        return self._embedder

    # -- main API -----------------------------------------------------------
    def process_folder(
        self,
        folder: str,
        point: str,
        regex: str = config.TIMESTAMP_REGEX,
        fmt: str = config.TIMESTAMP_FORMAT,
        progress=None,
    ) -> list[VehicleRecord]:
        """Return a `VehicleRecord` for every vehicle detected in ``folder``.

        ``progress`` — optional callback ``(done, total, message)`` for GUIs.
        """
        import cv2

        frames = frame_loader.load_folder(folder, point=point, regex=regex, fmt=fmt)
        records: list[VehicleRecord] = []
        total = len(frames)

        for i, frame in enumerate(frames):
            if progress:
                progress(i, total, f"[{point}] {frame.filename}")
            image = cv2.imread(frame.path)
            if image is None:
                continue
            detections = self.detector.detect(image)
            if not detections:
                continue
            crops = [d.crop for d in detections]
            embeddings = self.embedder.embed_batch(crops)
            for det, emb in zip(detections, embeddings):
                records.append(
                    VehicleRecord(
                        point=point,
                        frame_path=frame.path,
                        timestamp=frame.timestamp,
                        bbox=det.bbox,
                        confidence=det.confidence,
                        class_name=det.class_name,
                        embedding=emb,
                        crop=det.crop if self.keep_crops else None,
                    )
                )

        if progress:
            progress(total, total, f"[{point}] done: {len(records)} vehicles")
        return records


def stack_embeddings(records: list[VehicleRecord]) -> np.ndarray:
    """Stack the embeddings of a record list into an ``(N, dim)`` array."""
    if not records:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([r.embedding for r in records]).astype(np.float32)


def record_timestamps(records: list[VehicleRecord]) -> list[datetime]:
    return [r.timestamp for r in records]
