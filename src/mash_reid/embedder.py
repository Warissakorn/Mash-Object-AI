"""Appearance embedders: crop (BGR image) -> L2-normalized feature vector.

The `Embedder` base class defines the swappable interface. The default
implementation is `ResNet50Embedder` (torchvision, ImageNet weights) which is
reliable and downloads easily. To upgrade to a dedicated Re-ID model (OSNet via
`torchreid`, CLIP, ...) just implement `embed_batch` in a new subclass and pass
it into the pipeline — no other module changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

import config


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize row-wise. Accepts (N, D) or (D,) arrays."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors) + eps
        return vectors / norm
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + eps
    return vectors / norms


class Embedder(ABC):
    """Swappable appearance-embedding interface.

    An embedder turns a list of BGR image crops into an ``(N, dim)`` array of
    L2-normalized feature vectors, so that cosine similarity == dot product.
    """

    #: Dimensionality of the produced vectors (set by subclasses).
    dim: int = 0

    @abstractmethod
    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Return an ``(len(crops), dim)`` float32 array, L2-normalized."""

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Convenience wrapper for a single crop -> ``(dim,)`` vector."""
        return self.embed_batch([crop])[0]


class ResNet50Embedder(Embedder):
    """Default embedder: torchvision ResNet50 backbone (ImageNet), 2048-D.

    The classifier head is removed; the global-pooled 2048-D feature is used
    and L2-normalized. torch/torchvision are imported lazily so this module can
    be imported without them present.
    """

    dim = 2048

    def __init__(
        self,
        device: str | None = None,
        input_size: tuple[int, int] = config.EMBED_INPUT_SIZE,
    ) -> None:
        import torch
        import torchvision
        from torchvision.models import ResNet50_Weights

        self._torch = torch
        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        # Drop the final fc layer -> output is the 2048-D pooled feature.
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        self.model = model

        # ImageNet normalization constants.
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._mean = self._mean.to(self.device)
        self._std = self._std.to(self.device)

    def _preprocess(self, crops: Sequence[np.ndarray]):
        import cv2

        torch = self._torch
        h, w = self.input_size
        batch = np.empty((len(crops), h, w, 3), dtype=np.float32)
        for i, crop in enumerate(crops):
            # BGR (OpenCV) -> RGB, resize, scale to [0, 1].
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            batch[i] = resized.astype(np.float32) / 255.0
        # NHWC -> NCHW
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        tensor = (tensor - self._mean) / self._std
        return tensor

    def embed_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        tensor = self._preprocess(crops)
        with torch.no_grad():
            feats = self.model(tensor)
        feats_np = feats.cpu().numpy().astype(np.float32)
        return l2_normalize(feats_np)
