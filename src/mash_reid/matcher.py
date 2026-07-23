"""Match vehicles across point A and point B.

Given per-detection embeddings and timestamps from two points, this computes
cosine similarity for every A x B pair, applies **temporal gating** (a B
detection may only match an A detection that occurred earlier, within a
plausible travel-time window), and ranks candidates.

Pure numpy + optional scipy (for one-to-one Hungarian assignment). No torch,
no network — trivially unit-testable with synthetic embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

import config


@dataclass
class MatchCandidate:
    """A ranked A->B match candidate."""

    a_index: int
    b_index: int
    similarity: float
    delta_seconds: float  # t_B - t_A


def cosine_similarity_matrix(
    emb_a: np.ndarray, emb_b: np.ndarray
) -> np.ndarray:
    """Return an ``(Na, Nb)`` cosine-similarity matrix.

    Inputs are expected to be L2-normalized already (the embedders guarantee
    this), so this is just a dot product; we normalize defensively anyway.
    """
    emb_a = np.asarray(emb_a, dtype=np.float32)
    emb_b = np.asarray(emb_b, dtype=np.float32)
    if emb_a.ndim != 2 or emb_b.ndim != 2:
        raise ValueError("embeddings must be 2-D (N, dim) arrays")
    if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
        return np.zeros((emb_a.shape[0], emb_b.shape[0]), dtype=np.float32)
    if emb_a.shape[1] != emb_b.shape[1]:
        raise ValueError("embedding dims differ between A and B")

    a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12)
    b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def temporal_gate_mask(
    times_a: Sequence[datetime],
    times_b: Sequence[datetime],
    min_travel_seconds: float = config.MIN_TRAVEL_SECONDS,
    max_travel_seconds: float = config.MAX_TRAVEL_SECONDS,
) -> np.ndarray:
    """Return a boolean ``(Na, Nb)`` mask of temporally-allowed pairs.

    Pair (a, b) is allowed iff
        min_travel_seconds <= (t_b - t_a) <= max_travel_seconds
    """
    na, nb = len(times_a), len(times_b)
    mask = np.zeros((na, nb), dtype=bool)
    ta = np.array([t.timestamp() for t in times_a], dtype=np.float64)
    tb = np.array([t.timestamp() for t in times_b], dtype=np.float64)
    if na == 0 or nb == 0:
        return mask
    delta = tb[None, :] - ta[:, None]  # (Na, Nb)
    mask = (delta >= min_travel_seconds) & (delta <= max_travel_seconds)
    return mask


def match(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    times_a: Sequence[datetime],
    times_b: Sequence[datetime],
    similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    min_travel_seconds: float = config.MIN_TRAVEL_SECONDS,
    max_travel_seconds: float = config.MAX_TRAVEL_SECONDS,
    top_k: int = config.TOP_K,
) -> dict[int, list[MatchCandidate]]:
    """Rank B candidates for each A detection.

    Returns a mapping ``a_index -> [MatchCandidate, ...]`` sorted by descending
    similarity, keeping at most ``top_k`` per query, filtered by both the
    temporal gate and ``similarity_threshold``.
    """
    sim = cosine_similarity_matrix(emb_a, emb_b)
    gate = temporal_gate_mask(
        times_a, times_b, min_travel_seconds, max_travel_seconds
    )
    # Blank out disallowed pairs.
    masked = np.where(gate, sim, -np.inf)

    delta = np.array(
        [[tb.timestamp() - ta.timestamp() for tb in times_b] for ta in times_a],
        dtype=np.float64,
    ) if len(times_a) and len(times_b) else np.zeros((len(times_a), len(times_b)))

    results: dict[int, list[MatchCandidate]] = {}
    na = sim.shape[0]
    for a_idx in range(na):
        row = masked[a_idx]
        # Candidate columns above threshold and inside the gate.
        valid = np.where((row >= similarity_threshold) & np.isfinite(row))[0]
        if valid.size == 0:
            results[a_idx] = []
            continue
        order = valid[np.argsort(-row[valid])][:top_k]
        results[a_idx] = [
            MatchCandidate(
                a_index=a_idx,
                b_index=int(b_idx),
                similarity=float(row[b_idx]),
                delta_seconds=float(delta[a_idx, b_idx]),
            )
            for b_idx in order
        ]
    return results


def match_one_to_one(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    times_a: Sequence[datetime],
    times_b: Sequence[datetime],
    similarity_threshold: float = config.SIMILARITY_THRESHOLD,
    min_travel_seconds: float = config.MIN_TRAVEL_SECONDS,
    max_travel_seconds: float = config.MAX_TRAVEL_SECONDS,
) -> list[MatchCandidate]:
    """Globally optimal one-to-one assignment via the Hungarian algorithm.

    Each A detection is matched to at most one B detection (and vice versa),
    maximizing total similarity subject to the temporal gate and threshold.
    Requires scipy.
    """
    from scipy.optimize import linear_sum_assignment

    sim = cosine_similarity_matrix(emb_a, emb_b)
    gate = temporal_gate_mask(
        times_a, times_b, min_travel_seconds, max_travel_seconds
    )
    if sim.size == 0:
        return []

    # Cost = -similarity for allowed pairs; a large cost for disallowed ones so
    # the solver avoids them.
    BIG = 1e6
    cost = np.where(gate, -sim, BIG)
    row_ind, col_ind = linear_sum_assignment(cost)

    ta = [t.timestamp() for t in times_a]
    tb = [t.timestamp() for t in times_b]

    out: list[MatchCandidate] = []
    for a_idx, b_idx in zip(row_ind, col_ind):
        if not gate[a_idx, b_idx]:
            continue
        s = float(sim[a_idx, b_idx])
        if s < similarity_threshold:
            continue
        out.append(
            MatchCandidate(
                a_index=int(a_idx),
                b_index=int(b_idx),
                similarity=s,
                delta_seconds=float(tb[b_idx] - ta[a_idx]),
            )
        )
    out.sort(key=lambda c: -c.similarity)
    return out
