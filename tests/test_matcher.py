"""Unit tests for the matcher — synthetic embeddings, no network, no torch."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from mash_reid import matcher
from mash_reid.embedder import l2_normalize


def _t(seconds: int) -> datetime:
    return datetime(2026, 7, 23, 10, 0, 0) + timedelta(seconds=seconds)


def test_identical_vectors_have_similarity_one():
    v = l2_normalize(np.array([[1.0, 2.0, 3.0, 4.0]]))
    sim = matcher.cosine_similarity_matrix(v, v)
    assert sim.shape == (1, 1)
    assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_orthogonal_vectors_have_low_similarity():
    a = l2_normalize(np.array([[1.0, 0.0]]))
    b = l2_normalize(np.array([[0.0, 1.0]]))
    sim = matcher.cosine_similarity_matrix(a, b)
    assert sim[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_empty_inputs_yield_empty_matrix():
    empty = np.zeros((0, 4), dtype=np.float32)
    other = l2_normalize(np.ones((2, 4), dtype=np.float32))
    assert matcher.cosine_similarity_matrix(empty, other).shape == (0, 2)
    assert matcher.cosine_similarity_matrix(other, empty).shape == (2, 0)


def test_temporal_gate_allows_forward_travel_within_window():
    times_a = [_t(0)]
    times_b = [_t(100)]  # 100s later — plausible
    mask = matcher.temporal_gate_mask(
        times_a, times_b, min_travel_seconds=0, max_travel_seconds=300
    )
    assert mask[0, 0]


def test_temporal_gate_blocks_backward_and_too_late():
    times_a = [_t(500)]
    # B before A (backward) and B far too late.
    times_b = [_t(100), _t(10_000)]
    mask = matcher.temporal_gate_mask(
        times_a, times_b, min_travel_seconds=0, max_travel_seconds=300
    )
    assert not mask[0, 0]  # backward in time
    assert not mask[0, 1]  # exceeds max travel


def test_match_prefers_most_similar_and_respects_gate():
    # A has one query vehicle.
    a = l2_normalize(np.array([[1.0, 0.0, 0.0]]))
    times_a = [_t(0)]

    # B has three: b0 identical but too early (gated out),
    #              b1 identical and in-window (should win),
    #              b2 different and in-window.
    b = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],  # identical
                [1.0, 0.0, 0.0],  # identical
                [0.0, 1.0, 0.0],  # orthogonal
            ]
        )
    )
    times_b = [_t(-50), _t(120), _t(130)]

    result = matcher.match(
        a,
        b,
        times_a,
        times_b,
        similarity_threshold=0.5,
        min_travel_seconds=0,
        max_travel_seconds=300,
        top_k=5,
    )
    cands = result[0]
    assert len(cands) == 1  # only b1 passes gate + threshold
    assert cands[0].b_index == 1
    assert cands[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert cands[0].delta_seconds == pytest.approx(120.0)


def test_match_threshold_filters_weak_pairs():
    a = l2_normalize(np.array([[1.0, 0.0]]))
    b = l2_normalize(np.array([[0.0, 1.0]]))  # orthogonal -> sim 0
    times_a = [_t(0)]
    times_b = [_t(60)]
    result = matcher.match(
        a, b, times_a, times_b, similarity_threshold=0.5, max_travel_seconds=300
    )
    assert result[0] == []


def test_match_top_k_limits_results():
    a = l2_normalize(np.array([[1.0, 0.0]]))
    b = l2_normalize(np.tile([1.0, 0.0], (10, 1)))  # 10 identical matches
    times_a = [_t(0)]
    times_b = [_t(10 + i) for i in range(10)]
    result = matcher.match(
        a, b, times_a, times_b, similarity_threshold=0.5,
        max_travel_seconds=1000, top_k=3,
    )
    assert len(result[0]) == 3


def test_one_to_one_assignment_is_unique():
    # Two identical A vehicles, two identical B vehicles -> Hungarian must
    # assign them 1:1, not collapse both A onto the same B.
    a = l2_normalize(np.array([[1.0, 0.0], [1.0, 0.0]]))
    b = l2_normalize(np.array([[1.0, 0.0], [1.0, 0.0]]))
    times_a = [_t(0), _t(1)]
    times_b = [_t(50), _t(51)]
    matches = matcher.match_one_to_one(
        a, b, times_a, times_b, similarity_threshold=0.5, max_travel_seconds=300
    )
    assert len(matches) == 2
    assert len({m.a_index for m in matches}) == 2
    assert len({m.b_index for m in matches}) == 2
