"""Unit tests for timestamp parsing and folder loading."""
import os
from datetime import datetime

import pytest

from mash_reid import frame_loader


def test_parse_default_pattern():
    ts = frame_loader.parse_timestamp_from_name("A_20260723_101530.jpg")
    assert ts == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_ignores_prefix_and_suffix():
    ts = frame_loader.parse_timestamp_from_name("cam-B-20260101_000000-frame12.png")
    assert ts == datetime(2026, 1, 1, 0, 0, 0)


def test_parse_returns_none_when_absent():
    assert frame_loader.parse_timestamp_from_name("no_timestamp_here.jpg") is None


def test_parse_custom_regex_and_format():
    ts = frame_loader.parse_timestamp_from_name(
        "shot@2026-07-23T10.15.30.jpg",
        regex=r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}\.\d{2}\.\d{2})",
        fmt="%Y-%m-%dT%H.%M.%S",
    )
    assert ts == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_regex_without_named_group_uses_whole_match():
    ts = frame_loader.parse_timestamp_from_name(
        "20260723_101530.jpg",
        regex=r"\d{8}_\d{6}",
    )
    assert ts == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_invalid_datetime_returns_none():
    # Matches the shape but 99 is not a valid month/day.
    assert frame_loader.parse_timestamp_from_name("A_20269999_101530.jpg") is None


def test_resolve_falls_back_to_mtime(tmp_path):
    f = tmp_path / "plain_image.jpg"
    f.write_bytes(b"not-a-real-jpeg")
    ts, source = frame_loader.resolve_timestamp(str(f))
    assert source == "mtime"
    assert isinstance(ts, datetime)


def test_resolve_prefers_filename(tmp_path):
    f = tmp_path / "A_20260723_101530.jpg"
    f.write_bytes(b"x")
    ts, source = frame_loader.resolve_timestamp(str(f))
    assert source == "filename"
    assert ts == datetime(2026, 7, 23, 10, 15, 30)


def test_load_folder_sorts_by_timestamp(tmp_path):
    names = [
        "A_20260723_101530.jpg",
        "A_20260723_100000.jpg",
        "A_20260723_120000.jpg",
        "ignore_me.txt",  # non-image, must be skipped
    ]
    for n in names:
        (tmp_path / n).write_bytes(b"x")

    frames = frame_loader.load_folder(str(tmp_path), point="A")
    assert [f.filename for f in frames] == [
        "A_20260723_100000.jpg",
        "A_20260723_101530.jpg",
        "A_20260723_120000.jpg",
    ]
    assert all(f.point == "A" for f in frames)


def test_load_folder_raises_on_missing_dir():
    with pytest.raises(NotADirectoryError):
        frame_loader.load_folder(os.path.join("definitely", "not", "here"))
