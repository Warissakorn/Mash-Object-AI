"""Load still frames from a folder and resolve each frame's timestamp.

Timestamp resolution order (first hit wins):
    1. Parse from the filename via a configurable regex + strptime format.
    2. Read EXIF DateTimeOriginal / DateTime (if Pillow can read it).
    3. Fall back to the file modification time (mtime).

Everything here is pure-Python + Pillow; no network, no ML — so it is cheap to
unit test.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:  # Pillow is a light dependency and always available in this project.
    from PIL import Image, ExifTags

    _EXIF_DATETIME_TAGS = {
        tag_id
        for tag_id, name in ExifTags.TAGS.items()
        if name in ("DateTimeOriginal", "DateTime")
    }
except Exception:  # pragma: no cover - Pillow missing is unusual
    Image = None  # type: ignore
    _EXIF_DATETIME_TAGS = set()

import config


@dataclass
class Frame:
    """A single still image plus the metadata we need for matching."""

    path: str
    timestamp: datetime
    timestamp_source: str  # "filename" | "exif" | "mtime"
    point: str = ""  # "A" / "B" — filled in by the caller

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)


def parse_timestamp_from_name(
    filename: str,
    regex: str = config.TIMESTAMP_REGEX,
    fmt: str = config.TIMESTAMP_FORMAT,
) -> Optional[datetime]:
    """Return the timestamp encoded in ``filename`` or ``None``.

    The regex must expose the value in a named group ``ts`` (the default), or
    else the whole match is used. ``fmt`` is a ``datetime.strptime`` format.
    """
    match = re.search(regex, filename)
    if not match:
        return None
    # Prefer the named group "ts"; fall back to the whole match if absent.
    if "ts" in match.groupdict():
        raw = match.group("ts")
    else:
        raw = match.group(0)
    try:
        return datetime.strptime(raw, fmt)
    except ValueError:
        return None


def _read_exif_timestamp(path: str) -> Optional[datetime]:
    if Image is None:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return None
            for tag_id in _EXIF_DATETIME_TAGS:
                value = exif.get(tag_id)
                if value:
                    # EXIF datetimes look like "2026:07:23 10:15:30".
                    try:
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    except (ValueError, TypeError):
                        continue
    except Exception:
        return None
    return None


def resolve_timestamp(
    path: str,
    regex: str = config.TIMESTAMP_REGEX,
    fmt: str = config.TIMESTAMP_FORMAT,
) -> tuple[datetime, str]:
    """Resolve a timestamp for ``path`` using the filename → EXIF → mtime chain.

    Returns ``(timestamp, source)`` where source is one of
    ``"filename"``, ``"exif"``, ``"mtime"``.
    """
    filename = os.path.basename(path)

    ts = parse_timestamp_from_name(filename, regex, fmt)
    if ts is not None:
        return ts, "filename"

    ts = _read_exif_timestamp(path)
    if ts is not None:
        return ts, "exif"

    return datetime.fromtimestamp(os.path.getmtime(path)), "mtime"


def load_folder(
    folder: str,
    point: str = "",
    regex: str = config.TIMESTAMP_REGEX,
    fmt: str = config.TIMESTAMP_FORMAT,
    extensions: tuple[str, ...] = config.IMAGE_EXTENSIONS,
) -> list[Frame]:
    """Scan ``folder`` for images and return sorted ``Frame`` objects.

    Frames are sorted by timestamp so downstream code can rely on ordering.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a folder: {folder}")

    ext_lower = tuple(e.lower() for e in extensions)
    frames: list[Frame] = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        if not name.lower().endswith(ext_lower):
            continue
        ts, source = resolve_timestamp(full, regex, fmt)
        frames.append(
            Frame(path=full, timestamp=ts, timestamp_source=source, point=point)
        )

    frames.sort(key=lambda f: f.timestamp)
    return frames
