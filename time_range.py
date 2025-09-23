from __future__ import annotations

from typing import Optional, Tuple


def _coerce_time_text(value: str) -> str:
    if value is None:
        return ""
    return value.strip()


def parse_hhmmss(value: str) -> int:
    """Parse ``hh:mm:ss`` time strings into seconds.

    Raises ValueError for invalid formats or out-of-range minutes/seconds.
    """
    text = _coerce_time_text(value)
    if not text:
        raise ValueError("Time must be in hh:mm:ss format")

    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError("Time must be in hh:mm:ss format")

    hours_str, minutes_str, seconds_str = parts
    if not (hours_str.isdigit() and minutes_str.isdigit() and seconds_str.isdigit()):
        raise ValueError("Time must be numeric hh:mm:ss")

    if not (len(hours_str) == 2 and len(minutes_str) == 2 and len(seconds_str) == 2):
        raise ValueError("Time must use two digits per component")

    minutes = int(minutes_str)
    seconds = int(seconds_str)
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Minutes and seconds must be < 60")

    hours = int(hours_str)
    return hours * 3600 + minutes * 60 + seconds


def normalize_time_range(start_value: str, end_value: str) -> Tuple[int, Optional[int]]:
    """Validate start/end values and return seconds.

    Returns (start_seconds, end_seconds_or_none). Empty strings are allowed
    to indicate "from start" and/or "to end". Raises ValueError if end is
    not strictly greater than start.
    """
    start_text = _coerce_time_text(start_value)
    end_text = _coerce_time_text(end_value)

    start_seconds: Optional[int]
    end_seconds: Optional[int]

    start_seconds = parse_hhmmss(start_text) if start_text else 0
    end_seconds = parse_hhmmss(end_text) if end_text else None

    if end_seconds is not None and end_seconds <= start_seconds:
        raise ValueError("End time must be greater than start time")

    return start_seconds, end_seconds
