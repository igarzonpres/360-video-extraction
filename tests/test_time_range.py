import pytest

from time_range import parse_hhmmss, normalize_time_range


@pytest.mark.parametrize(
    "value, expected",
    [
        ("00:00:00", 0),
        ("00:00:05", 5),
        ("00:01:00", 60),
        ("01:00:00", 3600),
        ("12:34:56", 45296),
    ],
)
def test_parse_hhmmss_valid(value, expected):
    assert parse_hhmmss(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "1:00:00",
        "00:0:00",
        "00:00:60",
        "00:60:00",
        "abc",
        "99:99",
        "00:00:00:00",
    ],
)
def test_parse_hhmmss_invalid(value):
    with pytest.raises(ValueError):
        parse_hhmmss(value)


def test_normalize_time_range_defaults_to_full_video():
    assert normalize_time_range("", "") == (0, None)


def test_normalize_time_range_with_start_only():
    assert normalize_time_range("00:01:00", "") == (60, None)


def test_normalize_time_range_with_end_only():
    assert normalize_time_range("", "00:02:00") == (0, 120)


def test_normalize_time_range_with_both():
    assert normalize_time_range("00:01:00", "00:02:30") == (60, 150)


@pytest.mark.parametrize(
    "start, end",
    [
        ("00:02:00", "00:01:59"),
        ("00:02:00", "00:02:00"),
    ],
)
def test_normalize_time_range_requires_end_after_start(start, end):
    with pytest.raises(ValueError):
        normalize_time_range(start, end)
