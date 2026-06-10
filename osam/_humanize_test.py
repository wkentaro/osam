import datetime

import pytest

from ._humanize import naturalsize
from ._humanize import naturaltime


@pytest.mark.parametrize(
    "size, expected",
    [
        (0, "0B"),
        (1, "1.00 B"),
        (1023, "1023.00 B"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1024**2, "1.00 MB"),
        (1024**3, "1.00 GB"),
        (1024**4, "1.00 TB"),
        (1024**5, "1.00 PB"),
    ],
)
def test_naturalsize(size: int, expected: str) -> None:
    assert naturalsize(size) == expected


@pytest.mark.parametrize(
    "delta, expected",
    [
        (datetime.timedelta(days=2), "2 days ago"),
        (datetime.timedelta(hours=23), "23 hours ago"),
        (datetime.timedelta(hours=2), "2 hours ago"),
        (datetime.timedelta(seconds=3599), "59 minutes ago"),
        (datetime.timedelta(minutes=5), "5 minutes ago"),
        (datetime.timedelta(seconds=120), "2 minutes ago"),
        (datetime.timedelta(seconds=59), "59 seconds ago"),
        (datetime.timedelta(seconds=10), "10 seconds ago"),
        (datetime.timedelta(seconds=0), "0 seconds ago"),
    ],
)
def test_naturaltime(delta: datetime.timedelta, expected: str) -> None:
    now = datetime.datetime(2024, 1, 10, 12, 0, 0)
    assert naturaltime(now - delta, now=now) == expected


def test_naturaltime_default_now() -> None:
    dt = datetime.datetime.now() - datetime.timedelta(seconds=5)
    assert naturaltime(dt) == "5 seconds ago"
