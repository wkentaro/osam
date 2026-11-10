import datetime
import math


def naturalsize(*, size: int) -> str:
    if size == 0:
        return "0B"
    base = 1024
    prefixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    power = int(math.log(size, base))
    prefix = prefixes[power]
    size /= base**power
    return f"{size:.2f} {prefix}"


def naturaltime(*, dt: datetime.datetime, now: datetime.datetime | None = None) -> str:
    if now is None:
        now = datetime.datetime.now()
    # A file time ahead of the clock reads as now, not as "23 hours ago".
    delta = max(now - dt, datetime.timedelta(0))
    count: int
    unit: str
    if delta.days > 0:
        count, unit = delta.days, "day"
    elif delta.seconds < 60:
        count, unit = delta.seconds, "second"
    elif delta.seconds < 3600:
        count, unit = delta.seconds // 60, "minute"
    else:
        count, unit = delta.seconds // 3600, "hour"
    return f"{count} {unit}{'' if count == 1 else 's'} ago"
