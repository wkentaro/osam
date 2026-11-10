import datetime
import math


def naturalsize(*, size: int) -> str:
    base = 1024
    prefixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    power = min(int(math.log(size, base)), len(prefixes) - 1) if size else 0
    return f"{size / base**power:.2f} {prefixes[power]}"


def naturaltime(*, dt: datetime.datetime, now: datetime.datetime | None = None) -> str:
    if now is None:
        now = datetime.datetime.now()
    delta = now - dt
    if delta.days > 0:
        return f"{delta.days} days ago"
    elif delta.seconds < 60:
        return f"{delta.seconds} seconds ago"
    elif delta.seconds < 3600:
        return f"{delta.seconds // 60} minutes ago"
    else:
        return f"{delta.seconds // 3600} hours ago"
