import datetime
import math


def naturalsize(size: int) -> str:
    """Convert byte size to human readable format."""
    if size == 0:
        return "0B"
    base = 1024
    prefixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    power = int(math.log(size, base))
    prefix = prefixes[power]
    size /= base**power
    return f"{size:.2f} {prefix}"


def naturaltime(dt: datetime.datetime) -> str:
    """Convert datetime to human readable format."""
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
