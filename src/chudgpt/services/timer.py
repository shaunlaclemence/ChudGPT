from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    duration: float = 0.0


@contextmanager
def timer() -> Generator[Timer, None, None]:
    """Measure wall-clock duration of the enclosed block, in seconds.

    Usage:
        with timer() as t:
            do_something()
        print(t.duration)
    """
    t = Timer()
    start = time.monotonic()
    try:
        yield t
    finally:
        t.duration = time.monotonic() - start
