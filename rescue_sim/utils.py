import os
import random
from typing import Iterable, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def choice_weighted(rng: random.Random, items: list[Tuple[object, float]]):
    total = sum(w for _, w in items)
    r = rng.random() * total
    upto = 0.0
    for it, w in items:
        upto += w
        if upto >= r:
            return it
    return items[-1][0]


def manhattan_toroidal(a, b, w: int, h: int) -> int:
    dx = min(abs(a[0] - b[0]), w - abs(a[0] - b[0]))
    dy = min(abs(a[1] - b[1]), h - abs(a[1] - b[1]))
    return dx + dy


def avg(nums: Iterable[float]) -> float:
    nums = list(nums)
    return sum(nums) / len(nums) if nums else 0.0
