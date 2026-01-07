from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import random

from .types import Action, TaskType, Position
from .utils import manhattan_toroidal


class Policy:
    def choose_action(self, state: str, rng: random.Random) -> str:
        raise NotImplementedError

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        pass


@dataclass
class RLPolicy(Policy):
    """
    Tabular Q-learning for high-level decisions.
    Only controls "task selection bias" (not low-level pathfinding).
    """
    epsilon: float = 0.10
    alpha: float = 0.25
    gamma: float = 0.90
    q: Dict[Tuple[str, str], float] = None

    def __post_init__(self):
        if self.q is None:
            self.q = {}

    def choose_action(self, state: str, rng: random.Random) -> str:
        actions = ["scan", "rescue", "stabilize", "clear", "recharge", "patrol"]
        if rng.random() < self.epsilon:
            return rng.choice(actions)
        # greedy
        qs = {a: self.q.get((state, a), 0.0) for a in actions}
        m = max(qs.values())
        best = [a for a, v in qs.items() if v == m]
        return rng.choice(best)

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        actions = ["scan", "rescue", "stabilize", "clear", "recharge", "patrol"]
        prev = self.q.get((state, action), 0.0)
        nxt = max(self.q.get((next_state, a), 0.0) for a in actions)
        self.q[(state, action)] = prev + self.alpha * (reward + self.gamma * nxt - prev)


def encode_state(agent, world, cfg) -> str:
    """
    Compact state encoding:
    - energy bucket
    - local hazard density bucket (within 1 step)
    - nearest survivor distance bucket (close/far)
    """
    e = agent.energy.value / agent.energy.max_energy
    e_bucket = "low" if e < 0.3 else "mid" if e < 0.7 else "high"

    # hazard within 1 move
    hazard_near = 0
    for nb in world.neighbors(agent.pos) + [agent.pos]:
        if world.get_cell(nb).hazard_ids:
            hazard_near += 1
    hz_bucket = "hz0" if hazard_near == 0 else "hz1" if hazard_near <= 2 else "hz2"

    # nearest survivor
    best = None
    for s in world.survivors.values():
        if s.dead or s.rescued:
            continue
        d = manhattan_toroidal(agent.pos, s.pos, world.width, world.height)
        best = d if best is None else min(best, d)
    if best is None:
        dist_bucket = "none"
    else:
        dist_bucket = "close" if best <= 3 else "far"

    return f"{e_bucket}|{hz_bucket}|{dist_bucket}"
