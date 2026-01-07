from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .types import HazardType, Position
from .utils import choice_weighted


@dataclass
class Hazard:
    id: int
    type: HazardType
    pos: Position
    intensity: int  # 1..N

    def damage_agent(self, cfg) -> int:
        if self.type == HazardType.FIRE:
            return cfg.fire_damage_agent * max(1, self.intensity)
        return cfg.gas_damage_agent * max(1, self.intensity)

    def damage_survivor(self, cfg) -> int:
        if self.type == HazardType.FIRE:
            return cfg.fire_damage_survivor * max(1, self.intensity)
        return cfg.gas_damage_survivor * max(1, self.intensity)
