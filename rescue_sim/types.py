from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


Position = Tuple[int, int]


class Terrain(Enum):
    ROAD = auto()
    RUBBLE = auto()
    BUILDING = auto()


class AgentType(Enum):
    SCOUT = "SCOUT"
    MEDIC = "MEDIC"
    LIFTER = "LIFTER"
    COORD = "COORD"


class HazardType(Enum):
    FIRE = "FIRE"
    GAS = "GAS"


class Severity(Enum):
    CRITICAL = 3
    MODERATE = 2
    MINOR = 1


class TaskType(Enum):
    SCAN = "SCAN"
    STABILIZE = "STABILIZE"
    EVACUATE = "EVACUATE"
    CLEAR = "CLEAR"
    RECHARGE = "RECHARGE"
    PATROL = "PATROL"


@dataclass(frozen=True)
class Action:
    name: str
    target: Position | None = None
