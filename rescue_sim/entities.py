from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from .types import AgentType, Position, Severity


@dataclass
class Energy:
    max_energy: int
    value: int

    def spend(self, n: int) -> bool:
        if self.value >= n:
            self.value -= n
            return True
        return False

    def gain(self, n: int) -> None:
        self.value = min(self.max_energy, self.value + n)


@dataclass
class Integrity:
    max_integrity: int
    value: int

    def damage(self, n: int) -> None:
        self.value = max(0, self.value - n)

    def alive(self) -> bool:
        return self.value > 0


@dataclass
class Agent:
    id: int
    type: AgentType
    pos: Position
    energy: Energy
    integrity: Integrity
    carry_capacity: int
    sensor_range: int
    med_kits: int = 0
    rescued_count: int = 0
    hazard_entries: int = 0  
    current_task: Optional[object] = None
    policy: Optional[object] = None
    memory: Dict[str, Any] = field(default_factory=dict)

    def alive(self) -> bool:
        return self.integrity.alive()


@dataclass
class Survivor:
    id: int
    pos: Position
    severity: Severity
    time_to_live: int
    trapped: bool
    rescued: bool = False
    stabilized: bool = False
    dead: bool = False

    # def deteriorate(self) -> None:
    #     if self.dead or self.rescued:
    #         return
    #     self.time_to_live -= 1
    #     if self.time_to_live <= 0:
    #         self.dead = True

    def deteriorate(self) -> None:
        if self.dead or self.rescued:
            return
        if self.stabilized:
            if self.time_to_live % 2 == 0:
                self.time_to_live -= 1
        else:
            self.time_to_live -= 1

        if self.time_to_live <= 0:
            self.dead = True


    def stabilize(self) -> None:
        if self.dead or self.rescued:
            return
        self.stabilized = True
        self.time_to_live += 25

    def evacuate(self) -> None:
        if self.dead:
            return
        self.rescued = True
