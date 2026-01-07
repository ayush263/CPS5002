from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from .types import TaskType, Position


@dataclass
class Task:
    type: TaskType
    target: Optional[Position]
    priority: int
    assigned_agent_id: Optional[int] = None
    done: bool = False
