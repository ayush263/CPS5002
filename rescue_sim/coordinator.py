from __future__ import annotations
from typing import List, Optional
from .tasks import Task
from .types import TaskType, AgentType
from .utils import manhattan_toroidal


class CoordinatorAgent:
    """
    Assigns tasks based on global information (central dispatcher).
    This supports multi-agent coordination (requirement g/h).
    """

    def assign_tasks(self, world, cfg) -> List[Task]:
        tasks: List[Task] = []

        # Build tasks from active survivors (priority: critical > moderate > minor)
        for s in world.survivors.values():
            if s.dead or s.rescued:
                continue
            base = 100
            if s.severity.value == 3:
                base = 200
            elif s.severity.value == 2:
                base = 150
            # urgency increases as TTL decreases
            urgency = max(0, 120 - s.time_to_live)
            pr = base + urgency
            ttype = TaskType.STABILIZE if not s.stabilized else TaskType.EVACUATE
            tasks.append(Task(type=ttype, target=s.pos, priority=pr))

        # Create clearing tasks for blocked cells near survivors
        for s in world.survivors.values():
            if s.dead or s.rescued:
                continue
            for nb in world.neighbors(s.pos):
                if world.get_cell(nb).blocked:
                    tasks.append(Task(type=TaskType.CLEAR, target=nb, priority=80))

        # Always add some scan/patrol tasks (map expansion)
        tasks.append(Task(type=TaskType.SCAN, target=None, priority=40))
        tasks.append(Task(type=TaskType.PATROL, target=None, priority=30))

        # Sort by priority desc
        tasks.sort(key=lambda t: t.priority, reverse=True)
        return tasks

    def pick_task_for_agent(self, agent, tasks: List[Task], world) -> Optional[Task]:
        """
        Greedy assignment: choose best task compatible with agent type
        and closest among high priority tasks.
        """
        compatible = []
        for t in tasks:
            if t.done:
                continue
            if t.type in (TaskType.STABILIZE, TaskType.EVACUATE) and agent.type != AgentType.MEDIC:
                continue
            if t.type == TaskType.CLEAR and agent.type != AgentType.LIFTER:
                continue
            # scouts can scan/patrol
            # everyone can patrol/recharge (handled later)
            compatible.append(t)

        if not compatible:
            return None

        # pick among top N by priority, then nearest
        top = compatible[:10]
        best = None
        best_score = None
        for t in top:
            if t.target is None:
                # neutral
                dist = 5
            else:
                dist = manhattan_toroidal(agent.pos, t.target, world.width, world.height)
            score = -t.priority + dist * 2
            if best_score is None or score < best_score:
                best_score = score
                best = t
        return best
