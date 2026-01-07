from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random

from .types import Position, Terrain
from .utils import clamp


@dataclass
class Cell:
    terrain: Terrain = Terrain.ROAD
    blocked: bool = False
    supplies: int = 0
    agent_ids: List[int] = field(default_factory=list)
    survivor_ids: List[int] = field(default_factory=list)
    hazard_ids: List[int] = field(default_factory=list)


class GridWorld:
    """
    20x20 toroidal grid. Holds cells plus registries for agents/survivors/hazards.
    """

    def __init__(self, width: int, height: int, rng: random.Random):
        self.width = width
        self.height = height
        self.rng = rng
        self.grid: List[List[Cell]] = [[Cell() for _ in range(width)] for _ in range(height)]

        self.agents: Dict[int, object] = {}
        self.survivors: Dict[int, object] = {}
        self.hazards: Dict[int, object] = {}

        self._generate_city_terrain()

    def wrap(self, pos: Position) -> Position:
        x, y = pos
        return x % self.width, y % self.height

    def get_cell(self, pos: Position) -> Cell:
        x, y = self.wrap(pos)
        return self.grid[y][x]

    def neighbors(self, pos: Position) -> List[Position]:
        x, y = pos
        return [self.wrap((x+1, y)), self.wrap((x-1, y)), self.wrap((x, y+1)), self.wrap((x, y-1))]

    def random_empty_pos(self) -> Position:
        # "empty" = not blocked + no survivor (agents can share)
        for _ in range(5000):
            p = (self.rng.randrange(self.width), self.rng.randrange(self.height))
            c = self.get_cell(p)
            if not c.blocked and not c.survivor_ids:
                return p
        return (0, 0)

    def _generate_city_terrain(self) -> None:
        """
        Simple procedural city: roads, rubble, buildings, some blocked cells.
        """
        for y in range(self.height):
            for x in range(self.width):
                r = self.rng.random()
                cell = self.grid[y][x]
                if r < 0.60:
                    cell.terrain = Terrain.ROAD
                elif r < 0.85:
                    cell.terrain = Terrain.RUBBLE
                else:
                    cell.terrain = Terrain.BUILDING

                # Some cells become blocked (collapsed)
                if cell.terrain in (Terrain.RUBBLE, Terrain.BUILDING):
                    if self.rng.random() < 0.12:
                        cell.blocked = True

    def add_supplies(self, pos: Position, amount: int) -> None:
        self.get_cell(pos).supplies += max(0, amount)

    def take_supplies(self, pos: Position, amount: int) -> int:
        cell = self.get_cell(pos)
        taken = min(cell.supplies, amount)
        cell.supplies -= taken
        return taken

    def move_agent(self, agent_id: int, new_pos: Position) -> bool:
        agent = self.agents[agent_id]
        old_pos = agent.pos
        new_pos = self.wrap(new_pos)

        if self.get_cell(new_pos).blocked:
            return False

        self.get_cell(old_pos).agent_ids.remove(agent_id)
        self.get_cell(new_pos).agent_ids.append(agent_id)
        agent.pos = new_pos
        return True

    def place_agent(self, agent) -> None:
        self.agents[agent.id] = agent
        self.get_cell(agent.pos).agent_ids.append(agent.id)

    def place_survivor(self, survivor) -> None:
        self.survivors[survivor.id] = survivor
        self.get_cell(survivor.pos).survivor_ids.append(survivor.id)

    def place_hazard(self, hazard) -> None:
        self.hazards[hazard.id] = hazard
        self.get_cell(hazard.pos).hazard_ids.append(hazard.id)

    def remove_survivor_from_cell(self, survivor_id: int, pos: Position) -> None:
        cell = self.get_cell(pos)
        if survivor_id in cell.survivor_ids:
            cell.survivor_ids.remove(survivor_id)

    def remove_hazard_from_cell(self, hazard_id: int, pos: Position) -> None:
        cell = self.get_cell(pos)
        if hazard_id in cell.hazard_ids:
            cell.hazard_ids.remove(hazard_id)

    def set_blocked(self, pos: Position, blocked: bool) -> None:
        self.get_cell(pos).blocked = blocked

    def count_blocked(self) -> int:
        return sum(1 for y in range(self.height) for x in range(self.width) if self.grid[y][x].blocked)

    def ascii_map(self, focus: Optional[Position] = None) -> str:
        """
        Debug map:
        . road, ^ rubble, # building, X blocked
        S survivor, A agent, H hazard (priority overlay)
        """
        out = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                c = self.grid[y][x]
                ch = "."
                if c.terrain == Terrain.RUBBLE:
                    ch = "^"
                elif c.terrain == Terrain.BUILDING:
                    ch = "#"
                if c.blocked:
                    ch = "X"
                if c.hazard_ids:
                    ch = "H"
                if c.survivor_ids:
                    ch = "S"
                if c.agent_ids:
                    ch = "A"
                if focus == pos:
                    ch = "@"
                row.append(ch)
            out.append("".join(row))
        return "\n".join(out)
