from __future__ import annotations
import random
from typing import Dict, Any, Optional, List

from .config import SimConfig
from .types import AgentType, HazardType, Severity, TaskType, Action
from .world import GridWorld
from .entities import Agent, Survivor, Energy, Integrity
from .hazards import Hazard
from .tasks import Task
from .policies import RLPolicy, encode_state
from .coordinator import CoordinatorAgent
from .metrics import Metrics
from .utils import ensure_dir, manhattan_toroidal, choice_weighted, avg


class Simulation:
    """
    Disaster-Zone Rescue Simulation:
    - 20x20 toroidal city grid
    - rescue agents (scout/medic/lifter) + coordinator assignment
    - survivors with TTL
    - hazards that spread + aftershock events
    - resource constraints (energy, integrity, medkits)
    - ethical score
    - online learning (RLPolicy) influences agent behaviour
    - logs + plots support
    """

    def __init__(self, seed: int = 42, max_steps: int = 300, out_dir: str = "out/run1"):
        self.cfg = SimConfig(max_steps=max_steps)
        self.rng = random.Random(seed)
        self.out_dir = out_dir
        ensure_dir(out_dir)

        self.world = GridWorld(self.cfg.width, self.cfg.height, self.rng)
        self.metrics = Metrics()
        self.coordinator = CoordinatorAgent()

        self.step = 0
        self._next_id = 1

        # spawn all entities
        self._spawn_supplies()
        self._spawn_survivors()
        self._spawn_agents()
        self._spawn_hazards()

        # used for event-based scoring
        self._prev_dead = 0
        self._prev_agents_lost = 0

    def _id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # --------------------- spawning -----------------------------------------

    def _spawn_supplies(self):
        for _ in range(self.cfg.n_supply_cells):
            p = self.world.random_empty_pos()
            self.world.add_supplies(p, self.rng.randint(3, 8))

    def _spawn_survivors(self):
        for _ in range(self.cfg.n_survivors):
            p = self.world.random_empty_pos()
            severity = choice_weighted(
                self.rng,
                [
                    (Severity.CRITICAL, 0.25),
                    (Severity.MODERATE, 0.45),
                    (Severity.MINOR, 0.30),
                ],
            )
            ttl = self.rng.randint(self.cfg.ttl_min, self.cfg.ttl_max)
            trapped = (self.rng.random() < 0.35)
            s = Survivor(id=self._id(), pos=p, severity=severity, time_to_live=ttl, trapped=trapped)
            self.world.place_survivor(s)

    def _spawn_agents(self):
        # Scouts: scan/patrol bias
        for _ in range(self.cfg.n_scout):
            p = self.world.random_empty_pos()
            a = Agent(
                id=self._id(),
                type=AgentType.SCOUT,
                pos=p,
                energy=Energy(self.cfg.agent_max_energy, self.cfg.agent_max_energy),
                integrity=Integrity(self.cfg.agent_max_integrity, self.cfg.agent_max_integrity),
                carry_capacity=0,
                sensor_range=4,
                med_kits=0,
                policy=RLPolicy(epsilon=0.12),
            )
            self.world.place_agent(a)

        # Medics: stabilize/evacuate, medkits from supplies
        for _ in range(self.cfg.n_medic):
            p = self.world.random_empty_pos()
            a = Agent(
                id=self._id(),
                type=AgentType.MEDIC,
                pos=p,
                energy=Energy(self.cfg.agent_max_energy, self.cfg.agent_max_energy),
                integrity=Integrity(self.cfg.agent_max_integrity, self.cfg.agent_max_integrity),
                carry_capacity=1,
                sensor_range=3,
                med_kits=2,
                policy=RLPolicy(epsilon=0.10),
            )
            self.world.place_agent(a)

        # Lifters: clear debris
        for _ in range(self.cfg.n_lifter):
            p = self.world.random_empty_pos()
            a = Agent(
                id=self._id(),
                type=AgentType.LIFTER,
                pos=p,
                energy=Energy(self.cfg.agent_max_energy, self.cfg.agent_max_energy),
                integrity=Integrity(self.cfg.agent_max_integrity, self.cfg.agent_max_integrity),
                carry_capacity=0,
                sensor_range=3,
                med_kits=0,
                policy=RLPolicy(epsilon=0.10),
            )
            self.world.place_agent(a)

    def _spawn_hazards(self):
        for _ in range(self.cfg.n_fire):
            p = self.world.random_empty_pos()
            h = Hazard(id=self._id(), type=HazardType.FIRE, pos=p, intensity=self.rng.randint(1, 2))
            self.world.place_hazard(h)

        for _ in range(self.cfg.n_gas):
            p = self.world.random_empty_pos()
            h = Hazard(id=self._id(), type=HazardType.GAS, pos=p, intensity=self.rng.randint(1, 2))
            self.world.place_hazard(h)

    # --------------------- simulation core ----------------------------------

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        while self.step < self.cfg.max_steps and not self._done():
            if verbose and self.step % 25 == 0:
                print(f"\n=== Step {self.step} ===")
                print(self.world.ascii_map())
            self.tick()

        self.metrics.export_csv(self.out_dir)

        return {
            "steps": self.step,
            "survivors_saved": self.metrics.survivors_saved,
            "survivors_dead": self.metrics.survivors_dead,
            "agents_lost": self.metrics.agents_lost,
            "ethical_score": round(self.metrics.ethical_score, 2),
        }

    def _done(self) -> bool:
        # done if all survivors are rescued or dead, or all agents lost
        active_survivors = [s for s in self.world.survivors.values() if not s.dead and not s.rescued]
        living_agents = [a for a in self.world.agents.values() if a.alive()]
        return (len(active_survivors) == 0) or (len(living_agents) == 0)

    def tick(self) -> None:
        self.step += 1

        # 1) coordinator generates tasks
        tasks = self.coordinator.assign_tasks(self.world, self.cfg)

        # 2) each agent acts
        for agent in list(self.world.agents.values()):
            if not agent.alive():
                continue

            state = encode_state(agent, self.world, self.cfg)
            high_action = agent.policy.choose_action(state, self.rng) if agent.policy else "patrol"

            chosen_task = self.coordinator.pick_task_for_agent(agent, tasks, self.world)

            chosen_task = self._apply_rl_bias(agent, chosen_task, high_action)

            agent.current_task = chosen_task
            prev_state = state

            reward_before = (self.metrics.ethical_score, self.metrics.survivors_saved, self.metrics.agents_lost)

            self._execute_task(agent, chosen_task)

            # 3) survivors deteriorate
            for s in self.world.survivors.values():
                prev_dead = s.dead
                s.deteriorate()
                if not prev_dead and s.dead:
                    self.metrics.survivors_dead += 1
                    self.metrics.ethical_score += self.cfg.survivor_death_penalty

            # 4) hazards evolve (spread + aftershock)
            self._hazards_step()

            # 5) apply hazard damage on co-located agents and survivors
            self._apply_hazard_damage()

            # 6) RL update (local reward shaping)
            next_state = encode_state(agent, self.world, self.cfg)
            reward_after = (self.metrics.ethical_score, self.metrics.survivors_saved, self.metrics.agents_lost)
            r = self._compute_agent_reward(reward_before, reward_after, agent)
            if agent.policy:
                agent.policy.update(prev_state, high_action, r, next_state)

        # 7) log step metrics for plots
        self._log_step()

        # 8) remove dead agents count (integrity==0)
        self._recount_agents_lost()

    # --------------------- decision helpers ----------------------------------

    def _apply_rl_bias(self, agent: Agent, task: Optional[Task], high_action: str) -> Optional[Task]:
        # Hard safety: if energy low, prioritize recharge
        if agent.energy.value < int(agent.energy.max_energy * 0.20):
            return Task(type=TaskType.RECHARGE, target=None, priority=999)

        # RL bias for different high-level modes
        if high_action == "scan" and agent.type == AgentType.SCOUT:
            return Task(type=TaskType.SCAN, target=None, priority=80)
        if high_action == "clear" and agent.type == AgentType.LIFTER:
            target = self._nearest_blocked(agent.pos)
            return Task(type=TaskType.CLEAR, target=target, priority=90) if target else task
        if high_action == "stabilize" and agent.type == AgentType.MEDIC:
            target = self._nearest_survivor(agent.pos, want_stabilize=True)
            return Task(type=TaskType.STABILIZE, target=target, priority=95) if target else task
        if high_action == "rescue" and agent.type == AgentType.MEDIC:
            target = self._nearest_survivor(agent.pos, want_stabilize=False)
            return Task(type=TaskType.EVACUATE, target=target, priority=90) if target else task
        if high_action == "patrol":
            return Task(type=TaskType.PATROL, target=None, priority=20)

        return task

    def _nearest_survivor(self, pos, want_stabilize: bool):
        best = None
        best_d = None
        for s in self.world.survivors.values():
            if s.dead or s.rescued:
                continue
            if want_stabilize and s.stabilized:
                continue
            if (not want_stabilize) and (not s.stabilized):
                continue
            d = manhattan_toroidal(pos, s.pos, self.world.width, self.world.height)
            if best_d is None or d < best_d:
                best, best_d = s.pos, d
        return best

    def _nearest_blocked(self, pos):
        best = None
        best_d = None
        for y in range(self.world.height):
            for x in range(self.world.width):
                c = self.world.grid[y][x]
                if not c.blocked:
                    continue
                p = (x, y)
                d = manhattan_toroidal(pos, p, self.world.width, self.world.height)
                if best_d is None or d < best_d:
                    best, best_d = p, d
        return best

    # --------------------- task execution ------------------------------------

    def _execute_task(self, agent: Agent, task: Optional[Task]) -> None:
        if task is None:
            task = Task(type=TaskType.PATROL, target=None, priority=0)

        if task.type == TaskType.RECHARGE:
            self._do_recharge(agent)
            return

        if task.type == TaskType.SCAN:
            self._do_scan(agent)
            return

        if task.type == TaskType.PATROL:
            self._do_patrol(agent)
            return

        # tasks with targets: move toward then act if on target
        if task.target is not None:
            self._move_towards(agent, task.target)

            if agent.pos == task.target:
                if task.type == TaskType.STABILIZE:
                    self._do_stabilize(agent)
                elif task.type == TaskType.EVACUATE:
                    self._do_evacuate(agent)
                elif task.type == TaskType.CLEAR:
                    self._do_clear(agent)

    def _move_towards(self, agent: Agent, target):
        if not agent.energy.spend(self.cfg.move_cost):
            return
        x, y = agent.pos
        tx, ty = target

        candidates = self.world.neighbors(agent.pos)
        best = min(
            candidates,
            key=lambda p: manhattan_toroidal(p, target, self.world.width, self.world.height),
        )
        moved = self.world.move_agent(agent.id, best)
        if not moved:
            self.metrics.ethical_score += self.cfg.reckless_zone_penalty

    def _do_scan(self, agent: Agent):
        if agent.type != AgentType.SCOUT:
            return
        if not agent.energy.spend(self.cfg.scan_cost):
            return
        # scanning can "discover" supplies or reduce blocked probability nearby (represents mapping)
        for nb in self.world.neighbors(agent.pos) + [agent.pos]:
            c = self.world.get_cell(nb)
            if c.supplies == 0 and self.rng.random() < 0.08:
                c.supplies += self.rng.randint(1, 3)

    def _do_patrol(self, agent: Agent):
        if not agent.energy.spend(self.cfg.move_cost):
            return
        nbs = self.world.neighbors(agent.pos)
        self.world.move_agent(agent.id, self.rng.choice(nbs))

    def _do_recharge(self, agent: Agent):
        # Recharge at any location (represents portable solar station / base link), but better on supply cells
        bonus = self.cfg.recharge_gain
        if self.world.get_cell(agent.pos).supplies > 0:
            bonus += 5
        agent.energy.gain(bonus)

        # also pickup supplies (medkits) if medic
        if agent.type == AgentType.MEDIC:
            got = self.world.take_supplies(agent.pos, 1)
            agent.med_kits += got

    def _do_stabilize(self, agent: Agent):
        if agent.type != AgentType.MEDIC:
            return
        if not agent.energy.spend(self.cfg.stabilize_cost):
            return

        # find survivor on cell
        cell = self.world.get_cell(agent.pos)
        if not cell.survivor_ids:
            return
        sid = cell.survivor_ids[0]
        s = self.world.survivors[sid]
        if s.dead or s.rescued or s.stabilized:
            return
        if agent.med_kits <= 0:
            # ethics penalty: arrived without resources (planning failure)
            self.metrics.ethical_score += self.cfg.abandon_penalty
            return

        agent.med_kits -= 1
        s.stabilize()

    def _do_evacuate(self, agent: Agent):
        if agent.type != AgentType.MEDIC:
            return
        if not agent.energy.spend(self.cfg.evacuate_cost):
            return

        cell = self.world.get_cell(agent.pos)
        if not cell.survivor_ids:
            return
        sid = cell.survivor_ids[0]
        s = self.world.survivors[sid]
        if s.dead or s.rescued:
            return
        # evacuation preferred if stabilized (otherwise risk)
        if not s.stabilized:
            self.metrics.ethical_score += self.cfg.reckless_zone_penalty

        s.evacuate()
        self.world.remove_survivor_from_cell(s.id, s.pos)
        self.metrics.survivors_saved += 1
        agent.rescued_count += 1

        # ethics: severity-based rewards
        if s.severity == Severity.CRITICAL:
            self.metrics.ethical_score += self.cfg.save_critical_bonus
        elif s.severity == Severity.MODERATE:
            self.metrics.ethical_score += self.cfg.save_moderate_bonus
        else:
            self.metrics.ethical_score += self.cfg.save_minor_bonus

    def _do_clear(self, agent: Agent):
        if agent.type != AgentType.LIFTER:
            return
        if not agent.energy.spend(self.cfg.clear_debris_cost):
            return
        # clearing means unblocking the cell (if blocked)
        c = self.world.get_cell(agent.pos)
        if c.blocked:
            c.blocked = False

    # --------------------- hazards -------------------------------------------

    def _hazards_step(self):
        # spread
        for h in list(self.world.hazards.values()):
            if h.type == HazardType.FIRE:
                if self.rng.random() < self.cfg.fire_spread_chance:
                    self._spread_hazard(h)
            elif h.type == HazardType.GAS:
                if self.rng.random() < self.cfg.gas_spread_chance:
                    self._spread_hazard(h)

        # aftershock event: can create new blocked cells or intensify hazards
        if self.rng.random() < self.cfg.aftershock_chance:
            self._aftershock()

    def _spread_hazard(self, hazard: Hazard):
        if len(self.world.hazards) >= 80:
            return
        # choose neighbor cell that isn't blocked; if already hazard, intensify
        nbs = self.world.neighbors(hazard.pos)
        target = self.rng.choice(nbs)
        cell = self.world.get_cell(target)
        if cell.blocked:
            return
        # check if hazard of same type already exists
        existing = None
        for hid in cell.hazard_ids:
            h2 = self.world.hazards[hid]
            if h2.type == hazard.type:
                existing = h2
                break
        if existing:
            existing.intensity = min(2, existing.intensity + 1)
        else:
            hnew = Hazard(id=self._id(), type=hazard.type, pos=target, intensity=1)
            self.world.place_hazard(hnew)

    def _aftershock(self):
        # pick random cells: some become blocked, hazards intensify
        for _ in range(self.rng.randint(3, 7)):
            p = (self.rng.randrange(self.world.width), self.rng.randrange(self.world.height))
            c = self.world.get_cell(p)
            if not c.blocked and self.rng.random() < 0.35:
                c.blocked = True
            if c.hazard_ids and self.rng.random() < 0.5:
                hid = self.rng.choice(c.hazard_ids)
                self.world.hazards[hid].intensity = min(3, self.world.hazards[hid].intensity + 1)

    def _apply_hazard_damage(self):
        # damage agents and survivors on hazard cells
        for h in list(self.world.hazards.values()):
            cell = self.world.get_cell(h.pos)

            # agents
            for aid in list(cell.agent_ids):
                a = self.world.agents[aid]
                if not a.alive():
                    continue
                a.integrity.damage(h.damage_agent(self.cfg))
                a.hazard_entries += 1
                # ethics: repeated hazard entry indicates risky behaviour
                if a.hazard_entries % 3 == 0:
                    self.metrics.ethical_score += self.cfg.reckless_zone_penalty

            # survivors
            for sid in list(cell.survivor_ids):
                s = self.world.survivors[sid]
                if s.dead or s.rescued:
                    continue
                # hazards reduce TTL quickly (proxy damage)
                s.time_to_live -= max(1, h.damage_survivor(self.cfg) // 4)
                if s.time_to_live <= 0 and not s.dead:
                    s.dead = True
                    self.metrics.survivors_dead += 1
                    self.metrics.ethical_score += self.cfg.survivor_death_penalty

    # --------------------- metrics/logging -----------------------------------

    def _recount_agents_lost(self):
        lost = 0
        for a in self.world.agents.values():
            if not a.alive():
                lost += 1
                
        if lost > self._prev_agents_lost:
            diff = lost - self._prev_agents_lost
            self.metrics.agents_lost += diff
            self.metrics.ethical_score += self.cfg.agent_loss_penalty * diff
        self._prev_agents_lost = lost

    def _log_step(self):
        alive_survivors = sum(1 for s in self.world.survivors.values() if not s.dead and not s.rescued)
        dead_survivors = sum(1 for s in self.world.survivors.values() if s.dead)
        total_hazard_intensity = sum(h.intensity for h in self.world.hazards.values())
        avg_energy = avg([a.energy.value for a in self.world.agents.values() if a.alive()])

        row = {
            "step": self.step,
            "saved": self.metrics.survivors_saved,
            "dead": dead_survivors,
            "alive": alive_survivors,
            "agents_lost": self.metrics.agents_lost,
            "ethical_score": round(self.metrics.ethical_score, 4),
            "total_hazard_intensity": float(total_hazard_intensity),
            "avg_agent_energy": round(avg_energy, 2),
        }
        self.metrics.log_step(row)

    def _compute_agent_reward(self, before, after, agent: Agent) -> float:
        # local reward shaping: ethics gain + saved gain - loss penalties
        ethics0, saved0, lost0 = before
        ethics1, saved1, lost1 = after
        r = (ethics1 - ethics0) + (saved1 - saved0) * 5.0 - (lost1 - lost0) * 6.0
        # small penalty if energy too low (encourage recharging)
        if agent.energy.value < int(agent.energy.max_energy * 0.15):
            r -= 1.0
        return r
