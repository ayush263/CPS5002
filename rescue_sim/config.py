from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    width: int = 20
    height: int = 20

    # Entities count
    n_scout: int = 2
    n_medic: int = 3
    n_lifter: int = 2

    n_survivors: int = 18
    n_supply_cells: int = 10

    # Hazards
    n_fire: int = 6
    n_gas: int = 4
    n_aftershock: int = 1 

    # Dynamics
    max_steps: int = 300

    # Energy/Integrity
    agent_max_energy: int = 120
    agent_max_integrity: int = 100

    # Survivor TTL range
    ttl_min = 120
    ttl_max = 220     

    # Hazard tuning
    fire_spread_chance = 0.06
    gas_spread_chance = 0.04
    aftershock_chance = 0.01

    
    # Damage tuning
    fire_damage_survivor = 4
    gas_damage_survivor = 3
    fire_damage_agent = 3
    gas_damage_agent = 2
 

    # Action costs
    move_cost: int = 2
    scan_cost: int = 3
    stabilize_cost: int = 6
    evacuate_cost: int = 10
    clear_debris_cost: int = 8
    recharge_gain: int = 20

    # Ethics scoring
    save_critical_bonus: float = 15.0
    save_moderate_bonus: float = 10.0
    save_minor_bonus: float = 6.0
    survivor_death_penalty: float = -4.0
    abandon_penalty: float = -5.0
    agent_loss_penalty: float = -6.0
    reckless_zone_penalty: float = -2.0 
