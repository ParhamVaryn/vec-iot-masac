from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    # Map
    width_m: float = 1000.0
    height_m: float = 1000.0

    # Nodes
    num_vehicles: int = 60
    num_edges: int = 3
    num_uavs: int = 3

    # Time
    dt_s: float = 1.0
    num_steps: int = 40

    # Vehicle mobility
    vehicle_min_speed_mps: float = 4.0
    vehicle_max_speed_mps: float = 14.0
    turn_noise_std_rad: float = 0.12

    # UAV mobility / coverage
    uav_max_speed_mps: float = 35.0
    uav_coverage_radius_m: float = 220.0

    # Task generation
    task_generation_probability: float = 0.12
    task_min_size_mbit: float = 1.0
    task_max_size_mbit: float = 8.0
    task_min_cycles_giga: float = 0.3
    task_max_cycles_giga: float = 1.5

    # Edge compute capacities
    edge_cpu_ghz: tuple = (4.0, 5.0, 6.0)

    # Randomness
    random_seed: int = 42
