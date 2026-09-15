import numpy as np

from .entities import Vehicle, EdgeServer, UAV, Task
from .mobility import move_vehicle
from .uav_control import compute_traffic_centers, move_uavs_toward_centers
from .baseline import assign_tasks_directly


class VECEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed)
        self.time_s = 0.0
        self.step_index = 0
        self.next_task_id = 0

        self.vehicles = self._create_vehicles()
        self.edges = self._create_edges()
        self.uavs = self._create_uavs()

        self.last_cluster_centers = None
        self.last_task_assignments = {}

    def _create_vehicles(self):
        vehicles = []

        # Three initial traffic hotspots make K-Means behavior easy to see.
        hotspot_centers = np.array([
            [220.0, 250.0],
            [760.0, 260.0],
            [520.0, 760.0],
        ])

        for i in range(self.cfg.num_vehicles):
            center = hotspot_centers[i % len(hotspot_centers)]
            position = center + self.rng.normal(0.0, 85.0, size=2)
            position[0] = np.clip(position[0], 0, self.cfg.width_m)
            position[1] = np.clip(position[1], 0, self.cfg.height_m)

            vehicles.append(
                Vehicle(
                    vehicle_id=i,
                    x=float(position[0]),
                    y=float(position[1]),
                    speed_mps=float(
                        self.rng.uniform(
                            self.cfg.vehicle_min_speed_mps,
                            self.cfg.vehicle_max_speed_mps,
                        )
                    ),
                    heading_rad=float(self.rng.uniform(-np.pi, np.pi)),
                )
            )
        return vehicles

    def _create_edges(self):
        # Fixed infrastructure nodes.
        positions = [
            (150.0, 500.0),
            (850.0, 500.0),
            (500.0, 900.0),
        ]

        edges = []
        for i in range(self.cfg.num_edges):
            x, y = positions[i % len(positions)]
            cpu = self.cfg.edge_cpu_ghz[i % len(self.cfg.edge_cpu_ghz)]
            edges.append(EdgeServer(i, x, y, cpu))
        return edges

    def _create_uavs(self):
        # Start UAVs near the map center; K-Means control will reposition them.
        angles = np.linspace(0, 2 * np.pi, self.cfg.num_uavs, endpoint=False)
        radius = 70.0
        center = np.array([self.cfg.width_m / 2, self.cfg.height_m / 2])

        uavs = []
        for i, angle in enumerate(angles):
            p = center + radius * np.array([np.cos(angle), np.sin(angle)])
            uavs.append(
                UAV(
                    uav_id=i,
                    x=float(p[0]),
                    y=float(p[1]),
                    coverage_radius_m=self.cfg.uav_coverage_radius_m,
                )
            )
        return uavs

    def _generate_tasks(self):
        tasks = []
        for vehicle in self.vehicles:
            if self.rng.random() < self.cfg.task_generation_probability:
                tasks.append(
                    Task(
                        task_id=self.next_task_id,
                        vehicle_id=vehicle.vehicle_id,
                        created_at_s=self.time_s,
                        size_mbit=float(
                            self.rng.uniform(
                                self.cfg.task_min_size_mbit,
                                self.cfg.task_max_size_mbit,
                            )
                        ),
                        required_cycles_giga=float(
                            self.rng.uniform(
                                self.cfg.task_min_cycles_giga,
                                self.cfg.task_max_cycles_giga,
                            )
                        ),
                    )
                )
                self.next_task_id += 1
        return tasks

    def step(self):
        # 1) Vehicles move.
        for vehicle in self.vehicles:
            move_vehicle(vehicle, self.cfg, self.rng)

        # 2) Find current high-density traffic regions.
        vehicle_positions = np.array([v.position for v in self.vehicles])
        centers = compute_traffic_centers(
            vehicle_positions,
            self.cfg.num_uavs,
            self.cfg.random_seed + self.step_index,
        )
        self.last_cluster_centers = centers

        # 3) UAVs move physically toward those centers.
        move_uavs_toward_centers(self.uavs, centers, self.cfg)

        # 4) Vehicles probabilistically generate tasks.
        new_tasks = self._generate_tasks()

        # 5) Temporary baseline: new tasks go directly to nearest Edge.
        self.last_task_assignments = assign_tasks_directly(
            new_tasks, self.vehicles, self.edges
        )

        self.step_index += 1
        self.time_s += self.cfg.dt_s

        return {
            "time_s": self.time_s,
            "new_tasks": new_tasks,
            "assignments": self.last_task_assignments,
        }

    def queue_lengths(self):
        return [len(edge.queue) for edge in self.edges]
