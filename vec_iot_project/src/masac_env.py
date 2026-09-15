from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from .system_model import NetworkConfig, RouteCandidate, RouteMetrics, VECSystemModel
from .xml_loader import TaskRecord, VehicleRecord
from .uav_control import compute_traffic_centers, assign_centers_to_uavs


@dataclass
class EnvironmentConfig:
    width_m: float = 2000.0
    height_m: float = 2000.0
    num_edges: int = 10
    num_uavs: int = 5
    dt_s: float = 1.0
    uav_speed_mps: float = 35.0

    # Reward weights
    load_penalty: float = 0.40
    loss_penalty: float = 1.50
    energy_penalty: float = 0.15
    deadline_penalty: float = 2.00


class MASACVECEnv:
    """Task-level CTDE VEC environment.

    Key fixes in this version:
    - queues are served using CPU_hz * elapsed_time, not exponential decay;
    - `advance()` is intended to be called once per timestamp;
    - active task-generating vehicles are tracked separately from all cars on
      the map for radio contention;
    - local device queues are modeled as well as edge queues;
    - edge load is a real queued-work quantity in seconds;
    - LOCAL bitrate is NaN/N/A rather than infinity.
    """

    def __init__(
        self,
        vehicles_by_id: Dict[str, VehicleRecord],
        edge_positions=None,
        uav_positions=None,
        env_cfg: EnvironmentConfig = EnvironmentConfig(),
        net_cfg: NetworkConfig = NetworkConfig(),
    ):
        self.env_cfg = env_cfg
        self.net = VECSystemModel(net_cfg)
        self.vehicles_by_id = vehicles_by_id
        # Ten fixed edge servers, evenly distributed over the
        # 2000 m x 2000 m simulation area as a 5 x 2 grid.
        default_edge_positions = [
            (200.0, 500.0),    # E0
            (600.0, 500.0),    # E1
            (1000.0, 500.0),   # E2
            (1400.0, 500.0),   # E3
            (1800.0, 500.0),   # E4
            (200.0, 1500.0),   # E5
            (600.0, 1500.0),   # E6
            (1000.0, 1500.0),  # E7
            (1400.0, 1500.0),  # E8
            (1800.0, 1500.0),  # E9
        ]
        self.edges_xy = list(edge_positions) if edge_positions is not None else default_edge_positions
        if len(self.edges_xy) != int(self.env_cfg.num_edges):
            raise ValueError(
                f"Expected {self.env_cfg.num_edges} edge positions, "
                f"got {len(self.edges_xy)}"
            )
        # The UAV count is now controlled by EnvironmentConfig.num_uavs.
        # K=0 is supported for a true no-UAV experiment.  For K>0, create
        # exactly K initial positions; K-Means will move them toward traffic
        # hotspots as the simulation advances.
        k = int(self.env_cfg.num_uavs)
        if k < 0:
            raise ValueError("num_uavs must be >= 0")

        if uav_positions is not None:
            if len(uav_positions) != k:
                raise ValueError(
                    f"Expected {k} UAV positions for num_uavs={k}, "
                    f"got {len(uav_positions)}"
                )
            self.uavs_xy = [
                (float(x), float(y)) for x, y in uav_positions
            ]
        elif k == 0:
            self.uavs_xy = []
        else:
            # Deterministic, evenly spaced initialization around the map
            # center.  This avoids hard-coding exactly three UAV positions.
            cx = float(self.env_cfg.width_m) / 2.0
            cy = float(self.env_cfg.height_m) / 2.0
            radius = 0.20 * min(
                float(self.env_cfg.width_m),
                float(self.env_cfg.height_m),
            )
            angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
            self.uavs_xy = [
                (
                    float(np.clip(cx + radius * np.cos(a), 0.0, self.env_cfg.width_m)),
                    float(np.clip(cy + radius * np.sin(a), 0.0, self.env_cfg.height_m)),
                )
                for a in angles
            ]

        self.candidates = self.net.build_candidates(
            self.env_cfg.num_edges, k
        )

        self.edge_queue_cycles = np.zeros(self.env_cfg.num_edges, dtype=np.float64)
        self.local_queue_cycles: Dict[str, float] = {
            vid: 0.0 for vid in vehicles_by_id
        }
        self.active_task_creators = set(vehicles_by_id.keys())
        self.current_time_s = self._snapshot_time(vehicles_by_id)

        self.last_metrics: RouteMetrics | None = None
        self.last_assignment: RouteCandidate | None = None

    @staticmethod
    def _snapshot_time(vehicles_by_id):
        if not vehicles_by_id:
            return None
        return float(next(iter(vehicles_by_id.values())).time_s)

    @property
    def action_dim(self):
        return len(self.candidates)

    def set_active_task_creators(self, creator_ids):
        """Vehicles that actually generate tasks at the current timestamp."""
        self.active_task_creators = set(creator_ids)

    def _active_vehicle_positions(self):
        out = []
        for vid in self.active_task_creators:
            v = self.vehicles_by_id.get(vid)
            if v is not None:
                out.append((v.x, v.y))
        if not out:
            out = [(v.x, v.y) for v in self.vehicles_by_id.values()]
        return out

    def _serve_queues(self, elapsed_s: float):
        if elapsed_s <= 0:
            return

        cpus = np.asarray(self.net.cfg.edge_cpu_hz, dtype=float)
        for e in range(len(self.edge_queue_cycles)):
            served = cpus[e % len(cpus)] * elapsed_s
            self.edge_queue_cycles[e] = max(
                0.0, float(self.edge_queue_cycles[e]) - float(served)
            )

        local_served = float(self.net.cfg.local_cpu_hz) * elapsed_s
        for vid in list(self.local_queue_cycles):
            self.local_queue_cycles[vid] = max(
                0.0, float(self.local_queue_cycles[vid]) - local_served
            )

    def advance(self, vehicles_by_id: Dict[str, VehicleRecord]):
        """Advance mobility, CPU service and UAV positions to a new timestamp."""
        new_time = self._snapshot_time(vehicles_by_id)
        elapsed = 0.0
        if self.current_time_s is not None and new_time is not None:
            elapsed = max(0.0, new_time - self.current_time_s)
        self._serve_queues(elapsed)

        self.vehicles_by_id = vehicles_by_id
        self.current_time_s = new_time

        # Remove vanished vehicle queues; add newly appearing vehicles.
        live = set(vehicles_by_id)
        self.local_queue_cycles = {
            vid: self.local_queue_cycles.get(vid, 0.0) for vid in live
        }

        # Dynamic cooperative K-Means UAV positioning.
        # K=0 intentionally skips K-Means and represents the no-UAV case.
        k = int(self.env_cfg.num_uavs)
        if k > 0 and len(vehicles_by_id) >= k:
            pos = np.asarray(
                [(v.x, v.y) for v in vehicles_by_id.values()], dtype=float
            )
            seed = int(round(new_time or 0.0))
            centers = compute_traffic_centers(pos, k, seed=seed)
            current = np.asarray(self.uavs_xy, dtype=float)
            assignment = assign_centers_to_uavs(current, centers)
            max_step = self.env_cfg.uav_speed_mps * max(elapsed, self.env_cfg.dt_s)
            updated = []
            for i, center_idx in enumerate(assignment):
                delta = centers[center_idx] - current[i]
                dist = float(np.linalg.norm(delta))
                if dist == 0.0 or dist <= max_step:
                    p = centers[center_idx]
                else:
                    p = current[i] + delta / dist * max_step
                updated.append((float(p[0]), float(p[1])))
            self.uavs_xy = updated

    def edge_load_seconds(self) -> np.ndarray:
        """Queued processing workload at each edge in seconds."""
        cpus = np.asarray(self.net.cfg.edge_cpu_hz, dtype=float)
        return np.asarray(
            [
                self.edge_queue_cycles[e] / max(cpus[e % len(cpus)], 1.0)
                for e in range(len(self.edge_queue_cycles))
            ],
            dtype=float,
        )

    def _load_cv(self) -> float:
        loads = self.edge_load_seconds()
        positive_total = float(np.sum(loads))
        if positive_total <= 1e-12:
            return 0.0
        mean = float(np.mean(loads))
        return float(np.std(loads) / max(mean, 1e-12))

    def _global_features(self) -> np.ndarray:
        vs = list(self.vehicles_by_id.values())
        if not vs:
            veh_stats = np.zeros(6, np.float32)
        else:
            xy = np.asarray([[v.x, v.y] for v in vs], dtype=np.float32)
            sp = np.asarray([v.speed for v in vs], dtype=np.float32)
            veh_stats = np.array(
                [
                    xy[:, 0].mean() / self.env_cfg.width_m,
                    xy[:, 1].mean() / self.env_cfg.height_m,
                    xy[:, 0].std() / self.env_cfg.width_m,
                    xy[:, 1].std() / self.env_cfg.height_m,
                    sp.mean() / 20.0,
                    sp.std() / 20.0,
                ],
                np.float32,
            )

        edge_stats = []
        edge_loads = self.edge_load_seconds()
        for i, (x, y) in enumerate(self.edges_xy):
            cpu = self.net.cfg.edge_cpu_hz[i % len(self.net.cfg.edge_cpu_hz)]
            edge_stats.extend(
                [
                    x / self.env_cfg.width_m,
                    y / self.env_cfg.height_m,
                    min(edge_loads[i] / 10.0, 10.0),
                    cpu / 1e10,
                ]
            )

        uav_stats = []
        for x, y in self.uavs_xy:
            uav_stats.extend(
                [x / self.env_cfg.width_m, y / self.env_cfg.height_m]
            )
        return np.asarray(np.r_[veh_stats, edge_stats, uav_stats], dtype=np.float32)

    def observation(
        self, task: TaskRecord, vehicle: VehicleRecord
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_queue_s = self.local_queue_cycles.get(vehicle.id, 0.0) / max(
            self.net.cfg.local_cpu_hz, 1.0
        )

        local = np.array(
            [
                vehicle.x / self.env_cfg.width_m,
                vehicle.y / self.env_cfg.height_m,
                vehicle.speed / 20.0,
                min(vehicle.priority / 5.0, 1.0),
                min(task.data_size_mbit / 10.0, 1.0),
                min(task.cycles_per_bit / 5.0, 1.0),
                min(task.required_memory / 5000.0, 1.0),
                min(task.criticality / 5.0, 1.0),
                min(max(task.slack_s, 0.0) / 20.0, 1.0),
                min(max(task.deadline_s - task.time_s, 0.0) / 30.0, 1.0),
                min(local_queue_s / 10.0, 10.0),
            ],
            dtype=np.float32,
        )

        chans = []
        for exy in self.edges_xy:
            chans.append(self.net.path_gain((vehicle.x, vehicle.y), exy))
        for uxy in self.uavs_xy:
            chans.append(self.net.path_gain((vehicle.x, vehicle.y), uxy))
            chans.append(
                self.net.distance_m((vehicle.x, vehicle.y), uxy) / 2500.0
            )

        obs = np.r_[local, np.asarray(chans, np.float32)]
        state = np.r_[obs, self._global_features()]
        mask = self.net.valid_action_mask(
            vehicle,
            self.uavs_xy,
            self.edges_xy,
            self.candidates,
            task=task,
        )
        return obs.astype(np.float32), state.astype(np.float32), mask

    def evaluate_action(
        self, task: TaskRecord, vehicle: VehicleRecord, action: int
    ) -> RouteMetrics:
        candidate = self.candidates[action]
        return self.net.evaluate(
            task,
            vehicle,
            self.edges_xy,
            self.edge_queue_cycles,
            self.uavs_xy,
            candidate,
            self._active_vehicle_positions(),
            local_queue_cycles=self.local_queue_cycles.get(vehicle.id, 0.0),
        )

    def step_task(
        self, task: TaskRecord, vehicle: VehicleRecord, action: int
    ) -> Tuple[float, Dict]:
        mask = self.net.valid_action_mask(
            vehicle,
            self.uavs_xy,
            self.edges_xy,
            self.candidates,
            task=task,
        )
        if action < 0 or action >= len(mask) or not mask[action]:
            return -10.0, {"invalid": True, "success": 0.0}

        metric = self.evaluate_action(task, vehicle, action)
        self.last_metrics = metric
        self.last_assignment = self.candidates[action]

        cycles = self.net.cycles(task)
        if metric.final_edge is not None:
            self.edge_queue_cycles[int(metric.final_edge)] += cycles
        elif self.candidates[action].kind == "LOCAL":
            self.local_queue_cycles[vehicle.id] = (
                self.local_queue_cycles.get(vehicle.id, 0.0) + cycles
            )

        edge_loads = self.edge_load_seconds()
        imbalance = self._load_cv()

        deadline_window = float(max(task.deadline_s - task.time_s, 0.0))
        missed = (
            float(max(metric.latency_s - deadline_window, 0.0))
            if deadline_window > 0.0
            else 0.0
        )

        reward = -(
            metric.latency_s
            + self.env_cfg.energy_penalty * metric.energy_j
            + self.env_cfg.loss_penalty * metric.packet_loss
            + self.env_cfg.load_penalty * imbalance
            + self.env_cfg.deadline_penalty * missed
        )

        final_edge_load = float("nan")
        if metric.final_edge is not None:
            final_edge_load = float(edge_loads[int(metric.final_edge)])

        info = {
            "latency_s": float(metric.latency_s),
            "energy_j": float(metric.energy_j),
            "packet_loss": float(metric.packet_loss),
            "avg_rate_bps": float(metric.avg_rate_bps),
            "load_imbalance": float(imbalance),
            "deadline_miss_s": float(missed),
            "route": metric.route,
            "final_edge": metric.final_edge,
            "final_edge_load": final_edge_load,
            "success": float(missed <= 1e-12),
        }
        for edge_id, load_s in enumerate(edge_loads):
            info[f"edge{edge_id}_load"] = float(load_s)
        return float(reward), info
