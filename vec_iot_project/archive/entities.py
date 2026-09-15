from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Task:
    task_id: int
    vehicle_id: int
    created_at_s: float
    size_mbit: float
    required_cycles_giga: float


@dataclass
class Vehicle:
    vehicle_id: int
    x: float
    y: float
    speed_mps: float
    heading_rad: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class EdgeServer:
    edge_id: int
    x: float
    y: float
    cpu_ghz: float
    queue: List[Task] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class UAV:
    uav_id: int
    x: float
    y: float
    coverage_radius_m: float

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def set_position(self, p: np.ndarray) -> None:
        self.x = float(p[0])
        self.y = float(p[1])
