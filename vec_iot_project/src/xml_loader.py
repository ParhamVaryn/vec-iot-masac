"""Streaming SUMO-style XML loaders.

The original project used ElementTree.parse(), which loads 200k+ vehicle rows
and 260k+ tasks into memory. The streaming functions below preserve the same
fields while keeping the dataset practical for RL experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, Tuple


@dataclass(frozen=True)
class VehicleRecord:
    time_s: float
    id: str
    x: float
    y: float
    speed: float
    priority: float


@dataclass(frozen=True)
class TaskRecord:
    time_s: float
    id: str
    creator: str
    data_size_mbit: float
    cycles_per_bit: float
    required_memory: float
    criticality: int
    deadline_s: float
    slack_s: float


def iter_vehicle_timesteps(path: str) -> Iterator[Tuple[float, Dict[str, VehicleRecord]]]:
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event != "end" or elem.tag != "timestep":
            continue
        time_s = float(elem.attrib["time"])
        vehicles: Dict[str, VehicleRecord] = {}
        for v in elem.findall("vehicle"):
            vehicles[v.attrib["id"]] = VehicleRecord(
                time_s=time_s,
                id=v.attrib["id"],
                x=float(v.attrib["x"]),
                y=float(v.attrib["y"]),
                speed=float(v.attrib["speed"]),
                priority=float(v.attrib.get("priority", 1.0)),
            )
        elem.clear()
        yield time_s, vehicles


def iter_task_timesteps(path: str) -> Iterator[Tuple[float, Tuple[TaskRecord, ...]]]:
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event != "end" or elem.tag != "timestep":
            continue
        time_s = float(elem.attrib["time"])
        tasks = []
        for t in elem.findall("task"):
            tasks.append(
                TaskRecord(
                    time_s=time_s,
                    id=t.attrib["id"],
                    creator=t.attrib["creator"],
                    data_size_mbit=float(t.attrib["data_size"]),
                    cycles_per_bit=float(t.attrib["cycles_per_bit"]),
                    required_memory=float(t.attrib["required_memory"]),
                    criticality=int(t.attrib.get("criticality_level", 1)),
                    deadline_s=float(t.attrib.get("deadline", time_s)),
                    slack_s=float(t.attrib.get("slack_time", 0.0)),
                )
            )
        elem.clear()
        yield time_s, tuple(tasks)


def load_vehicles(path: str):
    """Backward-compatible flattening loader."""
    rows = []
    for _, data in iter_vehicle_timesteps(path):
        rows.extend(data.values())
    return rows


def load_tasks(path: str):
    """Backward-compatible flattening loader."""
    rows = []
    for _, data in iter_task_timesteps(path):
        rows.extend(data)
    return rows
