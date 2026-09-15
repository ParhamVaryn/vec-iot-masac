import numpy as np


def nearest_edge_index(vehicle, edges) -> int:
    """Phase-1 baseline: select the geographically nearest edge server."""
    distances = [
        np.linalg.norm(vehicle.position - edge.position)
        for edge in edges
    ]
    return int(np.argmin(distances))


def assign_tasks_directly(tasks, vehicles, edges) -> dict[int, int]:
    """
    Assign every new task to the nearest edge server.

    Returns:
        dict[task_id] = edge_id

    This is a baseline only. Later MASAC will replace this fixed policy and
    choose both destination and communication route.
    """
    vehicle_by_id = {v.vehicle_id: v for v in vehicles}
    assignments = {}

    for task in tasks:
        vehicle = vehicle_by_id[task.vehicle_id]
        edge_idx = nearest_edge_index(vehicle, edges)
        edges[edge_idx].queue.append(task)
        assignments[task.task_id] = edges[edge_idx].edge_id

    return assignments
