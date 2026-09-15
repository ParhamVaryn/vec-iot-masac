import numpy as np
from sklearn.cluster import KMeans


def compute_traffic_centers(vehicle_positions: np.ndarray, num_uavs: int, seed: int) -> np.ndarray:
    """Return K-Means centers of the current vehicle positions."""
    model = KMeans(n_clusters=num_uavs, n_init=10, random_state=seed)
    model.fit(vehicle_positions)
    return model.cluster_centers_


def assign_centers_to_uavs(uav_positions: np.ndarray, centers: np.ndarray) -> list[int]:
    """
    Greedy one-to-one assignment.

    This prevents every UAV from chasing the same center. For the small
    number of UAVs in this course simulation, a greedy assignment is enough
    for Phase 1. Later this can be replaced by Hungarian matching.
    """
    remaining = set(range(len(centers)))
    assignments = []

    for uav_pos in uav_positions:
        best_index = min(
            remaining,
            key=lambda idx: float(np.linalg.norm(uav_pos - centers[idx]))
        )
        assignments.append(best_index)
        remaining.remove(best_index)

    return assignments


def move_uavs_toward_centers(uavs, centers: np.ndarray, cfg) -> None:
    current_positions = np.array([u.position for u in uavs])
    assignments = assign_centers_to_uavs(current_positions, centers)

    max_step = cfg.uav_max_speed_mps * cfg.dt_s

    for uav, center_idx in zip(uavs, assignments):
        target = centers[center_idx]
        delta = target - uav.position
        distance = float(np.linalg.norm(delta))

        if distance <= max_step or distance == 0:
            new_pos = target
        else:
            new_pos = uav.position + delta / distance * max_step

        uav.set_position(new_pos)
