import numpy as np


def move_vehicle(vehicle, cfg, rng: np.random.Generator) -> None:
    # Smooth random heading perturbation.
    vehicle.heading_rad += rng.normal(0.0, cfg.turn_noise_std_rad)

    dx = vehicle.speed_mps * cfg.dt_s * np.cos(vehicle.heading_rad)
    dy = vehicle.speed_mps * cfg.dt_s * np.sin(vehicle.heading_rad)

    vehicle.x += dx
    vehicle.y += dy

    # Reflect from map boundaries instead of teleporting.
    if vehicle.x < 0:
        vehicle.x = -vehicle.x
        vehicle.heading_rad = np.pi - vehicle.heading_rad
    elif vehicle.x > cfg.width_m:
        vehicle.x = 2 * cfg.width_m - vehicle.x
        vehicle.heading_rad = np.pi - vehicle.heading_rad

    if vehicle.y < 0:
        vehicle.y = -vehicle.y
        vehicle.heading_rad = -vehicle.heading_rad
    elif vehicle.y > cfg.height_m:
        vehicle.y = 2 * cfg.height_m - vehicle.y
        vehicle.heading_rad = -vehicle.heading_rad
