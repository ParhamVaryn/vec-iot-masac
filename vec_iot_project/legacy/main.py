from pathlib import Path

from src.config import SimulationConfig
from src.environment import VECEnvironment
from src.visualization import draw_environment, save_animation


def main():
    cfg = SimulationConfig()
    env = VECEnvironment(cfg)

    print("=== VEC Phase-1 Simulation ===")
    print(f"Vehicles: {cfg.num_vehicles}")
    print(f"Edge servers: {cfg.num_edges}")
    print(f"UAVs: {cfg.num_uavs}")
    print()

    for _ in range(cfg.num_steps):
        info = env.step()
        print(
            f"t={info['time_s']:>4.0f}s | "
            f"new_tasks={len(info['new_tasks']):>2} | "
            f"edge_queues={env.queue_lengths()}"
        )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    draw_environment(
        env,
        title="Phase 1: Vehicles, Edge Servers and Dynamic K-Means UAVs",
        save_path=output_dir / "final_snapshot.png",
        show=False,
    )

    # A separate fresh environment is used for the animation.
    animation_env = VECEnvironment(cfg)
    save_animation(
        animation_env,
        output_dir / "uav_mobility.gif",
        frames=30,
    )

    print()
    print("Created:")
    print(" - outputs/final_snapshot.png")
    print(" - outputs/uav_mobility.gif")


if __name__ == "__main__":
    main()
