"""Train discrete MASAC on the temporal VEC-IoT traces.

This version advances the environment once per timestamp and uses the real next
VEC decision state in replay transitions. It also tracks only finite network
rates (LOCAL has no bitrate) and passes the active task-generating vehicles to
the radio model.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

from src.xml_loader import iter_task_timesteps, iter_vehicle_timesteps
from src.masac_env import MASACVECEnv, EnvironmentConfig
from src.masac import DiscreteMASAC, MASACConfig


def build_vehicle_index(path):
    return {t: v for t, v in iter_vehicle_timesteps(path)}


def build_task_groups(task_path, vehicle_index, max_tasks):
    groups = []
    total = 0
    for t, tasks in iter_task_timesteps(task_path):
        if t not in vehicle_index:
            continue
        aligned = tuple(task for task in tasks if task.creator in vehicle_index[t])
        if not aligned:
            continue
        if max_tasks and total + len(aligned) > max_tasks:
            aligned = aligned[: max_tasks - total]
        if aligned:
            groups.append((t, aligned))
            total += len(aligned)
        if max_tasks and total >= max_tasks:
            break
    return groups


def finite_mean(rows, key):
    vals = []
    for r in rows:
        if key not in r:
            continue
        try:
            v = float(r[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def write_training_log(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicles", default="datasets/vehicles.xml")
    ap.add_argument("--tasks", default="datasets/tasks.xml")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-tasks", type=int, default=15000)
    ap.add_argument("--save", default="outputs/masac.pt")
    ap.add_argument("--log", default="outputs/training_metrics.csv")
    ap.add_argument(
        "--num-uavs",
        type=int,
        default=5,
        help="Number of UAV relays (default: 5). Use 0 for the no-UAV case.",
    )
    args = ap.parse_args()

    if args.num_uavs < 0:
        ap.error("--num-uavs must be >= 0")
    env_cfg = EnvironmentConfig(num_uavs=args.num_uavs)

    vehicles = build_vehicle_index(args.vehicles)
    groups = build_task_groups(args.tasks, vehicles, args.max_tasks)
    if not groups:
        raise RuntimeError("No aligned task/vehicle events found.")

    t0, first_tasks = groups[0]
    first_task = first_tasks[0]
    env0 = MASACVECEnv(vehicles[t0], env_cfg=env_cfg)
    env0.set_active_task_creators(task.creator for task in first_tasks)
    obs0, state0, _ = env0.observation(first_task, vehicles[t0][first_task.creator])
    agent = DiscreteMASAC(len(obs0), len(state0), env0.action_dim, MASACConfig())

    print(f"MASAC device: {agent.device}")
    print(f"Number of UAVs (K): {args.num_uavs}")
    print(f"Number of edge servers: {env_cfg.num_edges}")
    print(f"Training timestamps: {len(groups)}")
    print(f"Tasks per episode: {sum(len(x[1]) for x in groups)}")

    episode_logs = []

    for ep in range(args.episodes):
        env = MASACVECEnv(vehicles[t0], env_cfg=env_cfg)
        recent = []
        pending = None

        for t, tasks in groups:
            env.advance(vehicles[t])
            env.set_active_task_creators(task.creator for task in tasks)

            for task in tasks:
                vehicle = vehicles[t][task.creator]
                obs, state, mask = env.observation(task, vehicle)

                # Complete the preceding replay transition with the actual
                # next decision state now that it is available.
                if pending is not None:
                    p_state, p_action, p_reward, p_mask = pending
                    agent.replay.add(
                        p_state,
                        p_action,
                        p_reward,
                        state,
                        False,
                        p_mask,
                        mask,
                    )
                    agent.update()

                # The 300 m hard restriction can legitimately leave a
                # memory-constrained task with no feasible route. Do not let
                # the policy silently choose a masked distant DIRECT action.
                if not mask.any():
                    if pending is not None:
                        p_state, p_action, p_reward, p_mask = pending
                        agent.replay.add(
                            p_state,
                            p_action,
                            p_reward,
                            state,
                            True,
                            p_mask,
                            p_mask,
                        )
                        agent.update()
                        pending = None

                    recent.append({
                        "latency_s": float("nan"),
                        "energy_j": 0.0,
                        "packet_loss": 1.0,
                        "avg_rate_bps": float("nan"),
                        "route": "NO_FEASIBLE_ROUTE",
                        "final_edge": None,
                        "success": 0.0,
                    })
                    continue

                action = agent.act(obs, mask, deterministic=False)
                reward, info = env.step_task(task, vehicle, action)
                recent.append(info)
                pending = (state, action, reward, mask)

        # Terminal transition for the last task of the episode.
        if pending is not None:
            p_state, p_action, p_reward, p_mask = pending
            agent.replay.add(
                p_state,
                p_action,
                p_reward,
                p_state,
                True,
                p_mask,
                p_mask,
            )
            agent.update()

        success_rate = (
            float(np.mean([float(r.get("success", 0.0)) for r in recent]))
            if recent else float("nan")
        )
        network_rows = [
            r for r in recent
            if np.isfinite(float(r.get("avg_rate_bps", float("nan"))))
        ]

        metrics = {
            "episode": ep + 1,
            "tasks": len(recent),
            "avg_latency_s": finite_mean(recent, "latency_s"),
            "avg_energy_j": finite_mean(recent, "energy_j"),
            "avg_packet_loss": finite_mean(network_rows, "packet_loss"),
            "avg_network_rate_mbps": finite_mean(network_rows, "avg_rate_bps") / 1e6,
            "success_rate_percent": success_rate * 100.0,
            "network_tasks": len(network_rows),
            "local_tasks": len(recent) - len(network_rows),
        }
        episode_logs.append(metrics)

        print(
            f"episode={ep+1:03d} tasks={len(recent):5d} "
            f"lat={metrics['avg_latency_s']:.4f}s "
            f"energy={metrics['avg_energy_j']:.4e}J "
            f"net_rate={metrics['avg_network_rate_mbps']:.3f}Mbps "
            f"loss={metrics['avg_packet_loss']:.4f} "
            f"success={metrics['success_rate_percent']:.2f}% "
            f"network={metrics['network_tasks']} local={metrics['local_tasks']} "
            f"replay={len(agent.replay)}"
        )

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    agent.save(args.save)
    write_training_log(args.log, episode_logs)
    print(f"Saved model: {args.save}")
    print(f"Saved training log: {args.log}")


if __name__ == "__main__":
    main()
