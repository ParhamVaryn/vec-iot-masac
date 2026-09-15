"""MASAC-only evaluation for the VEC-IoT project.

Outputs the mandatory project metrics without a nearest-edge baseline:
- total latency;
- matched UAV-relay versus non-UAV communication scenario latency;
- total energy;
- packet loss and network bitrate;
- actual load distribution among edge servers;
- task success rate;
- UAV presence and positions.

LOCAL execution is excluded from bitrate averages because it has no network
transmission. The no-UAV scenario comparison uses DIRECT and WIRED_1HOP routes,
not LOCAL. Only tasks feasible in BOTH scenario families are retained, so the
UAV and no-UAV bars use the exact same matched task set.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np

from src.xml_loader import iter_task_timesteps, iter_vehicle_timesteps
from src.masac_env import MASACVECEnv, EnvironmentConfig
from src.masac import DiscreteMASAC, MASACConfig


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


def write_csv(path, rows, fields_if_empty=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows))) if rows else (fields_if_empty or [])
    with path.open("w", newline="", encoding="utf-8") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            if rows:
                writer.writerows(rows)


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


def scenario_record(metric, task, vehicle, t, candidate):
    return {
        "time_s": t,
        "task": task.id,
        "agent": vehicle.id,
        "scenario": "UAV-relay" if candidate.kind == "UAV_RELAY" else "No-UAV-relay",
        "action_kind": candidate.kind,
        "route": metric.route,
        "latency_s": float(metric.latency_s),
        "avg_latency_s": float(metric.latency_s),
        "energy_j": float(metric.energy_j),
        "packet_loss": float(metric.packet_loss),
        "avg_rate_bps": float(metric.avg_rate_bps),
        "final_edge": metric.final_edge,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vehicles", default="datasets/vehicles.xml")
    p.add_argument("--tasks", default="datasets/tasks.xml")
    p.add_argument("--model", default="outputs/masac.pt")
    p.add_argument("--max-tasks", type=int, default=10000)
    p.add_argument("--csv", default="outputs/evaluation.csv")
    p.add_argument("--scenario-csv", default="outputs/evaluation_scenarios.csv")
    p.add_argument("--edge-csv", default="outputs/evaluation_edge_load.csv")
    p.add_argument("--uav-csv", default="outputs/uav_positions.csv")
    p.add_argument("--uav-presence-csv", default="outputs/uav_presence.csv")
    p.add_argument(
        "--num-uavs",
        type=int,
        default=5,
        help="Number of UAV relays (default: 5). Use 0 for the no-UAV case.",
    )
    args = p.parse_args()

    if args.num_uavs < 0:
        p.error("--num-uavs must be >= 0")
    env_cfg = EnvironmentConfig(num_uavs=args.num_uavs)

    vehicles = build_vehicle_index(args.vehicles)
    groups = build_task_groups(args.tasks, vehicles, args.max_tasks)
    if not groups:
        raise RuntimeError("No aligned task/vehicle records were found.")

    t0, tasks0 = groups[0]
    task0 = tasks0[0]
    env = MASACVECEnv(vehicles[t0], env_cfg=env_cfg)
    env.set_active_task_creators(task.creator for task in tasks0)
    obs0, state0, _ = env.observation(task0, vehicles[t0][task0.creator])
    agent = DiscreteMASAC(len(obs0), len(state0), env.action_dim, MASACConfig())
    agent.load(args.model)
    print(f"Loaded MASAC model on {agent.device}")
    print(f"Number of UAVs (K): {args.num_uavs}")
    print(f"Number of edge servers: {env_cfg.num_edges}")

    selected_rows = []
    scenario_rows = []
    edge_rows = []
    uav_rows = []
    presence_rows = []

    for t, tasks in groups:
        env.advance(vehicles[t])
        env.set_active_task_creators(task.creator for task in tasks)

        # Record UAV placement once per timestamp.
        presence_rows.append({"time_s": t, "uav_count": len(env.uavs_xy)})
        for uav_id, xy in enumerate(env.uavs_xy):
            uav_rows.append({
                "time_s": t,
                "uav_id": uav_id,
                "x": float(xy[0]),
                "y": float(xy[1]),
            })

        for task in tasks:
            vehicle = vehicles[t][task.creator]
            obs, _, mask = env.observation(task, vehicle)

            # ----------------------------------------------------------
            # Required scenario comparison on the SAME pre-action state.
            # UAV-relay: feasible UAV_RELAY routes.
            # No-UAV: feasible DIRECT / WIRED_1HOP routes only.
            # LOCAL is intentionally excluded from this communication
            # scenario comparison.
            # ----------------------------------------------------------
            uav_options = []
            no_uav_options = []
            for c in env.candidates:
                if not mask[c.index]:
                    continue
                if c.kind not in {"UAV_RELAY", "DIRECT", "WIRED_1HOP"}:
                    continue
                metric = env.evaluate_action(task, vehicle, c.index)
                row = scenario_record(metric, task, vehicle, t, c)
                if c.kind == "UAV_RELAY":
                    uav_options.append(row)
                else:
                    no_uav_options.append(row)

            # Keep ONLY matched tasks so the mandatory UAV-vs-no-UAV
            # comparison is apples-to-apples.  A task contributes two rows
            # (one per scenario) only when BOTH scenario families are feasible
            # in the exact same pre-action state.
            if uav_options and no_uav_options:
                pair_id = f"{t}:{task.id}:{vehicle.id}"
                best_uav = min(uav_options, key=lambda r: r["latency_s"])
                best_no_uav = min(no_uav_options, key=lambda r: r["latency_s"])
                best_uav["pair_id"] = pair_id
                best_no_uav["pair_id"] = pair_id
                best_uav["matched"] = 1
                best_no_uav["matched"] = 1
                scenario_rows.extend([best_uav, best_no_uav])

            # ----------------------------------------------------------
            # Actual trained MASAC decision.
            # ----------------------------------------------------------
            if not mask.any():
                # No legal route exists under LOCAL resource constraints,
                # the 300 m direct-V2E limit, UAV coverage, and one-hop
                # wired adjacency. Record a failed task instead of silently
                # violating the hard 300 m rule.
                info = {
                    "latency_s": float("nan"),
                    "energy_j": 0.0,
                    "packet_loss": 1.0,
                    "avg_rate_bps": float("nan"),
                    "load_imbalance": float("nan"),
                    "deadline_miss_s": float("nan"),
                    "route": "NO_FEASIBLE_ROUTE",
                    "final_edge": None,
                    "final_edge_load": float("nan"),
                    "success": 0.0,
                    "time_s": t,
                    "task": task.id,
                    "agent": vehicle.id,
                }
            else:
                action = agent.act(obs, mask, deterministic=True)
                _, info = env.step_task(task, vehicle, action)
                info = dict(info)
                info.update({"time_s": t, "task": task.id, "agent": vehicle.id})

            selected_rows.append(info)

            # Record real post-task queue load for every edge. This is the
            # actual load distribution, not a pre-task all-zero snapshot.
            loads = env.edge_load_seconds()
            for edge_id, load_s in enumerate(loads):
                edge_rows.append({
                    "time_s": t,
                    "task": task.id,
                    "edge": edge_id,
                    "load": float(load_s),
                    "queued_work_s": float(load_s),
                    "queue_cycles": float(env.edge_queue_cycles[edge_id]),
                })

    network_rows = [
        r for r in selected_rows
        if np.isfinite(float(r.get("avg_rate_bps", float("nan"))))
    ]
    uav_scen = [r for r in scenario_rows if r["scenario"] == "UAV-relay"]
    no_uav_scen = [r for r in scenario_rows if r["scenario"] == "No-UAV-relay"]

    route_counts = defaultdict(int)
    for r in selected_rows:
        route_counts[str(r.get("route", "UNKNOWN"))] += 1

    print("\n================ MASAC EVALUATION ================")
    print(f"Tasks                         = {len(selected_rows)}")
    print(f"Average total latency         = {finite_mean(selected_rows, 'latency_s'):.6f} s")
    print(f"Average total energy          = {finite_mean(selected_rows, 'energy_j'):.8f} J")
    if network_rows:
        print(f"Network packet loss           = {finite_mean(network_rows, 'packet_loss'):.6f}")
        print(f"Average NETWORK data rate     = {finite_mean(network_rows, 'avg_rate_bps')/1e6:.3f} Mbps")
        print(f"Network/offloaded tasks       = {len(network_rows)}")
    else:
        print("Network packet loss           = N/A (no network routes selected)")
        print("Average NETWORK data rate     = N/A (no network routes selected)")
        print("Network/offloaded tasks       = 0")
    print(f"Task success rate             = {100*np.mean([float(r.get('success',0)) for r in selected_rows]):.2f}%")
    no_route_count = sum(1 for r in selected_rows if r.get("route") == "NO_FEASIBLE_ROUTE")
    print(f"No-feasible-route tasks       = {no_route_count}")

    print("\nRequired MATCHED communication scenario comparison:")
    print(f"  Matched task pairs          = {len(uav_scen)}")
    print(f"  UAV-relay samples           = {len(uav_scen)}")
    print(f"  No-UAV samples              = {len(no_uav_scen)}")
    if len(uav_scen) != len(no_uav_scen):
        raise RuntimeError("Internal error: matched scenario counts are unequal.")
    if uav_scen:
        print(f"  UAV-relay avg latency       = {finite_mean(uav_scen, 'latency_s'):.6f} s")
        print(f"  UAV-relay avg bitrate       = {finite_mean(uav_scen, 'avg_rate_bps')/1e6:.3f} Mbps")
    if no_uav_scen:
        print(f"  No-UAV avg latency          = {finite_mean(no_uav_scen, 'latency_s'):.6f} s")
        print(f"  No-UAV avg bitrate          = {finite_mean(no_uav_scen, 'avg_rate_bps')/1e6:.3f} Mbps")

    print("\nMASAC route distribution:")
    for route, count in sorted(route_counts.items()):
        print(f"  {route:20s} = {count}")

    # Average edge load is queued processing work (seconds) after each task.
    print("\nAverage edge load (queued processing work):")
    for edge_id in range(env.env_cfg.num_edges):
        vals = [r["load"] for r in edge_rows if r["edge"] == edge_id]
        print(f"  Edge {edge_id}: {float(np.mean(vals)) if vals else 0.0:.6f} s")

    avg_presence = float(np.mean([r["uav_count"] for r in presence_rows])) if presence_rows else 0.0
    print(f"\nAverage UAV presence on map  = {avg_presence:.3f} UAVs/timestep")

    write_csv(args.csv, selected_rows)
    write_csv(
        args.scenario_csv,
        scenario_rows,
        fields_if_empty=[
            "time_s", "task", "agent", "scenario", "action_kind", "route",
            "latency_s", "avg_latency_s", "energy_j", "packet_loss",
            "avg_rate_bps", "final_edge", "pair_id", "matched",
        ],
    )
    write_csv(args.edge_csv, edge_rows)
    write_csv(
        args.uav_csv,
        uav_rows,
        fields_if_empty=["time_s", "uav_id", "x", "y"],
    )
    write_csv(args.uav_presence_csv, presence_rows)

    print("\nGenerated:")
    for path in [
        args.csv,
        args.scenario_csv,
        args.edge_csv,
        args.uav_csv,
        args.uav_presence_csv,
    ]:
        print(" ", path)


if __name__ == "__main__":
    main()
