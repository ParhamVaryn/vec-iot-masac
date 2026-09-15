from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation


def draw_environment(env, title=None, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(8, 8))

    vx = [v.x for v in env.vehicles]
    vy = [v.y for v in env.vehicles]
    ex = [e.x for e in env.edges]
    ey = [e.y for e in env.edges]
    ux = [u.x for u in env.uavs]
    uy = [u.y for u in env.uavs]

    ax.scatter(vx, vy, s=22, label="Vehicles")
    ax.scatter(ex, ey, s=180, marker="s", label="Edge Servers")
    ax.scatter(ux, uy, s=150, marker="^", label="UAV Relays")

    for edge in env.edges:
        ax.annotate(
            f"E{edge.edge_id}\nQ={len(edge.queue)}",
            (edge.x, edge.y),
            xytext=(6, 6),
            textcoords="offset points",
        )

    for uav in env.uavs:
        circle = Circle(
            (uav.x, uav.y),
            uav.coverage_radius_m,
            fill=False,
            linewidth=1.2,
            alpha=0.45,
        )
        ax.add_patch(circle)
        ax.annotate(
            f"U{uav.uav_id}",
            (uav.x, uav.y),
            xytext=(6, 6),
            textcoords="offset points",
        )

    if env.last_cluster_centers is not None:
        ax.scatter(
            env.last_cluster_centers[:, 0],
            env.last_cluster_centers[:, 1],
            marker="x",
            s=100,
            label="K-Means Centers",
        )

    ax.set_xlim(0, env.cfg.width_m)
    ax.set_ylim(0, env.cfg.height_m)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"VEC Simulation - t={env.time_s:.0f}s")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_animation(env, output_path, frames=30):
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(_):
        env.step()
        ax.clear()

        vx = [v.x for v in env.vehicles]
        vy = [v.y for v in env.vehicles]
        ex = [e.x for e in env.edges]
        ey = [e.y for e in env.edges]
        ux = [u.x for u in env.uavs]
        uy = [u.y for u in env.uavs]

        ax.scatter(vx, vy, s=22, label="Vehicles")
        ax.scatter(ex, ey, s=180, marker="s", label="Edge Servers")
        ax.scatter(ux, uy, s=150, marker="^", label="UAV Relays")

        for uav in env.uavs:
            ax.add_patch(
                Circle(
                    (uav.x, uav.y),
                    uav.coverage_radius_m,
                    fill=False,
                    linewidth=1.0,
                    alpha=0.35,
                )
            )

        if env.last_cluster_centers is not None:
            ax.scatter(
                env.last_cluster_centers[:, 0],
                env.last_cluster_centers[:, 1],
                marker="x",
                s=100,
                label="K-Means Centers",
            )

        ax.set_xlim(0, env.cfg.width_m)
        ax.set_ylim(0, env.cfg.height_m)
        ax.set_title(f"Dynamic UAV Placement - t={env.time_s:.0f}s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_aspect("equal", adjustable="box")

    animation = FuncAnimation(fig, update, frames=frames, interval=250)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer="pillow", fps=4)
    plt.close(fig)
