"""VEC-IoT physical/network/computation model.

The equations implemented here follow the supplied system model:

  xi_ij(tau) = 10*n*log10(d) + 20*log10(fc)
               + 10*n*log10(4*pi/(3*10^8))
  R_ij(tau)  = eta*B*log2(1 + S/(N+I))
  S          = p_ij(tau)*G_ij(tau)
  N          = xi_ij(tau)*d_ij(tau)*sigma^2_ij(tau)
  V2E I      = (N_r(tau)-1)*p_ir(tau)*g_r(tau)
  E2C I      = sum_{s != r} p_s(tau)*g_r(tau)
  T_exe      = C_v,k * D_v,k / f_core
  E_exe      = zeta * (f_core)^gamma * C_v,k
  E_trans    = p_v * D_v,k / R_ij

Two implementation details are made explicit because the source model does not
specify how eta is allocated or provide a packet-loss equation:

1) eta is derived from the configured resource-block scheduler, but N_r in the
   supplied V2E interference equation is the full number of vehicles connected
   to edge r, exactly as stated in the project model.
2) packet loss is a documented SINR-to-loss surrogate used only for evaluation.

Local execution has no network link, so avg_rate_bps is NaN rather than infinity.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .xml_loader import TaskRecord, VehicleRecord


@dataclass(frozen=True)
class NetworkConfig:
    # Radio / propagation
    bandwidth_hz: float = 100e6
    carrier_frequency_hz: float = 5.9e9
    path_loss_exponent_mean: float = 3.0
    path_loss_exponent_std: float = 0.35
    region_size_m: float = 500.0
    noise_power_w: float = 1.0e-13  # sigma^2

    vehicle_tx_power_w: float = 0.5
    edge_tx_power_w: float = 0.5
    uav_tx_power_w: float = 0.5

    # The mathematical model gives eta but does not prescribe a scheduler.
    # We partition B into orthogonal resource blocks. Vehicles assigned to the
    # same block are the co-channel users represented by N_r in the I formula.
    radio_resource_blocks: int = 20

    # Packet-loss surrogate parameters; no packet-loss equation was supplied.
    packet_loss_threshold_db: float = -12.0
    packet_loss_slope_db: float = 2.0

    min_rate_bps: float = 1.0
    max_rate_bps: float = 1.0e9

    # Topology
    wired_rate_bps: float = 1.0e9
    wired_energy_per_bit_j: float = 2.0e-10

    # Hard routing constraints discussed for the revised topology.
    direct_edge_radius_m: float = 300.0
    uav_coverage_radius_m: float = 500.0

    # Compute resources. These are simulation parameters, not equations.
    local_cpu_hz: float = 1.0e9
    edge_cpu_hz: Tuple[float, ...] = (
        4.0e9, 4.5e9, 5.0e9, 5.5e9, 6.0e9,
        4.0e9, 4.5e9, 5.0e9, 5.5e9, 6.0e9,
    )
    local_memory_capacity: float = 1536.0

    local_energy_coeff: float = 1.0e-27
    edge_energy_coeff: float = 1.0e-28
    local_energy_gamma: float = 2.0
    edge_energy_gamma: float = 2.0


@dataclass
class RouteCandidate:
    index: int
    kind: str
    name: str
    ingress_edge: Optional[int] = None
    final_edge: Optional[int] = None
    uav: Optional[int] = None


@dataclass
class RouteMetrics:
    latency_s: float
    energy_j: float
    packet_loss: float
    avg_rate_bps: float
    final_edge: Optional[int]
    route: str


class VECSystemModel:
    def __init__(
        self,
        cfg: NetworkConfig = NetworkConfig(),
        wired_neighbors: Optional[Dict[int, Sequence[int]]] = None,
    ):
        self.cfg = cfg
        # One-hop wired adjacency for the fixed 5 x 2 edge grid.
        # A task can move only to one directly neighboring edge.
        self.wired_neighbors = wired_neighbors or {
            0: (1, 5),
            1: (0, 2, 6),
            2: (1, 3, 7),
            3: (2, 4, 8),
            4: (3, 9),
            5: (0, 6),
            6: (1, 5, 7),
            7: (2, 6, 8),
            8: (3, 7, 9),
            9: (4, 8),
        }
        self._region_exponents: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Path loss / channel gain
    # ------------------------------------------------------------------
    def _region_key(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor(x / self.cfg.region_size_m)),
            int(math.floor(y / self.cfg.region_size_m)),
        )

    def _path_loss_exponent(self, x: float, y: float) -> float:
        """Draw n once per structural region from the required Gaussian."""
        key = self._region_key(x, y)
        if key not in self._region_exponents:
            seed = ((key[0] * 73856093) ^ (key[1] * 19349663)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            n = rng.normal(
                self.cfg.path_loss_exponent_mean,
                self.cfg.path_loss_exponent_std,
            )
            self._region_exponents[key] = max(float(n), 1.0e-6)
        return self._region_exponents[key]

    def distance_m(self, a_xy, b_xy) -> float:
        return max(
            float(
                np.linalg.norm(
                    np.asarray(a_xy, dtype=float) - np.asarray(b_xy, dtype=float)
                )
            ),
            1.0,
        )

    def path_loss_db(self, a_xy, b_xy) -> float:
        """Exact xi_i,j(tau) equation supplied in the project model."""
        d = self.distance_m(a_xy, b_xy)
        n = self._path_loss_exponent(*a_xy)
        fc = max(float(self.cfg.carrier_frequency_hz), 1.0e-12)
        return float(
            10.0 * n * math.log10(d)
            + 20.0 * math.log10(fc)
            + 10.0 * n * math.log10((4.0 * math.pi) / (3.0e8))
        )

    def channel_gain(self, a_xy, b_xy) -> float:
        """G_i,j(tau) represented as linear power gain from xi in dB."""
        return float(10.0 ** (-self.path_loss_db(a_xy, b_xy) / 10.0))

    # Backward-compatible name used by MASAC observation code.
    path_gain = channel_gain

    def received_power(self, src_xy, dst_xy, tx_power_w: float) -> float:
        """S = p_i,j(tau) * G_i,j(tau)."""
        return float(max(float(tx_power_w), 0.0) * self.channel_gain(src_xy, dst_xy))

    def effective_noise(self, src_xy, dst_xy) -> float:
        """N = xi_i,j(tau) * d_i,j(tau) * sigma^2_i,j(tau).

        The source equation writes xi directly in this product even though xi
        was introduced in dB. We reproduce that stated equation numerically and
        clamp only to avoid a non-positive denominator.
        """
        xi = self.path_loss_db(src_xy, dst_xy)
        d = self.distance_m(src_xy, dst_xy)
        sigma2 = max(float(self.cfg.noise_power_w), 0.0)
        return max(float(xi * d * sigma2), 1.0e-30)

    # ------------------------------------------------------------------
    # Interference / radio allocation
    # ------------------------------------------------------------------
    def radio_allocation(self, connected_count: int) -> float:
        """Return eta, the allocated bandwidth fraction for one link.

        The supplied rate equation contains eta but does not prescribe how it
        is chosen.  The configured resource-block scheduler is therefore used
        only to determine eta.  It does NOT alter N_r in the interference
        equation; N_r remains the full number of vehicles connected to edge r.
        """
        n = max(int(connected_count), 1)
        k = max(int(self.cfg.radio_resource_blocks), 1)
        occupied = min(k, n)
        return 1.0 / float(occupied)

    def v2e_interference(
        self,
        receiver_xy,
        desired_src_xy,
        connected_vehicle_count: int,
    ) -> float:
        """I_i,r(tau) = (N_r(tau)-1) * p_i,r(tau) * g_r(tau).

        N_r is exactly the number of vehicles connected to edge r, as stated
        in the supplied system model.
        """
        nr = max(int(connected_vehicle_count), 1)
        p_ir = float(self.cfg.vehicle_tx_power_w)
        g_r = self.channel_gain(desired_src_xy, receiver_xy)
        return float(max(nr - 1, 0) * p_ir * g_r)

    def e2c_interference(
        self,
        receiver_xy,
        desired_src_xy,
        interfering_edge_count: int,
    ) -> float:
        """I_r,cs(tau) = sum_{s != r} p_s(tau) * g_r(tau).

        The supplied equation uses the same g_r for every s in the sum.  With
        equal configured edge transmit powers this is evaluated literally as
        N_interferers * p_s * g_r.  For the UAV->edge relay hop, using this
        E2C expression is an explicit project-level modeling assumption because
        no separate UAV->edge interference equation was supplied.
        """
        n_interferers = max(int(interfering_edge_count), 0)
        p_s = float(self.cfg.edge_tx_power_w)
        g_r = self.channel_gain(desired_src_xy, receiver_xy)
        return float(n_interferers * p_s * g_r)

    def _packet_loss_surrogate(self, sinr: float) -> float:
        if not np.isfinite(sinr) or sinr <= 0.0:
            return 1.0
        sinr_db = 10.0 * math.log10(max(sinr, 1.0e-30))
        z = (sinr_db - self.cfg.packet_loss_threshold_db) / max(
            self.cfg.packet_loss_slope_db, 1.0e-9
        )
        # Numerically stable logistic: high SINR -> low loss.
        if z >= 0:
            ez = math.exp(-min(z, 700.0))
            return float(ez / (1.0 + ez))
        ez = math.exp(min(z, 700.0))
        return float(1.0 / (1.0 + ez))

    def link_rate(
        self,
        src_xy,
        dst_xy,
        *,
        allocated_fraction: float,
        signal_tx_power_w: float,
        interference_w: float,
    ) -> Tuple[float, float]:
        """R = eta*B*log2(1 + S/(N+I))."""
        S = self.received_power(src_xy, dst_xy, signal_tx_power_w)
        N = self.effective_noise(src_xy, dst_xy)
        I = max(float(interference_w), 0.0)
        sinr = S / max(N + I, 1.0e-30)

        eta = float(np.clip(allocated_fraction, 1.0e-9, 1.0))
        R = eta * float(self.cfg.bandwidth_hz) * math.log2(1.0 + max(sinr, 0.0))
        R = float(np.clip(R, self.cfg.min_rate_bps, self.cfg.max_rate_bps))
        return R, self._packet_loss_surrogate(sinr)

    # ------------------------------------------------------------------
    # Computation / energy
    # ------------------------------------------------------------------
    @staticmethod
    def bits(task: TaskRecord) -> float:
        # The original dataset expresses data_size in MB; convert MB -> bit.
        return float(task.data_size_mbit) * 8.0e6

    @classmethod
    def cycles(cls, task: TaskRecord) -> float:
        # C_v,k * D_v,k total required cycles.
        return cls.bits(task) * float(task.cycles_per_bit)

    @staticmethod
    def compute_time_from_cycles(
        cycles: float,
        cpu_hz: float,
        queued_cycles: float = 0.0,
    ) -> float:
        return (max(float(queued_cycles), 0.0) + max(float(cycles), 0.0)) / max(
            float(cpu_hz), 1.0
        )

    @classmethod
    def compute_time(cls, task: TaskRecord, cpu_hz: float, queued_cycles: float = 0.0) -> float:
        return cls.compute_time_from_cycles(cls.cycles(task), cpu_hz, queued_cycles)

    @staticmethod
    def compute_energy(task: TaskRecord, cpu_hz: float, coeff: float, gamma: float) -> float:
        """E_exe = zeta * (f_core)^gamma * C_v,k, exactly as supplied.

        C_v,k is the task's cycles-per-bit field.  D_v,k is deliberately NOT
        multiplied into this energy equation; D_v,k remains part of T_exe via
        the total-cycle quantity C_v,k * D_v,k.
        """
        C_vk = max(float(task.cycles_per_bit), 0.0)
        return float(
            float(coeff)
            * (max(float(cpu_hz), 1.0) ** float(gamma))
            * C_vk
        )

    # ------------------------------------------------------------------
    # Route candidates and validity
    # ------------------------------------------------------------------
    def build_candidates(self, num_edges: int, num_uavs: int):
        actions = [RouteCandidate(0, "LOCAL", "LOCAL")]
        idx = 1

        for e in range(num_edges):
            actions.append(RouteCandidate(idx, "DIRECT", f"DIRECT_E{e}", final_edge=e))
            idx += 1

        for u in range(num_uavs):
            for e in range(num_edges):
                actions.append(
                    RouteCandidate(
                        idx,
                        "UAV_RELAY",
                        f"UAV{u}->E{e}",
                        final_edge=e,
                        uav=u,
                    )
                )
                idx += 1

        for ingress, final in permutations(range(num_edges), 2):
            if final in self.wired_neighbors.get(ingress, ()):
                actions.append(
                    RouteCandidate(
                        idx,
                        "WIRED_1HOP",
                        f"E{ingress}->E{final}",
                        ingress_edge=ingress,
                        final_edge=final,
                    )
                )
                idx += 1
        return actions

    def valid_action_mask(self, vehicle, uavs_xy, edges_xy, candidates, task=None):
        mask = np.zeros(len(candidates), dtype=bool)

        # Heterogeneous-resource constraint: a task that exceeds the local
        # device memory cannot be executed locally and must be offloaded.
        local_ok = True
        if task is not None:
            local_ok = float(task.required_memory) <= float(self.cfg.local_memory_capacity)
        mask[0] = local_ok

        vehicle_xy = (vehicle.x, vehicle.y)

        for c in candidates:
            if c.kind == "DIRECT":
                # Vehicle -> edge is legal only when that edge is at most
                # 300 m away. Distant direct V2E transmission is forbidden.
                d_edge = self.distance_m(vehicle_xy, edges_xy[c.final_edge])
                mask[c.index] = d_edge <= self.cfg.direct_edge_radius_m

            elif c.kind == "UAV_RELAY":
                # Vehicle -> UAV -> edge. Both wireless hops must satisfy
                # the UAV coverage constraint.
                du = self.distance_m(vehicle_xy, uavs_xy[c.uav])
                de = self.distance_m(uavs_xy[c.uav], edges_xy[c.final_edge])
                mask[c.index] = (
                    du <= self.cfg.uav_coverage_radius_m
                    and de <= self.cfg.uav_coverage_radius_m
                )

            elif c.kind == "WIRED_1HOP":
                # The first hop is still Vehicle -> ingress edge, so the
                # same 300 m direct-V2E restriction applies. The second hop
                # is exactly one wired hop to a direct neighbor.
                d_ingress = self.distance_m(vehicle_xy, edges_xy[c.ingress_edge])
                mask[c.index] = (
                    d_ingress <= self.cfg.direct_edge_radius_m
                    and c.final_edge in self.wired_neighbors.get(c.ingress_edge, ())
                )

        # Do NOT re-enable a distant DIRECT route as a fallback. If LOCAL is
        # infeasible and no legal network route exists, the caller records
        # NO_FEASIBLE_ROUTE instead of violating the 300 m hard constraint.
        return mask

    @staticmethod
    def _nearest_edge_index(xy, edges_xy) -> int:
        return int(
            np.argmin(
                [
                    np.linalg.norm(np.asarray(xy, dtype=float) - np.asarray(e, dtype=float))
                    for e in edges_xy
                ]
            )
        )

    def _connected_vehicle_count(self, active_vehicle_xy, edges_xy, edge_idx: int) -> int:
        if not active_vehicle_xy:
            return 1
        return max(
            sum(
                1
                for xy in active_vehicle_xy
                if self._nearest_edge_index(xy, edges_xy) == int(edge_idx)
            ),
            1,
        )

    # ------------------------------------------------------------------
    # Route evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        task,
        vehicle,
        edges_xy,
        edge_queue_cycles,
        uavs_xy,
        candidate,
        active_vehicle_xy=(),
        local_queue_cycles: float = 0.0,
    ):
        bits = self.bits(task)
        cycles = self.cycles(task)
        vxy = (vehicle.x, vehicle.y)

        if candidate.kind == "LOCAL":
            t = self.compute_time_from_cycles(
                cycles,
                self.cfg.local_cpu_hz,
                queued_cycles=local_queue_cycles,
            )
            e = self.compute_energy(
                task,
                self.cfg.local_cpu_hz,
                self.cfg.local_energy_coeff,
                self.cfg.local_energy_gamma,
            )
            # No communication link exists for local execution.
            return RouteMetrics(t, e, 0.0, float("nan"), None, candidate.name)

        if candidate.kind == "DIRECT":
            edge = int(candidate.final_edge)
            exy = edges_xy[edge]
            connected = self._connected_vehicle_count(active_vehicle_xy, edges_xy, edge)
            eta = self.radio_allocation(connected)
            I = self.v2e_interference(exy, vxy, connected)
            rate, loss = self.link_rate(
                vxy,
                exy,
                allocated_fraction=eta,
                signal_tx_power_w=self.cfg.vehicle_tx_power_w,
                interference_w=I,
            )
            tx_time = bits / rate
            tx_energy = self.cfg.vehicle_tx_power_w * tx_time
            cpu = self.cfg.edge_cpu_hz[edge % len(self.cfg.edge_cpu_hz)]
            queue_time = float(edge_queue_cycles[edge]) / max(cpu, 1.0)
            comp_time = cycles / max(cpu, 1.0)
            comp_energy = self.compute_energy(
                task, cpu, self.cfg.edge_energy_coeff, self.cfg.edge_energy_gamma
            )
            return RouteMetrics(
                tx_time + queue_time + comp_time,
                tx_energy + comp_energy,
                loss,
                rate,
                edge,
                candidate.name,
            )

        if candidate.kind == "UAV_RELAY":
            uxy = uavs_xy[candidate.uav]
            edge = int(candidate.final_edge)
            exy = edges_xy[edge]

            # First wireless hop: vehicle -> UAV. The source model's V2E
            # co-channel form is used as the V2X access-link interference form.
            access_edge = self._nearest_edge_index(vxy, edges_xy)
            connected = self._connected_vehicle_count(active_vehicle_xy, edges_xy, access_edge)
            eta = self.radio_allocation(connected)
            I1 = self.v2e_interference(uxy, vxy, connected)
            r1, l1 = self.link_rate(
                vxy,
                uxy,
                allocated_fraction=eta,
                signal_tx_power_w=self.cfg.vehicle_tx_power_w,
                interference_w=I1,
            )

            # Second wireless hop: UAV -> edge, using the supplied E2C
            # interference sum over other edge transmissions.
            interfering_edge_count = max(len(edges_xy) - 1, 0)
            I2 = self.e2c_interference(
                exy, uxy, interfering_edge_count
            )
            r2, l2 = self.link_rate(
                uxy,
                exy,
                allocated_fraction=1.0,
                signal_tx_power_w=self.cfg.uav_tx_power_w,
                interference_w=I2,
            )

            t1 = bits / r1
            t2 = bits / r2
            tx_energy = self.cfg.vehicle_tx_power_w * t1 + self.cfg.uav_tx_power_w * t2
            cpu = self.cfg.edge_cpu_hz[edge % len(self.cfg.edge_cpu_hz)]
            queue_time = float(edge_queue_cycles[edge]) / max(cpu, 1.0)
            comp_time = cycles / max(cpu, 1.0)
            comp_energy = self.compute_energy(
                task, cpu, self.cfg.edge_energy_coeff, self.cfg.edge_energy_gamma
            )
            loss = 1.0 - (1.0 - l1) * (1.0 - l2)
            return RouteMetrics(
                t1 + t2 + queue_time + comp_time,
                tx_energy + comp_energy,
                loss,
                min(r1, r2),
                edge,
                candidate.name,
            )

        if candidate.kind == "WIRED_1HOP":
            ingress = int(candidate.ingress_edge)
            final = int(candidate.final_edge)
            ingress_xy = edges_xy[ingress]

            connected = self._connected_vehicle_count(active_vehicle_xy, edges_xy, ingress)
            eta = self.radio_allocation(connected)
            I = self.v2e_interference(ingress_xy, vxy, connected)
            rate, loss = self.link_rate(
                vxy,
                ingress_xy,
                allocated_fraction=eta,
                signal_tx_power_w=self.cfg.vehicle_tx_power_w,
                interference_w=I,
            )
            tx_time = bits / rate
            wired_time = bits / self.cfg.wired_rate_bps
            wired_energy = bits * self.cfg.wired_energy_per_bit_j

            cpu = self.cfg.edge_cpu_hz[final % len(self.cfg.edge_cpu_hz)]
            queue_time = float(edge_queue_cycles[final]) / max(cpu, 1.0)
            comp_time = cycles / max(cpu, 1.0)
            comp_energy = self.compute_energy(
                task, cpu, self.cfg.edge_energy_coeff, self.cfg.edge_energy_gamma
            )
            return RouteMetrics(
                tx_time + wired_time + queue_time + comp_time,
                self.cfg.vehicle_tx_power_w * tx_time + wired_energy + comp_energy,
                loss,
                min(rate, self.cfg.wired_rate_bps),
                final,
                candidate.name,
            )

        raise ValueError(f"Unsupported route kind: {candidate.kind}")
