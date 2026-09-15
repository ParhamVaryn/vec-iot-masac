# Multi-Agent Reinforcement Learning for Vehicular Edge Computing

### Discrete MASAC with UAV-Assisted Edge Intelligence for Dynamic Computation Offloading

**PyTorch · Multi-Agent Reinforcement Learning · Soft Actor-Critic · Edge Intelligence · UAV Networks**

---

## Overview

This repository presents an end-to-end implementation of a **dynamic Vehicular Edge Computing (VEC) environment** optimized through **Multi-Agent Reinforcement Learning**.

The system models a realistic edge-intelligent transportation scenario where mobile vehicles generate heterogeneous computational tasks and must autonomously determine optimal computation placement and communication routes.

The implemented environment supports:

- Local computation
- Direct vehicle-to-edge offloading
- UAV-assisted relay communication
- Wired multi-hop edge routing

The objective is to learn adaptive offloading policies that jointly optimize:

- Latency
- Energy consumption
- Communication reliability
- Edge resource utilization
- Load balancing

The simulation environment integrates:

- 10 fixed edge servers
- 5 mobile UAV relays
- Vehicle mobility
- Dynamic task generation
- Local and edge processing queues
- Communication and computation models

---

# Key Contributions

## 1. Dynamic Vehicular Edge Computing Environment

A complete VEC simulation environment was designed and implemented, modeling the interaction between:

- Mobile vehicles generating heterogeneous workloads
- Edge servers with limited computational capacity
- UAV relays adapting to traffic distribution
- Communication links with distance-dependent constraints
- Processing queues evolving over time

Vehicle and task states are processed through timestamp-aligned streaming inputs, enabling realistic temporal progression rather than static task assignment.

UAV positions are dynamically updated using K-Means clustering over vehicle locations, allowing relay placement to adapt to changing traffic patterns.

---

# 2. Multi-Agent Soft Actor-Critic Decision Framework

The decision-making component implements a discrete **Multi-Agent Soft Actor-Critic (MASAC)** framework using a:

**Centralized Training, Decentralized Execution (CTDE)** architecture.

The system models vehicles as decision-making agents with shared policy parameters.

During training:

- The Actor receives local observations.
- Centralized Critics evaluate decisions using global system state information.

During execution:

- The learned Actor selects feasible actions using only local observations and action constraints.

The implementation includes:

- Shared-policy Actor network
- Twin Critics
- Target Critics
- Replay Buffer
- Entropy-regularized policy optimization
- Soft target updates

---

# 3. Large Discrete Action Space with Feasibility Constraints

The computation offloading problem is modeled using a structured discrete action space:

```
A = A_local ∪ A_direct ∪ A_UAV ∪ A_wired
```

For the implemented configuration:

| Action Type | Number of Actions |
|---|---:|
| Local execution | 1 |
| Direct edge offloading | 10 |
| UAV relay routes | 50 |
| Wired one-hop routing | 26 |
| **Total** | **87** |

Invalid decisions are removed through **hard feasibility masking** before policy selection.

Constraints include:

- Vehicle-edge communication distance
- UAV coverage limitations
- Edge neighborhood topology
- Resource feasibility

This prevents the policy from selecting physically impossible actions.

---

# 4. Multi-Objective Reinforcement Learning Optimization

The reward function converts system-level outcomes into a unified optimization objective.

The learned policy considers:

- End-to-end latency
- Execution energy
- Packet-loss surrogate
- Edge load imbalance
- Deadline violations

This allows the agent to learn trade-offs between competing objectives rather than optimizing a single metric.

---

# System Architecture

```
Vehicle / Task Data
        |
        v
+-----------------------+
|   VEC Environment     |
|-----------------------|
| Mobility              |
| Task Generation       |
| Edge Resources        |
| UAV Control           |
| Queue Dynamics        |
| Communication Model   |
+-----------------------+
        |
        v
+-----------------------+
|     MASAC Agent       |
|-----------------------|
| Shared Actor          |
| Twin Critics          |
| Replay Buffer         |
| Action Masking        |
+-----------------------+
        |
        v
Optimized Offloading Policy
```

---

# Training Pipeline

```
Current System Snapshot

        |
        v

Observation + Global State + Action Mask

        |
        v

MASAC Policy Decision

        |
        v

Environment Transition

        |
        v

Reward Computation

        |
        v

Replay Buffer

        |
        v

Actor / Critic Updates
```

---

# Repository Structure

```
VEC-IoT-MASAC/

├── src/
│   ├── masac.py
│   ├── masac_env.py
│   ├── system_model.py
│   ├── uav_control.py
│   └── xml_loader.py
│
├── train_masac.py
├── evaluate_masac.py
├── run_phase3.py
│
├── VEC_IoT_MASAC_Final.ipynb
│
├── vec_iot_project/
│
└── latex_report/
    └── VEC-IoT-MASAC_Technical_Report.pdf
```

---

# Technical Report

A detailed technical report documents:

- Environment architecture
- Mathematical system modeling
- UAV positioning strategy
- Action-space construction
- Hard constraint formulation
- MASAC architecture
- CTDE training procedure
- Evaluation methodology

The report provides detailed derivations and implementation explanations for the complete framework.

---

# Project Motivation

Vehicular edge computing requires intelligent decision-making under rapidly changing conditions.

A practical solution must simultaneously reason about:

- where computation should execute,
- how data should be transmitted,
- how resources should be balanced,
- and how decisions should adapt to mobility.

This project explores these challenges through an integrated simulation and reinforcement learning framework combining **edge intelligence, autonomous decision-making, and multi-agent optimization**.
