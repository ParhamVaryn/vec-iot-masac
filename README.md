# Multi-Agent Reinforcement Learning for Vehicular Edge Computing

### Discrete MASAC with UAV-Assisted Edge Intelligence for Dynamic Computation Offloading

**PyTorch · Multi-Agent Reinforcement Learning · Soft Actor-Critic ·
Edge Intelligence · UAV Networks**

------------------------------------------------------------------------

## Overview

This repository presents an end-to-end implementation of a dynamic
Vehicular Edge Computing (VEC) environment optimized through Multi-Agent
Reinforcement Learning.

The system models a realistic edge-intelligent transportation scenario
where mobile vehicles generate heterogeneous computational tasks and
autonomously determine computation placement and communication routes.

The environment supports:

-   Local computation
-   Direct vehicle-to-edge offloading
-   UAV-assisted relay communication
-   Wired multi-hop edge routing

The objective is to learn adaptive offloading policies that optimize:

-   Latency
-   Energy consumption
-   Communication reliability
-   Edge resource utilization
-   Load balancing

The simulation environment integrates:

-   10 fixed edge servers
-   5 mobile UAV relays
-   Vehicle mobility
-   Dynamic task generation
-   Local and edge processing queues
-   Communication and computation models

------------------------------------------------------------------------

# Key Contributions

## Dynamic Vehicular Edge Computing Environment

A complete VEC simulation environment was designed and implemented,
modeling:

-   Mobile vehicles generating heterogeneous workloads
-   Edge servers with limited computational capacity
-   UAV relays adapting to traffic distribution
-   Communication links with distance-dependent constraints
-   Processing queues evolving over time

Vehicle and task states are processed through timestamp-aligned
streaming inputs, enabling realistic temporal progression.

UAV positions dynamically adapt through traffic-density clustering to
follow changing vehicle distributions.

------------------------------------------------------------------------

# Multi-Agent Soft Actor-Critic Decision Framework

The decision-making component implements a discrete Multi-Agent Soft
Actor-Critic (MASAC) framework using:

**Centralized Training, Decentralized Execution (CTDE).**

Vehicles act as decision-making agents with shared policy parameters.

The implementation includes:

-   Shared-policy Actor network
-   Twin Critics
-   Target Critics
-   Replay Buffer
-   Entropy-regularized optimization
-   Soft target updates

------------------------------------------------------------------------

# Large Discrete Action Space with Feasibility Constraints

The computation offloading problem is modeled as:

``` text
A = A_local ∪ A_direct ∪ A_UAV ∪ A_wired
```

Implemented action space:

  Action Type                Number of Actions
  ------------------------ -------------------
  Local execution                            1
  Direct edge offloading                    10
  UAV relay routes                          50
  Wired one-hop routing                     26
  **Total**                             **87**

Invalid decisions are removed through hard feasibility masking before
policy selection.

------------------------------------------------------------------------

# Multi-Objective Reinforcement Learning Optimization

The reward function considers:

-   End-to-end latency
-   Execution energy
-   Packet-loss surrogate
-   Edge load imbalance
-   Deadline violations

This enables learning trade-offs between competing objectives.

------------------------------------------------------------------------

# Repository Structure

``` text
VEC-IoT-MASAC/

├── README.md
├── VEC_IoT_MASAC_Final.ipynb
│
├── report/
│   └── VEC_IoT_MASAC_Technical_Report.pdf
│
└── vec_iot_project/
    ├── train_masac.py
    ├── evaluate_masac.py
    ├── run_phase3.py
    ├── diagnose_network.py
    │
    ├── datasets/
    │   ├── tasks.xml
    │   └── vehicles.xml
    │
    ├── src/
    │   ├── masac.py
    │   ├── masac_env.py
    │   ├── system_model.py
    │   ├── uav_control.py
    │   └── xml_loader.py
    │
    ├── outputs/
    │   ├── evaluation results
    │   ├── training metrics
    │   └── trained MASAC checkpoint
    │
    └── archive/
        Previous experimental implementations
```

------------------------------------------------------------------------

# Technical Report

A detailed technical report documents:

-   Environment architecture
-   Mathematical system modeling
-   UAV positioning strategy
-   Action-space construction
-   Hard constraint formulation
-   MASAC architecture
-   CTDE training procedure
-   Evaluation methodology

------------------------------------------------------------------------

# Project Motivation

Vehicular edge computing requires intelligent decision-making under
rapidly changing conditions.

This project explores these challenges through an integrated simulation
and reinforcement learning framework combining edge intelligence,
autonomous decision-making, and multi-agent optimization.
