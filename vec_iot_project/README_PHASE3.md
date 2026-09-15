# Phase 3 MASAC upgrade

This upgrade adds the missing joint routing/offloading layer while preserving the original Phase-1 K-Means UAV mobility.

## What is now modeled

- Local execution, direct E2E vehicle-to-edge offloading, UAV-relay offloading, and one-hop wired edge-to-edge forwarding.
- Gaussian region-to-region path-loss exponent variation, received power, noise, interference-aware SINR and Shannon data rate.
- Dynamic bandwidth allocation through an active-link fraction.
- Queueing delay at edge servers and CPU-cycle based compute latency.
- Local/edge dynamic compute energy and radio/wired transmission energy.
- Packet-loss surrogate from SINR (the exact displayed packet-error formula is not present in the extracted DOCX, so this is configurable).
- MASAC-style CTDE: shared vehicle actors + centralized twin critics + entropy regularization + action masking.
- Task state contains size, cycles/bit, memory, criticality, deadline/slack, vehicle position/speed and channel previews.
- Metrics: average latency, average energy, packet loss, average data rate, load imbalance, deadline miss, and route distribution.
- Streaming XML parsing so the 200k+ vehicle rows / 260k+ task rows are not all materialized as Python objects at once.

## Run

```bash
python train_masac.py --episodes 40 --max-tasks 15000
python evaluate_masac.py --max-tasks 10000
```

The exact path-loss and packet-error equations from the assignment should replace the configurable surrogate coefficients in `src/system_model.py` because the equation graphics in `SystemModel.docx` are not represented in the extracted text.
