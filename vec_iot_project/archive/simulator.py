import math

# Calibrated VEC/MEC parameters (Phase 3.1)
VEHICLE_CPU = 1.0e9
EDGE_CPU = 12.0e9
UAV_CPU = 6.0e9
BANDWIDTH = 20e6
TX_POWER = 0.5
NOISE = 1e-13
CHANNEL_GAIN = 1.0
ENERGY_COEFF = 1e-27


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def rate(d):
    # simplified log-distance channel model
    # reference-distance path loss model
    gain = CHANNEL_GAIN / (max(d, 1) ** 3.0)
    snr = TX_POWER * gain / NOISE
    return max(BANDWIDTH * math.log2(1 + snr), 1e5)


def bits(task):
    return task.data_size * 8e6


def cycles(task):
    return bits(task) * task.cycles_per_bit


def compute_time(task, cpu):
    return cycles(task) / cpu


def compute_energy(task, cpu):
    return ENERGY_COEFF * cycles(task) * cpu * cpu


def transmission(task, distance):
    r = rate(distance)
    t = bits(task) / r
    e = TX_POWER * t
    return t, e


def local(task):
    t = compute_time(task, VEHICLE_CPU)
    e = compute_energy(task, VEHICLE_CPU)
    return t, e


def edge(task, vehicle, edge_pos=(1000, 1000)):
    tx_t, tx_e = transmission(task, dist((vehicle.x, vehicle.y), edge_pos))
    comp_t = compute_time(task, EDGE_CPU)
    return tx_t + comp_t, tx_e


def uav_relay(task, vehicle, uav=(800,800), edge_pos=(1000,1000)):
    t1, e1 = transmission(task, dist((vehicle.x, vehicle.y), uav))
    t2, e2 = transmission(task, dist(uav, edge_pos))
    comp = compute_time(task, EDGE_CPU)
    relay = compute_time(task, UAV_CPU)
    return t1+t2+relay+comp, e1+e2


def decide(task, vehicle):
    values = {
        "LOCAL": local(task),
        "EDGE": edge(task, vehicle),
        "UAV_RELAY": uav_relay(task, vehicle)
    }
    choice = min(values, key=lambda x: values[x][0])
    return choice, values
