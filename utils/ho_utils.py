"""Opt-in first-order HO interruption and paired exogenous traffic helpers."""

import hashlib

import numpy as np


def interruption_slots(duration_ms, slot_len, slots_per_frame):
    """Require slot-aligned, sub-frame interruption (no silent rounding)."""
    value = float(duration_ms) / (1000.0 * slot_len)
    if not np.isfinite(value) or value < 0 or not np.isclose(value, round(value)):
        raise ValueError("HO interruption must be a nonnegative integer number of slots")
    value = int(round(value))
    if value >= slots_per_frame:
        raise ValueError("This first-order HO model requires interruption < one frame")
    return value


def capacity_factors(vehicles, num_bs, current_connection, blocked_slots, frame_slots):
    """BS-by-vehicle factors; initial access is not a handover."""
    if not 0 <= blocked_slots < frame_slots:
        raise ValueError("blocked_slots must lie in [0, frame_slots)")
    factors = np.ones((num_bs, len(vehicles)), dtype=float)
    if blocked_slots == 0:
        return factors
    if current_connection is None:
        raise ValueError("HO-aware capacities require the current association")
    for column, veh in enumerate(vehicles):
        if veh in current_connection:
            factors[:, column] = 1.0 / (1.0 - blocked_slots / frame_slots)
            factors[int(current_connection[veh]), column] = 1.0
    return factors


def make_paired_traffic(args, timeline, seed):
    """Generate once, replay identically across policies, independent of their RNG.

    Initial queues follow the existing uniform integer initialization. Arrivals
    retain independent Poisson counts and the existing per-vehicle mean rates.
    Separate RNG streams prevent queue initialization from shifting arrivals.
    """
    vehicle_order = sorted({v for frame in timeline.values() for v in frame}, key=repr)
    rate_rng, queue_rng, arrival_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]
    spread = args.random_factor_range4data_rate
    rates = {v: args.data_rate * rate_rng.uniform(1 - spread, 1 + spread)
             for v in vehicle_order}
    arrivals, initial_queues = {}, {}
    previous = set()
    digest = hashlib.sha256()
    for index, (frame, entries) in enumerate(timeline.items()):
        current = set(entries)
        initial_queues[frame] = {}
        arrivals[frame] = {}
        for veh in sorted(current, key=repr):
            digest.update(repr((frame, veh, rates[veh])).encode())
            if veh not in previous:
                upper = int(args.lat_slot_ub * rates[veh] * args.slot_len * 0.5)
                q0 = int(queue_rng.integers(max(upper, 1)))
                initial_queues[frame][veh] = q0
                digest.update(repr(q0).encode())
            if index > 0:
                values = arrival_rng.poisson(rates[veh] * args.slot_len,
                                            size=args.slots_per_frame)
                values.setflags(write=False)
                arrivals[frame][veh] = values
                digest.update(values.tobytes())
        previous = current
    return {"rates": rates, "initial_queues": initial_queues,
            "arrivals": arrivals, "sha256": digest.hexdigest(), "seed": seed}
