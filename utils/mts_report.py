"""MTS matching driven only by the frozen prediction report and causal feedback.

The original MTS implementation is intentionally untouched. Beam selection is
deferred to a paid physical probe at execution time, never to this controller.
"""
from dataclasses import dataclass
import math
import numpy as np

from utils.alg_utils import estimate_num_RB_allocated_perBS
from utils.dql_hbt import effective_sinr_db
from utils.mts_gs_hbf import MTSLinkCandidate, capacity_aware_gale_shapley
from utils.pql_ba import macro_gain_db
from utils.pql_ba_adapted import _capacity_per_rb_bps, _interference_db


@dataclass(frozen=True)
class PredictionReport:
    source_frame: float
    position: np.ndarray
    gain: np.ndarray
    interference: np.ndarray
    beams: np.ndarray


@dataclass(frozen=True)
class ReportCommand:
    target_bs: int
    report: PredictionReport


def reports_from_records(records, frame, config, k=5):
    """Whitelist the actual reporting interface; no H or Oracle label access."""
    reports = {}
    for v in sorted(records, key=str):
        raw = records[v]["shared_prediction"]
        arrays = [np.array(records[v]["pos"], dtype=float, copy=True),
                  np.array(raw["gain"], dtype=float, copy=True),
                  np.array(raw["interference"], dtype=float, copy=True),
                  np.array(raw["beam"], copy=True)]
        pos, gain, interference, beams = arrays
        assert pos.shape == (2,) and gain.shape == interference.shape == (config.num_micro_bs,)
        assert beams.shape == (config.num_micro_bs,k)
        assert np.issubdtype(beams.dtype,np.integer)
        assert np.isfinite(gain).all() and np.isfinite(interference).all()
        assert (beams >= 0).all() and (beams < config.full_sweep_pilots).all()
        assert all(len(set(row)) == k for row in beams)
        for array in arrays:
            array.setflags(write=False)
        reports[v] = PredictionReport(float(frame), *arrays)
    return reports


def estimate_report_load(args, reports, connection, rates, config, bs_locations,
                         last_measurements, macro_loc=(0.,0.)):
    """Use predicted interference and the last measured serving gain if known.

    A predicted optimum is only a candidate-link estimate; it is not substituted
    for a measured gain of an arbitrary retained serving beam.
    """
    desired, interference = {}, {}
    for v,r in reports.items():
        macro = macro_gain_db(args,r.position,np.asarray(macro_loc))
        desired[v] = np.r_[macro,r.gain]
        interference[v] = np.r_[macro,r.interference]
        if v in last_measurements:
            observed_bs, observed_gain = last_measurements[v]
            if observed_bs == connection[v] and observed_bs > 0:
                desired[v][observed_bs] = observed_gain
    estimated = estimate_num_RB_allocated_perBS(args,connection,bs_locations,
        list(reports),desired,rates,infer_g_dict=interference)
    capacities = np.array([args.num_RB_macro]+[args.num_RB_micro]*config.num_micro_bs)
    return estimated, np.clip(estimated/capacities,0.,1.5)


def build_report_candidates(args,reports,connection,queue,upper,rates,load,config,k=5,
                            macro_loc=(0.,0.),pilot_average_override=None):
    """Same preference equations as MTS; replace privileged channel inputs.

    ``pilot_average_override`` exists solely for an exact equation parity test;
    production runs always use the actual finite-probe overhead below.
    """
    assert config.local_tracking_interval_frames == 1
    load = np.asarray(load,dtype=float)
    assert load.shape == (config.num_bs,) and np.isfinite(load).all()
    duration = args.slots_per_frame*args.slot_len
    capacities = np.array([args.num_RB_macro]+[args.num_RB_micro]*config.num_micro_bs)
    # Five probes replace (not supplement) the normal tracking pilot in one slot.
    nominal_pilots = config.tracking_pilots + (k-config.tracking_pilots)/args.slots_per_frame
    if pilot_average_override is not None:
        nominal_pilots = pilot_average_override
    result = {}
    for v,r in reports.items():
        current = connection[v]
        ratio = float(queue[v]/max(upper[v],1e-12))
        pressure = float(np.clip((ratio-config.pressure_start_ratio)/
            (config.pressure_full_ratio-config.pressure_start_ratio),0,1)) if config.pressure_adaptive else 0.
        def blend(normal,urgent):
            return float(normal+pressure*(urgent-normal))
        load_weight = blend(config.load_weight,config.urgent_load_weight)
        energy_weight = blend(config.energy_weight,config.urgent_energy_weight)
        ho_penalty = blend(config.handover_penalty,config.urgent_handover_penalty)
        hysteresis = blend(config.handover_hysteresis,config.urgent_handover_hysteresis)
        queue_weight = blend(config.bs_queue_weight,config.urgent_bs_queue_weight)
        drain_weight = blend(config.queue_drain_weight,config.urgent_queue_drain_weight)
        bits = rates[v]*duration + drain_weight*min(queue[v],config.queue_drain_cap_ratio*upper[v])
        links = []
        for bs in range(config.num_bs):
            if bs == 0:
                gain = macro_gain_db(args,r.position,np.asarray(macro_loc))
                inter = -np.inf
                pilots,power,tx,rx = 0.,args.p_macro,None,None
            else:
                gain = float(r.gain[bs-1])
                inter = _interference_db(args,bs,np.r_[-180.,r.interference],load)
                pilots,power = nominal_pilots,args.p_micro
                pair = int(r.beams[bs-1,0])
                tx,rx = pair//config.num_rx_beams,pair%config.num_rx_beams
            sinr = effective_sinr_db(args,bs,gain,inter)
            capacity = _capacity_per_rb_bps(args,bs,gain,inter,pilots)
            demand = float(np.clip(bits/max(capacity*duration,1e-12),config.minimum_demand_rb,capacities[bs]))
            mbps = capacity/1e6
            score = (config.rate_weight*math.log1p(max(mbps,0.))
                     -load_weight*(load[bs]+demand/capacities[bs])
                     -energy_weight*power/max(mbps,1e-9)-ho_penalty*(bs!=current))
            bs_score = (queue_weight*min(ratio,10.)+config.bs_rate_weight*math.log1p(max(mbps,0.))
                        -config.bs_demand_weight*demand/capacities[bs]+config.bs_stay_bonus*(bs==current))
            corrected = demand/(1-config.ho_interruption_ms/(1000*duration)) if bs!=current else demand
            if sinr >= config.sinr_threshold_db or bs in (0,current):
                links.append(MTSLinkCandidate(v,bs,tx,rx,gain,sinr,capacity,corrected,float(score),float(bs_score)))
        staying = next((x for x in links if x.bs==current),None)
        other = max((x.vehicle_score for x in links if x.bs!=current),default=-np.inf)
        if staying is not None and other < staying.vehicle_score+hysteresis:
            from dataclasses import replace
            links = [replace(x,vehicle_score=max(x.vehicle_score,other+1e-6)) if x.bs==current else x for x in links]
        result[v] = links
    return result


def report_matching(args,reports,connection,queue,upper,rates,load,config,k=5,macro_loc=(0.,0.)):
    links = build_report_candidates(args,reports,connection,queue,upper,rates,load,config,k,macro_loc)
    matching = capacity_aware_gale_shapley(links,[args.num_RB_macro]+[args.num_RB_micro]*config.num_micro_bs,
        current_bs=connection,admission_capacity_factor=config.admission_capacity_factor)
    return {v:ReportCommand(int(c.bs),reports[v]) for v,c in matching.assignments.items()},matching
