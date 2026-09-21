"""Causal prediction interface and directional queues for the revision grid.

This explicit entry point leaves the historical simulator's defaults intact.
Predictions in record x always target x+1. Newly appearing vehicles initially
use the macro BS; missing reports are never filled with private micro CSI.
"""
import collections
from types import SimpleNamespace
import numpy as np

from utils import alg_utils as alg
from utils.directional_service import DirectionalService, beam_average_gain_db
from utils.gap_refinement import GAPRefinementConfig
from utils.gpu_phy import GPUFramePHY
from utils.ho_utils import interruption_slots
from utils.pql_ba import macro_gain_db


def predicted_records(records, frame):
    result = {}
    for v, r in records.items():
        p = r['shared_prediction']
        if not np.isclose(p['source_frame'], frame, rtol=0, atol=1e-7) or not np.isclose(p['target_frame'], frame+.1, rtol=0, atol=1e-7):
            raise ValueError('Prediction provenance/timing mismatch')
        result[v] = p
    return result


def oracle_prediction(record):
    return dict(gain=np.asarray(record['g_opt_beam']),
                interference=beam_average_gain_db(record['h']),
                beam=np.asarray(record['best_beam_pair_idx']).reshape(4, 1))


def run_revised_meet(args, locations, timeline, method, traffic, seed=1,
                     device='cpu', ho_ms=10., k=5, diagnostics=None,
                     service_diagnostics=None, gap_diagnostics=None,
                     progress_callback=None):
    allowed = {'meet_cobra', 'oracle_mc', 'reactive_obra', 'wo_gap_ho', 'wo_pet_bf', 'wo_otr_ra'}
    if method not in allowed:
        raise ValueError(method)
    frames = sorted(timeline)
    if len(frames) < 4 or not np.allclose(np.diff(frames), .1):
        raise ValueError('Need contiguous frames including context and warmup')
    oracle = method == 'oracle_mc'
    reactive = method == 'reactive_obra'
    random_bf = method in ('reactive_obra', 'wo_pet_bf')
    greedy = method in ('reactive_obra', 'wo_gap_ho')
    n, slots = len(frames)-1, args.slots_per_frame
    ho_slots = interruption_slots(ho_ms, args.slot_len, slots)
    caps = np.array([args.num_RB_macro]+[args.num_RB_micro]*4)
    powers = np.array([args.p_macro]+[args.p_micro]*4)
    bs_locations = np.asarray([(0., 0.), *locations])
    rates = traffic['rates']
    upper = {v: args.lat_slot_ub*args.slot_len*r for v, r in rates.items()}
    connection_last = {v: 0 for v in timeline[frames[0]]}
    queues_last = dict(traffic['initial_queues'][frames[0]])
    commands_last = {}
    occupancy_last = caps.astype(float).copy()
    previous_beams, measured_history = {}, {}
    result = SimpleNamespace(**{name: np.zeros(n) for name in
        ('energy_record', 'handover_record', 'violation_probability_record',
         'average_queue_record', 'pilot_record', 'decision_record', 'trigger_record',
         'optimizer_failure_record', 'optimizer_overflow_record')})
    result.rb_allocated_record = np.zeros((n, 5))
    result.queue_per_vehicle_record = collections.OrderedDict()
    result.association_record = collections.OrderedDict()
    np.random.seed(seed)
    for fi, frame in enumerate(frames[1:]):
        records = timeline[frame]
        ids = sorted(records, key=str)
        prev_records = timeline[frames[fi]]
        reports = predicted_records(records, frame) if not (oracle or reactive) else None
        prev_reports = predicted_records(prev_records, frames[fi]) if not (oracle or reactive) else None
        connection = {v: int(commands_last.get(v, connection_last.get(v, 0))) if v in connection_last else 0 for v in ids}
        switched = {v for v in ids if v in connection_last and connection[v] != connection_last[v]}
        q = {v: np.zeros(slots+1) for v in ids}
        for v in ids:
            q[v][0] = queues_last[v] if v in connection_last else traffic['initial_queues'][frame][v]
        macro = {v: macro_gain_db(args, records[v]['pos'], np.zeros(2)) for v in ids}
        planning_gain, planning_interference, current_interference, beam_candidates, beam_threshold = {}, {}, {}, {}, {}
        for v in ids:
            if oracle:
                current = oracle_prediction(records[v])
                future_record = timeline[frames[fi+2]].get(v, records[v]) if fi+2 < len(frames) else records[v]
                future = oracle_prediction(future_record)
            elif reactive:
                # Explicit privileged current-measurement reference, never NN.
                current = future = oracle_prediction(records[v])
            else:
                future = reports[v]
                current = prev_reports.get(v)
                if current is None:
                    if connection[v] != 0:
                        raise ValueError('Micro service has no causal report')
                    current = dict(gain=np.full(4, -180.), interference=np.full(4, -180.),
                                   beam=np.tile(np.arange(k), (4, 1)))
            planning_gain[v] = np.r_[macro[v], future['gain']].astype(float)
            planning_interference[v] = np.r_[macro[v], future['interference']]
            current_interference[v] = np.r_[macro[v], current['interference']]
            beam_candidates[v] = np.asarray(current['beam'])[:, :k]
            beam_threshold[v] = np.asarray(current['gain'])
            # Only actually observed serving links enter robustness smoothing.
            if not (oracle or reactive):
                for bs, (last_frame, gain) in measured_history.get(v, {}).items():
                    weight = .1**(round((frame-last_frame)/.1)+1)
                    planning_gain[v][bs] = weight*gain+(1-weight)*planning_gain[v][bs]
        planning_pilots = {v: np.full(4, 1 if oracle else k) for v in ids}
        captured = []
        def capture(c, a, b):
            captured[:] = [(c.copy(), a.copy(), b.copy())]
        ho = alg.HO_EE_Greedy_offload if greedy else alg.HO_EE_GAP_APX_SINR_conservative_adaptive
        commands, occupancy_next = ho(args, ids, q, rates,
            {v: records[v]['pos'] for v in ids}, planning_gain, bs_locations,
            infer_g_dict=planning_interference, num_pilot_dict=planning_pilots,
            current_connection=connection, ho_capacity_correction=True, ho_interruption_slots=ho_slots,
            gap_cap_rb_usage=True, gap_refinement_config=GAPRefinementConfig(2, None, 1.1),
            gap_refinement_diagnostics=gap_diagnostics, gap_problem_callback=capture,
            vio_prob_history=result.violation_probability_record[:fi])
        occupancy_next = np.clip(occupancy_next, 0, caps)
        physical = GPUFramePHY(args, records, frame, seed, device)
        assert physical.ids == ids
        if random_bf:
            blocked_mask = np.array([[v in switched and slot < ho_slots for v in ids] for slot in range(slots)])
            gains, _, pairs, pilots = physical.random_tracking(previous_beams,
                (int(seed)*999983+round(frame*10)) % 2**32, k, blocked_mask)
        else:
            gains, _, pairs, pilots = physical.pet(beam_candidates, beam_threshold, 1 if oracle else k)
        evaluator = DirectionalService(physical, pairs, connection, seed, frame, service_diagnostics)
        del physical
        association = {bs: [v for v in ids if connection[v] == bs] for bs in range(5)}
        arrival = traffic['arrivals'][frame]
        energy = pilot_sum = 0.
        ra = alg.RA_OTR3_SINR if method in ('reactive_obra', 'wo_otr_ra') else alg.RA_OTR_SINR
        for slot in range(slots):
            blocked = switched if slot < ho_slots else set()
            gain_slot, pilot_slot = {}, {}
            for j, v in enumerate(ids):
                gain_slot[v] = np.r_[macro[v], np.full(4, -180.)]
                bs = connection[v]
                if bs > 0:
                    gain_slot[v][bs] = gains[slot, j, bs-1]
                pilot_slot[v] = pilots[slot, j].copy() if v not in blocked else np.zeros(4)
                if bs > 0:
                    pilot_sum += pilot_slot[v][bs-1]
            allocation = collections.OrderedDict((v, 0) for v in blocked)
            rb = np.zeros(5)
            for bs in range(5):
                local = ra(args, slot_idx=slot, BS_id=bs, veh_set=[v for v in association[bs] if v not in blocked],
                    veh_data_rate_dict=rates, Q_ub_dict=upper, q_dict=q, a_dict=arrival,
                    g_dict=gain_slot, num_pilot_dict=pilot_slot, BS_association_dict=association,
                    infer_g_dict=current_interference,
                    est_num_RB_allocated_perBS=caps if method in ('reactive_obra','wo_otr_ra') else occupancy_last)
                allocation.update(local)
                rb[bs] = sum(local.values())
            q = evaluator.update(args, slot_idx=slot, RA_dict=allocation, connection_dict=connection,
                backlog_queue_dict=q, a_dict=arrival, g_dict=gain_slot, num_pilot_dict=pilot_slot)
            for v in blocked:
                assert allocation[v] == 0 and q[v][slot+1] == q[v][slot]+arrival[v][slot]
            result.rb_allocated_record[fi] += rb/slots
            energy += float(rb@powers)*args.slot_len
        for j, v in enumerate(ids):
            bs = connection[v]
            first = ho_slots if v in switched else 0
            if bs > 0:
                measured_history.setdefault(v, {})[bs] = (frame, float(10*np.log10(np.mean(10**(gains[first:, j, bs-1]/10)))))
        previous_beams = {v: pairs[-1, j].copy() for j, v in enumerate(ids)}
        measured_history = {v: history for v, history in measured_history.items() if v in connection}
        result.energy_record[fi] = energy
        result.pilot_record[fi] = pilot_sum/(slots*len(ids))
        result.handover_record[fi] = len(switched)
        result.average_queue_record[fi] = np.mean([q[v][1:] for v in ids])
        result.violation_probability_record[fi] = np.mean([q[v][1:] > upper[v] for v in ids])
        result.queue_per_vehicle_record[fi] = {v: q[v][1:].copy() for v in ids}
        result.association_record[fi] = dict(connection)
        diagnostic = dict(frame=float(frame), association=dict(connection), switched=sorted(switched,key=str),
            blocked_vehicle_slots=ho_slots*len(switched), active_vehicle_slots=slots*len(ids),
            prediction_source_frame=float(frames[fi]), prediction_target_frame=float(frame),
            estimated_rb=occupancy_last.tolist())
        if oracle and captured:
            c, a, b = captured[0]
            relaxed = alg.solve_gap_lp(c, a, b)
            diagnostic['relaxation_feasible'] = relaxed is not None
            diagnostic['relaxed_p2_power_w'] = float(relaxed[1]) if relaxed is not None else float(caps@powers)
        if diagnostics is not None:
            diagnostics.append(diagnostic)
        commands_last, occupancy_last = commands, occupancy_next
        connection_last = connection
        queues_last = {v: q[v][-1] for v in ids}
        if progress_callback:
            progress_callback(fi+1, n)
    return result
