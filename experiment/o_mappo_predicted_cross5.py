#!/usr/bin/env python3
"""Prediction-only O-MAPPO decisions with paid HO32/cross5 physical service.

This experiment leaves the formal baselines and manuscript untouched. Decision
records deliberately omit H, optimal-beam labels and raw pilot observations.
The same slot simulator is used for PPO rollouts, validation and final tests.
"""
import collections
import dataclasses
import os
from pathlib import Path
import sys
from types import SimpleNamespace

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from numba import njit
from utils import o_mappo as om
from utils.gpu_phy import GPUFramePHY
from utils.directional_service import DirectionalService
from utils.ho_utils import make_paired_traffic, interruption_slots
from experiment.o_mappo_slot_tracking import track_frame
from experiment.o_mappo_target_check import estimate_bounded
from experiment.o_mappo_optimizer_information import fixed_allocation
from experiment.pql_ba_experiment import paper_args
from utils.hierarchical_beam import coarse_codebook


def bounded_prediction_load(args, connection, gains, interference, rates):
    """Vectorized arithmetic of estimate_bounded, retaining its stopping rule."""
    ids = list(connection)
    caps = np.array([args.num_RB_macro]+[args.num_RB_micro]*4, dtype=float)
    occupied = np.minimum(np.full(5, args.num_RB_micro, dtype=float), caps)
    if not ids: return np.zeros(5)
    association = np.array([connection[v] for v in ids])
    desired = 10**(np.array([gains[v] for v in ids])/10)
    interfering = 10**(np.array([interference[v][1:] for v in ids])/10)
    arrival = np.array([rates[v] for v in ids])
    bandwidth = np.array([args.RB_intervel_macro]+[args.RB_intervel_micro]*4)
    power = np.array([args.p_macro]+[args.p_micro]*4)
    noise = args.N0*bandwidth*10**(np.array([args.NF_macro_dB]+[args.NF_micro_dB]*4)/10)
    for _ in range(10):
        parts = interfering*(args.p_micro*occupied[None, 1:]/args.num_RB_micro)
        inter = np.zeros_like(desired)
        # Explicit sums avoid subtractive cancellation of a strong serving link.
        for b in range(4): inter[:, b+1] = parts[:, [j for j in range(4) if j != b]].sum(1)
        demand = arrival[:, None]/(1e-10+bandwidth*np.log2(1+power*desired/(noise+inter))+1e-10)
        totals = np.bincount(association, weights=demand[np.arange(len(ids)), association], minlength=5)
        updated = np.clip(totals, 0, caps)
        converged = np.allclose(occupied, updated, atol=1)
        occupied = updated
        if converged: break
    return occupied


def public_records(records, frame):
    """A whitelist, not a shallow copy that retains hidden physical fields."""
    result = {}
    for v, r in records.items():
        p = r['shared_prediction']
        if not (np.isclose(p['source_frame'], frame, atol=1e-7, rtol=0)
                and np.isclose(p['target_frame'], frame+.1, atol=1e-7, rtol=0)):
            raise ValueError('Prediction report has the wrong source/target time')
        gains = {key: np.asarray(p[key], dtype=float).copy() for key in ('gain', 'interference')}
        if any(a.shape != (4,) or not np.isfinite(a).all() for a in gains.values()):
            raise ValueError('Malformed prediction report')
        result[v] = dict(pos=np.asarray(r['pos']).copy(), angle=float(r.get('angle', 0)),
                         v=float(r.get('v', 0)), shared_prediction=gains)
    return result


def context(args, records, connection, rates):
    """Shared E_all occupancy refinement, with predictions replacing true gains."""
    gains, interference, serving = {}, {}, {}
    for v, r in records.items():
        macro = om.macro_gain_db(args, r['pos'], np.zeros(2))
        gains[v] = np.r_[macro, r['shared_prediction']['gain']]
        interference[v] = np.r_[macro, r['shared_prediction']['interference']]
        serving[v] = float(gains[v][connection[v]])
    rb = bounded_prediction_load(args, connection, gains, interference, rates)
    caps = np.array([args.num_RB_macro]+[args.num_RB_micro]*4)
    return SimpleNamespace(gains=gains, interference=interference, serving=serving,
                           rb=rb, load=rb/caps, caps=caps)


def decide(args, records, states, queues, rates, cfg, policy, previous_throughput,
           explore=False):
    """No physical-channel arguments are accepted at this boundary."""
    ids = sorted(records, key=str)
    connection = {v: int(states[v].action) for v in ids}
    ctx = context(args, records, connection, rates)
    counts = np.bincount(list(connection.values()), minlength=5)
    local, due = {}, []
    for v in ids:
        s, r, b = states[v], records[v], connection[v]
        s.distance_since_event += float(np.linalg.norm(r['pos']-s.last_position))
        s.last_position = r['pos'].copy()
        if s.distance_since_event+1e-9 >= cfg.zone_size_m:
            s.distance_since_event %= cfg.zone_size_m
            due.append(v)
        inter = om._interference_db(args, b, ctx.interference[v], ctx.load)
        sinr = om.effective_sinr_db(args, b, ctx.serving[v], inter)
        local[v] = om.make_local_state(cfg, r['pos'], r['angle'], r['v'], b, sinr,
            queues[v]/(rates[v]*.02), rates[v]/1e6, ctx.load, counts, inter,
            s.last_handover, previous_throughput, s.previous_rb_fraction,
            s.tx_beam, s.rx_beam, predicted_link_state=(sinr, inter, ctx.load))
    global_state = om.make_global_state(np.stack([local[v] for v in ids]), len(ids),
                                         feature_count=om.critic_local_feature_count(cfg))
    updates = {}
    optimization = None
    if due:
        batch = np.stack([local[v] for v in due])
        actions, logps, values = policy.act(batch, global_state, explore=explore)
        triggered = [v for v, a in zip(due, actions) if int(a)]
        backlog = {v: queues[v]+rates[v]*.1 for v in ids}
        fixed = fixed_allocation(args, states, backlog, ctx.serving, ctx.interference,
                                 ctx.load, ctx.rb, cfg)
        optimization = om.optimize_triggered_targets(args, records, states, triggered,
            backlog, fixed, ctx.load, cfg, None, None, np.zeros(2), solver='milp')
        for j, v in enumerate(due):
            action = int(actions[j])
            target = int(optimization.targets[v]) if action else connection[v]
            updates[v] = dict(command=om.OMAPPOCommand(action, target), local=batch[j].copy(),
                global_state=global_state.copy(), action=action, logp=float(logps[j]),
                value=float(values[j]))
    return ctx, updates, optimization


def apply_command(state, command):
    """Only update association. No free nominal beam search before acquisition."""
    before = state.action
    state.last_trigger = 0 if command is None else int(command.trigger)
    if command is not None:
        state.action = int(command.target_bs)
    changed = state.action != before
    state.last_handover = changed
    state.current_sweep_pilots = 32 if changed and state.action > 0 else 0
    if state.action == 0:
        state.tx_beam = state.rx_beam = None
    elif changed:
        # Placeholder only: private track_frame overwrites it in the first
        # post-interruption acquisition slot. The actor sees no unmeasured beam.
        state.tx_beam = state.rx_beam = 0
    return om.OMAPPOActionOutcome(changed, changed, state.current_sweep_pilots)


@njit(cache=True)
def _select_paid(fine, wide, current, first, acquire):
    """Private batched samples; selection reads only 16+16 or 5 paid entries."""
    slots, n, _ = fine.shape
    chosen = np.zeros((slots,n),dtype=np.int64)
    measured = np.zeros((slots,n))
    pilots = np.zeros((slots,n),dtype=np.int64)
    for s in range(slots):
        for j in range(n):
            if s < first[j]:
                chosen[s,j]=current[j]; continue
            if acquire[j] and s == first[j]:
                sector = np.argmax(wide[s,j])
                tx0,rx0 = (sector%8)*4,(sector//8)*4
                candidates = np.array([(tx0+t)*8+rx0+r for t in range(4) for r in range(4)])
                pilots[s,j]=32
            else:
                t,r=current[j]//8,current[j]%8
                candidates=np.array([current[j],((t-1)%32)*8+r,((t+1)%32)*8+r,t*8+(r-1)%8,t*8+(r+1)%8])
                pilots[s,j]=5
            best=candidates[0]
            for pair in candidates[1:]:
                if fine[s,j,pair] > fine[s,j,best]: best=pair
            current[j]=best;chosen[s,j]=best;measured[s,j]=fine[s,j,best]
    return chosen,measured,pilots,current


@torch.inference_mode()
def track_frame_batched(physical, connection, states, ho_slots=10):
    """Equivalent paid searches with GPU batching instead of 100 host round trips.

Full beam responses are private simulator buffers, NEVER policy inputs. Only
the scheduled measurements are consulted by _select_paid; cost is unchanged.
"""
    ids, device = physical.ids, physical.device
    slots,n=physical.h.shape[:2]
    gains=np.full((slots,n),-180.);pilots=np.zeros((slots,n),dtype=int)
    pairs=np.zeros((slots,n,4),dtype=int)
    micro=np.array([j for j,v in enumerate(ids) if connection[v]>0],dtype=int)
    if not len(micro):return gains,pilots,pairs,{}
    bs=np.array([connection[ids[j]]-1 for j in micro])
    h=physical.h.permute(0,1,3,2,4)[:,torch.as_tensor(micro,device=device),torch.as_tensor(bs,device=device)]
    fine=torch.einsum('svrt,tk->svrk',h,physical.tx)
    fine=torch.einsum('svrk,rl->svkl',fine,physical.rx.conj()).abs().flatten(-2).cpu().numpy()
    wt=torch.tensor(coarse_codebook(32,8),dtype=h.dtype,device=device)
    wr=torch.tensor(coarse_codebook(8,2),dtype=h.dtype,device=device)
    wide=torch.einsum('ra,svrt,tk->svak',wr.conj(),h,wt).abs().flatten(-2).cpu().numpy()
    initial=np.array([states[ids[j]].tx_beam*8+states[ids[j]].rx_beam for j in micro])
    blocked=np.array([ho_slots if states[ids[j]].last_handover else 0 for j in micro])
    acquisition=np.array([states[ids[j]].current_sweep_pilots==32 for j in micro])
    chosen,measured,paid,final=_select_paid(fine,wide,initial,blocked,acquisition)
    gains[:,micro]=20*np.log10(measured/16+1e-9)
    pilots[:,micro]=paid;pairs[:,micro,bs]=chosen
    return gains,pilots,pairs,{ids[j]:int(p) for j,p in zip(micro,final)}


@njit(cache=True)
def _serve(q0, arrivals, bs, b, order, caps, blocked, pilots, own, cross,
           permutations, macro_b, noise, p_micro, bandwidth_dt, threshold, pilot_factor):
    """Exact OTR rounding/priority and explicit-RB directional queue updates."""
    slots, n = b.shape
    cap = caps[1]
    q = np.empty((n, slots+1))
    q[:, 0] = q0
    allocation = np.zeros((slots, n), dtype=np.int64)
    served = np.zeros(n)
    for slot in range(slots):
        remaining = caps.copy()
        for j in order[slot]:
            if slot < blocked[j]:
                continue
            ratio = q[j, slot]/b[slot, j]
            needed = np.ceil(ratio) if q[j, slot] > .9*threshold[j] else np.floor(ratio)
            k = int(min(needed, remaining[bs[j]]))
            allocation[slot, j] = k
            remaining[bs[j]] -= k
        owners = np.full((4, cap), -1, dtype=np.int64)
        # Same sorted-user concatenation and RNG permutations as DirectionalService.
        for cell in range(1, 5):
            offset = 0
            for j in range(n):
                if bs[j] == cell:
                    for _ in range(allocation[slot, j]):
                        owners[cell-1, permutations[slot, cell-1, offset]] = j
                        offset += 1
        for j in range(n):
            service = 0.
            if bs[j] == 0:
                service = allocation[slot, j]*macro_b[j]
            else:
                for rb in range(cap):
                    if owners[bs[j]-1, rb] != j:
                        continue
                    inter = 0.
                    for other in range(1, 5):
                        k = owners[other-1, rb]
                        if other != bs[j] and k >= 0:
                            inter += p_micro*cross[slot, j, k]
                    service += np.log2(1+p_micro*own[slot, j]/(noise+inter))
                service *= bandwidth_dt*(1-min(pilots[slot, j]*pilot_factor, 1.))
            served[j] += min(q[j, slot], service)
            q[j, slot+1] = max(q[j, slot]-service, 0.)+arrivals[j, slot]
    return q, allocation, served


def serve_frame(args, physical, states, connection, queues, arrivals, interference,
                rb_estimate, seed, frame):
    """Private PHY: only paid serving-link samples and realized feedback leave it."""
    ids = physical.ids
    gains, pilots, pairs, final = track_frame_batched(physical, connection, states,
        interruption_slots(10, args.slot_len, args.slots_per_frame))
    evaluator = DirectionalService(physical, pairs, connection, seed, frame)
    bs = np.array([connection[v] for v in ids])
    caps = np.array([args.num_RB_macro]+[args.num_RB_micro]*4, dtype=np.int64)
    slots, n = pilots.shape
    macro = np.array([om.macro_gain_db(args, physical.records[v]['pos'], np.zeros(2)) for v in ids])
    ng = args.N0*args.RB_intervel_macro*10**(args.NF_macro_dB/10)
    macro_b = args.slot_len*args.RB_intervel_macro*np.log2(1+args.p_macro*10**(macro/10)/ng)
    noise = args.N0*args.RB_intervel_micro*10**(args.NF_micro_dB/10)
    inter = np.array([sum(10**(interference[v][k]/10)*args.p_micro*rb_estimate[k]/caps[k]
                         for k in range(1, 5) if k != connection[v]) for v in ids])
    b = (1-np.minimum(pilots*args.pilot_overhead_factor, 1))*args.RB_intervel_micro*args.slot_len*np.log2(
        1+args.p_micro*10**(gains/10)/(noise+inter[None]))
    b[:, bs == 0] = macro_b[bs == 0]
    if not np.isfinite(b).all() or np.any(b <= 0):
        raise ValueError('Invalid achievable bit count')
    # NumPy's default argsort matches the existing OTR-RA (including its ties).
    order = np.concatenate([np.flatnonzero(bs == k)[np.argsort(-b[:, bs == k], axis=1)]
                            for k in range(5)], axis=1)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), round(frame*10), 24681357]))
    permutations = np.array([[rng.permutation(caps[1]) for _ in range(4)] for _ in range(slots)])
    blocked = np.array([10 if states[v].last_handover else 0 for v in ids])
    q, allocation, served = _serve(np.array([queues[v] for v in ids]),
        np.array([arrivals[v] for v in ids]), bs, b, order, caps, blocked, pilots,
        evaluator.own, evaluator.cross, permutations, macro_b, noise, args.p_micro,
        args.RB_intervel_micro*args.slot_len, np.full(n, args.data_rate*.02), args.pilot_overhead_factor)
    if np.any(allocation[np.arange(slots)[:, None] < blocked[None]]):
        raise AssertionError('RBs allocated during HO interruption')
    for v, beam in final.items():
        states[v].tx_beam, states[v].rx_beam = divmod(beam, 8)
    return q, allocation, served, pilots, gains


def configuration(reference):
    return dataclasses.replace(reference, state_variant='predicted_adapted',
        information_mode='shared_prediction', tracking_pilots=5, ho_interruption_ms=10,
        entropy_coefficient=.002)


def simulate(timeline, policy, rate, seed, device, learn=False, progress=None):
    """One common exact environment; decisions precede service as in formal runs.

Actions issued in frame x execute in x+1. Rewards from frame x therefore
belong to the preceding action, not to the pending command issued in x.
"""
    args = paper_args(rate*1e6)
    cfg = policy.config
    assert cfg.information_mode == 'shared_prediction' and cfg.tracking_pilots == 5
    traffic = make_paired_traffic(args, timeline, seed)
    frames = sorted(timeline)
    states, queues = {}, {}
    def new_state(v, r, frame):
        states[v] = om.OMAPPOLearnerState(action=0, rx_beam=None, tx_beam=None,
            pending_action=None, last_position=np.asarray(r['pos']).copy(), distance_since_event=cfg.zone_size_m)
        queues[v] = float(traffic['initial_queues'][frame][v])
    for v, r in timeline[frames[0]].items():
        new_state(v, r, frames[0])
    n = len(frames)-1
    result = SimpleNamespace(**{k: np.zeros(n) for k in ('energy_record', 'handover_record',
        'violation_probability_record', 'average_queue_record', 'pilot_record',
        'optimizer_failure_record', 'optimizer_overflow_record', 'decision_record', 'trigger_record')})
    result.rb_allocated_record = np.zeros((n, 5))
    result.queue_per_vehicle_record = collections.OrderedDict()
    result.association_record = collections.OrderedDict()
    previous_throughput = 0.
    memory, rewards, probes, diagnostic = om.OMAPPOMemory(), [], [], []
    reward_cfg = om.o_mappo_reward_presets()['qos_energy020_load1']
    torch.manual_seed(seed)
    for fi, frame in enumerate(frames[1:]):
        records = timeline[frame]
        ids = sorted(records, key=str)
        for v in set(states)-set(ids):
            if states[v].transition_frames:
                r = om._finalize_transition(states[v], v, memory, 0., True)
                if r is not None: rewards.append(r)
            else:
                for t in reversed(memory.transitions):
                    if t.vehicle == v:
                        t.done = True; t.next_value = 0.; break
            del states[v], queues[v]
        for v in ids:
            if v not in states: new_state(v, records[v], frame)
        outcomes = {}
        for v in ids:
            s = states[v]
            outcomes[v] = apply_command(s, s.pending_command)
            s.pending_command = None
        connection = {v: states[v].action for v in ids}
        # Do not expose a placeholder post-HO beam as a measured beam to the actor.
        fresh = [v for v in ids if states[v].current_sweep_pilots == 32]
        for v in fresh: states[v].tx_beam = states[v].rx_beam = None
        public = public_records(records, frame)
        ctx, updates, optimization = decide(args, public, states, queues,
            traffic['rates'], cfg, policy, previous_throughput, explore=learn)
        for v in fresh: states[v].tx_beam = states[v].rx_beam = 0
        # The previous frame's prediction is the only unmeasured interference
        # input used for service allocation in the current frame.
        previous = public_records(timeline[frames[fi]], frames[fi])
        current = {v: dict(public[v], shared_prediction=previous[v]['shared_prediction'])
                   for v in ids if v in previous}
        for v in ids:
            if v not in current:
                if connection[v] != 0: raise AssertionError('No causal report for a micro user')
                current[v] = dict(public[v], shared_prediction=dict(gain=np.full(4, -180.), interference=np.full(4, -180.)))
        ra_ctx = context(args, current, connection, traffic['rates'])
        physical = GPUFramePHY(args, records, frame, seed, device)
        physical.records = records
        q, allocation, served, pilots, gains = serve_frame(args, physical, states, connection,
            queues, traffic['arrivals'][frame], ra_ctx.interference, ra_ctx.rb, seed, frame)
        del physical
        counts = np.bincount(list(connection.values()), minlength=5)
        rb = np.array([allocation[:, np.array(list(connection.values())) == k].sum()/args.slots_per_frame for k in range(5)])
        per_user_rb = allocation.mean(0)
        q_end = {v: float(q[j, -1]) for j, v in enumerate(ids)}
        step = om.OMAPPOFluidStep(q_end, dict(zip(ids, served)), dict(zip(ids, per_user_rb)),
            {v: per_user_rb[j]*(args.p_macro if connection[v] == 0 else args.p_micro) for j, v in enumerate(ids)},
            {}, {}, {}, rb/ctx.caps, counts, connection)
        frame_rewards = om._adapted_rewards(reward_cfg, ids, step, rate*1e6*.02,
            rate*1e6*.1, outcomes, cfg.reward_sweep_reference_pilots or cfg.full_sweep_pilots)
        # Finish the reward interval before installing the action for x+1.
        for j, v in enumerate(ids):
            s = states[v]
            if s.transition_local_state is not None:
                s.transition_reward += frame_rewards[v]
                s.transition_frames += 1
            s.previous_rb_fraction = per_user_rb[j]/ctx.caps[connection[v]]
            if v in updates:
                u = updates[v]
                r = om._finalize_transition(s, v, memory, u['value'], False)
                if r is not None: rewards.append(r)
                s.pending_command = u['command']
                s.transition_local_state = u['local']
                s.transition_global_state = u['global_state']
                s.transition_action_binary = u['action']
                s.transition_log_probability = u['logp']
                s.transition_value = u['value']
                s.transition_reward = 0.; s.transition_frames = 0
                if not learn and len(probes) < 256: probes.append(u['local'])
        previous_throughput = float(served.sum()/sum(traffic['arrivals'][frame][v].sum() for v in ids))
        queues = q_end
        result.rb_allocated_record[fi] = rb
        result.energy_record[fi] = float(rb@np.array([args.p_macro]+[args.p_micro]*4)*.1)
        result.handover_record[fi] = sum(x.handover for x in outcomes.values())
        # Match the existing paper simulator's end-of-slot samples.
        result.queue_per_vehicle_record[fi] = {v: q[j, 1:].copy() for j, v in enumerate(ids)}
        result.association_record[fi] = connection.copy()
        result.violation_probability_record[fi] = np.mean(q[:, 1:] > rate*1e6*.02)
        result.average_queue_record[fi] = q[:, 1:].mean()
        result.pilot_record[fi] = pilots.mean()
        result.decision_record[fi] = len(updates)
        result.trigger_record[fi] = sum(u['action'] for u in updates.values())
        if optimization is not None:
            result.optimizer_failure_record[fi] = not optimization.solver_success
            result.optimizer_overflow_record[fi] = optimization.overflow.sum()
        diagnostic.append(dict(frame=float(frame), prediction_source=float(frame),
            prediction_target=round(frame+.1, 7), ra_prediction_source=float(frames[fi]),
            predicted_rb=ctx.rb.tolist(), ra_predicted_rb=ra_ctx.rb.tolist(),
            handovers=int(result.handover_record[fi]), probes=int(pilots.sum())))
        if progress: progress(fi+1, n)
    for v, s in states.items():
        # Pending final-frame commands have no observed consequence.
        if s.transition_frames:
            r = om._finalize_transition(s, v, memory, 0., True)
            if r is not None: rewards.append(r)
        elif memory.transitions:
            # The preceding interval must be terminal, not bootstrap into an
            # unexecuted final command. Search only this vehicle's last record.
            for t in reversed(memory.transitions):
                if t.vehicle == v:
                    t.done = True; t.next_value = 0.; break
    return result, traffic, memory, dict(frames=diagnostic,
        mean_event_reward=float(np.mean(rewards)) if rewards else 0., probes=np.asarray(probes),
        actor_input='31 public/predicted features; no H or beam labels',
        service='100 physical slots per frame; exact directional RB interference')
