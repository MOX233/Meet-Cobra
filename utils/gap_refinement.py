"""Opt-in, bounded GAP-HO fixed-point refinement.

No new setting affects the legacy path. Two iterations with tolerance=None
reproduce its two solves and final repair, including its reserve and pilot
estimators. HO interruption changes capacity coefficients only.
"""

import collections
from dataclasses import dataclass
import math
import time

import numpy as np

from utils.ho_utils import capacity_factors
from utils.mox_utils import dB2lin


@dataclass(frozen=True)
class GAPRefinementConfig:
    max_iterations: int = 3
    tolerance_rb: float | None = None
    relaxation_factor: float = 1.1

    def __post_init__(self):
        if (isinstance(self.max_iterations, bool)
                or not isinstance(self.max_iterations, (int, np.integer))
                or self.max_iterations < 1):
            raise ValueError('max_iterations must be a positive integer')
        if self.tolerance_rb is not None and (
                not math.isfinite(self.tolerance_rb) or self.tolerance_rb < 0):
            raise ValueError('tolerance_rb must be finite and nonnegative, or None')
        if not math.isfinite(self.relaxation_factor) or self.relaxation_factor <= 1:
            raise ValueError('relaxation_factor must be finite and greater than one')


def iterate_assignment(initial_load, demand_function, assignment_function, config):
    """Apply the undamped manuscript update; stopping is tested before repair.

    The residual is max_m |new frame-average load_m - input load_m| in RBs.
    A repeated iterate is recorded, never treated as convergence or used to
    override the specified stopping rule. None disables early stopping; zero
    instead requires exact equality.
    """
    load = np.asarray(initial_load, dtype=float).copy()
    traces = []
    seen = [load.copy()]
    for iteration in range(config.max_iterations):
        demand = demand_function(load, iteration)
        assignment = assignment_function(demand)
        new_load = (demand * assignment).sum(axis=1)
        if not np.isfinite(new_load).all():
            raise ValueError('Nonfinite load in GAP refinement')
        residual = float(np.max(np.abs(new_load - load)))
        traces.append(dict(
            residual_rb=residual, input_load=load.tolist(),
            implied_load=new_load.tolist(),
            repeated_load=any(np.array_equal(new_load, old) for old in seen)))
        load = new_load
        seen.append(load.copy())
        if config.tolerance_rb is not None and residual <= config.tolerance_rb:
            return demand, assignment, traces, 'tolerance'
    return demand, assignment, traces, 'iteration_limit'


def refined_gap_handover(args, veh_set_cur, backlog_queue_dict,
                         veh_data_rate_dict, pred_loc_dict, pred_g_dict,
                         BS_loc_array, **kwargs):
    from utils import alg_utils

    started = time.perf_counter()
    config = kwargs['gap_refinement_config']
    if isinstance(config, dict):
        config = GAPRefinementConfig(**config)
    if not isinstance(config, GAPRefinementConfig):
        raise TypeError('gap_refinement_config must be GAPRefinementConfig or dict')
    vehicles = list(veh_set_cur)
    num_bs = len(BS_loc_array)
    diagnostics = kwargs.get('gap_refinement_diagnostics')
    if not vehicles:
        if diagnostics is not None:
            diagnostics.append(dict(iterations=0, stop_reason='empty',
                                    traces=[], elapsed_s=time.perf_counter()-started))
        return collections.OrderedDict(), np.zeros(num_bs)

    factors = capacity_factors(
        vehicles, num_bs, kwargs.get('current_connection'),
        kwargs.get('ho_interruption_slots', 0)
        if kwargs.get('ho_capacity_correction', False) else 0,
        args.slots_per_frame)
    history = kwargs.get('vio_prob_history', [])
    # Deliberately retain the legacy heuristic; this study isolates refinement.
    reserve = (0 if len(history) == 0 else
               (np.asarray(history[-100:]) > args.vio_prob_threshold).mean() * .1)
    capacity = np.array([args.num_RB_macro] + [args.num_RB_micro]*(num_bs-1), dtype=float)
    planning_capacity = np.array([int(k*(1-reserve)) for k in capacity], dtype=float)
    powers = np.array([args.p_macro] + [args.p_micro]*(num_bs-1))
    pred_g_db = np.zeros((len(vehicles), num_bs))
    for i, veh in enumerate(vehicles):
        pred_g_db[i, :] = pred_g_dict[veh]
    rates = np.array([veh_data_rate_dict[v] for v in vehicles])
    interfering_gains = kwargs.get('infer_g_dict')
    pilots = kwargs.get('num_pilot_dict')

    def demand_function(load, iteration):
        matrix = np.zeros((len(vehicles), num_bs))
        for bs in range(num_bs):
            bandwidth = args.RB_intervel_micro if bs > 0 else args.RB_intervel_macro
            p = args.p_micro if bs > 0 else args.p_macro
            nf = args.NF_micro_dB if bs > 0 else args.NF_macro_dB
            gain = dB2lin(pred_g_db[:, bs])
            if interfering_gains is not None:
                if iteration == 0:
                    interference = np.array([
                        sum(dB2lin(interfering_gains[v][other])*args.p_micro
                            for other in range(1, num_bs) if other != bs)
                        for v in vehicles])
                else:
                    interference = np.array([
                        sum(dB2lin(interfering_gains[v][other])*args.p_micro
                            * (load[other]/args.num_RB_micro)
                            for other in range(1, num_bs) if other != bs)
                        for v in vehicles])
            else:
                # Preserve the fallback branch too (the experiment uses Oracle
                # interfering gains, so the load-dependent branch is active).
                interference = np.array([
                    sum(dB2lin(pred_g_dict[v][other]-alg_utils.min_bf_gain_dB)
                        * args.p_micro for other in range(1, num_bs) if other != bs)
                    for v in vehicles])
            if bs == 0:
                interference *= 0
            overhead = np.zeros(len(vehicles))
            if pilots is not None:
                for i, v in enumerate(vehicles):
                    overhead[i] = (min(pilots[v][bs-1]*args.pilot_overhead_factor, 1)
                                   if bs > 0 else 0)
            matrix[:, bs] = rates / (1e-10 + (1-overhead)*bandwidth*np.log2(
                1+p*gain/(args.N0*bandwidth*dB2lin(nf)+interference)))
        return matrix.swapaxes(0, 1)

    def assignment_function(demand):
        return alg_utils.alg_GAP_APX_adap(
            c=demand*powers[:, None], a=demand*factors,
            b=planning_capacity, adap_mtp=config.relaxation_factor)

    demand, assignment, traces, stop_reason = iterate_assignment(
        capacity, demand_function, assignment_function, config)
    before_repair = assignment.copy()
    assignment, feasible = alg_utils._ITERATIVE_OFFLOAD(
        assignment, demand*factors, planning_capacity, powers,
        T_COST=demand*powers[:, None])
    loads = (demand*assignment).sum(axis=1)
    if diagnostics is not None:
        diagnostics.append(dict(
            iterations=len(traces), stop_reason=stop_reason, traces=traces,
            tolerance_rb=config.tolerance_rb, max_iterations=config.max_iterations,
            relaxation_factor=config.relaxation_factor, reserve_ratio=float(reserve),
            pre_repair_residual_rb=traces[-1]['residual_rb'],
            repair_changed_vehicles=int(np.sum(
                before_repair.argmax(axis=0) != assignment.argmax(axis=0))),
            capacity_feasible_after_repair=bool(feasible),
            post_repair_frame_average_load=loads.tolist(),
            planning_capacity=planning_capacity.tolist(),
            post_repair_capacity_load=(demand*factors*assignment).sum(axis=1).tolist(),
            elapsed_s=time.perf_counter()-started))
    commands = collections.OrderedDict((v, assignment[:, i].argmax())
                                       for i, v in enumerate(vehicles))
    return commands, loads
