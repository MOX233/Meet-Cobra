"""Exact paired simulator for prediction-report MTS. Original files unchanged."""
import collections
import dataclasses
import time
import numpy as np
import torch

from utils.alg_utils import RA_OTR_SINR, update_BS_association_state
from utils.gpu_phy import GPUFramePHY
from utils.ho_utils import interruption_slots
from utils.mts_gs_hbf import MTSLinkState
from utils.mts_gs_hbf_sim import MTSGSHBFSimulationResult
from utils.mts_report import reports_from_records, estimate_report_load, report_matching
from utils.pql_ba import macro_gain_db, no_bf_gain_db
from utils.queue_utils import update4slot_vehset_backlog_queue


@torch.inference_mode()
def probe_and_hold(physical,connection,states,reports,switched,ho_slots,k=5,tracking_pilots=1):
    """Private environment: measure only K reported candidates on the serving BS.

    Probe at slot 0, or at the first available slot after HO interruption.
    Only these observations choose the beam. Later H values compute physical
    service and cannot affect this selection. A normal tracking pilot is charged
    in every other active slot; probe-slot K includes its tracking observation.
    """
    ids = physical.ids
    active = [j for j,v in enumerate(ids) if connection[v]>0]
    pilots = np.zeros((physical.args.slots_per_frame,len(ids)),dtype=int)
    chosen = {}
    if active:
        index = torch.tensor(active,device=physical.device)
        bs = torch.tensor([connection[ids[j]]-1 for j in active],device=physical.device)
        slots = torch.tensor([ho_slots if ids[j] in switched else 0 for j in active],device=physical.device)
        candidates = torch.tensor(np.stack([reports[ids[j]].beams[connection[ids[j]]-1,:k] for j in active]),
                                  dtype=torch.long,device=physical.device)
        h = physical.h.permute(0,1,3,2,4)[slots,index,bs]
        tx = physical.tx[:,candidates//physical.args.M_r].permute(1,0,2)
        rx = physical.rx[:,candidates%physical.args.M_r].permute(1,0,2)
        response = torch.einsum('vrt,vtk->vrk',h,tx)
        gain = physical.db(torch.einsum('vrk,vrk->vk',response.conj(),rx).abs())
        best = candidates.gather(1,gain.argmax(1)[:,None])[:,0].cpu().numpy()
        for row,j in enumerate(active):
            v = ids[j]
            pair = int(best[row])
            assert pair in reports[v].beams[connection[v]-1]
            states[v].tx_beam,states[v].rx_beam = pair//physical.args.M_r,pair%physical.args.M_r
            first = ho_slots if v in switched else 0
            pilots[first:,j] = tracking_pilots
            pilots[first,j] = k
            chosen[v] = pair
    for v in ids:
        if connection[v] == 0:
            states[v].tx_beam = states[v].rx_beam = None
    service = physical.fixed_pairs(connection,states)
    for j,v in enumerate(ids):
        if v in switched:
            pilots[:ho_slots,j] = 0
            service[:ho_slots,j] = -180. # Unobserved and unused while interrupted.
    return service,pilots,chosen


def run_sim_mts_report(args,micro_bs_locations,timeline,config,traffic_trace,seed=1,
                       physics_device='cuda:0',ho_interruption_ms=10.,k=5,
                       diagnostics=None,progress_callback=None):
    config = dataclasses.replace(config,ho_interruption_ms=ho_interruption_ms)
    config.validate()
    assert config.local_tracking_interval_frames == 1 and k*args.pilot_overhead_factor < 1
    assert traffic_trace['seed'] == seed
    slots = args.slots_per_frame
    blocked_slots = interruption_slots(ho_interruption_ms,args.slot_len,slots)
    frames = list(timeline)
    n = len(frames)-1
    rates = traffic_trace['rates']
    upper = {v:args.lat_slot_ub*rate*args.slot_len for v,rate in rates.items()}
    states = {v:MTSLinkState() for v in timeline[frames[0]]}
    queue_last = {v:float(traffic_trace['initial_queues'][frames[0]][v]) for v in states}
    commands_last = {}
    measured_last = {}
    reports_last = reports_from_records(timeline[frames[0]],frames[0],config,k)
    bs_locations = np.asarray([(0.,0.),*micro_bs_locations])
    bs_dict = collections.OrderedDict(enumerate(bs_locations))
    capacities = np.array([args.num_RB_macro]+[args.num_RB_micro]*config.num_micro_bs)
    dictionary_fields = {'queue_per_vehicle_record','association_record','action_record'}
    result = MTSGSHBFSimulationResult(**{field.name:
        (collections.OrderedDict() if field.name in dictionary_fields else
         np.zeros((n,config.num_bs)) if field.name=='rb_allocated_record' else np.zeros(n))
        for field in dataclasses.fields(MTSGSHBFSimulationResult)})
    for fi,frame in enumerate(frames[1:]):
        records = timeline[frame]
        ids = sorted(records,key=str)
        current = set(ids)
        states = {v:state for v,state in states.items() if v in current}
        measured_last = {v:val for v,val in measured_last.items() if v in current}
        queue = {}
        for v in ids:
            queue[v] = np.zeros(slots+1)
            if v not in states:
                states[v] = MTSLinkState()
                queue[v][0] = traffic_trace['initial_queues'][frame][v]
            else:
                queue[v][0] = queue_last[v]
        switched = set()
        for v in ids:
            if v in commands_last:
                command = commands_last[v]
                assert np.isclose(command.report.source_frame,frames[fi])
                if states[v].action != command.target_bs:
                    switched.add(v)
                states[v].action = command.target_bs
        connection = collections.OrderedDict((v,states[v].action) for v in ids)
        association,_ = update_BS_association_state(bs_dict,connection)
        reports = reports_from_records(records,frame,config,k)
        estimated_rb,load = estimate_report_load(args,reports,connection,rates,config,bs_locations,measured_last)
        start = time.perf_counter()
        if fi % config.association_interval_frames == 0:
            commands,matching = report_matching(args,reports,connection,{v:queue[v][0] for v in ids},upper,rates,load,config,k)
            result.proposal_record[fi] = matching.proposal_count
            result.unassigned_record[fi] = len(matching.unassigned)
            result.optimizer_overflow_record[fi] = np.maximum(matching.used_capacity-capacities,0).sum()
            result.association_epoch_record[fi] = 1
            result.decision_record[fi] = len(commands)
            result.trigger_record[fi] = sum(c.target_bs != connection[v] for v,c in commands.items())
        else:
            commands = {}
        result.optimizer_time_record[fi] = time.perf_counter()-start
        # Physical execution is strictly separated from the already-made HO decision.
        physical = GPUFramePHY(args,records,frame,seed,physics_device)
        assert physical.ids == ids
        micro = [v for v in ids if connection[v]>0]
        assert all(v in reports_last for v in micro)
        assert all(np.isclose(reports_last[v].source_frame,frames[fi]) for v in micro)
        previous_pairs = {v:(states[v].tx_beam,states[v].rx_beam) for v in ids}
        gains,pilots,chosen = probe_and_hold(physical,connection,states,reports_last,switched,blocked_slots,k,config.tracking_pilots)
        del physical
        result.beam_switch_record[fi] = sum(v in switched or previous_pairs[v]!=(states[v].tx_beam,states[v].rx_beam) for v in ids)
        result.local_sweep_record[fi] = len(micro)
        result.local_tracking_epoch_record[fi] = 1
        # Same slot-level interference convention as MEET and original MTS.
        # This true environment quantity is NEVER passed to report_matching.
        physical_interference = {v:np.r_[macro_gain_db(args,records[v]['pos'],np.zeros(2)),no_bf_gain_db(records[v]['h'])] for v in ids}
        arrivals = traffic_trace['arrivals'][frame]
        energy = 0.
        for slot in range(slots):
            blocked = switched if slot < blocked_slots else set()
            gain_slot,pilot_slot = {},{}
            for j,v in enumerate(ids):
                values = np.full(config.num_bs,-180.)
                bs = connection[v]
                values[bs] = physical_interference[v][0] if bs==0 else gains[slot,j]
                gain_slot[v] = values
                count = np.zeros(config.num_micro_bs)
                if bs>0:
                    count[bs-1] = pilots[slot,j]
                pilot_slot[v] = count
            allocation = collections.OrderedDict((v,0) for v in blocked)
            rb = np.zeros(config.num_bs,dtype=int)
            for bs in range(config.num_bs):
                local = RA_OTR_SINR(args,slot_idx=slot,BS_id=bs,
                    veh_set=[v for v in association[bs] if v not in blocked],
                    veh_data_rate_dict=rates,Q_ub_dict=upper,q_dict=queue,a_dict=arrivals,
                    g_dict=gain_slot,num_pilot_dict=pilot_slot,BS_association_dict=association,
                    infer_g_dict=physical_interference,est_num_RB_allocated_perBS=estimated_rb)
                allocation.update(local)
                rb[bs] = sum(local.values())
            assert (rb<=capacities).all() and (rb>=0).all()
            queue = update4slot_vehset_backlog_queue(args,slot_idx=slot,RA_dict=allocation,
                veh_set=ids,connection_dict=connection,backlog_queue_dict=queue,a_dict=arrivals,
                g_dict=gain_slot,infer_g_dict=physical_interference,num_RB_allocated_perBS=rb,
                num_pilot_dict=pilot_slot,sinr_flag=True)
            for v in blocked:
                assert allocation[v]==0 and not pilot_slot[v].any()
                assert queue[v][slot+1]==queue[v][slot]+arrivals[v][slot]
            result.rb_allocated_record[fi] += rb/slots
            energy += float(rb @ np.array([args.p_macro]+[args.p_micro]*config.num_micro_bs))*args.slot_len
        for j,v in enumerate(ids):
            bs = connection[v]
            measured_last[v] = (bs,physical_interference[v][0] if bs==0 else float(gains[-1,j]))
        result.energy_record[fi] = energy
        result.handover_record[fi] = len(switched)
        result.pilot_record[fi] = pilots.mean()
        result.queue_per_vehicle_record[fi] = {v:queue[v][1:].copy() for v in ids}
        result.average_queue_record[fi] = np.mean([queue[v][1:] for v in ids])
        result.violation_probability_record[fi] = np.mean([queue[v][1:]>upper[v] for v in ids])
        result.association_record[fi] = dict(connection)
        result.action_record[fi] = {v:c.target_bs for v,c in commands.items()}
        if diagnostics is not None:
            diagnostics.append(dict(frame=float(frame),association=dict(connection),
                switched=list(sorted(switched,key=str)),probed_pairs=chosen,
                probe_slots={v:blocked_slots if v in switched else 0 for v in micro},
                source_frame=float(frames[fi]),total_probe_count=k*len(micro),
                blocked_vehicle_slots=blocked_slots*len(switched),
                active_vehicle_slots=slots*len(ids),estimated_rb=estimated_rb.tolist()))
        commands_last = commands
        reports_last = reports
        queue_last = {v:queue[v][-1] for v in ids}
        if progress_callback:
            progress_callback(fi+1,n)
    return result
