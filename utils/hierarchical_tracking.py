"""Paid, causal hierarchical acquisition and five-point beam tracking.

This experimental module does not change any formal baseline defaults.
Only probed serving-link responses select beams; unprobed full-codebook gains
are never supplied to the controller. Fine beams use the existing convention.
"""
import numpy as np
import torch
from utils.hierarchical_beam import coarse_codebook


def cross_neighbors(pair, num_tx=32, num_rx=8):
    """Current beam plus four axial neighbours, including circular boundaries."""
    t, r = pair // num_rx, pair % num_rx
    return torch.stack((pair, ((t-1) % num_tx)*num_rx+r,
                        ((t+1) % num_tx)*num_rx+r,
                        t*num_rx+(r-1) % num_rx,
                        t*num_rx+(r+1) % num_rx), dim=-1)


def probe_candidates(h, candidates, tx, rx):
    """Measure only supplied pairs for a batch of serving channels (V,R,T)."""
    wt = tx[:, candidates // rx.shape[0]].permute(1, 0, 2)
    wr = rx[:, candidates % rx.shape[0]].permute(1, 0, 2)
    response = torch.einsum('vrt,vtk->vrk', h, wt)
    return torch.einsum('vrk,vrk->vk', response, wr.conj()).abs()


def acquire32(h, tx, rx):
    """Sixteen wide pairs, followed by sixteen fine pairs in the best sector."""
    if tx.shape != (32, 32) or rx.shape != (8, 8):
        raise ValueError('32-probe acquisition requires the 32 by 8 codebooks')
    wt = torch.tensor(coarse_codebook(32, 8), dtype=h.dtype, device=h.device)
    wr = torch.tensor(coarse_codebook(8, 2), dtype=h.dtype, device=h.device)
    wide = torch.einsum('ra,vrt,tk->vak', wr.conj(), h, wt).abs()
    best = wide.flatten(1).argmax(1)
    ts, rs = best % 8, best // 8
    offset = torch.arange(4, device=h.device)
    fine = ((ts[:, None, None]*4+offset[None, :, None])*8
            +rs[:, None, None]*4+offset[None, None, :]).flatten(1)
    responses = probe_candidates(h, fine, tx, rx)
    index = responses.argmax(1)
    return fine.gather(1, index[:, None])[:, 0], responses.gather(1, index[:, None])[:, 0]


@torch.inference_mode()
def search_frame(physical, connection, states, switched, ho_slots, local_five):
    """Acquire at each link's first available slot; then hold or track.

    Returns per-slot gains, probe counts and actual beam indices for the
    directional RB evaluator. Five probes include the current-beam observation.
    Slot evolution is causal even though the simulator has the whole H tensor.
    """
    args, ids, device = physical.args, physical.ids, physical.device
    slots, vehicles = args.slots_per_frame, len(ids)
    pairs = torch.zeros((slots, vehicles, 4), dtype=torch.long, device=device)
    values = torch.full((slots, vehicles), -180., dtype=torch.float64, device=device)
    pilots = np.zeros((slots, vehicles), dtype=int)
    micro = np.array([j for j,v in enumerate(ids) if connection[v]>0], dtype=int)
    if not len(micro):
        for state in states.values(): state.tx_beam = state.rx_beam = None
        return values.cpu().numpy(), pilots, pairs.cpu().numpy(), {}
    index = torch.as_tensor(micro, device=device)
    bs = torch.tensor([connection[ids[j]]-1 for j in micro], device=device)
    # Extraction of private physical samples is not a measurement. The
    # selection below evaluates only the 32 or 5 paid candidates at that slot.
    h = physical.h.permute(0,1,3,2,4)[:,index,bs]
    first = np.array([ho_slots if ids[j] in switched else 0 for j in micro])
    current = torch.zeros(len(micro), dtype=torch.long, device=device)
    acquired = {}
    for slot in range(slots):
        new = np.flatnonzero(first == slot)
        old = np.flatnonzero(first < slot)
        if len(new):
            ni = torch.as_tensor(new, device=device)
            chosen, amplitude = acquire32(h[slot,ni], physical.tx, physical.rx)
            current[ni] = chosen
            values[slot,index[ni]] = physical.db(amplitude)
            pilots[slot,micro[new]] = 32
            acquired[slot] = (new.copy(), chosen.clone())
        if len(old):
            oi = torch.as_tensor(old, device=device)
            candidates = cross_neighbors(current[oi]) if local_five else current[oi,None]
            measured = probe_candidates(h[slot,oi], candidates, physical.tx, physical.rx)
            best = measured.argmax(1)
            current[oi] = candidates.gather(1,best[:,None])[:,0]
            values[slot,index[oi]] = physical.db(measured.gather(1,best[:,None])[:,0])
            pilots[slot,micro[old]] = 5 if local_five else 1
        pairs[slot,index,bs] = current
    chosen = {}
    for slot,(local, initial) in acquired.items():
        for j,p in zip(micro[local], initial.cpu().tolist()): chosen[ids[j]] = p
    final = current.cpu().tolist()
    for j,p in zip(micro,final):
        states[ids[j]].tx_beam, states[ids[j]].rx_beam = p//args.M_r,p%args.M_r
    for v in ids:
        if connection[v]==0: states[v].tx_beam = states[v].rx_beam = None
    return values.cpu().numpy(), pilots, pairs.cpu().numpy(), chosen
