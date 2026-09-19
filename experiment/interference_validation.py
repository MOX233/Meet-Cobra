#!/usr/bin/env python3
"""Paired R2C1 validation without modifying the frozen Fig.5--8 simulator.

Private channel tensors are used ONLY by the service evaluator, never by HO,
BF, or RA. The legacy mode is an exact rerun with diagnostic instrumentation.
Alternative modes replace micro-tier queue service only; decisions then react
to the resulting queues. Explicit indices are independent random permutations
at each BS, conditional on its deterministic per-vehicle RB counts.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

OUTPUT = ROOT / 'experiment/results/interference_validation_20260919'
MODES = ('legacy', 'frame_average', 'average_expected', 'directional_expected',
         'average_explicit', 'directional_explicit')
COLUMNS = ('frame', 'slot', 'vehicle_index', 'bs', 'rb', 'queue_bits', 'arrivals_bits',
           'signal_w', 'pilot_efficiency') + tuple('I_' + m for m in MODES) + tuple('R_' + m for m in MODES)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def rb_owners(bs, counts, capacity, rng, n_bs=4):
    """Each BS uses a distinct random ordering; intracell indices never overlap."""
    owners = np.full((n_bs, capacity), -1, dtype=int)
    assert np.all(counts >= 0)
    for j in range(1, n_bs + 1):
        users = np.flatnonzero(bs == j)
        occupants = np.repeat(users, counts[users])
        assert len(occupants) <= capacity
        owners[j - 1, rng.permutation(capacity)[:len(occupants)]] = occupants
    return owners


def expected_interference(gain, bs, counts, capacity, power):
    """gain[v,w]: victim v, transmitter beam serving w; omit own/macro BS."""
    mask = (bs[:, None] != bs[None, :]) & (bs[None, :] > 0) & (bs[:, None] > 0)
    return power / capacity * ((gain * mask) @ counts)


def explicit_interference(gain, bs, owners, power):
    """I[v,r] sums powers of independent other-cell data streams on RB r."""
    result = np.zeros((len(bs), owners.shape[1]))
    for j, row in enumerate(owners, 1):
        result += power * gain[:, np.maximum(row, 0)] * (row[None, :] >= 0) * (bs[:, None] != j)
    result[bs == 0] = 0
    assigned = owners[np.maximum(bs - 1, 0)] == np.arange(len(bs))[:, None]
    assigned[bs == 0] = False
    return result, assigned


@torch.inference_mode()
def directional_gains(physical, pairs, bs):
    """Use the victim's serving RX beam and EACH interferer's serving TX beam.

    The normalized DFT codebooks match GPUFramePHY.pet. H has shape S,V,R,B,T.
    Return S,V,V cross gains, S,V,B codebook means, and own selected gains.
    """
    device = physical.device
    slots, vehicles = physical.h.shape[:2]
    b = torch.as_tensor(np.maximum(bs - 1, 0), device=device)
    vidx = torch.arange(vehicles, device=device)
    sidx = torch.arange(slots, device=device)
    chosen = torch.as_tensor(pairs, device=device)[:, vidx, b]
    tx_idx, rx_idx = chosen // physical.args.M_r, chosen % physical.args.M_r
    rx = physical.rx[:, rx_idx].permute(1, 2, 0)
    response = torch.einsum('svrbt,svr->svbt', physical.h, rx.conj())
    response = torch.einsum('svbt,tk->svbk', response, physical.tx)
    gain = response.abs().square() / (physical.args.M_r * physical.args.M_t)
    cross = gain[sidx[:, None, None], vidx[None, :, None], b[None, None, :], tx_idx[:, None, :]]
    own = gain[sidx[:, None], vidx[None, :], b[None, :], tx_idx]
    mean = physical.h.abs().square().mean(dim=(2, 4))
    return cross.cpu().numpy(), mean.cpu().numpy(), own.cpu().numpy()


def prepare():
    from experiment import revision_cap_grid as grid
    previous, sha = grid.validate(check_inputs=True)
    code = dict(previous['code_sha256'])
    for name in ('experiment/interference_validation.py', 'test_interference_validation.py'):
        code[name] = digest(ROOT / name)
    protocol = dict(version=1, frozen_grid_protocol=sha, code_sha256=code,
        checkpoint_cache_sha256=digest(grid.old.CACHE), modes=list(MODES),
        parent_git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        rollback_tag='pre-r2c1-interference-validation-20260919',
        traffic='Original paired Poisson arrivals and vehicle trace; 300 frames, omit first two for metrics',
        physics='Original Sionna RT complex matrices and frozen elementwise Rician-amplitude fading; FP64 CUDA; no new ray tracing',
        directional='Normalized victim serving RX and actual cochannel transmitter serving TX; independent stream powers; frequency-flat per-link H',
        indices='Each slot: independent uniform BS permutations conditional on allocated counts; independent keyed RNG, not policy/fading RNG',
        pilot_and_ho='Unchanged scalar PET pilot efficiency, zero service during 10ms HO interruption; no waveform-level pilot interference',
        legacy='Original frame-level maximum-element squared gain (including original dB numerical floor), expected RB overlap',
        frame_average='Frame-level Frobenius/codebook mean, expected overlap; diagnostic bridge only',
        average_expected='Slot-level Frobenius/codebook mean, expected overlap: manuscript surrogate',
        directional_expected='Slot-level actual selected RX/TX beams, expected random overlap',
        average_explicit='Slot-level codebook mean, explicit RB indices and sum of per-RB rates',
        directional_explicit='Slot-level actual selected RX/TX beams, explicit RB indices and sum of per-RB rates',
        scope='MEET decisions and predictors remain unchanged. Legacy = matched-schedule replay. Other modes change physical queue service only, not optimizer information.',
        caveats=['Existing BF report timing retained to isolate interference.',
                 'Explicit random frequency mapping is the stated assumption, not a frequency-optimizing OTR-RA implementation.',
                 'Seeds vary arrivals and fading on the same trajectory/map/model, not independent deployments.',
                 'Closed-loop tests do not yet retrain the interference predictor or harmonize all optimizer gain inputs.'])
    path = OUTPUT / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        assert not list((OUTPUT/'runs').glob('*.json')), 'Accepted results exist; use a new directory'
        write_json(OUTPUT/'preflight_history'/f'protocol_{digest(path)}.json', json.loads(path.read_text()))
        write_json(path, protocol)
    elif not path.exists():
        write_json(path, protocol)
    print('PROTOCOL', digest(path), flush=True)


class Audit:
    def __init__(self, mode, seed):
        self.mode, self.seed = mode, seed
        self.frames = 0
        self.rows = []
        self.overlap = np.zeros((4, 4, 2))
        self.frame_log = []
        self.diagonal_max_error = 0.

    def capture(self, physical, pairs, values):
        self.physical, self.ids = physical, physical.ids
        self.pairs, self.values = pairs, values
        self.frame = physical.validation_frame
        self.frame_mean = physical.validation_frame_mean
        self.frames += 1
        self.ready = False

    def slot(self, original, args, **kw):
        ids = self.ids
        slot = kw['slot_idx']
        bs = np.array([kw['connection_dict'][v] for v in ids], dtype=int)
        k = np.array([kw['RA_dict'][v] for v in ids], dtype=int)
        if not self.ready:
            assert slot == 0
            self.cross, self.mean, own = directional_gains(self.physical, self.pairs, bs)
            reconstructed = 20 * np.log10(np.sqrt(own) + 1e-9)
            selected = self.values[:, np.arange(len(ids)), np.maximum(bs - 1, 0)]
            valid = (bs > 0)[None, :] & (selected > -179)
            err = float(np.max(np.abs((reconstructed - selected)[valid]), initial=0))
            assert err < 1e-7, err
            self.diagonal_max_error = max(self.diagonal_max_error, err)
            self.bs = bs.copy()
            self.ready = True
            self.physical = None  # Do not retain the full channel tensor across frames.
            self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, round(float(self.frame)*10), 24681357]))
        np.testing.assert_array_equal(bs, self.bs)
        owners = rb_owners(bs, k, args.num_RB_micro, self.rng)
        # Macro users never contribute cross-tier interference in the frozen model.
        bsrc = np.maximum(bs - 1, 0)
        g_frame = self.frame_mean[:, bsrc]
        g_avg = self.mean[slot][:, bsrc]
        g_dir = self.cross[slot]
        legacy_gain = 10 ** (np.array([kw['infer_g_dict'][v][1:] for v in ids]) / 10)
        g_leg = legacy_gain[:, bsrc]
        ie = [expected_interference(g, bs, k, args.num_RB_micro, args.p_micro)
              for g in (g_leg, g_frame, g_avg, g_dir)]
        ia_rb, assigned = explicit_interference(g_avg, bs, owners, args.p_micro)
        id_rb, _ = explicit_interference(g_dir, bs, owners, args.p_micro)
        np.testing.assert_array_equal(assigned.sum(1), np.where(bs > 0, k, 0))
        ie += [(v * assigned).sum(1) / np.maximum(k, 1) for v in (ia_rb, id_rb)]
        signal = args.p_micro * 10 ** (np.array([kw['g_dict'][v][b] for v, b in zip(ids, bs)]) / 10)
        noise = args.N0 * args.RB_intervel_micro * 10 ** (args.NF_micro_dB / 10)
        efficiency = 1 - np.minimum(np.array([kw['num_pilot_dict'][v][b-1] if b else 0
                                             for v, b in zip(ids, bs)]) * args.pilot_overhead_factor, 1)
        factor = efficiency * args.RB_intervel_micro * args.slot_len
        rates = [factor * k * np.log2(1 + signal / (noise + inter)) for inter in ie[:4]]
        rates += [factor * (np.log2(1 + signal[:, None] / (noise + inter)) * assigned).sum(1)
                  for inter in (ia_rb, id_rb)]
        q = np.array([kw['backlog_queue_dict'][v][slot] for v in ids])
        arrivals = np.array([kw['a_dict'][v][slot] for v in ids])
        active = (bs > 0) & (k > 0)
        # Reproduce the old queues exactly; then replace only micro service if asked.
        result = original(args, **kw)
        q_legacy = np.maximum(q - rates[0], 0) + arrivals
        np.testing.assert_allclose([result[v][slot+1] for v in np.array(ids)[bs > 0]],
                                   q_legacy[bs > 0], rtol=1e-12, atol=1e-7)
        if self.mode != 'legacy':
            alternative = rates[MODES.index(self.mode)]
            for index in np.flatnonzero(bs > 0):
                result[ids[index]][slot+1] = max(q[index] - alternative[index], 0) + arrivals[index]
        if self.frames > 2:
            data = np.column_stack((np.full(len(ids), self.frames - 1), np.full(len(ids), slot),
                np.arange(len(ids)), bs, k, q, arrivals, signal, efficiency, *ie, *rates))[active]
            self.rows.append(data)
            occupied = owners >= 0
            for i in range(4):
                for j in range(i+1, 4):
                    self.overlap[i,j,0] += occupied[i].sum() * occupied[j].sum() / args.num_RB_micro
                    self.overlap[i,j,1] += np.count_nonzero(occupied[i] & occupied[j])
        if slot == args.slots_per_frame - 1:
            self.frame_log.append(dict(frame=float(self.frame), ids=[str(v) for v in ids], bs=bs.tolist()))
        return result

    def summarize(self, args):
        data = np.concatenate(self.rows)
        self.rows.clear()
        col = {name: data[:, i] for i, name in enumerate(COLUMNS)}
        noise = args.N0 * args.RB_intervel_micro * 10 ** (args.NF_micro_dB / 10)
        ref_i, ref_r, weight = col['I_directional_explicit'], col['R_directional_explicit'], col['rb']
        summary = dict(active_micro_vehicle_slots=len(data), allocated_micro_rb_slots=float(weight.sum()),
            noise_w=noise, directional_diagonal_max_error_db=self.diagonal_max_error,
            active_rb_mean_reference_inr_db=float(10*np.log10((ref_i*weight).sum()/weight.sum()/noise)),
            overlap_expected_realized=self.overlap.tolist(), modes={}, comparisons={})
        for mode in MODES:
            inter, rate = col['I_'+mode], col['R_'+mode]
            sinr_error = 10*np.log10((noise+ref_i)/(noise+inter))
            relative_rate = rate / ref_r - 1
            summary['modes'][mode] = dict(
                interference_sum_ratio=float(np.sum(inter*weight)/np.sum(ref_i*weight)),
                sinr_of_mean_interference_error_db_mean=float(sinr_error.mean()),
                sinr_of_mean_interference_abs_error_db_median=float(np.median(np.abs(sinr_error))),
                sinr_of_mean_interference_abs_error_db_p95=float(np.quantile(np.abs(sinr_error),.95)),
                potential_service_ratio=float(rate.sum()/ref_r.sum()),
                relative_service_error_median=float(np.median(relative_rate)),
                relative_service_abs_error_p95=float(np.quantile(np.abs(relative_rate),.95)),
                useful_departure_ratio=float(np.minimum(rate,col['queue_bits']).sum()/np.minimum(ref_r,col['queue_bits']).sum()))
        for name, estimate, reference in (
            ('gain_only','average_expected','directional_expected'),
            ('overlap_only_directional','directional_expected','directional_explicit'),
            ('overlap_only_average','average_expected','average_explicit'),
            ('legacy_to_frame_mean','legacy','frame_average'),
            ('frame_to_slot_mean','frame_average','average_expected')):
            s, r = col['I_'+estimate], col['I_'+reference]
            err = 10*np.log10((noise+r)/(noise+s))
            summary['comparisons'][name] = dict(interference_sum_ratio=float((s*weight).sum()/(r*weight).sum()),
                sinr_error_mean_db=float(err.mean()),sinr_abs_error_median_db=float(np.median(np.abs(err))),
                sinr_abs_error_p95_db=float(np.quantile(np.abs(err),.95)),
                potential_service_ratio=float(col['R_'+estimate].sum()/col['R_'+reference].sum()))
        return summary, data


def case(mode, rate, seed, gpu):
    from experiment import revision_cap_grid as grid
    import utils.gpu_phy as phy_module
    import utils.sim_utils as sim_module
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    for name, sha in protocol['code_sha256'].items():
        assert digest(ROOT/name) == sha, 'Code changed: ' + name
    assert digest(grid.old.CACHE) == protocol['checkpoint_cache_sha256']
    key = f'{mode}_rate{rate}_seed{seed}'
    destination = OUTPUT/'runs'/f'{key}.json'
    if destination.exists():
        row = json.loads(destination.read_text())
        assert row['protocol_sha256'] == digest(OUTPUT/'protocol.json')
        print('EXISTS', key, flush=True)
        return
    started = time.monotonic()
    audit = Audit(mode, seed)
    original_phy = phy_module.GPUFramePHY
    original_queue = sim_module.update4slot_vehset_backlog_queue

    class CapturedPHY(original_phy):
        def __init__(self, args, records, frame, seed, device):
            super().__init__(args, records, frame, seed, device)
            self.validation_frame = frame
            h = np.stack([records[v]['h'] for v in self.ids])
            self.validation_frame_mean = (np.abs(h)**2).mean(axis=(1,3))

        def pet(self, *args, **kwargs):
            result = super().pet(*args, **kwargs)
            audit.capture(self, result[2], result[0])
            return result

    def queue_update(args, **kw):
        return audit.slot(original_queue, args, **kw)

    phy_module.GPUFramePHY = CapturedPHY
    sim_module.update4slot_vehset_backlog_queue = queue_update
    print('START', key, 'GPU', gpu, flush=True)
    try:
        metrics, raw, diagnostics, traffic, _ = grid.simulate('meet_cobra', rate, seed, gpu)
    finally:
        phy_module.GPUFramePHY = original_phy
        sim_module.update4slot_vehset_backlog_queue = original_queue
    assert audit.frames == 300
    match = None
    if mode == 'legacy':
        reference = grid.OUTPUT/'raw'/f'meet_cobra_rate{rate}_seed{seed}.npz'
        with np.load(reference, allow_pickle=False) as prior:
            assert set(raw) == set(prior.files)
            for name in raw:
                np.testing.assert_array_equal(raw[name], prior[name], err_msg=name)
        match = dict(path=str(reference.relative_to(ROOT)), sha256=digest(reference), all_arrays_exact=True)
    args = grid.old.shared.paper_args(rate*1e6)
    summary, samples = audit.summarize(args)
    rawpath = OUTPUT/'raw'/f'{key}.npz'
    rawpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rawpath, **raw)
    samplepath = OUTPUT/'samples'/f'{key}.npz'
    samplepath.parent.mkdir(parents=True, exist_ok=True)
    # Summaries use FP64. Compact saved diagnostic samples are FP32.
    np.savez_compressed(samplepath, columns=np.array(COLUMNS), data=samples.astype(np.float32))
    del samples
    diagpath = OUTPUT/'diagnostics'/f'{key}.json'
    write_json(diagpath, grid.json_native(dict(ho=diagnostics['ho'], frames=audit.frame_log)))
    row = dict(mode=mode,rate_mbps=rate,seed=seed,metrics=metrics,diagnostics=summary,
        control_match=match,traffic_sha256=traffic['sha256'],elapsed_s=time.monotonic()-started,
        protocol_sha256=digest(OUTPUT/'protocol.json'),raw_sha256=digest(rawpath),
        samples_sha256=digest(samplepath),diagnostics_sha256=digest(diagpath))
    write_json(destination, row)
    print('DONE', key, json.dumps(dict(metrics=metrics,comparisons=summary['comparisons'])), flush=True)


def queue(args):
    jobs = [(mode, rate, seed) for mode in args.modes.split(',')
            for seed in map(int,args.seeds.split(',')) for rate in map(int,args.rates.split(','))]
    gpus = list(map(int,args.gpus.split(',')))
    buckets = [jobs[i::len(gpus)] for i in range(len(gpus))]
    def worker(gpu, items):
        for mode, rate, seed in items:
            key = f'{mode}_rate{rate}_seed{seed}'
            cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'case',
                   '--mode', mode, '--rate', str(rate), '--seed', str(seed), '--gpu', str(gpu)]
            logpath = OUTPUT/'logs'/f'{key}.log'
            logpath.parent.mkdir(parents=True, exist_ok=True)
            with logpath.open('a') as log:
                process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONHASHSEED='0'))
            if process.returncode:
                raise RuntimeError(f'{key} failed: see {logpath}')
            print('COMPLETED', key, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker,gpu,bucket) for gpu,bucket in zip(gpus,buckets)]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','case','queue'])
    parser.add_argument('--mode', choices=MODES, default='legacy')
    parser.add_argument('--rate', type=int, default=13)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--modes', default='legacy')
    parser.add_argument('--rates', default='1,13,23,29,35')
    parser.add_argument('--seeds', default='1')
    parser.add_argument('--gpus', default='0,1,2,3,4')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'case': case(args.mode,args.rate,args.seed,args.gpu)
    else: queue(args)


if __name__ == '__main__':
    main()
