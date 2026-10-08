#!/usr/bin/env python3
"""Safe output-isolated entrypoints for R1 data generation and preprocessing.

Never overwrites an existing output directory. SUMO and RT delegate to the
existing generators. System-input preprocessing preserves the arithmetic in
utils.sim_utils.get_default_sim_params without loading legacy NN weights.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fresh_output(path):
    path = path.resolve()
    if path.parent != ROOT/'experiment/results' or not path.name.startswith('data_rebuild_'):
        raise ValueError('Use a new experiment/results/data_rebuild_NAME directory')
    path.mkdir(exist_ok=False)
    return path


def legacy_args():
    from utils.options import args_parser
    previous = sys.argv
    try:
        sys.argv = [previous[0]]
        args = args_parser()
    finally:
        sys.argv = previous
    args.Lambda = 1.0
    args.random_factor = 2
    args.num_steps = 1000
    args.trajectoryInfo_path = './sumo_data/trajectory_Lbd1.00.csv'
    return args


def prepare_system(timeline, seed=1):
    """Same mode-0/Np8/10-frame input preprocessing as the retained legacy code."""
    import numpy as np
    from utils.beam_utils import generate_dft_codebook, beamPairId_to_beamIdPair
    from utils.data_utils import preprocess_input_np, generate_complex_gaussian_vector
    from utils.mox_utils import lin2dB
    np.random.seed(seed)
    tx, rx = generate_dft_codebook(32), generate_dft_codebook(8)
    output = collections.OrderedDict()
    previous = None
    for frame, records in timeline.items():
        output[frame] = {}
        for vehicle, original in records.items():
            r = dict(original)
            h = r['h']
            if h.shape != (8, 4, 32):
                raise ValueError(f'Unexpected H shape {h.shape}; expected (8,4,32)')
            idx = np.abs(np.matmul(np.matmul(h, tx).T.conjugate(), rx).transpose([1, 0, 2]).reshape(4, -1)).argmax(axis=-1)
            pairs = beamPairId_to_beamIdPair(idx, 32, 8)
            gain = np.zeros(4).astype(np.float32)
            for bs in range(4):
                gain[bs] = 1/np.sqrt(8*32)*np.abs(np.matmul(np.matmul(h[:, bs, :], tx[:, pairs[bs, 0]]).T.conjugate(), rx[:, pairs[bs, 1]]))
                gain[bs] = 2*lin2dB(gain[bs])
            csi = np.sqrt(.1)*np.matmul(h, tx)[:, :, :32:4].sum(axis=-2).reshape(-1)
            noise = generate_complex_gaussian_vector(csi.shape, scale=np.sqrt(1e-14), mean=0.)
            x = preprocess_input_np((csi+noise).astype(np.complex64)).reshape(1, -1)
            if previous is not None and vehicle in output[previous]:
                x = np.concatenate((output[previous][vehicle]['CSI_preprocessed'], x), axis=0)[-10:]
            r.update(best_beam_pair_idx=idx, best_beam_idx_pair=pairs, g_opt_beam=gain,
                     g_avg=2*lin2dB(np.abs(h).mean(axis=0).mean(axis=-1)), CSI_preprocessed=x)
            output[frame][vehicle] = r
        previous = frame
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    for name in ('sumo', 'raytrace', 'system-input'):
        p = sub.add_parser(name)
        p.add_argument('--output', type=Path, required=True)
        if name != 'sumo':
            p.add_argument('--source', type=Path, required=True)
        if name == 'raytrace':
            p.add_argument('--start', type=int, required=True)
            p.add_argument('--end', type=int, required=True)
            p.add_argument('--gpu', type=int, required=True)
            p.add_argument('--workers', type=int, default=1)
        if name == 'system-input':
            p.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if hasattr(args, 'source'):
        args.source = args.source.resolve(strict=True)
    command = sys.argv.copy()
    out = fresh_output(args.output)
    metadata = dict(command=command, phase=args.phase, complete=False,
                    source_sha256=digest(args.source) if hasattr(args, 'source') else None)
    try:
        if args.phase == 'system-input':
            with args.source.open('rb') as stream:
                raw = pickle.load(stream)
            timeline = prepare_system(raw, args.seed)
            product = out/'prepared_system.pkl'
            with product.open('wb') as stream:
                pickle.dump(timeline, stream, protocol=4)
            metadata.update(frames=len(timeline), start=min(timeline), end=max(timeline),
                            seed=args.seed, note='Regenerated pilot noise; use retained input for identical submitted realization')
        else:
            settings = legacy_args()
            (out/'sumo_data').mkdir()
            if args.phase == 'raytrace':
                if not (0 <= args.start < args.end <= 1000 and args.workers >= 1 and args.gpu >= 0):
                    raise ValueError('Invalid interval, GPU index or worker count')
                # The old child script has an import fallback that installs a
                # package; preflight here so a missing dependency fails instead.
                os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
                import sionna.rt  # noqa: F401
                (out/'sumo_data/trajectory_Lbd1.00.csv').symlink_to(args.source)
                for name in ('scene_from_sionna.xml', 'meshes', 'utils', 'generate_data_3Dbeam_subprocess.py'):
                    (out/name).symlink_to(ROOT/name)
                (out/'sionna_result').mkdir()
                os.environ['PATH'] = str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
                from utils.data_utils import run_sionna_sim
                os.chdir(out)
                run_sionna_sim(settings, args.start, args.end, max_workers=args.workers,
                               gpu=args.gpu, Lambda=1., N_t_V=32, N_r_V=8)
                products = list((out/'sionna_result').glob('*.pkl'))
                if len(products) != 1:
                    raise RuntimeError('Expected one merged RT file')
                product = products[0]
                with product.open('rb') as stream:
                    timeline = pickle.load(stream)
                import numpy as np
                expected = np.round(np.arange(args.start, args.end+.05, .1), 1)
                if len(timeline) != len(expected) or not np.allclose(sorted(timeline), expected):
                    raise RuntimeError('RT output is incomplete; inspect child logs before using it')
                metadata.update(frames=len(timeline), start=min(timeline), end=max(timeline))
            else:
                if not os.environ.get('SUMO_HOME') or not shutil.which('sumo'):
                    raise RuntimeError('Set SUMO_HOME and activate an environment containing SUMO')
                from utils.sumo_utils import sumo_run_with_trajectoryInfo
                os.chdir(out)
                sumo_run_with_trajectoryInfo(settings)
                product = out/'sumo_data/trajectory_Lbd1.00.csv'
                if not product.is_file() or product.stat().st_size == 0:
                    raise RuntimeError('Empty SUMO trajectory')
        metadata.update(complete=True, product=str(product), product_sha256=digest(product))
    finally:
        (out/'generation.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
