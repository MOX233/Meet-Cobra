#!/usr/bin/env python3
"""Run a disjoint subset of an existing HO protocol, e.g. late high-load cases.

Do not run a case concurrently in another process. The primary sweep resumes
these case files when it reaches them; this launcher does not rewrite aggregates.
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import ho_interruption_experiment as exp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rate', type=float, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--durations-ms', type=float, nargs='+', required=True)
    options = parser.parse_args()
    protocol = json.loads((options.output / 'protocol.json').read_text())
    assert options.rate in protocol['rates_mbps']
    assert set(options.seeds).issubset(protocol['seeds'])
    assert set(options.durations_ms).issubset(protocol['durations_ms'])
    for path, digest in protocol['source_sha256'].items():
        assert hashlib.sha256((exp.ROOT / path).read_bytes()).hexdigest() == digest
    exp.torch.set_num_threads(1)
    saved = sys.argv[:]
    sys.argv = [saved[0]]
    try:
        args, locations, timeline, *_ = exp.get_default_sim_params(
            str(options.output / '_loader'), cut_ratio=protocol['seconds'] / 150,
            load_predictors=False)
    finally:
        sys.argv = saved
    assert exp.jsonable(vars(args)) == protocol['args']
    cache = exp.oracle_cache(args, locations, timeline)
    context = (args, locations, timeline, cache, options.output, options.durations_ms)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(options.seeds), mp_context=multiprocessing.get_context('spawn'),
            initializer=exp.initialize_worker, initargs=(context,)) as pool:
        for future in concurrent.futures.as_completed([
                pool.submit(exp.run_pair, (options.rate, seed)) for seed in options.seeds]):
            future.result()
    print('SELECTED CASES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
