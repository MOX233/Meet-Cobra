#!/usr/bin/env python3
"""Verify lossless shared arrays and real spawn workers before long training."""
from pathlib import Path
import pickle
import concurrent.futures
import multiprocessing as mp
import numpy as np
from experiment import train_o_mappo_hierarchical32 as train


def main():
    root=Path('experiment/results/o_mappo_h32_retrained_20260924_v4')
    train.initialize_worker(str(root))
    actual=train.array_slice(710,1)
    with (root/'exact_validation.pkl').open('rb') as stream:
        source=pickle.load(stream)
    for frame,records in actual.items():
        assert list(records)==list(source[frame])
        for vehicle,record in records.items():
            for key in ('h','pos','v','angle'):
                np.testing.assert_array_equal(record[key],source[frame][vehicle][key])
    policy=Path('experiment/results/o_mappo_h32_retrained_20260924/reference.pt').resolve()
    jobs=[(policy,r,710,2,1000+r,True) for r in (1,15,25,35)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),
            initializer=train.initialize_worker,initargs=(str(root.resolve()),)) as pool:
        result=list(pool.map(train.rollout,jobs))
    print('PASS lossless original data and vehicle IDs; four spawn workers, four loads',
          [len(transitions) for _,transitions in result],flush=True)


if __name__=='__main__':main()
