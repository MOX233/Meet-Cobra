"""Shadow-check compiled matching on actual NN-driven GAP-HO matrices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import o_mappo_shared_frontend as exp
from utils.compiled_matching import km_algorithm_compiled
import utils.alg_utils as alg
import numpy as np


def main():
    source = alg.km_algorithm
    checks = []
    def checked(cost):
        old = source(cost)
        new = km_algorithm_compiled(cost)
        assert old[0] == new[0], "Matching or tie order differs"
        assert old[1] == new[1], "Matching total weight differs"
        checks.append(dict(shape=list(cost.shape), matching_equal=True, total_weight_equal=True))
        return new
    alg.km_algorithm = checked
    timeline = exp.temporal_slice(exp.read_pickle(exp.OUTPUT / "test_prepared.pkl"), 800, 800.5)
    # Matching needs actual NN-driven matrices, not a GPU context. Keep this
    # short arithmetic audit separate from the paired-Rician test runs.
    exp.CONTEXT = (exp.OUTPUT, timeline, None, False, True, None, False)
    # Unique audit seed avoids reusing the debug cache and skipping checks.
    result = exp.evaluate_one(("meet_cobra", 13, 991, 10))
    assert len(checks) == 10, len(checks)
    exp.write_json(exp.OUTPUT / "compiled_matching_audit.json", dict(checks=checks,
                   matrices=len(checks), result=result, tolerance=1e-10, fastmath=False))
    print("Real GAP-HO matching matrices checked:", len(checks), flush=True)


if __name__ == "__main__":
    main()
