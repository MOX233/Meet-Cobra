#!/usr/bin/env python3
"""One-off numerical and timing check for batched legacy predictors."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment.paper_methods_multiseed import load_common
from utils.sim_utils import _predict_vehicle_batch


args, _, timeline, _, beam, gain, interference, _ = load_common(
    1, Path("/tmp/meet_cobra_batchcheck"), 800, 800.3, 0
)
frame = list(timeline)[-1]
csi = {
    vehicle: timeline[frame][vehicle]["CSI_preprocessed"].astype(np.float32)
    for vehicle in timeline[frame]
}
print("vehicles", len(csi))
for name, model, top_k in (
    ("gain", gain, None),
    ("beam", beam, 5),
    ("interference", interference, None),
):
    started = time.time()
    individual = {
        vehicle: (
            model.predict(value[None], args.device, top_k)[0]
            if top_k is not None
            else model.predict(value[None], args.device)[0]
        )
        for vehicle, value in csi.items()
    }
    individual_s = time.time() - started
    started = time.time()
    batched = _predict_vehicle_batch(
        model, csi, args.device, K=top_k, batch_size=512
    )
    batch_s = time.time() - started
    exact = all(np.array_equal(individual[v], batched[v]) for v in csi)
    max_error = max(
        float(np.max(np.abs(individual[v] - batched[v]))) for v in csi
    )
    print(
        name,
        "individual_s",
        individual_s,
        "batch_s",
        batch_s,
        "speedup",
        individual_s / batch_s,
        "exact",
        exact,
        "max_error",
        max_error,
    )
