"""Check paired Rician arithmetic across the GPUs used by the two studies."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiment.pql_ba_experiment import paper_args
from utils.gpu_phy import GPUFramePHY


def main():
    args = paper_args()
    rng = np.random.default_rng(910)
    records = {str(v): {"h": (rng.normal(size=(8, 4, 32)) +
                              1j * rng.normal(size=(8, 4, 32))) * 1e-6}
               for v in range(4)}
    beam = {v: rng.integers(0, 256, (4, 5)) for v in records}
    gain = {v: rng.uniform(-120, -70, 4) for v in records}
    checks = []
    for seed in (1, 2, 3):
        values = []
        for gpu in (3, 5):
            phy = GPUFramePHY(args, records, 801.1, seed, f"cuda:{gpu}")
            values.append((phy.h.cpu().numpy(), *phy.pet(beam, gain)))
            del phy
            torch.cuda.empty_cache()
        for a, b in zip(*values):
            np.testing.assert_array_equal(a, b)
        checks.append(dict(seed=seed, channel_and_pet_bitwise_equal=True))
    output = ROOT / "experiment/results/o_mappo_report_input_20260917"
    output.mkdir(exist_ok=True)
    (output / "gpu_pairing_audit.json").write_text(json.dumps(dict(
        gpu_indices=[3, 5], devices=[torch.cuda.get_device_name(i) for i in (3, 5)],
        checks=checks, torch_version=torch.__version__), indent=2) + "\n")
    print("GPU 3/5 paired channel and PET outputs exactly equal for all audit seeds.")


if __name__ == "__main__":
    main()
