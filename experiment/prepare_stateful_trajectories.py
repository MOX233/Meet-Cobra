#!/usr/bin/env python3
"""Build compact chronological streams for stateful TBPTT training.

The output contains clean superposed pilot observations at frame x and labels
at frame x+1. Pilot noise is generated online during training, so every epoch
can use a fresh realization without storing the 5.4-GB channel timeline.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment.benchmark_nn_overhead import digest, json_write
from utils.beam_utils import generate_dft_codebook
import numpy as np

DEFAULT_SOURCE = ROOT / "sionna_result/trajectoryInfo_lbd1.00_200_800_3Dbeam_tx(1,32)_rx(1,8)_freq2.8e+10.pkl"


def frame_values(records, pilot_power=0.1, pilot_count=8, interference_label="legacy-max"):
    """Vectorized labels and noiseless pilot observations for one frame."""
    h = np.stack([record["h"] for record in records]).astype(np.complex64, copy=False)
    # The repository DFT codebook uses exp(-j2*pi*n*k/N), i.e., np.fft.fft.
    tx_response = np.fft.fft(h, axis=-1)
    stride = h.shape[-1] // pilot_count
    clean_csi = (np.sqrt(pilot_power) * tx_response[..., :pilot_count * stride:stride]
                 .sum(axis=-2).reshape(len(h), -1)).astype(np.complex64)
    # Matches abs(((h @ DFT_tx).T.conj() @ DFT_rx).transpose(...)).
    response = np.abs(np.fft.fft(np.conj(tx_response), axis=1).transpose(0, 2, 3, 1))
    flat = response.reshape(len(h), h.shape[2], -1)
    beam = flat.argmax(-1).astype(np.int16)
    desired = (20 * np.log10(flat.max(-1) / np.sqrt(h.shape[1] * h.shape[3]) + 1e-9)).astype(np.float32)
    if interference_label == "beam-average":
        from utils.directional_service import beam_average_gain_db
        interference = beam_average_gain_db(h).astype(np.float32)
    elif interference_label == "legacy-max":
        interference = (20 * np.log10(np.abs(h).max(axis=(1, 3)) + 1e-9)).astype(np.float32)
    else:
        raise ValueError("Unknown interference label")
    return clean_csi, beam, desired, interference


def audit_fft(records, outputs, count=32):
    clean, beam, desired, _ = outputs
    tx = generate_dft_codebook(32)
    rx = generate_dft_codebook(8)
    for i in range(min(count, len(records))):
        h = records[i]["h"]
        tx_response = h @ tx
        expected_csi = (np.sqrt(0.1) * tx_response[:, :, ::4].sum(axis=-2).reshape(-1)).astype(np.complex64)
        response = np.abs((tx_response.T.conj() @ rx).transpose(1, 0, 2)).reshape(4, -1)
        expected_beam = response.argmax(-1)
        expected_gain = 20 * np.log10(response.max(-1) / 16 + 1e-9)
        np.testing.assert_allclose(clean[i], expected_csi, rtol=2e-5, atol=2e-10)
        np.testing.assert_array_equal(beam[i], expected_beam)
        np.testing.assert_allclose(desired[i], expected_gain, rtol=0, atol=2e-5)


def build(source, output, interference_label="legacy-max"):
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    started = time.monotonic()
    with source.open("rb") as handle:
        timeline = pickle.load(handle)
    frames = sorted(timeline)
    streams = collections.defaultdict(list)
    audited = False
    for fi, frame in enumerate(frames):
        ids = list(timeline[frame])
        records = [timeline[frame][v] for v in ids]
        values = frame_values(records, interference_label=interference_label)
        if not audited:
            audit_fft(records, values)
            audited = True
        for i, vehicle in enumerate(ids):
            streams[str(vehicle)].append((float(frame), *(value[i] for value in values)))
        if (fi + 1) % 500 == 0 or fi + 1 == len(frames):
            print(f"Prepared {fi+1}/{len(frames)} frames; elapsed {time.monotonic()-started:.1f}s", flush=True)

    pieces = {name: [] for name in ("clean_csi", "beam", "desired_gain", "interfering_gain",
                                     "input_frame", "target_frame")}
    offsets, vehicle_ids, segment_indices = [0], [], []
    gap_count = 0
    for vehicle in sorted(streams, key=lambda x: float(x)):
        rows = streams[vehicle]
        starts = [0]
        for i in range(1, len(rows)):
            if not np.isclose(rows[i][0] - rows[i-1][0], 0.1, atol=1e-7, rtol=0):
                starts.append(i)
                gap_count += 1
        starts.append(len(rows))
        for segment_number, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
            if end - start < 2:
                continue
            segment = rows[start:end]
            pieces["clean_csi"].append(np.stack([x[1] for x in segment[:-1]]))
            pieces["beam"].append(np.stack([x[2] for x in segment[1:]]))
            pieces["desired_gain"].append(np.stack([x[3] for x in segment[1:]]))
            pieces["interfering_gain"].append(np.stack([x[4] for x in segment[1:]]))
            pieces["input_frame"].append(np.asarray([x[0] for x in segment[:-1]], dtype=np.float32))
            pieces["target_frame"].append(np.asarray([x[0] for x in segment[1:]], dtype=np.float32))
            length = end - start - 1
            offsets.append(offsets[-1] + length)
            vehicle_ids.append(vehicle)
            segment_indices.append(segment_number)

    data = {name: np.concatenate(values) for name, values in pieces.items()}
    data.update(offsets=np.asarray(offsets, dtype=np.int64),
                vehicle_ids=np.asarray(vehicle_ids), segment_indices=np.asarray(segment_indices, dtype=np.int16))
    np.testing.assert_allclose(data["target_frame"] - data["input_frame"], .1, rtol=0, atol=4e-5)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **data)
    metadata = {
        "source": str(source.resolve()), "source_sha256": digest(source),
        "output": str(output.resolve()), "output_sha256": digest(output),
        "frames": len(frames), "frame_range": [frames[0], frames[-1]],
        "vehicle_frames_in_source": sum(len(x) for x in timeline.values()),
        "vehicles": len(streams), "trajectory_segments": len(vehicle_ids), "gaps": gap_count,
        "prediction_samples": int(len(data["clean_csi"])),
        "minimum_segment_length": int(np.diff(data["offsets"]).min()),
        "maximum_segment_length": int(np.diff(data["offsets"]).max()),
        "pilot_count": 8, "pilot_power_w": 0.1, "pilot_noise_power_w": 1e-14,
        "label_alignment": "clean CSI at x predicts beam and gains computed from h at x+1",
        "gain_units": "dB; training normalizes as gain/20+7",
        "interference_label": interference_label,
        "fft_parity_audit": "passed against repository DFT matrices on first frame",
        "elapsed_seconds": time.monotonic() - started,
    }
    json_write(output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interference-label", choices=("legacy-max", "beam-average"), default="legacy-max")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    build(args.source, args.output, args.interference_label)


if __name__ == "__main__":
    main()
