#!/usr/bin/env python3
"""Read-only mobility/data audit; no inference, training, or simulation runs.

Writes new audit artifacts only. Speeds are aligned to the actual serialized
channel records, not to an assumed SUMO speed or a reconstructed new trace.
"""
import argparse
import gc
import hashlib
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "experiment/results/revision_directional_20260922"
SPLIT = ROOT / "experiment/results/stateful_tbptt_unified_split_20260913/vehicle_split_seed20.npz"
BINS = [0., 1., 20., 40., float("inf")]
LABELS = ["[0,1)", "[1,20)", "[20,40)", "[40,infinity)"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def load_timeline(path):
    with path.open("rb") as f:
        return pickle.load(f)


def mobility_rows(timeline):
    return pd.DataFrame(
        [(int(round(float(t) * 10)), str(v), float(r["v"]),
          float(r["pos"][0]), float(r["pos"][1]))
         for t, rows in timeline.items() for v, r in rows.items()],
        columns=["step", "vehicle", "speed_mps", "x", "y"],
    )


def summarize(name, rows):
    speed = rows.speed_mps.to_numpy() * 3.6
    assert len(speed) and np.isfinite(speed).all() and speed.min() >= 0
    result = dict(dataset=name, vehicle_frames=len(speed),
                  serialized_vehicle_ids=int(rows.vehicle.nunique()),
                  mean_kmh=float(speed.mean()), minimum_kmh=float(speed.min()),
                  maximum_kmh=float(speed.max()), stopped_percent=float(100*np.mean(speed == 0)),
                  below_1_kmh_percent=float(100*np.mean(speed < 1)))
    result.update({f"p{p}_kmh": float(np.percentile(speed, p)) for p in (25, 50, 75, 90, 95, 99)})
    groups = []
    for lo, hi, label in zip(BINS[:-1], BINS[1:], LABELS):
        mask = (speed >= lo) & (speed < hi)
        subset = rows.loc[mask]
        groups.append(dict(dataset=name, speed_group_kmh=label,
                           vehicle_frames=int(mask.sum()), percent=float(mask.mean()*100),
                           serialized_vehicle_ids=int(subset.vehicle.nunique()),
                           time_frames=int(subset.step.nunique())))
    return result, groups


def npy_shape(z, name):
    with z.open(name + ".npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        return shape, str(dtype)


def compare_csv(rows, csv_index):
    ref = csv_index.reindex(pd.MultiIndex.from_frame(rows[["step", "vehicle"]]))
    missing = ref.speed_mps.isna().to_numpy()
    valid = ~missing
    speed_error = np.abs(rows.speed_mps.to_numpy()[valid]-ref.speed_mps.to_numpy()[valid])
    pos_error = np.linalg.norm(rows[["x", "y"]].to_numpy()[valid]-ref[["x", "y"]].to_numpy()[valid], axis=1)
    return dict(rows=len(rows), missing_csv_rows=int(missing.sum()),
                maximum_speed_error_mps=float(speed_error.max()),
                maximum_position_error_m=float(pos_error.max()),
                position_mismatch_rows=int((pos_error > 1e-6).sum()),
                speed_mismatch_rows=int((speed_error > 1e-8).sum())), ref


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new audit directory; existing results are never overwritten")
    args.output.mkdir(parents=True)
    csv_path = ROOT / "sumo_data/trajectory_Lbd1.00.csv"
    csv = pd.read_csv(csv_path, header=None,
                      names=["step", "sumo_vehicle", "veh_index", "x", "y", "speed_mps", "angle"])
    csv["vehicle"] = csv.sumo_vehicle.str.replace("flow", "", regex=False).astype(float).astype(str)
    # Reproduce the existing reader's last-assignment-wins behavior for auditing
    # only. This is not a fix to the underlying identifier conversion.
    duplicate = csv.duplicated(["step", "vehicle"], keep=False)
    reduced = csv.drop_duplicates(["step", "vehicle"], keep="last")
    csv_index = reduced.set_index(["step", "vehicle"])
    duplicate_keys = csv.loc[duplicate, ["step", "vehicle"]].drop_duplicates()
    duplicate_keys.to_csv(args.output / "colliding_frame_keys.csv", index=False)
    print("CSV loaded; checking actual channel records", flush=True)

    metadata = json.loads((REV / "training_data.json").read_text())
    source = Path(metadata["source"])
    timeline = load_timeline(source)
    train_rows = mobility_rows(timeline)
    del timeline
    gc.collect()
    train_check, _ = compare_csv(train_rows, csv_index)
    assert train_check["missing_csv_rows"] == 0
    assert train_check["position_mismatch_rows"] == train_check["speed_mismatch_rows"] == 0
    train_index = train_rows.set_index(["step", "vehicle"])
    with np.load(REV / "training_data.npz") as d:
        ids = np.repeat(d["vehicle_ids"], np.diff(d["offsets"]))
        target = np.rint(d["target_frame"]*10).astype(np.int64)
        current = np.rint(d["input_frame"]*10).astype(np.int64)
    np.testing.assert_array_equal(target-current, np.ones(len(target), dtype=np.int64))
    target_index = pd.MultiIndex.from_arrays([target, ids], names=["step", "vehicle"])
    input_index = pd.MultiIndex.from_arrays([current, ids], names=["step", "vehicle"])
    samples = train_index.reindex(target_index).reset_index()
    input_samples = train_index.reindex(input_index).reset_index()
    assert samples.speed_mps.notna().all() and input_samples.speed_mps.notna().all()
    with np.load(SPLIT) as split:
        train_ids, validation_ids = split["train_vehicle_ids"], split["validation_vehicle_ids"]
    train_mask = np.isin(ids, train_ids)
    validation_mask = np.isin(ids, validation_ids)
    assert np.all(train_mask ^ validation_mask)
    current_names = csv_index.reindex(input_index).sumo_vehicle.to_numpy()
    target_names = csv_index.reindex(target_index).sumo_vehicle.to_numpy()
    switches = current_names != target_names
    switch_rows = pd.DataFrame(dict(vehicle=ids[switches], input_step=current[switches],
                                    target_step=target[switches], source_sumo_id=current_names[switches],
                                    target_sumo_id=target_names[switches]))
    switch_rows.to_csv(args.output / "training_identity_transitions.csv", index=False)
    duplicate_index = pd.MultiIndex.from_frame(duplicate_keys)
    collision_samples = target_index.isin(duplicate_index)
    print("Training sample and speed alignment checked", flush=True)

    cache = load_timeline(REV / "test_predictions.pkl")
    test_rows = mobility_rows(cache)
    test_check, test_csv_rows = compare_csv(test_rows, csv_index)
    assert test_check["missing_csv_rows"] == 0
    assert test_check["position_mismatch_rows"] == test_check["speed_mismatch_rows"] == 0
    frames = sorted(cache)
    service_steps = np.rint(np.asarray(frames[1:])*10).astype(np.int64)
    test_index = test_rows.set_index(["step", "vehicle"])
    raw_path = REV / "grid/raw/meet_cobra_rate29_seed1.npz"
    with np.load(raw_path) as raw:
        qf, qv = raw["queue_frame"], raw["queue_vehicle"]
    service_index = pd.MultiIndex.from_arrays([service_steps[qf], qv], names=["step", "vehicle"])
    aligned = test_index.reindex(service_index).reset_index()
    assert aligned.speed_mps.notna().all()
    analyzed = aligned.loc[qf >= 2].copy()
    tested_pairs = []
    test_identity_changes = []
    test_names = test_csv_rows.sumo_vehicle.to_dict()
    for step, frame in enumerate(frames[1:]):
        previous, following = cache[frames[step]], cache[frame]
        for v in previous.keys() & following.keys():
            p = previous[v]["shared_prediction"]
            assert abs(p["target_frame"]-frame) < 1e-7
            assert np.asarray(p["beam"]).shape == (4, 5)
            assert np.asarray(p["gain"]).shape == np.asarray(p["interference"]).shape == (4,)
            target_step = int(round(frame*10))
            tested_pairs.append((target_step, str(v)))
            names = [test_names[(target_step-1, str(v))], test_names[(target_step, str(v))]]
            if names[0] != names[1]:
                test_identity_changes.append(dict(vehicle=str(v),target_step=target_step,old=names[0],new=names[1]))
    test_pred = test_index.reindex(pd.MultiIndex.from_tuples(tested_pairs, names=["step", "vehicle"])).reset_index()
    test_pred_scored = test_pred[test_pred.step >= service_steps[2]]
    del cache
    gc.collect()

    sets = {"training_targets": samples[train_mask], "validation_targets": samples[validation_mask],
            "training_inputs": input_samples[train_mask], "validation_inputs": input_samples[validation_mask],
            "system_test_scored": analyzed, "test_prediction_targets_scored": test_pred_scored}
    statistics, groups = [], []
    for name, rows in sets.items():
        stat, grouping = summarize(name, rows)
        statistics.append(stat)
        groups.extend(grouping)
    pd.DataFrame(statistics).to_csv(args.output / "speed_statistics.csv", index=False)
    pd.DataFrame(groups).to_csv(args.output / "speed_groups.csv", index=False)
    # Descriptive speed CDF data, not performance-based choice of bins.
    cdf = []
    for name in ("training_targets", "validation_targets", "system_test_scored"):
        ordered = np.sort(sets[name].speed_mps.to_numpy()*3.6)
        for threshold in np.arange(0, 75.1, .5):
            cdf.append(dict(dataset=name, speed_kmh=float(threshold),
                            cdf=float(np.searchsorted(ordered, threshold, side="right")/len(ordered))))
    pd.DataFrame(cdf).to_csv(args.output / "speed_cdf.csv", index=False)
    analyzed.assign(service_frame=qf[qf >= 2]).to_csv(args.output / "test_speed_alignment.csv", index=False)

    # Check availability and NPY headers for the 432 formal, current cases.
    inventory = []
    methods = ["meet_cobra", "oracle_mc", "reactive_obra", "wo_gap_ho", "wo_pet_bf", "wo_otr_ra", "o_mappo", "mts_report"]
    required = {"queue_bits", "queue_frame", "queue_vehicle"}
    for method in methods:
        for rate in range(1, 36, 2):
            for seed in (1, 2, 3):
                if method == "o_mappo":
                    path = ROOT / f"experiment/results/o_mappo_predicted_cross5_20260925/test/runs/predicted_cross5_rate{rate}_seed{seed}.npz"
                elif method == "mts_report":
                    path = ROOT / f"experiment/results/mts_h32_full_grid_20260925/raw/mts_h32_cross5_rate{rate}_seed{seed}.npz"
                else:
                    path = REV / f"grid/raw/{method}_rate{rate}_seed{seed}.npz"
                assert path.exists(), path
                with zipfile.ZipFile(path) as z:
                    members = {Path(x).stem for x in z.namelist()}
                    assert required <= members, (path, members)
                    shape, dtype = npy_shape(z, "queue_bits")
                    assert shape == (len(qf), 100), (path, shape)
                # Check every current MEET/Oracle result's join keys, without
                # inflating all queue arrays or running performance analysis.
                indices_checked = method in ("meet_cobra", "oracle_mc")
                if indices_checked:
                    with np.load(path) as d:
                        np.testing.assert_array_equal(d["queue_frame"], qf)
                        np.testing.assert_array_equal(d["queue_vehicle"], qv)
                inventory.append(dict(method=method, rate_mbps=rate, seed=seed,
                                      path=str(path.relative_to(ROOT)), queue_rows=shape[0],
                                      slots_per_row=shape[1], dtype=dtype, join_keys_checked=indices_checked))
    pd.DataFrame(inventory).to_csv(args.output / "result_inventory.csv", index=False)

    # Verify a representative actual queue payload and its published metrics.
    with np.load(raw_path) as raw:
        q = raw["queue_bits"]
        assert np.isfinite(q).all() and (q >= 0).all()
        mask = qf >= 2
        u = np.array([np.mean(q[qf == f] > 29e6*.020) for f in range(300)])
        np.testing.assert_allclose(u, raw["violation_probability"], atol=1e-12)
        delay = q[mask].ravel() * 1000 / 29e6
        metrics = json.loads((REV / "grid/runs/meet_cobra_rate29_seed1.json").read_text())["metrics"]
        np.testing.assert_allclose(100*u[2:].mean(), metrics["violation_percent"], atol=1e-12)
        np.testing.assert_allclose(np.percentile(delay, 99), metrics["p99_proxy_ms"], atol=1e-12)

    analysis_keys = pd.MultiIndex.from_frame(analyzed[["step", "vehicle"]])
    grouped_ids = reduced.groupby("vehicle").sumo_vehicle.nunique()
    report = dict(
        scope="Descriptive audit only: no new inference, training, simulation or manuscript edits",
        script_sha256=digest(Path(__file__)), csv_sha256=digest(csv_path), split_sha256=digest(SPLIT),
        training_source_metadata=metadata,
        training_alignment=train_check, test_alignment=test_check,
        training_samples=len(samples), train_samples=int(train_mask.sum()), validation_samples=int(validation_mask.sum()),
        train_vehicle_ids=len(train_ids), validation_vehicle_ids=len(validation_ids),
        unique_original_training_sumo_ids=int(pd.Series(target_names[train_mask]).nunique()),
        unique_original_validation_sumo_ids=int(pd.Series(target_names[validation_mask]).nunique()),
        original_sumo_train_validation_overlap=len(set(target_names[train_mask]) & set(target_names[validation_mask])),
        input_target_speed_mean_abs_difference_kmh=float(np.mean(abs(samples.speed_mps.to_numpy()-input_samples.speed_mps.to_numpy()))*3.6),
        system_service_frames=300, discarded_frames=2, analyzed_frames=298,
        system_vehicle_frames=len(aligned), analyzed_vehicle_frames=len(analyzed),
        analyzed_vehicle_slots=len(analyzed)*100,
        system_scored_frame_timestamps_s=[float(analyzed.step.min()/10),float(analyzed.step.max()/10)],
        prediction_pairs=len(test_pred), analyzed_prediction_pairs=len(test_pred_scored),
        scored_rows_without_previous_frame_prediction=len(analyzed)-len(test_pred_scored),
        existing_result_files=len(inventory), meet_oracle_join_key_checks=108,
        representative_queue_metrics_checked=True,
        id_conversion=dict(csv_rows=len(csv), csv_duplicate_frame_keys=len(duplicate_keys),
            numeric_ids_with_multiple_original_ids=int((grouped_ids > 1).sum()),
            training_target_rows_at_colliding_keys=int(collision_samples.sum()),
            training_cross_identity_next_frame_pairs=int(switches.sum()),
            training_cross_identity_pairs_train=int((switches & train_mask).sum()),
            training_cross_identity_pairs_validation=int((switches & validation_mask).sum()),
            system_scored_rows_at_colliding_keys=int(analysis_keys.isin(duplicate_index).sum()),
            system_prediction_identity_changes=test_identity_changes),
        speed_statistics=statistics, speed_groups=groups,
        availability=dict(validation_predictions="Labels, split, selected checkpoints and continuous trajectories available; new frozen-model validation inference required for speed-conditioned NN errors",
            test_predictions="Current cache stores top-5 indices and both predicted gains; next-frame truths available; no new inference required on 800-830 s",
            system_results="432 raw archives present; 108 MEET/Oracle join indices verified; representative queue payload and published metrics reproduce",
            individual_power="Stored energy and RB counts are per BS or system, not per vehicle; do not assign global power to speed bins",
            extended_test="Existing 800-950 s CSI trace available, but old prediction outputs use the prior interference predictor and cannot replace the final bundle"))
    (args.output / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("training_alignment", "test_alignment", "speed_statistics", "id_conversion", "existing_result_files", "analyzed_vehicle_frames", "analyzed_prediction_pairs")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
