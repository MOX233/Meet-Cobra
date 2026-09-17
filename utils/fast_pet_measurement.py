"""Vectorized arithmetic for the existing PET-BF physical measurements.

This is an opt-in simulator acceleration, not a policy or information source.
It preserves candidate order, the stopping test, and the charged pilot count.
"""
import collections
import numpy as np
from utils.channel_utils import rician_channel_gain
from utils.mox_utils import lin2dB


def measure_pet_batch(args, frame, veh_set, timeline_dir, bs_locations,
                      predicted_beams, predicted_gains, dft_tx, dft_rx,
                      bf_func, previous_beams, db_err_th=5, db_lb=-100,
                      rician_fading=False, K_BF=None):
    if bf_func != "topKbeam_savePilot":
        raise ValueError("Batch measurement implements PET-BF only")
    ids = list(veh_set)
    if not ids:
        return tuple(collections.OrderedDict() for _ in range(4))
    k = args.K if K_BF is None else K_BF
    channels = []
    # Keep the legacy random draw order exactly, including x/y interleaving.
    for v in ids:
        h = timeline_dir[frame][v]["h"]
        if rician_fading:
            h = h * np.sqrt(rician_channel_gain(args.K_rician, size=h.shape))
        channels.append(h)
    h = np.stack(channels)
    pairs = np.stack([predicted_beams[v][:, :k] for v in ids]).astype(int)
    tx = dft_tx[:, pairs // args.M_r].transpose(1, 2, 0, 3)
    rx = dft_rx[:, pairs % args.M_r].transpose(1, 0, 2, 3)
    projected = np.einsum("vrbt,vbtk->vrbk", h, tx)
    amplitude = np.abs(np.einsum("vrbk,vrbk->vbk", projected.conj(), rx)) / np.sqrt(args.M_r * args.M_t)
    gains = 2 * lin2dB(amplitude)
    predicted = np.stack([predicted_gains[v] for v in ids])
    accepted = (gains > predicted[..., None] - db_err_th) & (gains > db_lb)
    stop = np.where(accepted.any(-1), accepted.argmax(-1) + 1, k)
    tested = np.where(np.arange(k) < stop[..., None], gains, 2 * lin2dB(np.zeros_like(gains)))
    best = tested.argmax(-1)
    gain = tested.max(-1)
    beam = np.take_along_axis(pairs, best[..., None], -1)[..., 0]
    no_bf = 2 * lin2dB(np.abs(h).max(axis=(1, 3)))
    return tuple(collections.OrderedDict((v, array[i].copy()) for i, v in enumerate(ids))
                 for array in (gain, no_bf, beam, stop))
