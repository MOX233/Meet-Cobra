"""Two-level acquisition on the existing 32-by-8 DFT beam grid.

Coarse codewords use antenna deactivation, not a subset of narrow beams.
Each unit-norm codeword activates N/4 consecutive elements, centered on a
group of four fine DFT directions. Only the winning pair of sectors is
refined. This simple construction is not a reproduction of BMW-SS.
"""

from functools import lru_cache

import numpy as np

from utils.mox_utils import lin2dB


@lru_cache(maxsize=8)
def coarse_codebook(num_antennas: int, num_sectors: int) -> np.ndarray:
    """Unit total power, equal magnitude on active elements, zeros elsewhere."""
    if num_antennas <= 0 or num_sectors <= 0 or num_antennas % num_sectors:
        raise ValueError("Sectors must partition the fine DFT grid")
    group_size = num_antennas // num_sectors
    centers = group_size * np.arange(num_sectors) + (group_size - 1) / 2
    active = num_sectors
    weights = np.zeros((num_antennas, num_sectors), dtype=complex)
    weights[:active] = np.exp(
        -2j * np.pi * np.outer(np.arange(active), centers) / num_antennas
    ) / np.sqrt(active)
    weights.setflags(write=False)
    return weights


def hierarchical_beam_pair(channel, micro_index, dft_tx, dft_rx):
    """Measure 8x2 coarse pairs, then 4x4 fine pairs; return fine indices/gain.

    The full fine-grid gain matrix is deliberately never evaluated here.
    Like the existing beam search, measurements are noiseless channel gains.
    Data transmission uses the full arrays and the original fine codebooks.
    """
    if dft_tx.shape != (32, 32) or dft_rx.shape != (8, 8):
        raise ValueError("hierarchical32 requires the 32-TX/8-RX DFT grid")
    h = channel[:, micro_index, :]
    if h.shape != (8, 32):
        raise ValueError("Channel dimensions do not match the DFT grid")
    wide_tx = coarse_codebook(32, 8)
    wide_rx = coarse_codebook(8, 2)
    coarse_gains = np.abs(wide_rx.conj().T @ h @ wide_tx) ** 2
    rx_sector, tx_sector = np.unravel_index(np.argmax(coarse_gains), (2, 8))
    tx_candidates = 4 * tx_sector + np.arange(4)
    rx_candidates = 4 * rx_sector + np.arange(4)
    fine_gains = np.abs(
        dft_rx[:, rx_candidates].conj().T @ h @ dft_tx[:, tx_candidates]
    ) / np.sqrt(32 * 8)
    rx_local, tx_local = np.unravel_index(np.argmax(fine_gains), (4, 4))
    gain_db = float(2.0 * lin2dB(fine_gains[rx_local, tx_local]))
    return int(tx_candidates[tx_local]), int(rx_candidates[rx_local]), gain_db
