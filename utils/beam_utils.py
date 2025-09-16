#!/usr/bin/env python
import os
import sys
import numpy as np

def beamIdPair_to_beamPairId(beamIdPair, M_t, M_r):
    assert beamIdPair.shape[-1] == 2
    return beamIdPair[..., 0]*M_r + beamIdPair[..., 1]

def beamPairId_to_beamIdPair(beamPairId, M_t, M_r):
    j = (beamPairId % M_r).astype(np.int64)
    i = ((beamPairId - j) / M_r).astype(np.int64)
    return np.stack((i,j),axis=-1)

# 生成 DFT 码本
def generate_dft_codebook(M_t,N_codes=None):
    # 创建 DFT 矩阵
    # 这就是 M_t 个基向量的矩阵，包含 M_t 列，每列是一个 DFT 基向量
    if N_codes==None:
        dft_matrix = np.exp(-1j * 2 * np.pi * np.outer(np.arange(M_t), np.arange(M_t)) / M_t)
    else:
        dft_matrix = np.exp(-1j * 2 * np.pi * np.outer(np.arange(M_t), np.arange(N_codes)) / N_codes)
    return dft_matrix