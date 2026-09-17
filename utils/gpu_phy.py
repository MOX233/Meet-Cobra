"""Batched GPU evaluation of the existing slot-level Rician channel model.

No policy sees these tensors. A frame/seed-keyed CUDA generator produces the
same external fading for all compared methods, independently of association.
The distribution and link budget are unchanged; CUDA and NumPy seeds do not
produce identical samples. Use this backend consistently across comparisons.
"""
import math
import numpy as np
import torch
from utils.beam_utils import generate_dft_codebook


class GPUFramePHY:
    @torch.inference_mode()
    def __init__(self, args, records, frame, seed, device):
        self.args = args
        self.ids = sorted(records, key=str)
        self.device = torch.device(device)
        h = torch.as_tensor(np.stack([records[v]["h"] for v in self.ids]),
                            dtype=torch.complex128, device=self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed) * 1000003 + round(float(frame) * 10))
        shape = (args.slots_per_frame, len(self.ids), 2, *h.shape[1:])
        noise = torch.randn(shape, generator=generator, dtype=torch.float64, device=self.device) / math.sqrt(2)
        x, y = noise[:, :, 0], noise[:, :, 1]
        k = args.K_rician
        factor = ((k + 2 * math.sqrt(k) * x + x.square() + y.square()) / (k + 1)).clamp_min(0).sqrt()
        self.h = h[None] * factor
        self.tx = torch.as_tensor(generate_dft_codebook(args.M_t), device=self.device)
        self.rx = torch.as_tensor(generate_dft_codebook(args.M_r), device=self.device)

    def db(self, amplitude):
        return 20 * torch.log10(amplitude / math.sqrt(self.args.M_r * self.args.M_t) + 1e-9)

    @torch.inference_mode()
    def fixed_pairs(self, connection, learners):
        bs = torch.tensor([max(connection[v]-1, 0) for v in self.ids], device=self.device)
        tx = torch.tensor([learners[v].tx_beam or 0 for v in self.ids], device=self.device)
        rx = torch.tensor([learners[v].rx_beam or 0 for v in self.ids], device=self.device)
        index = torch.arange(len(self.ids), device=self.device)
        selected = self.h.permute(0, 1, 3, 2, 4)[:, index, bs]
        response = torch.einsum("svrt,vt->svr", selected, self.tx[:, tx].T)
        amplitude = torch.einsum("svr,vr->sv", response.conj(), self.rx[:, rx].T).abs()
        return self.db(amplitude).cpu().numpy()

    @torch.inference_mode()
    def pet(self, predicted_beams, predicted_gains, k=5):
        pairs = torch.tensor(np.stack([predicted_beams[v][:, :k] for v in self.ids]),
                             dtype=torch.long, device=self.device)
        tx = self.tx[:, pairs // self.args.M_r].permute(1, 2, 0, 3)
        rx = self.rx[:, pairs % self.args.M_r].permute(1, 0, 2, 3)
        response = torch.einsum("svrbt,vbtk->svrbk", self.h, tx)
        gains = self.db(torch.einsum("svrbk,vrbk->svbk", response.conj(), rx).abs())
        predicted = torch.as_tensor(np.stack([predicted_gains[v] for v in self.ids]), device=self.device)
        accepted = (gains > predicted[None, ..., None] - 5) & (gains > -100)
        stop = torch.where(accepted.any(-1), accepted.long().argmax(-1) + 1, k)
        tested = torch.where(torch.arange(k, device=self.device) < stop[..., None], gains, -180.)
        value, best = tested.max(-1)
        beam = pairs[None].expand(self.args.slots_per_frame, -1, -1, -1).gather(-1, best[..., None])[..., 0]
        no_bf = 20 * torch.log10(self.h.abs().amax(dim=(2, 4)) + 1e-9)
        return tuple(x.cpu().numpy() for x in (value, no_bf, beam, stop))
