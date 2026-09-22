"""Private physical service evaluator shared by the revised baselines.

Policy inputs never include the cross-gain tensors. Random frequency mapping
has its own frame/seed keyed stream; it cannot perturb arrivals or fading.
"""
import numpy as np
import torch


BEAM_AVERAGE_DB_CONVENTION = '20*log10(sqrt(mean(abs(H)**2))+1e-9)'


def beam_average_gain_db(channel):
    """Beam-averaged power in dB, with the original amplitude-domain epsilon.

    H ends in (R, BS, T). Adding 1e-9 before taking the amplitude logarithm
    preserves the original -180 dB value for a zero channel. This numerical
    convention does not change the physical evaluator's linear gains.
    """
    power = np.mean(np.abs(channel).astype(np.float64) ** 2, axis=(-3, -1))
    return 20 * np.log10(np.sqrt(power) + 1e-9)


def state_pairs(physical, connection, states):
    result = np.zeros((physical.args.slots_per_frame, len(physical.ids), 4), dtype=int)
    for j, v in enumerate(physical.ids):
        if connection[v] > 0:
            state = states[v]
            result[:, j, connection[v] - 1] = int(state.tx_beam) * physical.args.M_r + int(state.rx_beam)
    return result


class DirectionalService:
    @torch.inference_mode()
    def __init__(self, physical, pairs, connection, seed, frame, diagnostics=None):
        self.ids = physical.ids
        self.bs = np.array([connection[v] for v in self.ids], dtype=int)
        self.args = physical.args
        slots, vehicles = physical.h.shape[:2]
        device = physical.device
        b = torch.as_tensor(np.maximum(self.bs - 1, 0), device=device)
        vi = torch.arange(vehicles, device=device)
        si = torch.arange(slots, device=device)
        chosen = torch.as_tensor(pairs, dtype=torch.long, device=device)[:, vi, b]
        tx, rx = chosen // self.args.M_r, chosen % self.args.M_r
        response = torch.einsum('svrbt,svr->svbt', physical.h,
                                physical.rx[:, rx].permute(1, 2, 0).conj())
        response = torch.einsum('svbt,tk->svbk', response, physical.tx)
        gains = response.abs().square() / (self.args.M_r * self.args.M_t)
        self.cross = gains[si[:, None, None], vi[None, :, None], b[None, None, :], tx[:, None, :]].cpu().numpy()
        self.own = gains[si[:, None], vi[None, :], b[None, :], tx].cpu().numpy()
        self.mean = physical.h.abs().square().mean(dim=(2, 4)).cpu().numpy()
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), round(float(frame)*10), 24681357]))
        self.diagnostics = diagnostics
        self.stats = dict(frame=float(frame), rb_weight=0, directional_interference_sum=0.,
                          approximate_interference_sum=0., directional_service_bits=0.,
                          approximate_service_bits=0., slots=0)
        if diagnostics is not None:
            diagnostics.append(self.stats)

    def update(self, args, **kw):
        slot = kw['slot_idx']
        bs = np.array([kw['connection_dict'][v] for v in self.ids])
        np.testing.assert_array_equal(bs, self.bs)
        values = np.array([kw['RA_dict'][v] for v in self.ids], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError('RB allocations must be finite nonnegative integers')
        k = values.astype(int)
        cap = args.num_RB_micro
        owners = np.full((4, cap), -1, dtype=int)
        for j in range(1, 5):
            users = np.flatnonzero(bs == j)
            occupants = np.repeat(users, k[users])
            if len(occupants) > cap:
                raise ValueError('Micro-BS capacity exceeded')
            owners[j-1, self.rng.permutation(cap)[:len(occupants)]] = occupants
        if k[bs == 0].sum() > args.num_RB_macro:
            raise ValueError('Macro-BS capacity exceeded')
        interference = np.zeros((len(bs), cap))
        for j, row in enumerate(owners, 1):
            interference += args.p_micro * self.cross[slot][:, np.maximum(row, 0)] * (row[None, :] >= 0) * (bs[:, None] != j)
        assigned = owners[np.maximum(bs-1, 0)] == np.arange(len(bs))[:, None]
        assigned[bs == 0] = False
        np.testing.assert_array_equal(assigned.sum(1), np.where(bs > 0, k, 0))
        signal = args.p_micro * self.own[slot]
        noise = args.N0 * args.RB_intervel_micro * 10**(args.NF_micro_dB/10)
        efficiency = np.array([1-min(float(kw['num_pilot_dict'][v][b-1])*args.pilot_overhead_factor, 1)
                               if b else 1 for v, b in zip(self.ids, bs)])
        factor = efficiency * args.RB_intervel_micro * args.slot_len
        service = factor * (np.log2(1+signal[:, None]/(noise+interference))*assigned).sum(1)
        for i in np.flatnonzero(bs == 0):
            g = 10**(float(kw['g_dict'][self.ids[i]][0])/10)
            service[i] = k[i]*args.slot_len*args.RB_intervel_macro*np.log2(
                1+args.p_macro*g/(args.N0*args.RB_intervel_macro*10**(args.NF_macro_dB/10)))
        if not np.isfinite(service).all() or (service < 0).any():
            raise ValueError('Invalid physical service')
        queue = kw['backlog_queue_dict']
        for i, v in enumerate(self.ids):
            queue[v][slot+1] = max(queue[v][slot]-service[i], 0) + kw['a_dict'][v][slot]
        active = (bs > 0) & (k > 0)
        mask = (bs[:, None] != bs[None, :]) & (bs[None, :] > 0) & (bs[:, None] > 0)
        avg = args.p_micro/cap * ((self.mean[slot][:, np.maximum(bs-1, 0)]*mask) @ k)
        approx = factor*k*np.log2(1+signal/(noise+avg))
        self.stats['rb_weight'] += int(k[active].sum())
        self.stats['directional_interference_sum'] += float((interference*assigned).sum())
        self.stats['approximate_interference_sum'] += float((avg*k)[active].sum())
        self.stats['directional_service_bits'] += float(service[active].sum())
        self.stats['approximate_service_bits'] += float(approx[active].sum())
        self.stats['slots'] += 1
        return queue
