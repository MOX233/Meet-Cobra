"""Capacity-bounded occupancy feedback for the report-only MTS controller.

RB demand may exceed capacity. The occupied-RB estimate used to compute
interference may not. The original unbounded helper remains untouched.
"""
import numpy as np
from utils.pql_ba import macro_gain_db


def estimate_bounded_report_load(args,reports,connection,rates,config,bs_locations,
                                 last_measurements,macro_loc=(0.,0.)):
    ids=list(reports)
    caps=np.array([args.num_RB_macro]+[args.num_RB_micro]*config.num_micro_bs,dtype=float)
    if not ids:
        return np.zeros_like(caps),np.zeros_like(caps)
    desired,interference=[],[]
    for v in ids:
        r=reports[v]
        macro=macro_gain_db(args,r.position,np.asarray(macro_loc))
        gains=np.r_[macro,r.gain]
        if v in last_measurements:
            bs,gain=last_measurements[v]
            if bs==connection[v] and bs>0:
                gains[bs]=gain
        desired.append(gains)
        interference.append(np.r_[0.,10.**(r.interference/10.)])
    desired=10.**(np.array(desired)/10.)
    interference=np.array(interference)
    bandwidth=np.array([args.RB_intervel_macro]+[args.RB_intervel_micro]*config.num_micro_bs)
    power=np.array([args.p_macro]+[args.p_micro]*config.num_micro_bs)
    nf=np.array([args.NF_macro_dB]+[args.NF_micro_dB]*config.num_micro_bs)
    noise=args.N0*bandwidth*10.**(nf/10.)
    association=np.array([connection[v] for v in ids])
    arrival=np.array([rates[v] for v in ids])
    occupied=caps.copy()
    for _ in range(10):
        from_each_bs=interference*(occupied/caps*power)[None,:]
        total=from_each_bs.sum(1,keepdims=True)-from_each_bs
        total[:,0]=0.
        sinr=power[None,:]*desired/(noise[None,:]+np.maximum(total,0.))
        capacity=bandwidth[None,:]*np.log1p(sinr)/np.log(2.)
        required=arrival/(capacity[np.arange(len(ids)),association]+2e-10)
        demand=np.bincount(association,weights=required,minlength=config.num_bs)
        next_occupied=np.clip(demand,0.,caps)
        if np.allclose(occupied,next_occupied,atol=1):
            occupied=next_occupied
            break
        occupied=next_occupied
    assert np.isfinite(occupied).all() and (occupied<=caps).all()
    # Preserve the original matching overload cue (capped at 1.5), while RA
    # receives only physically possible occupied RBs, not unconstrained demand.
    return occupied,np.clip(demand/caps,0.,1.5)
