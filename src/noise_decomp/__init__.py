"""
noise_decomp: Tiny intrinsic/extrinsic noise decomposition 
(from dual-reporter gene expression measurements).

API
---
noise_decomp(r, g, normalize_means=True, ddof=0) -> dict
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, Iterable

ArrayLike = Iterable[float]

__all__ = ["noise_decomp"]

def _to_np(x: ArrayLike) -> np.ndarray:
    a = np.asarray(list(x), dtype=float).ravel()
    if a.size == 0:
        raise ValueError("Input array is empty.")
    return a

def noise_decomp(r: ArrayLike, g: ArrayLike, normalize_means: bool = True, ddof: int = 0) -> Dict[str, Any]:
    r = _to_np(r)
    g = _to_np(g)
    if r.size != g.size:
        raise ValueError("r and g must have the same length (paired measurements).")
    if normalize_means:
        mr, mg = r.mean(), g.mean()
        if mr == 0 or mg == 0:
            raise ValueError("Reporter means must be nonzero for normalization.")
        r = r * (mg / mr)
    mu = 0.5 * (r.mean() + g.mean())
    if mu == 0:
        raise ValueError("Common mean is zero; cannot compute noise.")
    V_int = 0.5 * np.mean((r - g) ** 2)
    V_tot = 0.5 * (np.var(r, ddof=ddof) + np.var(g, ddof=ddof))
    V_ext = np.cov(r, g, ddof=ddof)[0, 1]
    eta_int_sq = V_int / (mu ** 2)
    eta_ext_sq = V_ext / (mu ** 2)
    eta_tot_sq = V_tot / (mu ** 2)
    return {
        "eta_int": float(np.sqrt(max(eta_int_sq, 0.0))),
        "eta_ext": float(np.sqrt(max(eta_ext_sq, 0.0))),
        "eta_tot": float(np.sqrt(max(eta_tot_sq, 0.0))),
        "eta_int_sq": float(eta_int_sq),
        "eta_ext_sq": float(eta_ext_sq),
        "eta_tot_sq": float(eta_tot_sq),
        "mu": float(mu),
        "n": int(r.size),
    }
