"""
noise_decomp: Tiny intrinsic/extrinsic noise decomposition 
(from dual-reporter gene expression measurements).

API
---
noise_decomp(r, g, normalize_means=True, ddof=0) -> dict
"""
from __future__ import annotations
import numpy as np
from typing import Iterable, Dict, Any

ArrayLike = Iterable[float]
__all__ = ["noise_decomp"]

try:
    from ._cnoise import noise_decomp_c as _noise_decomp_fast
    _HAVE_CYTHON = True
except ImportError:
    _HAVE_CYTHON = False

def _to_np(x: ArrayLike) -> np.ndarray:
    a = np.asarray(list(x), dtype=float).ravel()
    if a.size == 0:
        raise ValueError("Input array is empty.")
    return a

def noise_decomp(
    r: ArrayLike,
    g: ArrayLike,
    normalize: str = "match_means",   # "none" | "match_means" | "ols"
    ddof: int = 1,
    clip_nonneg: bool = False,
) -> Dict[str, Any]:
    """
    Dual-reporter noise decomposition (Elowitz-style).
    Returns CV, CV^2 for intrinsic/extrinsic/total.

    normalize:
      - "none": use inputs as-is
      - "match_means": scale r by (mean_g / mean_r)
      - "ols": scale r by a = cov(r,g)/var(r) (gain correction)
    """
    r = _to_np(r); g = _to_np(g)
    if r.size != g.size:
        raise ValueError("r and g must have the same length.")
    n = r.size
    if n - ddof <= 0:
        raise ValueError("Need n > ddof (e.g., ddof=1 requires n>=2).")

    mr, mg = r.mean(), g.mean()
    if normalize == "match_means":
        if mr == 0 or mg == 0:
            raise ValueError("Reporter means must be nonzero for normalization.")
        r = r * (mg / mr)
    elif normalize == "ols":
        vr = r.var(ddof=ddof)
        cxy = np.cov(r, g, ddof=ddof)[0, 1]
        if vr == 0:
            raise ValueError("Variance of r is zero; cannot OLS-rescale.")
        a = cxy / vr
        r = a * r

    if _HAVE_CYTHON:
        return _noise_decomp_fast(r, g, ddof=ddof, clip_nonneg=clip_nonneg)

    mu = 0.5 * (r.mean() + g.mean())
    if mu == 0:
        raise ValueError("Common mean is zero; cannot compute noise.")

    # intrinsic: 0.5 * Var(r - g)
    V_int = 0.5 * np.var(r - g, ddof=ddof)
    # total: average of per-reporter variances
    V_tot = 0.5 * (np.var(r, ddof=ddof) + np.var(g, ddof=ddof))
    # extrinsic: covariance
    V_ext = np.cov(r, g, ddof=ddof)[0, 1]

    eta_int_sq = V_int / (mu ** 2)
    eta_ext_sq = V_ext / (mu ** 2)
    eta_tot_sq = V_tot / (mu ** 2)

    if clip_nonneg:
        eta_int_sq = max(eta_int_sq, 0.0)
        eta_ext_sq = max(eta_ext_sq, 0.0)
        eta_tot_sq = max(eta_tot_sq, 0.0)

    def _safe_sqrt(x):  # show NaNs if negative and not clipping
        return float(np.sqrt(x)) if x >= 0 else float("nan")

    return {
        "eta_int": _safe_sqrt(eta_int_sq),
        "eta_ext": _safe_sqrt(eta_ext_sq),
        "eta_tot": _safe_sqrt(eta_tot_sq),
        "eta_int_sq": float(eta_int_sq),
        "eta_ext_sq": float(eta_ext_sq),
        "eta_tot_sq": float(eta_tot_sq),
        "additivity_gap": float(eta_tot_sq - (eta_int_sq + eta_ext_sq)),
        "mu": float(mu),
        "n": int(n),
        "normalize": normalize,
        "ddof": ddof,
        "clipped": bool(clip_nonneg),
    }
