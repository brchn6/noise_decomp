# distutils: language = c
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def noise_decomp_c(np.ndarray[np.float64_t, ndim=1] r,
                  np.ndarray[np.float64_t, ndim=1] g,
                  int ddof=1,
                  bint clip_nonneg=False):
    cdef Py_ssize_t n = r.shape[0]
    if n != g.shape[0]:
        raise ValueError("r and g must have the same length.")
    if n - ddof <= 0:
        raise ValueError("Need n > ddof (e.g., ddof=1 requires n>=2).")
    cdef double mr = np.mean(r)
    cdef double mg = np.mean(g)
    cdef double mu = 0.5 * (mr + mg)
    if mu == 0:
        raise ValueError("Common mean is zero; cannot compute noise.")
    cdef np.ndarray[np.float64_t, ndim=1] diff = r - g
    cdef double V_int = 0.5 * np.var(diff, ddof=ddof)
    cdef double V_tot = 0.5 * (np.var(r, ddof=ddof) + np.var(g, ddof=ddof))
    cdef double V_ext = np.cov(r, g, ddof=ddof)[0, 1]
    cdef double eta_int_sq = V_int / (mu * mu)
    cdef double eta_ext_sq = V_ext / (mu * mu)
    cdef double eta_tot_sq = V_tot / (mu * mu)
    if clip_nonneg:
        eta_int_sq = max(eta_int_sq, 0.0)
        eta_ext_sq = max(eta_ext_sq, 0.0)
        eta_tot_sq = max(eta_tot_sq, 0.0)
    def _safe_sqrt(double x):
        return sqrt(x) if x >= 0 else float('nan')
    return {
        "eta_int": _safe_sqrt(eta_int_sq),
        "eta_ext": _safe_sqrt(eta_ext_sq),
        "eta_tot": _safe_sqrt(eta_tot_sq),
        "eta_int_sq": eta_int_sq,
        "eta_ext_sq": eta_ext_sq,
        "eta_tot_sq": eta_tot_sq,
        "additivity_gap": eta_tot_sq - (eta_int_sq + eta_ext_sq),
        "mu": mu,
        "n": n,
        "ddof": ddof,
        "clipped": bool(clip_nonneg),
    }
