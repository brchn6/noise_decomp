import numpy as np
from noise_decomp import noise_decomp

def test_shapes():
    r = [1, 2, 3, 4]
    g = [1.1, 2.1, 3.1, 4.1]
    res = noise_decomp(r, g)
    assert "eta_int" in res and "eta_ext" in res and "eta_tot" in res
    assert res["n"] == 4

def test_additivity():
    rng = np.random.default_rng(0)
    n = 2000
    mu = 1000.0
    E = np.exp(rng.normal(0, 0.2, n))     # extrinsic shared
    Ir = np.exp(rng.normal(0, 0.1, n))    # intrinsic private
    Ig = np.exp(rng.normal(0, 0.1, n))
    r = mu * E * Ir
    g = mu * E * Ig
    res = noise_decomp(r, g)
    assert abs((res["eta_int_sq"] + res["eta_ext_sq"]) - res["eta_tot_sq"]) < 1e-2
