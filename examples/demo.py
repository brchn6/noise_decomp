"""Small demo script for noise_decomp.

Run:
  python examples/demo.py
"""
from __future__ import annotations
import numpy as np
from noise_decomp import noise_decomp


def make_synthetic(n=1000, mu=1000.0, sigma_e=0.2, sigma_i=0.1, seed=0):
    rng = np.random.default_rng(seed)
    E = np.exp(rng.normal(0, sigma_e, n))
    Ir = np.exp(rng.normal(0, sigma_i, n))
    Ig = np.exp(rng.normal(0, sigma_i, n))
    r = mu * E * Ir
    g = mu * E * Ig
    return r, g


if __name__ == "__main__":
    r, g = make_synthetic()
    res = noise_decomp(r, g)
    print("Demo results:")
    for k, v in res.items():
        print(f"{k}: {v}")
