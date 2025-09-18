"""Command-line interface for noise_decomp.

Provides a tiny CLI to compute intrinsic/extrinsic noise from paired series.
Usage examples:
  noise-decomp --r 1,2,3 --g 1.1,2.1,3.1
  noise-decomp --rfile r.csv --gfile g.csv

The entry point is exposed as the console script `noise-decomp`.
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
from typing import List, Sequence

from . import noise_decomp as ndfunc  # package-level function


def _parse_list(s: str) -> List[float]:
    try:
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except Exception as exc:  # pragma: no cover - trivial parsing error
        raise argparse.ArgumentTypeError(f"Could not parse list: {exc}")


def _read_values_from_file(path: str) -> np.ndarray:
    # Try numpy loadtxt (CSV or whitespace-separated). Fall back to eval of string.
    try:
        arr = np.loadtxt(path, delimiter=",")
        return np.atleast_1d(arr).astype(float)
    except Exception:
        # last resort: read file and parse comma-separated
        txt = open(path, "r").read().strip()
        return np.asarray([float(x) for x in txt.replace("\n", ",").split(",") if x.strip() != ""])  # type: ignore


def _print_results(res: dict) -> None:
    # Print keys in a stable order
    keys = ["mu", "n", "eta_int", "eta_ext", "eta_tot", "eta_int_sq", "eta_ext_sq", "eta_tot_sq"]
    for k in keys:
        if k in res:
            print(f"{k}: {res[k]}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="noise-decomp", description="Compute intrinsic/extrinsic noise decomposition from paired measurements")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--r", help="comma-separated values for reporter R (e.g. 1,2,3)", type=_parse_list)
    grp.add_argument("--rfile", help="path to file with reporter R values (csv or whitespace separated)")
    grp2 = p.add_mutually_exclusive_group(required=True)
    grp2.add_argument("--g", help="comma-separated values for reporter G (e.g. 1.1,2.1,3.1)", type=_parse_list)
    grp2.add_argument("--gfile", help="path to file with reporter G values (csv or whitespace separated)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="do not normalize reporter means")
    p.add_argument("--ddof", type=int, default=0, help="delta degrees of freedom passed to variance/covariance computations (default: 0)")
    p.add_argument("--quiet", "-q", action="store_true", help="only print JSON result")
    args = p.parse_args(argv)

    try:
        if args.r is not None:
            r = np.asarray(args.r, dtype=float)
        else:
            r = _read_values_from_file(args.rfile)  # type: ignore[arg-type]

        if args.g is not None:
            g = np.asarray(args.g, dtype=float)
        else:
            g = _read_values_from_file(args.gfile)  # type: ignore[arg-type]
    except Exception as exc:
        p.error(f"Failed to read inputs: {exc}")

    try:
        res = ndfunc(r, g, normalize_means=args.normalize, ddof=args.ddof)
    except Exception as exc:
        p.error(f"Computation failed: {exc}")

    if args.quiet:
        # print compact JSON
        import json

        print(json.dumps(res))
    else:
        _print_results(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
