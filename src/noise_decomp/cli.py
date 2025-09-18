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
    try:
        # whitespace-separated (default) first
        arr = np.loadtxt(path)
        return np.atleast_1d(arr).astype(float)
    except Exception:
        pass
    try:
        # then try CSV explicitly
        arr = np.loadtxt(path, delimiter=",")
        return np.atleast_1d(arr).astype(float)
    except Exception:
        # last resort: tolerant parser
        txt = open(path, "r").read().strip()
        tokens = txt.replace("\n", " ").replace(",", " ").split()
        return np.asarray([float(x) for x in tokens])


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
    p.add_argument("--ddof", type=int, default=1, help="delta degrees of freedom passed to variance/covariance computations (default: 1)")
    p.add_argument("--quiet", "-q", action="store_true", help="only print JSON result")
    p.add_argument("--norm", choices=["match_means","ols","none"], help="normalization mode (overrides --no-normalize)")
    p.add_argument("--clip-nonneg", action="store_true", help="clip negative η² to 0 in the output")
    args = p.parse_args(argv)

    try:
        if args.r is not None:
            r = np.asarray(args.r, dtype=float)
        elif args.rfile:
            r = _read_values_from_file(args.rfile)
        else:
            raise ValueError("Must provide --r or --rfile")
        if args.g is not None:
            g = np.asarray(args.g, dtype=float)
        elif args.gfile:
            g = _read_values_from_file(args.gfile)
        else:
            raise ValueError("Must provide --g or --gfile")
        if args.norm:
            norm_mode = args.norm
        else:
            norm_mode = "none" if (hasattr(args, 'normalize') and args.normalize is False) else "match_means"
        res = ndfunc(r, g, normalize=norm_mode, ddof=args.ddof, clip_nonneg=args.clip_nonneg)
        if not args.quiet:
            _print_results(res)
            if "additivity_gap" in res:
                print(f"additivity_gap: {res['additivity_gap']}")
        else:
            import json
            print(json.dumps(res))
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
