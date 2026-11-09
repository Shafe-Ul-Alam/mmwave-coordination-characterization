"""
Leakage audit for DeepMIMO-style beam-selection splits.

This is the reusable benchmark contribution of the paper (contribution C2).
It answers one question about any split, however it was produced:

    What fraction of the receivers appearing in the evaluation groups were
    already seen during training?

In the original, conventional random-grouping split used throughout this line
of work, the answer was 45.30 %. Under both protocols released here it is 0.00 %.

Usage
-----
    python -m src.leakage_audit --train train_groups.npy \
                                --test  test_groups.npy \
                                [--val val_groups.npy] \
                                [--coords receiver_xyz.npy] \
                                [--json report.json]

`*_groups.npy` is an (n_groups, users_per_group) integer array of receiver IDs,
which is exactly what `splits.make_groups` produces. `--coords` additionally
enables the spatial check (minimum test-to-train separation), which catches the
subtler failure in which receiver IDs are disjoint but the positions are
effectively duplicates a few centimetres apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
def unique_receivers(groups: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(groups).ravel())


def id_leakage(train_groups: np.ndarray, eval_groups: np.ndarray) -> dict:
    """Fraction of unique evaluation receivers that also occur in training."""
    tr = unique_receivers(train_groups)
    ev = unique_receivers(eval_groups)
    shared = np.intersect1d(tr, ev)
    frac = float(shared.size) / float(ev.size) if ev.size else 0.0
    return {
        "unique_train_receivers": int(tr.size),
        "unique_eval_receivers": int(ev.size),
        "shared_receivers": int(shared.size),
        "leaked_fraction": frac,
        "leaked_percent": 100.0 * frac,
    }


def spatial_leakage(train_groups: np.ndarray, eval_groups: np.ndarray,
                    coords: np.ndarray, threshold_m: float = 1.0) -> dict:
    """Minimum and quantile distances from evaluation receivers to training ones."""
    coords = np.asarray(coords, dtype=np.float64)
    tr = coords[unique_receivers(train_groups)]
    ev = coords[unique_receivers(eval_groups)]

    nearest = np.empty(len(ev), dtype=np.float64)
    block = 1024
    for i in range(0, len(ev), block):
        chunk = ev[i:i + block]
        d = np.sqrt(((chunk[:, None, :] - tr[None, :, :]) ** 2).sum(-1))
        nearest[i:i + block] = d.min(axis=1)

    return {
        "min_separation_m": float(nearest.min()),
        "p05_separation_m": float(np.percentile(nearest, 5)),
        "median_separation_m": float(np.median(nearest)),
        "frac_within_threshold": float((nearest < threshold_m).mean()),
        "threshold_m": float(threshold_m),
    }


# ---------------------------------------------------------------------------
def audit(train_groups: np.ndarray,
          test_groups: np.ndarray,
          val_groups: Optional[np.ndarray] = None,
          coords: Optional[np.ndarray] = None,
          threshold_m: float = 1.0) -> Dict[str, dict]:
    report: Dict[str, dict] = {
        "train_vs_test": id_leakage(train_groups, test_groups),
    }
    if val_groups is not None:
        report["train_vs_val"] = id_leakage(train_groups, val_groups)
        report["val_vs_test"] = id_leakage(val_groups, test_groups)
    if coords is not None:
        report["spatial_train_vs_test"] = spatial_leakage(
            train_groups, test_groups, coords, threshold_m)
    report["verdict"] = {
        "clean": all(v["shared_receivers"] == 0
                     for k, v in report.items() if k.startswith(("train_vs", "val_vs")))
    }
    return report


def format_report(report: Dict[str, dict]) -> str:
    lines = ["=" * 62, "LEAKAGE AUDIT", "=" * 62]
    for key in ("train_vs_test", "train_vs_val", "val_vs_test"):
        if key not in report:
            continue
        r = report[key]
        lines.append(
            f"{key:<16s} shared {r['shared_receivers']:>8d} / "
            f"{r['unique_eval_receivers']:<8d} receivers   "
            f"= {r['leaked_percent']:6.2f} %")
    if "spatial_train_vs_test" in report:
        s = report["spatial_train_vs_test"]
        lines += [
            "-" * 62,
            f"min test-to-train separation   {s['min_separation_m']:.3f} m",
            f"5th percentile separation      {s['p05_separation_m']:.3f} m",
            f"median separation              {s['median_separation_m']:.3f} m",
            f"fraction within {s['threshold_m']:.2f} m        "
            f"{100*s['frac_within_threshold']:.2f} %",
        ]
    lines.append("=" * 62)
    lines.append("VERDICT: " + ("CLEAN — no receiver appears in two splits"
                                if report["verdict"]["clean"]
                                else "LEAKAGE DETECTED — do not report results from this split"))
    lines.append("=" * 62)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", required=True, help="training groups .npy")
    p.add_argument("--test", required=True, help="test groups .npy")
    p.add_argument("--val", default=None, help="validation groups .npy")
    p.add_argument("--coords", default=None,
                   help="receiver coordinates .npy, (n_receivers, 2 or 3)")
    p.add_argument("--threshold-m", type=float, default=1.0,
                   help="distance below which two receivers count as duplicates")
    p.add_argument("--json", default=None, help="write the full report here")
    a = p.parse_args(argv)

    tr = np.load(a.train)
    te = np.load(a.test)
    va = np.load(a.val) if a.val else None
    co = np.load(a.coords) if a.coords else None

    report = audit(tr, te, va, co, a.threshold_m)
    print(format_report(report))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nfull report written to {a.json}")
    return 0 if report["verdict"]["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
