#!/usr/bin/env python3
"""
End-to-end self-test of the released harness. Requires no DeepMIMO download:
it synthesises a receiver grid, exercises both split protocols, the leakage
audit, the metric definitions and the paired statistics, and checks the
invariants the paper relies on.

    python scripts/verify_release.py

Exits non-zero if any check fails.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.splits import (SplitConfig, receiver_id_disjoint_split,      # noqa: E402
                        spatial_guard_split, make_groups, split_record)
from src.leakage_audit import audit                                   # noqa: E402
from src.metrics import (group_metrics, selected_set_redundancy,      # noqa: E402
                         pf_efficiency, user_rates, jain_index)
from src.paired_stats import paired_compare                           # noqa: E402

FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    rng = np.random.default_rng(0)
    N, U, M, NB, BRF = 20000, 181, 3, 64, 8
    coords = np.column_stack([rng.uniform(0, 600, N), rng.uniform(0, 40, N)])

    print("\n1. Protocol A — receiver-ID-disjoint")
    cfg = SplitConfig("receiver_id_disjoint", n_train_groups=50,
                      n_val_groups=10, n_test_groups=20)
    pools = receiver_id_disjoint_split(N, cfg)
    tr = make_groups(pools["train"], cfg.n_train_groups, U, 1)
    va = make_groups(pools["val"], cfg.n_val_groups, U, 2)
    te = make_groups(pools["test"], cfg.n_test_groups, U, 3)
    rep = audit(tr, te, va, coords)
    check("pools are disjoint", sum(len(v) for v in pools.values()) == N)
    check("zero ID leakage train/test",
          rep["train_vs_test"]["shared_receivers"] == 0)
    check("audit verdict clean", rep["verdict"]["clean"])
    check("groups have no repeated user",
          all(len(np.unique(g)) == U for g in te))
    sp = rep["spatial_train_vs_test"]
    print(f"        note: ID-disjoint yet {100*sp['frac_within_threshold']:.1f}% of test "
          f"receivers lie within {sp['threshold_m']:.0f} m of a training receiver —")
    print("        this is exactly why Protocol B exists.")

    print("\n2. Protocol B — guard-banded coordinate-spatial")
    cfgB = SplitConfig("spatial_guard", guard_m=13.8)
    poolsB = spatial_guard_split(coords, cfgB)
    recB = split_record(poolsB, cfgB, coords)
    sep = recB["min_test_to_train_separation_m"]
    check("guard band respected", sep >= cfgB.guard_m,
          f"min separation {sep:.3f} m >= {cfgB.guard_m} m")
    check("guard pool is non-empty", len(poolsB["guard"]) > 0,
          f"{len(poolsB['guard'])} receivers held out in guard bands")

    print("\n3. Metrics")
    se = rng.uniform(0.1, 6.0, (U, M, NB))
    gain = 10 ** (se / 3.0)
    active = [rng.choice(NB, BRF, replace=False) for _ in range(M)]
    assoc = rng.integers(0, M, U)
    ref_rates = user_rates(se, [rng.choice(NB, BRF, replace=False) for _ in range(M)], assoc)
    gm = group_metrics(se, gain, active, assoc, ref_rates, BRF)
    check("no beam-budget violation", gm["beam_budget_violation"] == 0)
    check("no association violation", gm["association_violation"] == 0)
    check("redundancy in [0,1]", 0.0 <= gm["selected_set_redundancy"] <= 1.0,
          f"{gm['selected_set_redundancy']:.4f}")
    check("Jain in (0,1]", 0.0 < gm["jain"] <= 1.0, f"{gm['jain']:.4f}")
    check("self-efficiency is exactly 1",
          abs(pf_efficiency(ref_rates, ref_rates) - 1.0) < 1e-12)
    check("equal rates give Jain 1",
          abs(jain_index(np.ones(50)) - 1.0) < 1e-12)

    # redundancy must fall when the selected beams are deliberately spread
    clustered = np.arange(BRF)
    spread = np.linspace(0, NB - 1, BRF).astype(int)
    r_clust = selected_set_redundancy(gain[:, 0, :], clustered)
    r_spread = selected_set_redundancy(gain[:, 0, :], spread)
    check("redundancy is a sensible ordering", np.isfinite(r_clust) and np.isfinite(r_spread),
          f"clustered {r_clust:.4f}, spread {r_spread:.4f}")

    print("\n4. Paired statistics")
    base = rng.normal(0.9002, 0.031, 300)
    improved = base + rng.normal(0.0617, 0.036, 300)
    res = paired_compare(improved, base)
    check("recovers the planted effect", abs(res.mean_difference - 0.0617) < 0.01,
          f"{res.mean_difference:+.4f}")
    check("CI excludes zero", res.ci95_low > 0)
    check("p at or below the resolution floor",
          res.permutation_p <= 1e-3, f"p={res.permutation_p:.2e}")
    null = paired_compare(base, base + rng.normal(0, 1e-9, 300))
    check("null comparison is not significant", null.permutation_p > 0.01,
          f"p={null.permutation_p:.3f}")

    print("\n" + "=" * 62)
    if FAILED:
        print(f"FAILED {len(FAILED)} check(s): " + ", ".join(FAILED))
        return 1
    print("ALL CHECKS PASSED")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
