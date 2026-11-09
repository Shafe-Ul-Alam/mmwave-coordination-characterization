#!/usr/bin/env python3
"""
Build and record a leakage-free split from a config file.

    python scripts/make_splits.py --config configs/protocol_a_uid.yaml \
                                  --coords data/receiver_xyz.npy \
                                  --out splits/protocol_a

Writes, into --out:
    train_groups.npy / val_groups.npy / test_groups.npy   (n_groups, U) receiver IDs
    pools.npz                                             the three disjoint ID pools
    split_record.json                                     sizes, overlaps, separation
    audit.json                                            the leakage audit of what was built

The script refuses to finish if the audit is not clean, so a leaky split can
never be produced silently.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.splits import (SplitConfig, receiver_id_disjoint_split,      # noqa: E402
                        spatial_guard_split, make_groups, split_record)
from src.leakage_audit import audit, format_report                    # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--coords", required=True,
                   help="(n_receivers, 2 or 3) receiver positions in metres")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    with open(a.config) as f:
        raw = yaml.safe_load(f)

    coords = np.load(a.coords)
    n_receivers = len(coords)

    grp = raw.get("groups", {})
    spl = raw.get("split", {})
    cfg = SplitConfig(
        protocol=raw["protocol"],
        seed=int(raw.get("seed", 20240501)),
        train_frac=float(spl.get("train_frac", 0.70)),
        val_frac=float(spl.get("val_frac", 0.15)),
        users_per_group=int(grp.get("users_per_group", 181)),
        n_train_groups=int(grp.get("n_train_groups", 2000)),
        n_val_groups=int(grp.get("n_val_groups", 200)),
        n_test_groups=int(grp.get("n_test_groups", 300)),
        guard_m=float(spl.get("guard_m", 13.8)),
    )

    if cfg.protocol == "receiver_id_disjoint":
        pools = receiver_id_disjoint_split(n_receivers, cfg)
    elif cfg.protocol == "spatial_guard":
        pools = spatial_guard_split(coords, cfg)
    else:
        raise SystemExit(f"unknown protocol '{cfg.protocol}'")

    os.makedirs(a.out, exist_ok=True)
    groups = {}
    for name, n_groups, seed_off in (("train", cfg.n_train_groups, 1),
                                     ("val", cfg.n_val_groups, 2),
                                     ("test", cfg.n_test_groups, 3)):
        g = make_groups(pools[name], n_groups, cfg.users_per_group,
                        cfg.seed + seed_off)
        groups[name] = g
        np.save(os.path.join(a.out, f"{name}_groups.npy"), g)

    np.savez_compressed(os.path.join(a.out, "pools.npz"), **pools)

    rec = split_record(pools, cfg, coords)
    with open(os.path.join(a.out, "split_record.json"), "w") as f:
        json.dump(rec, f, indent=2)

    report = audit(groups["train"], groups["test"], groups["val"], coords)
    with open(os.path.join(a.out, "audit.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(format_report(report))
    print(f"\npools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))
    if "min_test_to_train_separation_m" in rec:
        print(f"min test-to-train separation: "
              f"{rec['min_test_to_train_separation_m']:.3f} m")
    print(f"written to {a.out}")

    if not report["verdict"]["clean"]:
        print("\nREFUSING to emit a leaky split.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
