"""
Leakage-free split protocols for DeepMIMO-based beam-selection studies.

This module implements the two protocols described in Section V-A of the paper:

  Protocol A  receiver-ID-disjoint
      Receiver identifiers are partitioned into three disjoint pools before any
      group is formed. A decision group is then drawn *within* one pool, so a
      receiver can never appear in more than one split.

  Protocol B  guard-banded coordinate-spatial holdout
      Receivers are projected onto the principal spatial axis of the receiver
      grid and cut into contiguous train / test / validation regions separated
      by physical guard bands, so that no test receiver is within `guard_m`
      metres of any training receiver.

Both protocols return integer index arrays into the receiver table, plus an
audit record that `leakage_audit.py` can verify independently.

Reference configuration reproduced by this module (paper Table II):
    Protocol A : 183,610 train / 39,345 val / 39,346 test receivers,
                 2000 / 200 / 300 groups of U = 181 users, zero overlap.
    Protocol B : minimum test-to-train separation 13.80 m.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np

__all__ = ["SplitConfig", "receiver_id_disjoint_split", "spatial_guard_split",
           "make_groups", "split_record"]


# ---------------------------------------------------------------------------
@dataclass
class SplitConfig:
    """Everything needed to reproduce a split exactly."""
    protocol: str                 # "receiver_id_disjoint" | "spatial_guard"
    seed: int = 20240501
    train_frac: float = 0.70
    val_frac: float = 0.15
    # group construction
    users_per_group: int = 181
    n_train_groups: int = 2000
    n_val_groups: int = 200
    n_test_groups: int = 300
    # protocol B only
    guard_m: float = 13.8

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
def receiver_id_disjoint_split(n_receivers: int, cfg: SplitConfig
                               ) -> Dict[str, np.ndarray]:
    """Protocol A. Partition receiver IDs into three disjoint pools.

    Parameters
    ----------
    n_receivers : total number of receiver positions in the scenario.
    cfg         : split configuration (uses seed, train_frac, val_frac).

    Returns
    -------
    dict with keys 'train', 'val', 'test' -> sorted int64 arrays of receiver IDs.
    """
    if not 0 < cfg.train_frac < 1 or not 0 < cfg.val_frac < 1:
        raise ValueError("train_frac and val_frac must lie in (0, 1)")
    if cfg.train_frac + cfg.val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test pool")

    rng = np.random.default_rng(cfg.seed)
    ids = rng.permutation(n_receivers)

    n_tr = int(round(cfg.train_frac * n_receivers))
    n_va = int(round(cfg.val_frac * n_receivers))

    pools = {
        "train": np.sort(ids[:n_tr]),
        "val": np.sort(ids[n_tr:n_tr + n_va]),
        "test": np.sort(ids[n_tr + n_va:]),
    }
    _assert_disjoint(pools)
    return pools


# ---------------------------------------------------------------------------
def spatial_guard_split(coords: np.ndarray, cfg: SplitConfig
                        ) -> Dict[str, np.ndarray]:
    """Protocol B. Contiguous spatial regions separated by physical guard bands.

    Parameters
    ----------
    coords : (n_receivers, 2 or 3) array of receiver positions in metres.
    cfg    : split configuration (uses train_frac, val_frac, guard_m).

    Returns
    -------
    dict with keys 'train', 'val', 'test', 'guard' -> sorted int64 arrays.

    The projection axis is the first principal component of the receiver cloud,
    which for a rectangular DeepMIMO grid is the long axis of the street.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be (n_receivers, >=2)")

    centred = coords - coords.mean(axis=0, keepdims=True)
    # principal axis via SVD (no covariance matrix, numerically safer)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    t = centred @ vt[0]                       # projection, metres

    order = np.argsort(t)
    t_sorted = t[order]
    n = len(t_sorted)

    # provisional cut positions along the axis
    cut_tr = t_sorted[int(round(cfg.train_frac * n)) - 1]
    cut_te = t_sorted[int(round((cfg.train_frac + (1 - cfg.train_frac
                                                   - cfg.val_frac)) * n)) - 1]

    g = cfg.guard_m
    train_m = t <= cut_tr - g / 2.0
    test_m = (t >= cut_tr + g / 2.0) & (t <= cut_te - g / 2.0)
    val_m = t >= cut_te + g / 2.0
    guard_m_ = ~(train_m | test_m | val_m)

    pools = {
        "train": np.sort(np.flatnonzero(train_m)),
        "val": np.sort(np.flatnonzero(val_m)),
        "test": np.sort(np.flatnonzero(test_m)),
        "guard": np.sort(np.flatnonzero(guard_m_)),
    }
    _assert_disjoint({k: v for k, v in pools.items() if k != "guard"})
    return pools


# ---------------------------------------------------------------------------
def make_groups(pool: np.ndarray, n_groups: int, users_per_group: int,
                seed: int) -> np.ndarray:
    """Draw spread decision groups *within* a single pool.

    Users are sampled without replacement inside a group; groups are sampled
    independently, mirroring the co-scheduled population of the paper. Because
    the pool is already disjoint from the other splits, no cross-split leakage
    can be introduced at this stage.

    Returns an (n_groups, users_per_group) int64 array of receiver IDs.
    """
    pool = np.asarray(pool)
    if users_per_group > len(pool):
        raise ValueError(f"pool of {len(pool)} receivers cannot fill a group "
                         f"of {users_per_group}")
    rng = np.random.default_rng(seed)
    out = np.empty((n_groups, users_per_group), dtype=np.int64)
    for g in range(n_groups):
        out[g] = rng.choice(pool, size=users_per_group, replace=False)
    return out


# ---------------------------------------------------------------------------
def split_record(pools: Dict[str, np.ndarray], cfg: SplitConfig,
                 coords: np.ndarray | None = None) -> dict:
    """Build the audit record that accompanies a released split."""
    rec = {
        "config": asdict(cfg),
        "pool_sizes": {k: int(len(v)) for k, v in pools.items()},
        "pairwise_overlap": {},
    }
    keys = [k for k in ("train", "val", "test") if k in pools]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            rec["pairwise_overlap"][f"{a}-{b}"] = int(
                np.intersect1d(pools[a], pools[b]).size)
    if coords is not None and "test" in pools and "train" in pools:
        rec["min_test_to_train_separation_m"] = float(
            _min_cross_distance(coords[pools["test"]], coords[pools["train"]]))
    return rec


# ---------------------------------------------------------------------------
def _assert_disjoint(pools: Dict[str, np.ndarray]) -> None:
    keys = list(pools)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            n = np.intersect1d(pools[a], pools[b]).size
            if n:
                raise AssertionError(f"{a}/{b} share {n} receiver IDs")


def _min_cross_distance(a: np.ndarray, b: np.ndarray,
                        block: int = 2048) -> float:
    """Minimum Euclidean distance between two point sets, blocked for memory."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    best = np.inf
    for i in range(0, len(a), block):
        chunk = a[i:i + block]
        d = np.sqrt(((chunk[:, None, :] - b[None, :, :]) ** 2).sum(-1))
        best = min(best, float(d.min()))
    return best
