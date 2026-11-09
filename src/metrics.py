"""
Objective and evaluation metrics used in the paper.

Every quantity reported in Tables III-VI of the manuscript is computed by one of
the functions below, so that a third party can recompute them from raw per-group
outputs without reimplementing the definitions.

Conventions
-----------
* `se[u, m, b]` is the spectral efficiency of user u on beam b of BS m.
* `active[m]` is the length-B_RF index array of beams BS m has switched on.
* `assoc[u]`  is the index of the BS serving user u.
* PF efficiency is a *ratio to a reference solution on the same group*; it is
  therefore scale invariant and is the headline metric. Rate-valued metrics
  (sum rate, 5th-percentile rate, Jain) are reported on the uncalibrated SNR
  scale, as disclosed in the paper's limitations.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = ["effective_se", "user_rates", "pf_utility", "geometric_mean_rate",
           "pf_efficiency", "jain_index", "selected_set_redundancy",
           "coverage_overlap", "group_metrics"]

_EPS = 1e-300


# ---------------------------------------------------------------------------
def effective_se(se: np.ndarray, active: Sequence[np.ndarray],
                 assoc: np.ndarray) -> np.ndarray:
    """e_u = max over the beams the serving BS actually has on."""
    se = np.asarray(se)
    assoc = np.asarray(assoc)
    out = np.empty(se.shape[0], dtype=np.float64)
    for m, beams in enumerate(active):
        sel = assoc == m
        if sel.any():
            out[sel] = se[sel][:, m, :][:, np.asarray(beams)].max(axis=1)
    return out


def user_rates(se: np.ndarray, active: Sequence[np.ndarray],
               assoc: np.ndarray) -> np.ndarray:
    """r_u = e_u / L_{a(u)} under equal intra-cell time sharing."""
    e = effective_se(se, active, assoc)
    assoc = np.asarray(assoc)
    loads = np.bincount(assoc, minlength=len(active)).astype(np.float64)
    loads[loads == 0] = 1.0                      # empty cell serves nobody
    return e / loads[assoc]


def pf_utility(rates: np.ndarray) -> float:
    """J_PF = sum_u log r_u  (Proposition 1)."""
    return float(np.log(np.maximum(np.asarray(rates), _EPS)).sum())


def geometric_mean_rate(rates: np.ndarray) -> float:
    r = np.maximum(np.asarray(rates, dtype=np.float64), _EPS)
    return float(np.exp(np.log(r).mean()))


def pf_efficiency(rates: np.ndarray, reference_rates: np.ndarray) -> float:
    """Headline metric: GM(method) / GM(reference) on the *same* group.

    Values above 1 are possible: the reference is a strong heuristic search,
    not a certified optimum.
    """
    return geometric_mean_rate(rates) / geometric_mean_rate(reference_rates)


def jain_index(rates: np.ndarray) -> float:
    r = np.asarray(rates, dtype=np.float64)
    s = r.sum()
    return float(s * s / (len(r) * np.square(r).sum())) if s > 0 else 0.0


# ---------------------------------------------------------------------------
def coverage_overlap(gain: np.ndarray) -> np.ndarray:
    """Pairwise coverage-overlap matrix O for one BS.

    `gain` is (n_users, n_beams) linear beamforming gain. Each beam's coverage
    profile is its gain vector normalised by each user's best available beam;
    O is the cosine similarity between profiles. This is the quantity the
    overlap-aware attention subtracts, and it is computed from one BS's own
    measurements only.
    """
    g = np.asarray(gain, dtype=np.float64)
    best = g.max(axis=1, keepdims=True)
    best[best <= 0] = 1.0
    c = g / best                                     # (n_users, n_beams)
    norms = np.linalg.norm(c, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    cn = c / norms
    return cn.T @ cn


def selected_set_redundancy(gain: np.ndarray, active: np.ndarray) -> float:
    """Mean off-diagonal coverage overlap within the selected beam subset.

    Lower is better: it means the active beams cover distinct user populations.
    """
    active = np.asarray(active)
    k = len(active)
    if k < 2:
        return 0.0
    O = coverage_overlap(gain)[np.ix_(active, active)]
    off = (O.sum() - np.trace(O)) / (k * (k - 1))
    return float(off)


# ---------------------------------------------------------------------------
def group_metrics(se: np.ndarray, gain: np.ndarray,
                  active: Sequence[np.ndarray], assoc: np.ndarray,
                  reference_rates: np.ndarray, b_rf: int) -> Dict[str, float]:
    """All per-group quantities reported in the paper, for one decision group."""
    rates = user_rates(se, active, assoc)
    loads = np.bincount(np.asarray(assoc), minlength=len(active))
    red = float(np.mean([selected_set_redundancy(gain[:, m, :], active[m])
                         for m in range(len(active))]))
    return {
        "pf_utility": pf_utility(rates),
        "gm_rate": geometric_mean_rate(rates),
        "pf_efficiency_vs_search": pf_efficiency(rates, reference_rates),
        "sum_rate": float(rates.sum()),
        "mean_rate": float(rates.mean()),
        "p5_rate": float(np.percentile(rates, 5)),
        "jain": jain_index(rates),
        "load_std": float(loads.std()),
        "selected_set_redundancy": red,
        # feasibility: these must be exactly zero in every reported result
        "beam_budget_violation": int(sum(len(np.unique(a)) != b_rf for a in active)),
        "association_violation": int(((np.asarray(assoc) < 0) |
                                      (np.asarray(assoc) >= len(active))).sum()),
    }
