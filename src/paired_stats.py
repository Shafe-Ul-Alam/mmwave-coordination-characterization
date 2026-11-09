"""
Paired group-level statistics, exactly as reported in the paper.

Every comparison in the manuscript is *paired*: both methods are evaluated on
the same decision groups, so each group contributes one difference and the
group-to-group difficulty cancels. This module implements the four quantities
reported for every comparison:

    mean difference, bootstrap 95 % CI, sign-flip permutation p, Cohen's d_z,
    and the per-group win rate.

The permutation test has a resolution floor of 1/(n_resamples+1); with the
default 20,000 resamples that floor is 4.99975e-05, which is why the paper
writes p < 5e-5 rather than an exact value.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

__all__ = ["PairedResult", "paired_compare"]


@dataclass
class PairedResult:
    n: int
    mean_difference: float
    ci95_low: float
    ci95_high: float
    permutation_p: float
    permutation_floor: float
    cohens_dz: float
    win_rate: float

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        p = (f"< {self.permutation_floor:.3g}"
             if self.permutation_p <= self.permutation_floor
             else f"= {self.permutation_p:.3g}")
        return (f"n={self.n}  mean {self.mean_difference:+.6f}  "
                f"95% CI [{self.ci95_low:+.6f}, {self.ci95_high:+.6f}]  "
                f"p {p}  d_z={self.cohens_dz:+.3f}  win={100*self.win_rate:.1f}%")


def paired_compare(a: np.ndarray, b: np.ndarray, *,
                   n_boot: int = 20000, n_perm: int = 20000,
                   seed: int = 12345,
                   alternative: str = "two-sided") -> PairedResult:
    """Compare method `a` against method `b` on the same groups.

    Positive `mean_difference` favours `a`.

    Parameters
    ----------
    a, b : (n_groups,) arrays of the same per-group metric.
    n_boot, n_perm : resample counts for the CI and the permutation test.
    seed : RNG seed; the released results use 12345.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of equal length")
    d = a - b
    n = d.size
    rng = np.random.default_rng(seed)

    # --- bootstrap CI over groups -----------------------------------------
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    # --- sign-flip permutation test ---------------------------------------
    obs = abs(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (signs * d).mean(axis=1)
    if alternative == "two-sided":
        hits = int((np.abs(perm_means) >= obs).sum())
    elif alternative == "greater":
        hits = int((perm_means >= d.mean()).sum())
    else:
        raise ValueError("alternative must be 'two-sided' or 'greater'")
    p = (hits + 1.0) / (n_perm + 1.0)

    # --- effect size and consistency --------------------------------------
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else 0.0
    win = float((d > 0).mean())

    return PairedResult(
        n=n,
        mean_difference=float(d.mean()),
        ci95_low=float(lo), ci95_high=float(hi),
        permutation_p=float(p),
        permutation_floor=1.0 / (n_perm + 1.0),
        cohens_dz=dz,
        win_rate=win,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":  # tiny self-test
    rng = np.random.default_rng(0)
    base = rng.normal(0.90, 0.03, size=300)
    improved = base + rng.normal(0.0617, 0.036, size=300)
    print("synthetic check:", paired_compare(improved, base))
