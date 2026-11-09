#!/usr/bin/env python3
"""Verify that the ported model and teacher code is faithful to the paper.

The models were extracted from the research notebook into ``src/models`` and
``src/expert``. The strongest cheap check that the extraction is faithful is
the parameter count: the paper reports exact counts, and any dropped layer,
changed width or missing head would move them. This script asserts them, then
exercises a forward pass and the teacher on a small synthetic instance.

Requires PyTorch. Run:  python scripts/verify_models.py
Exits non-zero on any failure, so it can be used in CI.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
except ImportError:
    print("PyTorch is required for this script. "
          "The split/metric/statistics checks in verify_release.py do not "
          "need it and can be run on their own.")
    sys.exit(2)

from src.models import (JointDecGATv14Batch, JointDecGATv14Ablation,
                        MatchedPooledMLP, MatchedCNN2D, EXPECTED_U, M, NB)
from src.expert import (spectral_efficiency, naive_top_np, swap_refine_fast,
                        score_geometric_mean, B_RF)

# counts exactly as reported in the paper (Table V and Section V-A)
EXPECTED_PARAMETERS = {
    "JointDecGATv14Batch": 221_125,
    "JointDecGATv14Ablation": 221_125,   # width-preserving by construction
    "MatchedPooledMLP": 219_859,
    "MatchedCNN2D": 222_166,
}

failures = []


def check(ok, label, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
          + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


print("=" * 62)
print("MODEL AND TEACHER VERIFICATION")
print("=" * 62)

print("\n1. Parameter counts match the paper")
for cls in (JointDecGATv14Batch, JointDecGATv14Ablation,
            MatchedPooledMLP, MatchedCNN2D):
    name = cls.__name__
    n = sum(p.numel() for p in cls().parameters())
    exp = EXPECTED_PARAMETERS[name]
    check(n == exp, f"{name}", f"{n:,} (expected {exp:,})")

print("\n2. Capacity matching claimed in the paper")
counts = {k: sum(p.numel() for p in c().parameters())
          for k, c in [("gnn", JointDecGATv14Batch), ("mlp", MatchedPooledMLP),
                       ("cnn", MatchedCNN2D)]}
spread = max(counts.values()) / min(counts.values()) - 1.0
check(spread <= 0.01, "all learned models within 1% of each other",
      f"spread {100 * spread:.2f}%")

print("\n3. Forward pass produces the documented shapes")
torch.manual_seed(0)
x = torch.randn(2, EXPECTED_U, M, NB)
for cls in (JointDecGATv14Batch, MatchedPooledMLP, MatchedCNN2D):
    model = cls().eval()
    with torch.no_grad():
        beam, assoc = model(x)
    ok = (tuple(beam.shape) == (2, M, NB)
          and tuple(assoc.shape) == (2, EXPECTED_U, M))
    check(ok, f"{cls.__name__} output shapes",
          f"beam {tuple(beam.shape)}, assoc {tuple(assoc.shape)}")

print("\n4. The GNN is permutation invariant (Proposition 5)")
model = JointDecGATv14Batch().eval()
perm = torch.randperm(EXPECTED_U)
with torch.no_grad():
    b1, a1 = model(x)
    b2, a2 = model(x[:, perm])
beam_dev = (b1 - b2).abs().max().item()
assoc_dev = (a1[:, perm] - a2).abs().max().item()
check(beam_dev < 1e-4, "beam scores invariant under user permutation",
      f"max deviation {beam_dev:.2e}")
check(assoc_dev < 1e-4, "association logits equivariant",
      f"max deviation {assoc_dev:.2e}")

print("\n5. The teacher runs and improves on its initialisation")
rng = np.random.default_rng(0)
gain = 10.0 ** rng.normal(-14.5, 1.2, size=(40, M, NB))
se = spectral_efficiency(gain)
init = naive_top_np(se, budget=B_RF)
refined = swap_refine_fast(se, init)
s_init = score_geometric_mean(se, init)
s_ref = score_geometric_mean(se, refined)
check(all(len(a) == B_RF for a in refined), "beam budget respected",
      f"|A_m| = {[len(a) for a in refined]}")
check(s_ref >= s_init - 1e-12, "swap refinement never worsens the objective",
      f"{s_init:.6e} -> {s_ref:.6e}")

print("\n" + "=" * 62)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    sys.exit(1)
print("ALL MODEL CHECKS PASSED")
print("=" * 62)
