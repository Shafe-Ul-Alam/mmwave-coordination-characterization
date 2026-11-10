# Verification record

Both self-tests were run on a clean clone of this repository in Google Colab
(CPU runtime, PyTorch pre-installed). Reproduce with:

```bash
git clone https://github.com/Shafe-Ul-Alam/mmwave-coordination-characterization.git
cd mmwave-coordination-characterization
python scripts/verify_release.py
python scripts/verify_models.py
```

---

## 1. `verify_release.py` — splits, metrics, statistics

```
1. Protocol A — receiver-ID-disjoint
  [PASS] pools are disjoint
  [PASS] zero ID leakage train/test
  [PASS] audit verdict clean
  [PASS] groups have no repeated user
        note: ID-disjoint yet 58.4% of test receivers lie within 1 m of a
        training receiver — this is exactly why Protocol B exists.

2. Protocol B — guard-banded coordinate-spatial
  [PASS] guard band respected  — min separation 14.032 m >= 13.8 m
  [PASS] guard pool is non-empty  — 857 receivers held out in guard bands

3. Metrics
  [PASS] no beam-budget violation
  [PASS] no association violation
  [PASS] redundancy in [0,1]  — 0.4411
  [PASS] Jain in (0,1]  — 0.9809
  [PASS] self-efficiency is exactly 1
  [PASS] equal rates give Jain 1
  [PASS] redundancy is a sensible ordering  — clustered 0.4494, spread 0.4313

4. Paired statistics
  [PASS] recovers the planted effect  — +0.0600
  [PASS] CI excludes zero
  [PASS] p at or below the resolution floor  — p=5.00e-05
  [PASS] null comparison is not significant  — p=0.277

ALL CHECKS PASSED
```

The note under Protocol A is worth reading rather than skipping: receiver-ID
disjointness alone leaves $58.4\%$ of held-out receivers within one metre of a
training receiver. That is the motivation for Protocol B, and it is why the
paper reports both.

---

## 2. `verify_models.py` — architectures and teacher

```
1. Parameter counts match the paper
  [PASS] JointDecGATv14Batch     — 221,125 (expected 221,125)
  [PASS] JointDecGATv14Ablation  — 221,125 (expected 221,125)
  [PASS] MatchedPooledMLP        — 219,859 (expected 219,859)
  [PASS] MatchedCNN2D            — 222,166 (expected 222,166)

2. Capacity matching claimed in the paper
  [PASS] MatchedPooledMLP within 1% of the proposed model  — -0.57%
  [PASS] MatchedCNN2D     within 1% of the proposed model  — +0.47%
         (spread between the two extremes is 1.05%; the paper does not
          claim this is below 1%)

3. Forward pass produces the documented shapes
  [PASS] JointDecGATv14Batch  — beam (2, 3, 64), assoc (2, 181, 3)
  [PASS] MatchedPooledMLP     — beam (2, 3, 64), assoc (2, 181, 3)
  [PASS] MatchedCNN2D         — beam (2, 3, 64), assoc (2, 181, 3)

4. The GNN is permutation invariant (Proposition 5)
  [PASS] beam scores invariant under user permutation  — max deviation 5.36e-07
  [PASS] association logits equivariant                — max deviation 2.09e-07

5. The teacher runs and improves on its initialisation
  [PASS] beam budget respected  — |A_m| = [8, 8, 8]
  [PASS] swap refinement never worsens the objective
         — 2.026927e-04 -> 3.277644e-04

ALL MODEL CHECKS PASSED
```

### Why the parameter counts are the important line

`src/models/` was extracted from the research notebook. The cheapest strong
check that the extraction is faithful is the parameter count: a dropped layer,
a changed hidden width, or a missing head all move it. All four match the
values reported in the paper exactly.

### A correction made during verification

The first run of check 2 failed. The test had compared the two baselines
against *each other* (a spread of $1.05\%$) rather than against the proposed
model, which is what the paper's capacity-matching claim is about ($-0.57\%$
and $+0.47\%$). The test was wrong, not the paper; it now asserts the claim the
paper actually makes, and the paper states both deviations explicitly so the
distinction cannot be misread.

### Proposition 5, measured

Check 4 is the empirical counterpart of the invariance proof. Deviations of
order $10^{-7}$ are floating-point reduction noise, consistent with the
qualification given in the paper: invariance holds to numerical rather than
exact precision because the beam features involve `argmax` and top-$k$
reductions whose summation order is not associative on a GPU.
