# Leakage-Free Beam Selection Benchmark and Coordination Characterisation

Companion code for:

> M. S. U. Alam and P. K. Choudhury, "When Does Inter-Cell Coordination Pay in Limited-Beam
> mmWave Selection? A Decoupling Theory and Leakage-Free Characterisation," submitted to
> *IEEE Transactions on Vehicular Technology*.

This repository releases contribution **C2** (the leakage-free evaluation protocol, as a reusable
benchmark) and **C4** (the training, evaluation and statistics harness) of that paper.

---

## Why this exists

Beam-selection studies built on DeepMIMO conventionally form decision groups by sampling receivers at
random and *then* splitting into train/test. In our own earlier work this let **45.30 % of unique test
receivers appear in training**. Leakage of exactly this kind is a documented cause of over-optimistic
results across machine-learning-based science (Kapoor & Narayanan, *Patterns* 2023); to our knowledge
it had not previously been audited in this literature.

The audit is one command, and it runs on **any** split, however it was produced:

```bash
python -m src.leakage_audit --train train_groups.npy --test test_groups.npy \
                            --val val_groups.npy --coords receiver_xyz.npy
```

It exits non-zero if any receiver appears in two splits, so it can be wired into CI.

---

## The two protocols

| | Protocol A | Protocol B |
|---|---|---|
| Name | receiver-ID-disjoint | guard-banded coordinate-spatial |
| Guarantee | no receiver ID in two splits | no test receiver within `guard_m` of any training receiver |
| Pools (paper) | 183,610 / 39,345 / 39,346 | principal-axis regions + guard bands |
| Groups (paper) | 2000 / 200 / 300 of U = 181 | 2000 / 200 / 300 of U = 181 |
| Measured leakage | 0.00 % | 0.00 %, min separation **13.80 m** |
| Difficulty | harder than the leaky split | harder still; the heuristic already scores 0.985 |

**Both are needed.** Protocol A removes identity leakage but not *positional* near-duplication: on a
dense receiver grid, two distinct receiver IDs can sit centimetres apart. Running `leakage_audit` with
`--coords` on a Protocol-A split makes this visible immediately — ID leakage 0.00 %, yet a large
fraction of test receivers lie within a metre of a training receiver. Protocol B is the answer to that,
and the reason the paper reports both.

---

## Layout

```
.
├── README.md
├── LICENSE                     MIT
├── CITATION.cff
├── requirements.txt
├── configs/
│   ├── protocol_a_uid.yaml     receiver-ID-disjoint, the headline protocol
│   ├── protocol_b_spatial.yaml guard-banded coordinate-spatial holdout
│   └── model_v14.yaml          architecture and training hyper-parameters
├── src/
│   ├── splits.py               both split protocols + group construction
│   ├── leakage_audit.py        the reusable audit (C2)
│   ├── metrics.py              PF objective, efficiency, redundancy, Jain, feasibility
│   └── paired_stats.py         bootstrap CI, permutation test, Cohen's d_z, win rate
├── scripts/
│   ├── make_splits.py          build and record a split from a config
│   └── verify_release.py       end-to-end self-test of everything above
└── results/
    └── README.md               which released CSV reproduces which table
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. build a leakage-free split (Protocol A)
python scripts/make_splits.py --config configs/protocol_a_uid.yaml \
                              --coords data/receiver_xyz.npy --out splits/protocol_a

# 2. audit it — this is the reusable part
python -m src.leakage_audit --train splits/protocol_a/train_groups.npy \
                            --test  splits/protocol_a/test_groups.npy \
                            --val   splits/protocol_a/val_groups.npy \
                            --coords data/receiver_xyz.npy \
                            --json  splits/protocol_a/audit.json

# 3. check the harness end to end (no DeepMIMO download required)
python scripts/verify_release.py
```

`data/receiver_xyz.npy` is the `(n_receivers, 3)` position table that the DeepMIMO loader returns for
the chosen scenario; we do not redistribute DeepMIMO itself. See `configs/protocol_a_uid.yaml` for the
exact scenario and transmitter/receiver set IDs used in the paper.

---

## Reproducing the paper's numbers

Every number in the manuscript comes from a released artifact, not from a re-run:

| Paper element | Source artifact |
|---|---|
| Table III (matched baselines) | `T21_clean_matched_learned_baseline_final_test.tex` |
| Table IV (paired statistics) | `T22_clean_matched_learned_baseline_paired_statistics.tex` |
| Table V (component ablation) | `T24_clean_multiseed_component_ablation.tex` |
| Table VI (latency / communication) | `T27_clean_runtime_communication_accounting.csv`, `clean_runtime_forward_summary.csv` |
| Figs. 5–7 | regenerated from the above by `make_figs.py` in the manuscript tree |
| Tables VIII–X, Fig. 9 (DeepSense) | `deepsense_multiscenario_E1.json`, `deepsense_loso_E2.json`, `deepsense_repr_E3.json` |

`src/paired_stats.py` recomputes any paired comparison from the per-group CSVs, so the confidence
intervals, permutation *p*-values, effect sizes and win rates can be checked independently.

---

## What is **not** in this release, and why

* **DeepMIMO and DeepSense 6G raw data.** Both are redistributed by their own authors under their own
  terms; we release the split indices and the code that builds them, not the channels.
* **Trained checkpoints.** Available on request; they are large and not needed to verify any claim in
  the paper, all of which rest on the released per-group metric CSVs.
* **The Colab research notebook.** The harness here is the cleaned, documented extraction of it. Model
  definitions and the expert-label generator are being ported from the notebook and land in
  `src/models/` and `src/expert/`; until then this release covers the protocol, metrics and statistics,
  which are the parts a reviewer or reuser actually needs to check the claims.

---

## Honest scope

This repository supports a **characterisation**, not an architecture claim. The paper's own headline
finding is that a parameter-matched CNN with no inter-cell messaging *beats* the graph model in the
deployed noise-limited regime, and that removing the graph's message-passing component *significantly
improves* it (p = 0.0019). The code is released so that this can be checked, not so that a graph model
can be promoted. If you reuse the protocol and find the opposite, that is a useful result and we would
like to hear about it.

---

## Licence and citation

MIT for the code (see `LICENSE`). The split indices and audit reports are released CC-BY-4.0. If you
use the protocol or the audit, please cite the paper (see `CITATION.cff`).

## Citing and obtaining this release

Repository: <https://github.com/Shafe-Ul-Alam/mmwave-coordination-characterization>

A tagged archive with a DOI is deposited on Zenodo; cite that DOI rather than
the branch. See `CITATION.cff`.
