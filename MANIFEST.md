# Release manifest

## Status of each component

| Component | Status | Location |
|---|---|---|
| Split protocols A and B | complete, tested | `src/splits.py` |
| Leakage audit | complete, tested | `src/leakage_audit.py` (non-zero exit on leakage) |
| Metric definitions | complete, tested | `src/metrics.py` |
| Paired statistics | complete, tested | `src/paired_stats.py` |
| **Model architectures** | **complete, tested** | `src/models/` — GNN, ablation, MLP, CNN |
| **Imitation teacher (Algorithm 1)** | **complete, tested** | `src/expert/` — popularity init + swap refinement |
| Config files | complete | `configs/` |
| Split builder CLI | complete, tested | `scripts/make_splits.py` |
| Result fetcher | complete | `scripts/fetch_results.py` |
| Self-test (splits, metrics, stats) | complete, passing | `scripts/verify_release.py` |
| Self-test (models, teacher) | complete | `scripts/verify_models.py` (needs PyTorch) |
| Result artifacts | fetched on demand | `results/`, see its README |
| Trained checkpoints | on request | large; no claim in the paper depends on them |
| DeepMIMO / DeepSense raw data | not redistributable | obtain from the original authors; `configs/` records the generator settings |

## How the model port is verified

`src/models/` and `src/expert/` were extracted from the research notebook.
The extraction is checked three ways by `scripts/verify_models.py`:

1. **Parameter counts** must equal the values reported in the paper exactly —
   221,125 (GNN and each ablation), 219,859 (MLP), 222,166 (CNN). Any dropped
   layer, changed width or missing head moves these.
2. **Forward shapes** must be `(B, M, Nb)` for beam scores and
   `(B, U, M)` for association logits.
3. **Permutation invariance** of the GNN (Proposition 5) is checked numerically,
   and the teacher is checked to respect the beam budget and never to worsen
   its own objective during swap refinement.

## Before publishing the repository

1. `python scripts/fetch_results.py --drive /path/to/MyDrive/Thesis`
2. `python scripts/verify_release.py` and `python scripts/verify_models.py`;
   commit both outputs.
3. Replace `<user>` in `README.md`, `CITATION.cff` and the paper's `\thanks`
   footnote with the real GitHub account.
4. Archive a tagged release on Zenodo; put the DOI in `CITATION.cff` and in the
   paper's reproducibility footnote.

## Known gaps, stated plainly

- The DeepSense **in-domain** results were produced before the split fix and
  carry ~1.6 % frame-level optimism. The corrected cross-site study is in
  `deepsense_E5_corrected.csv`.
- CPU latency measurements are provisional; see `results/README.md`.
- Training scripts are not included. The model and teacher definitions are, so
  the architecture and the labels are reproducible; reproducing the trained
  checkpoints requires the notebook's training cells.
