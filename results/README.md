# Released result artifacts

Every number in the manuscript is transcribed from one of the files listed
here. None is recomputed from memory.

Populate this directory with:

    python scripts/fetch_results.py --drive /path/to/MyDrive/Thesis

The script copies only the artifacts the paper cites, refuses to copy anything
from the superseded pre-leakage-fix split, and exits non-zero listing any file
whose producing cell has not been run.

## What each artifact supports

| Group | Files | Supports |
|---|---|---|
| Protocol A main result | `clean_multiseed_test_*`, `T13_*`, `T21_*`, `T22_*` | Tables V–VI |
| Operation-bypass ablation | `T24_*`, `T25_*` | Table VIII |
| Cost accounting | `T27_*`, `clean_runtime_forward_summary.csv` | latency and signalling |
| Robustness | `clean_validation_csi_noise_summary.csv`, `clean_validation_pf_iteration_summary.csv` | Sec. VII-D |
| Protocol B spatial holdout | `T17_*`, `spatial_test_diversity_diagnostic.json` | Sec. VII-B |
| Order-symmetry audit | `clean_TEST_user_order_symmetry_*` | Table IX |
| Order-augmentation control | `clean_TEST_order_augmented_symmetry_summary.csv` | Table X |
| Theory verification | `teacher_quality_reduced_instances.csv`, `Cell220T_*`, `Cell220U_*` | Table VII, Props. 3–4 |
| DeepSense, corrected | `deepsense_E5_corrected.csv`, `deepsense_E5_protocol.json` | Table XI |
| DeepSense, in-domain | `deepsense_multiscenario_E1.json` | Table XII — **uncorrected split** |

## Two honest caveats

**`deepsense_multiscenario_E1.json` comes from the uncorrected pipeline.** Its
in-domain numbers carry roughly 1.6 % frame-level optimism from windows
straddling split boundaries. The manuscript says so where it reports them.
The cross-site numbers in that file are superseded by
`deepsense_E5_corrected.csv`.

**CPU latency is provisional.** `clean_runtime_forward_summary.csv` contains a
timing ordering we cannot explain (the message-free variant times slower than
the full model), so absolute latencies there should be treated as
order-of-magnitude only. The manuscript flags this.

## Superseded — do not cite

Anything produced before the leakage fix, including the figures
`proposed 0.9513 / CNN 0.9423 / naive 0.8869`. Those came from a split in which
45.41 % of held-out test receivers had also appeared in training.
