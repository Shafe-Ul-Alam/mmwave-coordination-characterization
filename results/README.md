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
| Signalling | `T27_*` | Table VII, signalling column |
| Superseded latency | `clean_runtime_forward_summary.csv` | kept for provenance only — see caveat |
| Robustness | `clean_validation_csi_noise_summary.csv`, `clean_validation_pf_iteration_summary.csv` | Sec. VII-D |
| Protocol B spatial holdout | `T17_*`, `spatial_test_diversity_diagnostic.json` | Sec. VII-B |
| Order-symmetry audit | `clean_TEST_user_order_symmetry_*` | Table IX |
| Order-augmentation control | `clean_TEST_order_augmented_symmetry_summary.csv` | Table X |
| Theory verification | `teacher_quality_reduced_instances.csv`, `Cell220T_*`, `Cell220U_*` | Table VII, Props. 3–4 |
| Submodularity | `Cell220G3_submodularity_nested.json`, `Cell220G3b_zero_gain_audit.json` | Sec. IV-D, Thm. 1 |
| Latency, controlled | `Cell220G4b_latency_paired.json`, `Cell220G4c_latency_cpu.json` | Table VII, Sec. VIII-C |
| DeepSense, corrected | `deepsense_E5_corrected.csv`, `deepsense_E5_protocol.json` | Table XI |
| DeepSense, onset | `deepsense_G5_onset.csv`, `deepsense_G5_protocol.json` | Table XIV, Sec. IX-F |
| DeepSense, in-domain | `deepsense_multiscenario_E1.json` | Table XII — **uncorrected split** |
| Coupling across geometries | `epsilon_across_geometries.csv`, `.json` | Table XIII |

## Three honest caveats

**The submodularity test must be run in double precision.** The archived gain
arrays are `float32` and `F_m` accumulates several hundred nats, so a
single-precision evaluation reports violations at the `1e-5` level that are
round-off, not property failures. `Cell220G3_submodularity_nested.json` records
the double-precision result and the relative tolerance used; the first version
of that cell failed for exactly this reason and the JSON says so.


**`deepsense_multiscenario_E1.json` comes from the uncorrected pipeline.** Its
in-domain numbers carry roughly 1.6 % frame-level optimism from windows
straddling split boundaries. The manuscript says so where it reports them.
The cross-site numbers in that file are superseded by
`deepsense_E5_corrected.csv`.

**`clean_runtime_forward_summary.csv` is superseded.** It was collected
without warm-up, with each method timed in its own pass and no recorded thread
count. Under the controlled harness the proposed model's figure is unchanged
(23.06 -> 23.71 ms) but the CNN's falls 20x (2880.6 -> 146.9 ms) and the
"message-free is slower" inversion reverses sign. Use
`Cell220G4b_latency_paired.json` (GPU) and `Cell220G4c_latency_cpu.json` (CPU)
instead; both keep raw per-repetition samples and both include an
identical-computation control whose CI contains zero.

## Superseded — do not cite

Anything produced before the leakage fix, including the figures
`proposed 0.9513 / CNN 0.9423 / naive 0.8869`. Those came from a split in which
45.41 % of held-out test receivers had also appeared in training.
