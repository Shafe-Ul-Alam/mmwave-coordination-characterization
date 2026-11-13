#!/usr/bin/env python3
"""Copy the result artifacts the manuscript cites into ``results/``.

Run this once, from a machine (or Colab session) that has the project Drive
mounted. It copies only the files the paper actually transcribes numbers from,
reports anything missing, and refuses to copy artifacts from the superseded
pre-leakage-fix split.

    python scripts/fetch_results.py --drive /content/drive/MyDrive/Thesis
"""
import argparse
import shutil
import sys
from pathlib import Path

# artifact -> what it reproduces in the manuscript
WANTED = {
    # ---- DeepMIMO, Protocol A (receiver-ID disjoint) ----
    "spread64_clean_uid_v3/q1_data/clean_multiseed_test_group_metrics.csv":
        "per-group raw metrics; input to every paired test",
    "spread64_clean_uid_v3/q1_data/clean_multiseed_test_method_summary.csv":
        "per-method means",
    "spread64_clean_uid_v3/q1_data/clean_multiseed_test_paired_statistics.csv":
        "paired comparisons",
    "spread64_clean_uid_v3/q1_data/T13_clean_multiseed_final_test_table.tex":
        "Table V, naive and search-reference columns",
    "spread64_clean_uid_v3/q1_data/T21_clean_matched_learned_baseline_final_test.tex":
        "Table V, learned baselines",
    "spread64_clean_uid_v3/q1_data/T22_clean_matched_learned_baseline_paired_statistics.tex":
        "Table VI, paired statistics",
    "spread64_clean_uid_v3/q1_data/T24_clean_multiseed_component_ablation.tex":
        "Table VIII, operation-bypass ablation",
    "spread64_clean_uid_v3/q1_data/T25_clean_multiseed_component_ablation_statistics.tex":
        "Table VIII, significance",
    "spread64_clean_uid_v3/q1_data/T27_clean_runtime_communication_accounting.csv":
        "signalling column",
    "spread64_clean_uid_v3/q1_data/clean_runtime_forward_summary.csv":
        "SUPERSEDED latency (kept for provenance; see results/README.md)",
    "spread64_clean_uid_v3/q1_data/clean_validation_csi_noise_summary.csv":
        "CSI-noise robustness",
    "spread64_clean_uid_v3/q1_data/clean_validation_pf_iteration_summary.csv":
        "PF-refinement sensitivity",
    # ---- Protocol B (guard-banded coordinate-spatial) ----
    "spread64_spatial_v1/q1_data/T17_spatial_multiseed_final_test_table.tex":
        "spatial holdout results",
    "spread64_spatial_v1/q1_data/spatial_test_diversity_diagnostic.json":
        "spatial test-set diversity audit",
    # ---- order-symmetry audit and its control ----
    "spread64_clean_uid_v3/q1_data/clean_TEST_user_order_symmetry_summary.csv":
        "Table IX, native CNN order sensitivity (test split)",
    "spread64_clean_uid_v3/q1_data/clean_TEST_user_order_symmetry_paired_statistics.csv":
        "Table IX, paired statistics",
    "spread64_clean_uid_v3/q1_data/clean_TEST_order_augmented_symmetry_summary.csv":
        "Table X, order-augmented CNN control",
    # ---- theory verification ----
    "spread64_clean_uid_v3/q1_data/teacher_quality_reduced_instances.csv":
        "Table VII, solver quality vs exact optimum",
    "spread64_clean_uid_v3/q1_data/Cell220T_teacher_quality_and_epsilon.json":
        "eta and the epsilon distribution",
    "spread64_clean_uid_v3/q1_data/Cell220U_joint_epsilon_emin.json":
        "joint (eps_u, e_u) distribution for Proposition 4",
    "spread64_clean_uid_v3/q1_data/Cell220G3_submodularity_nested.json":
        "Sec. IV-D, nested-set submodularity test and the kappa sweep",
    "spread64_clean_uid_v3/q1_data/Cell220G3b_zero_gain_audit.json":
        "Sec. IV-D, zero-gain audit behind the floor discussion",
    "spread64_clean_uid_v3/q1_data/Cell220G4b_latency_paired.json":
        "Table VII, GPU latency with the identical-computation control",
    "spread64_clean_uid_v3/q1_data/Cell220G4c_latency_cpu.json":
        "Table VII, CPU latency under the same harness",
    # ---- DeepSense 6G ----
    "deepsense_corrected/deepsense_E5_corrected.csv":
        "Table XI, corrected cross-site protocol",
    "deepsense_corrected/deepsense_E5_protocol.json":
        "split and normalisation protocol record",
    "deepsense_corrected/deepsense_G5_onset.csv":
        "Table XIV, event-level blockage onset",
    "deepsense_corrected/deepsense_G5_protocol.json":
        "onset task definition and operating-point rule",
    "deepsense_multiscenario_E1.json":
        "in-domain results (UNCORRECTED split; see Sec. IX-B)",
    # ---- coupling across geometries and frequencies ----
    "geometry_sweep/epsilon_across_geometries.csv":
        "Table XIII, per-user eps across five scenarios",
    "geometry_sweep/epsilon_across_geometries.json":
        "Table XIII, summary quantiles and Prop. 4 coverage",
}

# never ship these: they came from the pre-fix split
FORBIDDEN_SUBSTRINGS = ("figs_final", "_SUPERSEDED", "legacy_evidence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True,
                    help="path to MyDrive/Thesis")
    ap.add_argument("--out", default=None, help="destination (default results/)")
    args = ap.parse_args()

    root = Path(args.drive)
    dest = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results"
    dest.mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for rel, what in WANTED.items():
        if any(b in rel for b in FORBIDDEN_SUBSTRINGS):
            continue
        src = root / rel
        if src.exists():
            shutil.copy2(src, dest / src.name)
            copied.append((src.name, what))
        else:
            missing.append((rel, what))

    print(f"copied {len(copied)} artifact(s) into {dest}")
    for n, w in copied:
        print(f"  + {n:62} {w}")
    if missing:
        print(f"\n{len(missing)} MISSING — the cell that produces each has not been run:")
        for n, w in missing:
            print(f"  - {n}")
            print(f"      ({w})")
        print("\nThe release is still usable; the manuscript simply cannot be")
        print("re-derived end to end until these are present.")
        return 1
    print("\nall cited artifacts present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
