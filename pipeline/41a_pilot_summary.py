"""
where-teacher-acts: the simulated validation mAP of the four pilot runs (EfficientNet-B1, QAT(A) and QAKD(B) with
ConvNeXt-S, 20 and 40 epochs), read from their run.json on the machine that holds the runs, for
41_paper_checks.py. Read-only apart from its output; refuses to overwrite it unless --force.

    python pipeline/41a_pilot_summary.py     ->  reports_qj/part2/pilot_sim.csv
"""
import os
import sys
import csv
import json
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
os.chdir(os.path.dirname(THIS))
import _qj_common as QJ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = os.path.join(QJ.REPORTS, "part2", "pilot_sim.csv")
    if os.path.exists(out) and not a.force:
        sys.exit(f"{out} exists; pass --force to overwrite it")
    rows = []
    for n in QJ.PILOT_EPOCHS:
        for arm, t in (("qat_a", None), ("qakd_b", QJ.PILOT_TEACHER)):
            meta = json.load(open(os.path.join(QJ.run_dir(arm, QJ.PILOT_STUDENT, t, n, QJ.FP32_SEED), "run.json")))
            rows.append({"arm": arm, "student": QJ.PILOT_STUDENT, "teacher": t or "", "ft_epochs": n,
                         "status": meta.get("status"), "val_mAP_sim": meta.get("val_mAP_sim"),
                         "best_epoch": meta.get("best_epoch")})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {rows}")


if __name__ == "__main__":
    main()
