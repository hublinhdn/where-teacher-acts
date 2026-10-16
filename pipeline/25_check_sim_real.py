"""
where-teacher-acts: does the simulated INT8 that selected each Part II checkpoint transfer to real INT8?
Read-only, validation fold only.

Every Part II run keeps the checkpoint with the best SIMULATED INT8 validation mAP (fake quantization in
the training graph, run.json "val_mAP_sim"); 22_eval_part2.py --fold val then measures the same
checkpoint as REAL INT8 on ONNX Runtime ("mAP_INT8_minmax"). In the pilot the two agreed at N = 40
(real / simulated 0.85 for QAT(A) and 0.84 for QAKD(B) of EfficientNet-B1) and failed at N = 20 (0.31 and
0.28): a short fine-tune let the simulation overstate the deployed model. This script prints the ratio
for every measured run and flags a ratio below --min-ratio (0.7, fixed from the pilot before any other
student was measured) as an early warning that the selection does not transfer for that run, so a
student's cells can be looked at before the rest of the grid builds on them. It is a monitoring check,
not a result of the paper. Exit 1 when a run is flagged.

    python pipeline/25_check_sim_real.py [--students efficientnet_b1_timm] [--ft-epochs 40] [--seeds 42]
"""
import os
import sys
import csv
import json
import argparse

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import _qj_common as QJ  # noqa: E402

SHORT = {"efficientnet_b5_timm": "EffB5", "seresnext101_32x4d_timm": "SERX101", "convnext_base_timm": "CNX-B",
         "convnext_small_timm": "CNX-S", "resnet101_tv": "R101", "densenet161_tv": "DN161"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ft-epochs", type=int, default=40)
    ap.add_argument("--students", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[QJ.FP32_SEED])
    ap.add_argument("--min-ratio", type=float, default=0.7)
    ap.add_argument("--flags-only", action="store_true", help="print only the flagged runs and the count")
    a = ap.parse_args()
    path = os.path.join(QJ.REPORTS, "part2", a.tag, "eval_val.csv")
    real = {}
    for r in csv.DictReader(open(path)):
        if (r["kind"] == "part2" and r["status"] == "SUCCESS" and r["ft_epochs"] == str(a.ft_epochs)
                and QJ.row_is_current(r)):
            real[(r["arm"], r["student"], r["teacher"] or None, int(r["seed"]))] = r
    flagged, shown = [], 0
    if not a.flags_only:
        print(f"{'student':24s} {'arm':7s} {'teacher':8s} {'seed':>4s} {'sim':>6s} {'epoch':>5s} {'INT8':>6s} "
              f"{'pct':>6s} {'ratio':>6s}")
    for s in a.students or QJ.STUDENTS:
        for seed in a.seeds:
            for arm in QJ.ARMS:
                for t in ([None] if arm == "qat_a" else QJ.TEACHERS):
                    try:
                        with open(os.path.join(QJ.run_dir(arm, s, t, a.ft_epochs, seed), "run.json")) as f:
                            st = json.load(f)
                    except (OSError, ValueError):
                        continue
                    r = real.get((arm, s, t, seed))
                    if st.get("status") != "done" or r is None:
                        continue
                    sim, i8 = float(st["val_mAP_sim"]), float(r["mAP_INT8_minmax"])
                    ratio = i8 / sim if sim > 0 else float("nan")
                    bad = not ratio >= a.min_ratio
                    flagged += [(arm, s, t, seed)] if bad else []
                    shown += 1
                    if a.flags_only and not bad:
                        continue
                    print(f"{s:24s} {arm:7s} {SHORT.get(t, '-'):8s} {seed:>4d} {sim:6.3f} "
                          f"{st.get('best_epoch', ''):>5} {i8:6.3f} {float(r['mAP_INT8_pct99.9']):6.3f} "
                          f"{ratio:6.2f}{'  <- below ' + str(a.min_ratio) if bad else ''}")
    print(f"\n{shown} runs with both numbers; {len(flagged)} below a real/simulated ratio of {a.min_ratio}")
    sys.exit(1 if flagged else 0)


if __name__ == "__main__":
    main()
