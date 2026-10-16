"""
where-teacher-acts: gate G0 (the dated plan, docs/PLAN_AND_LOG.md), decided on the validation fold only.

Reads the validation measurements of the pilot (22_eval_part2.py --fold val --plan pilot --fp32-refs):
EfficientNet-B1 with ConvNeXt-S, QAT(A) and QAKD(B) at N = 20 and N = 40, and the FP32 A and B they
started from. The old thresholds (0.45 and 0.55 INT8 MinMax) were 59% of A's and 67% of B's FP32 mAP on
an earlier study's data; the same fractions apply here: QAT(A) >= 0.59 x A and QAKD(B) >= 0.67 x B. The
chosen N is the smallest that passes both; it is not the N with the best score. Writes reports_qj/part2/g0.json.

    python pipeline/23_gate_g0.py [--tag main]
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

FRAC_A, FRAC_B = 0.59, 0.67


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    path = os.path.join(QJ.REPORTS, "part2", a.tag, "eval_val.csv")
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "SUCCESS" and QJ.row_is_current(r)]
    s, t = QJ.PILOT_STUDENT, QJ.PILOT_TEACHER

    def pick(kind, arm="", teacher="", n=""):
        hit = [r for r in rows if r["kind"] == kind and r["arm"] == arm and r["student"] == s
               and r["teacher"] == teacher and r["ft_epochs"] == str(n) and r["seed"] == str(QJ.FP32_SEED)]
        return float(hit[0]["mAP_FP32" if kind != "part2" else "mAP_INT8_minmax"]) if hit else None

    a_fp32, b_fp32 = pick("fp32_A"), pick("fp32_B", teacher=t)
    if a_fp32 is None or b_fp32 is None:
        sys.exit("FP32 references of the pilot are missing: run 22_eval_part2.py --fold val --plan pilot --fp32-refs")
    need_a, need_b = FRAC_A * a_fp32, FRAC_B * b_fp32
    result = {"student": s, "teacher": t, "A_fp32_val": a_fp32, "B_fp32_val": b_fp32,
              "threshold_qat_a": round(need_a, 4), "threshold_qakd_b": round(need_b, 4), "by_N": {}, "chosen_N": None}
    for n in QJ.PILOT_EPOCHS:
        qa, qb = pick("part2", "qat_a", "", n), pick("part2", "qakd_b", t, n)
        ok = qa is not None and qb is not None and qa >= need_a and qb >= need_b
        result["by_N"][n] = {"qat_a_int8": qa, "qakd_b_int8": qb, "pass": ok}
        print(f"N={n}: QAT(A) {qa} (need >= {need_a:.4f}), QAKD(B) {qb} (need >= {need_b:.4f}) -> {'PASS' if ok else 'fail'}")
        if ok and result["chosen_N"] is None:
            result["chosen_N"] = n
    out = os.path.join(QJ.REPORTS, "part2", "g0.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"G0: {'PASS, N = ' + str(result['chosen_N']) if result['chosen_N'] else 'FAIL (raise N or LR once, then stop and discuss)'} -> {out}")
    sys.exit(0 if result["chosen_N"] else 1)


if __name__ == "__main__":
    main()
