"""
where-teacher-acts: the FP32 gate (docs/PLAN_AND_LOG.md, section 2.9), read-only. Summarizes the FP32
phase from the run records and, when present, the test-fold benchmark, and checks the two things the
design needs before Part II:

  F1  every teacher beats every undistilled student on validation (a teacher worth distilling from)
  F2  the INT8 floors of the seven students still spread from low to high (MinMax), judged on the
      VALIDATION fold (22_eval_part2.py --fold val --kinds fp32_A), never on test: at least two
      students below 0.3 and two above 0.5, the spread the roster was chosen for. Test-fold
      floors from the benchmark are shown for information only.

Prints a table and writes reports_qj/fp32/gate_f.json. Safe to run while training continues.

    python pipeline/13_check_fp32.py [--bench-tag main]
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


def run_json(role, model, teacher=None, seed=QJ.FP32_SEED):
    try:
        with open(os.path.join(QJ.fp32_dir(role, model, teacher, seed), "run.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-tag", default="main")
    a = ap.parse_args()
    bench = {}
    path = os.path.join(QJ.REPORTS, "bench", a.bench_tag, "fp32_ptq.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["status"] == "SUCCESS":
                bench[(r["role"], r["model"], r["teacher"], r["seed"])] = r

    def b(role, model, teacher="", col="test_mAP_FP32"):
        r = bench.get((role, model, teacher, str(QJ.FP32_SEED)))
        return r[col] if r else ""

    print(f"{'teacher':28s} {'status':8s} {'val':>7s} {'test FP32':>9s} {'epoch':>5s} {'min':>5s}")
    teachers = {}
    for t in QJ.TEACHERS:
        st = run_json("teacher", t)
        teachers[t] = st.get("val_mAP")
        print(f"{t:28s} {st.get('status', 'missing'):8s} {st.get('val_mAP', ''):>7} {b('teacher', t):>9} "
              f"{st.get('best_epoch', ''):>5} {st.get('minutes', ''):>5}")
    print(f"\n{'student (A)':28s} {'status':8s} {'val':>7s} {'test FP32':>9s} {'test INT8':>9s} {'test pct':>8s} {'val INT8':>8s}")
    val_floor = {}
    vpath = os.path.join(QJ.REPORTS, "part2", a.bench_tag, "eval_val.csv")
    if os.path.exists(vpath):
        for r in csv.DictReader(open(vpath)):
            if (r["kind"] == "fp32_A" and r["status"] == "SUCCESS" and r["seed"] == str(QJ.FP32_SEED)
                    and QJ.row_is_current(r)):
                val_floor[r["student"]] = float(r["mAP_INT8_minmax"])   # seed 42; the seed repeats are not F2
    students, floors = {}, {}
    for s in QJ.STUDENTS:
        st = run_json("base", s)
        students[s] = st.get("val_mAP")
        c = b("base", s, col="test_mAP_INT8_minmax")
        floors[s] = val_floor.get(s)
        print(f"{s:28s} {st.get('status', 'missing'):8s} {st.get('val_mAP', ''):>7} {b('base', s):>9} {c:>9} "
              f"{b('base', s, col='test_mAP_INT8_pct99.9'):>8} {val_floor.get(s, ''):>8}")
    done_kd = sum(run_json("kd", s, t).get("status") == "done" for s in QJ.STUDENTS for t in QJ.TEACHERS)
    print(f"\nB runs finished: {done_kd} / {len(QJ.STUDENTS) * len(QJ.TEACHERS)}")

    tv = [v for v in teachers.values() if v is not None]
    sv = [v for v in students.values() if v is not None]
    f1 = (len(tv) == len(QJ.TEACHERS) and len(sv) == len(QJ.STUDENTS) and min(tv) > max(sv)) if tv and sv else None
    fl = [v for v in floors.values() if v is not None]
    f2 = (sum(v < 0.3 for v in fl) >= 2 and sum(v > 0.5 for v in fl) >= 2) if len(fl) == len(QJ.STUDENTS) else None
    weak = [t for t, v in teachers.items() if v is not None and sv and v <= max(sv)]
    print(f"F1 every teacher above every A on validation: {f1}" + (f"  (not above: {weak})" if weak else ""))
    print(f"F2 INT8 floors spread on validation (>= 2 below 0.3 and >= 2 above 0.5): {f2}")
    os.makedirs(os.path.join(QJ.REPORTS, "fp32"), exist_ok=True)
    with open(os.path.join(QJ.REPORTS, "fp32", "gate_f.json"), "w") as f:
        json.dump({"teachers_val": teachers, "students_val": students, "floors_val_minmax": floors,
                   "floors_basis": "validation fold 0, INT8 MinMax", "kd_done": done_kd, "F1": f1, "F2": f2,
                   "teachers_not_above": weak}, f, indent=2)
    sys.exit(0 if f2 else 1)


if __name__ == "__main__":
    main()
