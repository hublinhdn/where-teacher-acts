"""
where-teacher-acts: descriptive checks the manuscript reports about its own pipeline (declared 2026-10-16, item (5) of
the sensitivity analyses in the dated plan, docs/PLAN_AND_LOG.md). Read-only; every input already exists, on
the laptop after rsync.

  coverage.csv     per student, from the deployed graphs of the latency benchmark (latency_graphs/): the
                   QuantizeLinear nodes of PTQ(A) and of QAKD(B) (ConvNeXt-S) and the operators that produce the
                   quantized tensors; the learned clipping ranges exported as Clip nodes; how many Clip nodes ONNX
                   Runtime keeps in the INT8 graph, and how many of those sit between a dequantization and a
                   requantization (a tensor rounded twice)
  rebasing.csv     per student, on the stored embeddings of the validation analysis set (mech/main/emb): the mean
                   cosine of A, of B (ConvNeXt-S) and of the teacher ConvNeXt-S with each other, and the
                   distillation loss alpha (1 - cos) a run starting from A or from B begins with
  calibration.csv  the reference and consumer images among the 256 calibration images of each fold's draw
  simdeploy.csv    validation, MinMax: the simulated mAP that selects each fine-tuned checkpoint against its
                   deployed INT8 mAP, per student and cell (means over teachers), and the pooled contrasts in
                   both measures
  pilot.csv        the 20- and 40-epoch pilot (EfficientNet-B1 with ConvNeXt-S): simulated and deployed
                   validation mAP (reports_qj/part2/pilot_sim.csv, written on the server by 41a_pilot_summary.py,
                   and g0.json)

    python pipeline/41_paper_checks.py        ->  reports_qj/analysis/checks_main/
"""
import os
import sys
import csv
import json
import collections
import statistics as st

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ROOT = os.path.dirname(THIS)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import _qj_common as QJ  # noqa: E402

OUT = os.path.join(QJ.REPORTS, "analysis", "checks_main")
T = QJ.SEED_REPEAT_TEACHER


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"   wrote {os.path.join(OUT, name)} ({len(rows)} rows)")


def coverage():
    import onnx
    d = os.path.join(QJ.REPORTS, "bench", "main", "latency_graphs")
    man = {(r["student"], r["network"], r["precision"]): r["file"]
           for r in csv.DictReader(open(os.path.join(d, "manifest.csv")))}

    def stats(path):
        g = onnx.load(path).graph
        prod = {o: n for n in g.node for o in n.output}
        cons = collections.defaultdict(list)
        for n in g.node:
            for i in n.input:
                cons[i].append(n)
        q = [n for n in g.node if n.op_type == "QuantizeLinear"]
        by = collections.Counter((prod[n.input[0]].op_type if n.input[0] in prod else "input") for n in q)
        clips = [n for n in g.node if n.op_type == "Clip"]
        sandwiched = sum(1 for c in clips if c.input[0] in prod and prod[c.input[0]].op_type == "DequantizeLinear"
                         and cons.get(c.output[0]) and all(o.op_type == "QuantizeLinear" for o in cons[c.output[0]]))
        return len(q), by, len(clips), sandwiched

    rows = []
    for s in QJ.STUDENTS:
        qa, by_a, _, _ = stats(os.path.join(d, man[(s, "A", "INT8")]))
        qk, _, kept, sandwiched = stats(os.path.join(d, man[(s, "QAKD(B)", "INT8")]))
        f32 = onnx.load(os.path.join(d, man[(s, "QAKD(B)", "FP32")])).graph
        learned = sum(1 for n in f32.node if n.op_type == "Clip") - 1     # less the GeM clamp
        rows.append({"student": s, "q_ptqa": qa, "q_ptqa_conv": by_a.get("Conv", 0) + by_a.get("Gemm", 0),
                     "q_ptqa_other": qa - by_a.get("Conv", 0) - by_a.get("Gemm", 0),
                     "q_ptqa_producers": " ".join(f"{k}:{v}" for k, v in by_a.most_common()),
                     "q_qakdb": qk, "learned_ranges": learned, "clip_kept_int8": kept,
                     "clip_between_dq_q": sandwiched})
    write("coverage.csv", rows)


def rebasing():
    d = os.path.join(QJ.REPORTS, "mech", "main", "emb")

    def load(name):
        x = np.load(os.path.join(d, name)).astype(np.float64)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    t = load(f"T_fp32__-__{T}__fp32.npy")
    rows = []
    for s in QJ.STUDENTS:
        a, b = load(f"A_fp32__{s}__-__fp32.npy"), load(f"B_fp32__{s}__{T}__fp32.npy")
        cat, cbt, cab = (a * t).sum(1).mean(), (b * t).sum(1).mean(), (a * b).sum(1).mean()
        rows.append({"student": s, "n_images": len(t), "cos_A_T": round(cat, 4), "cos_B_T": round(cbt, 4),
                     "cos_A_B": round(cab, 4), "kd_loss_from_A": round(QJ.KD_ALPHA * (1 - cat), 3),
                     "kd_loss_from_B": round(QJ.KD_ALPHA * (1 - cbt), 3)})
    write("rebasing.csv", rows)


def calibration():
    from utils.data_utils import load_epill_full_data
    import _qj_split as S
    df = load_epill_full_data()
    rows = []
    for fold_name, exclude in (("test", {S.TEST_FOLD}), ("val", {S.VAL_FOLD, S.TEST_FOLD})):
        pool = df[~df["fold"].isin(list(exclude))].reset_index(drop=True)
        draw = pool.sample(n=QJ.CALIB_N, random_state=QJ.CALIB_SEED).reset_index(drop=True)
        rows.append({"fold": fold_name, "pool": len(pool), "pool_reference": int(pool["is_ref"].sum()),
                     "draw": len(draw), "draw_reference": int(draw["is_ref"].sum()),
                     "draw_consumer": int(len(draw) - draw["is_ref"].sum())})
    write("calibration.csv", rows)


def simdeploy():
    import importlib.util
    spec = importlib.util.spec_from_file_location("qj_analysis", os.path.join(THIS, "26_analysis.py"))
    AN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AN)
    os.chdir(ROOT)
    sim = {(r["arm"], r["student"], r["teacher"] or None, int(r["seed"])): float(r["val_mAP_sim"])
           for r in csv.DictReader(open(os.path.join(QJ.REPORTS, "part2", "train_summary.csv")))
           if r["status"] == "done" and r["ft_epochs"] == "40" and r["val_mAP_sim"]}
    dep = AN.load(os.path.join(QJ.REPORTS, "part2", "main", "eval_val.csv"), "mAP_INT8_minmax", 40)
    cells = (("qat_a", False), ("qat_b", True), ("qakd_a", True), ("qakd_b", True))
    rows, per = [], {}
    for s in QJ.STUDENTS:
        row = {"student": s}
        for arm, has_t in cells:
            keys = [(arm, s, t, 42) for t in (QJ.TEACHERS if has_t else (None,))]
            sv = [sim[k] for k in keys if k in sim]
            dv = [dep[k]["int8"] for k in keys if k in dep]
            row[f"{arm}_sim"], row[f"{arm}_dep"] = round(st.mean(sv), 4), round(st.mean(dv), 4)
            row[f"{arm}_ratio"] = round(st.mean(dep[k]["int8"] / sim[k] for k in keys), 3)
        per[s] = row
        rows.append(row)
    for name, x, y in (("during", "qakd_a", "qat_a"), ("qat_benefit", "qat_b", "qat_a"), ("added", "qakd_b", "qat_b"),
                       ("both", "qakd_b", "qat_a")):
        for kind in ("sim", "dep"):
            v = [per[s][f"{x}_{kind}"] - per[s][f"{y}_{kind}"] for s in QJ.STUDENTS]
            rows.append({"student": f"pooled {name} ({kind})", "qat_a_sim": round(st.mean(v), 4)})
    write("simdeploy.csv", rows)


def pilot():
    g0 = json.load(open(os.path.join(QJ.REPORTS, "part2", "g0.json")))
    sim_path = os.path.join(QJ.REPORTS, "part2", "pilot_sim.csv")
    sim = {(r["arm"], r["ft_epochs"]): float(r["val_mAP_sim"]) for r in csv.DictReader(open(sim_path))} \
        if os.path.exists(sim_path) else {}
    rows = []
    for n, res in sorted(g0["by_N"].items(), key=lambda kv: int(kv[0])):
        for arm, key in (("qat_a", "qat_a_int8"), ("qakd_b", "qakd_b_int8")):
            s = sim.get((arm, n))
            rows.append({"ft_epochs": n, "arm": arm, "val_sim": s if s is not None else "",
                         "val_deployed": res[key], "ratio": round(res[key] / s, 3) if s else ""})
    write("pilot.csv", rows)


def main():
    coverage()
    rebasing()
    calibration()
    simdeploy()
    pilot()


if __name__ == "__main__":
    main()
