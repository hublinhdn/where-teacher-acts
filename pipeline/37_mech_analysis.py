"""
where-teacher-acts: the mechanism analysis, from the measurements of 36_mech_measure.py (read-only, local).

Three questions, each tied to a result of the paper:

  1. Why the INT8 floor, and why the calibration rule moves it (RQ1): per-tensor quantization noise of
     the deployed graphs (isolated SQNR under the deployed scale and zero point) for A under MinMax and
     percentile 99.9, against the floor.
  2. Why quantization-aware fine-tuning repairs the floor (RQ2): the same noise for QAT(A), whose
     learned clipping ranges the deployed quantizer calibrates on.
  3. Why the teacher during quantization helps most at low floors (RQ3): how much of the teacher's
     retrieval geometry a deployed student keeps. Neighbour preservation NP@10 is the share of a query's
     ten nearest references (cosine, over the 2,000 references of the analysis set) that two models
     share, averaged over the 730 queries; it does not depend on the embedding basis, so it compares a
     student with its teacher, its FP32 start or its own FP32 graph alike. For starts distilled with the
     cosine objective, the mean cosine between student and teacher embeddings of the same image is
     reported too.

Writes reports_qj/analysis/mech_<tag>/{per_model.csv, per_pair.csv, per_student.csv, summary.md}.

    python pipeline/37_mech_analysis.py [--tag main] [--k 10]
"""
import os
import sys
import csv
import argparse
import statistics as st

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _qj_paper as P  # noqa: E402  (moves to the project root)
import _qj_common as QJ  # noqa: E402
import numpy as np  # noqa: E402

CNX_S = QJ.SEED_REPEAT_TEACHER


def mid(cell, student, teacher, rule):
    return f"{cell}__{student or '-'}__{teacher or '-'}__{rule}"


class Store:
    def __init__(self, root, k):
        self.root, self.k = root, k
        self.meta = {r["model"]: r for r in csv.DictReader(open(os.path.join(root, "models.csv")))}
        aset = list(csv.DictReader(open(os.path.join(root, "analysis_set.csv"))))
        self.q = np.array([r["role"] == "query" for r in aset])
        self._emb, self._nn = {}, {}

    def has(self, m):
        return m in self.meta and self.meta[m]["status"] == "SUCCESS" and \
            os.path.exists(os.path.join(self.root, "emb", m + ".npy"))

    def emb(self, m):
        if m not in self._emb:
            e = np.load(os.path.join(self.root, "emb", m + ".npy")).astype(np.float32)
            self._emb[m] = e / np.linalg.norm(e, axis=1, keepdims=True)
        return self._emb[m]

    def nn(self, m):
        """Indices of each query's k nearest references (cosine)."""
        if m not in self._nn:
            e = self.emb(m)
            s = e[self.q] @ e[~self.q].T
            self._nn[m] = np.argpartition(-s, self.k, axis=1)[:, :self.k]
        return self._nn[m]

    def np_at_k(self, a, b):
        na, nb = self.nn(a), self.nn(b)
        return float(np.mean([len(set(x) & set(y)) / self.k for x, y in zip(na, nb)]))

    def cos(self, a, b):
        return float(np.mean(np.sum(self.emb(a) * self.emb(b), axis=1)))

    def sqnr(self, m):
        p = os.path.join(self.root, "sqnr", m + ".csv")
        return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def bulk_bits(r):
    """Effective bits the deployed quantizer leaves to the bulk of a tensor (0.1 to 99.9 percentile span)."""
    sc, span = float(r["scale"]), float(r["p999"]) - float(r["p001"])
    return float(np.log2(max(0.0, span / sc) + 1)) if sc > 0 else float("nan")


def sqnr_summary(rows):
    v = np.array([float(r["sqnr_db"]) for r in rows])
    rr = np.array([float(r["range_ratio"]) for r in rows])
    bb = np.array([bulk_bits(r) for r in rows])
    worst = sorted(rows, key=lambda r: float(r["sqnr_db"]))[:3]
    pw = [float(r["sqnr_db"]) for r in rows if r["producer"] == "Pow"]
    return {"n_tensors": len(rows), "sqnr_min": float(v.min()), "sqnr_p05": float(np.percentile(v, 5)),
            "sqnr_median": float(np.median(v)), "n_below10": int((v < 10).sum()), "n_below20": int((v < 20).sum()),
            "range_ratio_median": float(np.nanmedian(rr)), "range_ratio_max": float(np.nanmax(rr)),
            "bulk_bits_median": float(np.nanmedian(bb)), "bulk_bits_p05": float(np.nanpercentile(bb, 5)),
            "n_bulk_below4": int(np.nansum(bb < 4)),
            "worst": "; ".join(f"{r['producer']}@{float(r['depth']):.2f}:{float(r['sqnr_db']):.1f}" for r in worst),
            "gem_pow_sqnr": pw[0] if pw else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    root = os.path.join(QJ.REPORTS, "mech", a.tag)
    out = os.path.join(QJ.REPORTS, "analysis", f"mech_{a.tag}")
    os.makedirs(out, exist_ok=True)
    S = Store(root, a.k)
    R = P.results()
    order = R["order"]

    # 1. per model: quantization noise of the deployed graph, isolated (36) and accumulated (36b)
    per_model = []
    for m, r in S.meta.items():
        rows = S.sqnr(m)
        if r["status"] == "SUCCESS" and rows:
            row = {"model": m, "cell": r["cell"], "student": r["student"], "teacher": r["teacher"],
                   "rule": r["rule"], **sqnr_summary(rows)}
            acc = os.path.join(root, "accum", m + ".csv")
            if os.path.exists(acc):
                ar = sorted(csv.DictReader(open(acc)), key=lambda x: float(x["depth"]))
                av = np.array([float(x["sqnr_accum_db"]) for x in ar])
                dp = np.array([float(x["depth"]) for x in ar])
                # the backbone's last quantized tensor: the features the retrieval head pools
                bb = [x for x in ar if x["tensor"].startswith("/features/")]
                row.update(acc_bb_end=float(bb[-1]["sqnr_accum_db"]) if bb else float("nan"),
                           acc_bb_min=min(float(x["sqnr_accum_db"]) for x in bb) if bb else float("nan"))
                row.update(acc_final=float(av[-1]), acc_min=float(av.min()),
                           acc_q1=float(np.median(av[dp <= 0.25])) if (dp <= 0.25).any() else float("nan"),
                           acc_q2=float(np.median(av[(dp > 0.25) & (dp <= 0.5)])) if ((dp > 0.25) & (dp <= 0.5)).any() else float("nan"),
                           acc_q3=float(np.median(av[(dp > 0.5) & (dp <= 0.75)])) if ((dp > 0.5) & (dp <= 0.75)).any() else float("nan"),
                           acc_q4=float(np.median(av[dp > 0.75])) if (dp > 0.75).any() else float("nan"))
            per_model.append(row)

    # 2. per pair: geometry kept from the FP32 start, the teacher and the network's own FP32 graph
    per_pair = []
    for s in order:
        A = mid("A_fp32", s, None, "fp32")
        for t in QJ.TEACHERS:
            B, T = mid("B_fp32", s, t, "fp32"), mid("T_fp32", None, t, "fp32")
            cells = {"A": A, "B": B, "PTQ_A": mid("PTQ_A", s, None, "mm"), "PTQ_B": mid("PTQ_B", s, t, "mm"),
                     "QAT_A": mid("QAT_A", s, None, "mm"), "QAT_B": mid("QAT_B", s, t, "mm"),
                     "QAKD_A": mid("QAKD_A", s, t, "mm"), "QAKD_B": mid("QAKD_B", s, t, "mm")}
            if not all(S.has(x) for x in list(cells.values()) + [T]):
                continue
            row = {"student": s, "teacher": t}
            for c, m in cells.items():
                row[f"np_T_{c}"] = S.np_at_k(m, T)                       # teacher's neighbours kept
            for c in ("PTQ_A", "QAT_A", "QAKD_A"):
                row[f"np_A_{c}"] = S.np_at_k(cells[c], A)               # undistilled start's neighbours kept
            for c in ("PTQ_B", "QAT_B", "QAKD_B"):
                row[f"np_B_{c}"] = S.np_at_k(cells[c], B)               # distilled start's neighbours kept
            for c in ("B", "PTQ_B", "QAT_B", "QAKD_B", "QAKD_A"):
                row[f"cos_T_{c}"] = S.cos(cells[c], T)                  # alignment with the teacher
            for c, ref in (("PTQ_A", A), ("PTQ_B", B), ("QAT_A", A), ("QAT_B", B), ("QAKD_A", A), ("QAKD_B", B)):
                row[f"cos_start_{c}"] = S.cos(cells[c], ref)
            per_pair.append(row)

    # 3. per student: means over teachers, with the floors and effects of the test fold
    pc_ok = {s: S.has(mid("PTQ_A", s, None, "pc")) and S.has(mid("PTQ_B", s, CNX_S, "pc")) for s in order}
    per_student = []
    for s in order:
        pr = [r for r in per_pair if r["student"] == s]
        row = {"student": s, "floor_mm": R["mm"]["stu"][s]["floor"], "floor_pc": R["pc"]["stu"][s]["floor"]}
        for k in ("gF", "ptq", "qat", "rescue", "during", "both", "added", "kept_ptq", "kept_qat", "kept_both"):
            row[f"mm_{k}"] = R["mm"]["stu"][s][k]
            row[f"pc_{k}"] = R["pc"]["stu"][s][k]
        if pr:
            for key in pr[0]:
                if key not in ("student", "teacher"):
                    row[key] = st.mean(r[key] for r in pr)
        A = mid("A_fp32", s, None, "fp32")
        if S.has(A):
            for rule in ("mm", "pc"):
                if S.has(mid("PTQ_A", s, None, rule)):
                    row[f"np_own_PTQ_A_{rule}"] = S.np_at_k(mid("PTQ_A", s, None, rule), A)
                    row[f"cos_own_PTQ_A_{rule}"] = S.cos(mid("PTQ_A", s, None, rule), A)
                Bs = mid("B_fp32", s, CNX_S, "fp32")
                if S.has(Bs) and S.has(mid("PTQ_B", s, CNX_S, rule)):
                    row[f"np_own_PTQ_Bcnxs_{rule}"] = S.np_at_k(mid("PTQ_B", s, CNX_S, rule), Bs)
        for m in per_model:
            if m["student"] == s and m["teacher"] in ("", CNX_S):
                tag = f"{m['cell']}_{m['rule']}"
                for key in ("n_tensors", "sqnr_min", "sqnr_p05", "sqnr_median", "n_below10", "n_below20",
                            "range_ratio_max", "bulk_bits_median", "bulk_bits_p05", "n_bulk_below4", "gem_pow_sqnr",
                            "worst", "acc_final", "acc_min", "acc_q1", "acc_q2", "acc_q3", "acc_q4", "acc_bb_end",
                            "acc_bb_min"):
                    if key in m:
                        row[f"{tag}_{key}"] = m[key]
        row["pc_complete"] = pc_ok[s]
        per_student.append(row)

    for name, rows in (("per_model.csv", per_model), ("per_pair.csv", per_pair), ("per_student.csv", per_student)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with open(os.path.join(out, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)

    # 4. summary
    L = [f"# where-teacher-acts mechanism analysis ({a.tag}), NP@{a.k} over {int(S.q.sum())} queries and "
         f"{int((~S.q).sum())} references of the validation analysis set", "",
         f"Models measured: {sum(r['status'] == 'SUCCESS' for r in S.meta.values())} of {len(S.meta)}; "
         f"pairs complete: {len(per_pair)} of 42", ""]
    if per_model:
        L += ["## Quantization noise of the deployed graphs (A and B with ConvNeXt-S, QAT(A), QAKD(B) with ConvNeXt-S)",
              "", "| student | cell | rule | tensors | min SQNR | 5th pct | median | < 10 dB | < 20 dB | max range ratio | GeM pow | worst three |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in order:
            for m in sorted((m for m in per_model if m["student"] == s), key=lambda m: (m["cell"], m["rule"])):
                L.append(f"| {P.SHORT[s]} | {m['cell']} | {m['rule']} | {m['n_tensors']} | {m['sqnr_min']:.1f} | "
                         f"{m['sqnr_p05']:.1f} | {m['sqnr_median']:.1f} | {m['n_below10']} | {m['n_below20']} | "
                         f"{m['range_ratio_max']:.1f} | {m['gem_pow_sqnr']:.1f} | {m['worst']} |")
    if per_student:
        L += ["", "## Per student (means over teachers where a teacher is involved)", "",
              "| student | floor mm | NP own PTQ(A) mm | NP own PTQ(A) pc | NP T: B | PTQ(B) | QAT(B) | QAKD(B) | "
              "cos T: B | QAT(B) | QAKD(B) | NP B: QAT(B) | QAKD(B) | added mm |", "|" + "---|" * 14]
        for r in per_student:
            g = lambda k, f=".3f": format(r[k], f) if k in r and r[k] == r[k] else "n/a"  # noqa: E731
            L.append(f"| {P.SHORT[r['student']]} | {r['floor_mm']:.3f} | {g('np_own_PTQ_A_mm')} | {g('np_own_PTQ_A_pc')} | "
                     f"{g('np_T_B')} | {g('np_T_PTQ_B')} | {g('np_T_QAT_B')} | {g('np_T_QAKD_B')} | {g('cos_T_B')} | "
                     f"{g('cos_T_QAT_B')} | {g('cos_T_QAKD_B')} | {g('np_B_QAT_B')} | {g('np_B_QAKD_B')} | "
                     f"{r['mm_added']:+.3f} |")
        # floor dependence and links to the effects
        full = [r for r in per_student if "np_T_QAKD_B" in r]
        if len(full) == len(order):
            L += ["", "## Spearman over the seven students (exact two-sided p)", ""]
            fl = [r["floor_mm"] for r in full]
            tests = [("floor_mm", "teacher NP lost by QAT(B) vs B", [r["np_T_B"] - r["np_T_QAT_B"] for r in full]),
                     ("floor_mm", "teacher NP given back by QAKD(B) vs QAT(B)", [r["np_T_QAKD_B"] - r["np_T_QAT_B"] for r in full]),
                     ("floor_mm", "start NP kept by QAT(B)", [r["np_B_QAT_B"] for r in full]),
                     ("mm_added", "teacher NP given back by QAKD(B) vs QAT(B)", [r["np_T_QAKD_B"] - r["np_T_QAT_B"] for r in full]),
                     ("mm_added", "teacher cosine given back by QAKD(B) vs QAT(B)", [r["cos_T_QAKD_B"] - r["cos_T_QAT_B"] for r in full]),
                     ("mm_kept_ptq", "NP own PTQ(B) minus PTQ(A), MinMax",
                      [r.get("np_own_PTQ_Bcnxs_mm", np.nan) - r.get("np_own_PTQ_A_mm", np.nan) for r in full]),
                     ("floor_mm", "NP own PTQ(A), MinMax", [r.get("np_own_PTQ_A_mm", np.nan) for r in full])]
            tests += [("floor_mm", "quantized tensors in the graph (PTQ(A))", [r.get("PTQ_A_mm_n_tensors", np.nan) for r in full]),
                      ("floor_mm", "median bulk bits, PTQ(A) MinMax", [r.get("PTQ_A_mm_bulk_bits_median", np.nan) for r in full]),
                      ("floor_mm", "accumulated SQNR at the backbone output, PTQ(A) MinMax", [r.get("PTQ_A_mm_acc_bb_end", np.nan) for r in full]),
                      ("floor_pc", "accumulated SQNR at the backbone output, PTQ(A) percentile", [r.get("PTQ_A_pc_acc_bb_end", np.nan) for r in full]),
                      ("floor_mm", "embedding cosine INT8 vs FP32, PTQ(A) MinMax", [r.get("cos_own_PTQ_A_mm", np.nan) for r in full]),
                      ("floor_pc", "embedding cosine INT8 vs FP32, PTQ(A) percentile", [r.get("cos_own_PTQ_A_pc", np.nan) for r in full]),
                      ("mm_rescue", "drift of QAT(A) from A (1 - NP)", [1 - r["np_A_QAT_A"] for r in full]),
                      ("mm_added", "drift of QAT(B) from B (1 - NP)", [1 - r["np_B_QAT_B"] for r in full]),
                      ("mm_rescue", "drift of QAT(B) from B (1 - NP)", [1 - r["np_B_QAT_B"] for r in full]),
                      ("floor_mm", "drift of QAT(A) from A (1 - NP)", [1 - r["np_A_QAT_A"] for r in full])]
            for xk, name, y in tests:
                x = [r[xk] for r in full]
                if any(v != v for v in y):
                    continue
                rho, p = P.spearman_two_sided(x, y)
                L.append(f"- {name} against {xk}: rho {rho:+.2f}, p {p:.3f}")
    text = "\n".join(L) + "\n"
    open(os.path.join(out, "summary.md"), "w").write(text)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
