"""
where-teacher-acts: the analysis planned in the dated plan (docs/PLAN_AND_LOG.md), on one fold, read-only.

Reads reports_qj/part2/<tag>/eval_<fold>.csv (22_eval_part2.py; a row is used only when it is current
and of this study's deploy graph, _qj_common.row_is_current) and, for the calibration noise of the test
fold, reports_qj/bench/<tag>/calib_noise.csv (B2). INT8 rule: MinMax (primary) or percentile 99.9
(--rule pct). Writes reports_qj/analysis/<fold>_<rule>_<tag>/{summary.md, per_student.csv, per_pair.csv}
and prints the summary.

Per pair (student x teacher, seed 42):
  gF        B_FP32 - A_FP32             FP32 distillation gain
  ptq       PTQ(B) - PTQ(A)             teacher before quantization, post-training quantization
  qat       QAT(B) - QAT(A)             teacher before, quantization-aware fine-tuning
  rescue    QAT(A) - PTQ(A)             what fine-tuning recovers without any teacher
  during    QAKD(A) - QAT(A)            teacher only during quantization
  both      QAKD(B) - QAT(A)            teacher before and during
  added     QAKD(B) - QAT(B)            teacher during on top of before (H3)
  b_vs_a    QAKD(B) - QAKD(A)           distilled start against undistilled, teacher on (RQ4)
  interaction (QAKD(B) - QAKD(A)) - (QAT(B) - QAT(A)), the 2 x 2 interaction of the Part II factorial
Pooled values are means over students of per-student means over teachers, with a cluster bootstrap over
students (10000 resamples, seed 42), because the student is the independent unit (n = 7). The floor is
PTQ(A); floor dependence is Spearman over the students with an exact one-sided permutation p. The
transfer coefficient beta (per student and regime) is the regression through the origin of the INT8
benefit on gF over the six teachers, signed and on absolute values.
Noise: calibration sd of the PTQ benefit from B2 (test fold only: same five calibration seeds for A and
B), training sd of the whole chain from the seed repeats (3 students x seeds 42, 123, 223) for QAT(A),
QAKD(B), the 'both' effect and gF. A pair's sign counts only when |effect| > 2 sd; for differences of
Part II cells without their own repeats (qat, added, during, b_vs_a) the sd of 'both' stands in, which
the summary states.

    python pipeline/26_analysis.py --fold test [--rule minmax|pct] [--tag main]
"""
import os
import sys
import csv
import math
import argparse
import itertools
import statistics as st

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _j3_common as C  # noqa: E402
os.chdir(C.PROJ_ROOT)
sys.path.insert(0, C.PROJ_ROOT)

import numpy as np  # noqa: E402
import _qj_common as QJ  # noqa: E402

S42 = QJ.FP32_SEED
SHORT = {"efficientnet_b5_timm": "EffB5", "seresnext101_32x4d_timm": "SERX101", "convnext_base_timm": "CNX-B",
         "convnext_small_timm": "CNX-S", "resnet101_tv": "R101", "densenet161_tv": "DN161"}
EFFECTS = ("gF", "ptq", "qat", "rescue", "during", "both", "added", "b_vs_a", "interaction")


def load(path, col, ft_epochs):
    """{(cell, student, teacher, seed): {'fp32': x, 'int8': y}} from the evaluation CSV."""
    out = {}
    for r in csv.DictReader(open(path)):
        if r["status"] != "SUCCESS" or not QJ.row_is_current(r):
            continue
        if r["kind"] == "part2" and r["ft_epochs"] != str(ft_epochs):
            continue
        cell = {"fp32_A": "A", "fp32_B": "B"}.get(r["kind"], r["arm"])
        out[(cell, r["student"], r["teacher"] or None, int(r["seed"]))] = {
            "fp32": float(r["mAP_FP32"]) if r["mAP_FP32"] else None, "int8": float(r[col])}
    return out


def pair_effects(V, s, t, seed=S42):
    def g(cell, tt, what="int8"):
        x = V.get((cell, s, tt, seed))
        return None if x is None else x[what]
    a32, b32, pa, pb = g("A", None, "fp32"), g("B", t, "fp32"), g("A", None), g("B", t)
    qa, qb, ka, kb = g("qat_a", None), g("qat_b", t), g("qakd_a", t), g("qakd_b", t)
    if None in (a32, b32, pa, pb, qa, qb, ka, kb):
        return None
    return {"gF": b32 - a32, "ptq": pb - pa, "qat": qb - qa, "rescue": qa - pa, "during": ka - qa,
            "both": kb - qa, "added": kb - qb, "b_vs_a": kb - ka, "interaction": (kb - ka) - (qb - qa),
            "A_fp32": a32, "B_fp32": b32, "PTQ_A": pa, "PTQ_B": pb, "QAT_A": qa, "QAT_B": qb,
            "QAKD_A": ka, "QAKD_B": kb}


def cluster_ci(per_student, level=0.95, n_boot=10000, seed=42):
    """Mean over students and its percentile cluster-bootstrap interval."""
    x = np.asarray(per_student, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def spearman_exact(x, y, alternative="less"):
    """Spearman rho with an exact permutation p (one-sided 'less' or 'greater')."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)

    def rho(a, b):
        a, b = np.asarray(a) - np.mean(a), np.asarray(b) - np.mean(b)
        return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
    r0 = rho(rx, ry)
    perms = [rho(rx, [ry[i] for i in p]) for p in itertools.permutations(range(len(ry)))]
    p = sum((v <= r0 + 1e-12) if alternative == "less" else (v >= r0 - 1e-12) for v in perms) / len(perms)
    return r0, p


def beta(gi, gf, absolute=False):
    gi, gf = np.asarray(gi, float), np.asarray(gf, float)
    if absolute:
        gi, gf = np.abs(gi), np.abs(gf)
    den = float((gf * gf).sum())
    return float((gi * gf).sum() / den) if den > 0 else float("nan")


def teacher_share(M):
    """Share of the total sum of squares of a students x teachers matrix due to the teacher."""
    M = np.asarray(M, float)
    m = M.mean()
    ss_t = M.shape[0] * ((M.mean(axis=0) - m) ** 2).sum()
    ss_s = M.shape[1] * ((M.mean(axis=1) - m) ** 2).sum()
    ss = ((M - m) ** 2).sum()
    return float(ss_t / ss), float(ss_s / ss)


def chain_noise(V, students):
    """Pooled sd over the seed repeats (seeds 42, 123, 223; teacher ConvNeXt-S) of the cells and effects."""
    t = QJ.SEED_REPEAT_TEACHER
    seeds = (S42,) + tuple(QJ.EXTRA_SEEDS)
    quants = {"QAT_A": [], "QAKD_B": [], "both": [], "gF": [], "ptq": []}
    used = []
    for s in students:
        vals = {k: [] for k in quants}
        for seed in seeds:
            def g(cell, tt, what="int8"):
                x = V.get((cell, s, tt, seed))
                return None if x is None else x[what]
            qa, kb, a32, b32, pa, pb = (g("qat_a", None), g("qakd_b", t), g("A", None, "fp32"), g("B", t, "fp32"),
                                        g("A", None), g("B", t))
            if None in (qa, kb, a32, b32, pa, pb):
                break
            for k, v in (("QAT_A", qa), ("QAKD_B", kb), ("both", kb - qa), ("gF", b32 - a32), ("ptq", pb - pa)):
                vals[k].append(v)
        else:
            used.append(s)
            for k in quants:
                quants[k].append(st.variance(vals[k]))
    return {k: math.sqrt(st.mean(v)) for k, v in quants.items() if v}, used


def calib_noise(path, rule):
    """Pooled sd over the five calibration seeds (B2) of the PTQ benefit PTQ(B) - PTQ(A), per student."""
    if not os.path.exists(path):
        return None
    tag = "minmax" if rule == "minmax" else "pct99.9"
    d = {}
    for r in csv.DictReader(open(path)):
        if r["status"] == "SUCCESS" and r["calibration"] == tag and (r.get("graph") or "plain") == QJ.deploy_graph(r["student"]):
            d[(r["student"], r["checkpoint"], r["calib_seed"])] = float(r["test_mAP_INT8"])
    variances = []
    for s in QJ.STUDENTS:
        diffs = [d[(s, "B", str(c))] - d[(s, "A", str(c))] for c in QJ.CALIB_SEEDS
                 if (s, "B", str(c)) in d and (s, "A", str(c)) in d]
        if len(diffs) >= 3:
            variances.append(st.variance(diffs))
    return math.sqrt(st.mean(variances)) if variances else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, choices=("val", "test"))
    ap.add_argument("--rule", default="minmax", choices=("minmax", "pct"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--ft-epochs", type=int, default=40)
    a = ap.parse_args()
    col = "mAP_INT8_minmax" if a.rule == "minmax" else "mAP_INT8_pct99.9"
    V = load(os.path.join(QJ.REPORTS, "part2", a.tag, f"eval_{a.fold}.csv"), col, a.ft_epochs)
    out_dir = os.path.join(QJ.REPORTS, "analysis", f"{a.fold}_{a.rule}_{a.tag}")
    os.makedirs(out_dir, exist_ok=True)

    pairs, missing = {}, []
    for s in QJ.STUDENTS:
        for t in QJ.TEACHERS:
            e = pair_effects(V, s, t)
            (pairs.__setitem__((s, t), e) if e else missing.append((s, t)))
    students = [s for s in QJ.STUDENTS if all((s, t) in pairs for t in QJ.TEACHERS)]
    floor = {s: V[("A", s, None, S42)]["int8"] for s in students}
    students.sort(key=lambda s: -floor[s])
    per_student = {s: {k: st.mean(pairs[(s, t)][k] for t in QJ.TEACHERS) for k in EFFECTS} for s in students}
    lines = [f"# where-teacher-acts analysis: fold {a.fold}, INT8 {a.rule}, tag {a.tag}", "",
             f"Pairs complete: {len(pairs)} of {len(QJ.STUDENTS) * len(QJ.TEACHERS)}; students complete: "
             f"{len(students)}" + (f"; missing pairs: {missing}" if missing else ""), ""]

    # 1. per student
    lines += ["## Per student (mean over the six teachers)", "",
              "| student | floor PTQ(A) | gF | PTQ | QAT | rescue | during | both | added | QAKD(B)-QAKD(A) | "
              "interaction | kept PTQ | kept QAT | kept both | beta PTQ | beta QAT | beta both |",
              "|" + "---|" * 17]
    rows_out = []
    for s in students:
        m = per_student[s]
        gis = {k: [pairs[(s, t)][k] for t in QJ.TEACHERS] for k in ("ptq", "qat", "both")}
        gfs = [pairs[(s, t)]["gF"] for t in QJ.TEACHERS]
        b = {k: beta(gis[k], gfs) for k in gis}
        ba = {k: beta(gis[k], gfs, absolute=True) for k in gis}
        lines.append(f"| {s} | {floor[s]:.3f} | {m['gF']:+.3f} | {m['ptq']:+.3f} | {m['qat']:+.3f} | "
                     f"{m['rescue']:+.3f} | {m['during']:+.3f} | {m['both']:+.3f} | {m['added']:+.3f} | "
                     f"{m['b_vs_a']:+.3f} | {m['interaction']:+.3f} | {m['ptq'] / m['gF']:.2f} | "
                     f"{m['qat'] / m['gF']:.2f} | {m['both'] / m['gF']:.2f} | {b['ptq']:.2f} | {b['qat']:.2f} | "
                     f"{b['both']:.2f} |")
        rows_out.append(dict(student=s, floor=floor[s], **{k: m[k] for k in EFFECTS},
                             **{f"beta_{k}": b[k] for k in b}, **{f"beta_abs_{k}": ba[k] for k in ba}))

    # 2. pooled, cluster bootstrap
    lines += ["", "## Pooled over students (cluster bootstrap 95% interval, 10000 resamples)", "",
              "| effect | mean | 95% interval | excludes 0 |", "|---|---|---|---|"]
    pooled = {}
    for k in EFFECTS:
        mu, lo, hi = cluster_ci([per_student[s][k] for s in students])
        pooled[k] = (mu, lo, hi)
        lines.append(f"| {k} | {mu:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {'yes' if lo > 0 or hi < 0 else 'no'} |")

    # 3. floor dependence
    lines += ["", "## Dependence on the floor (Spearman over students, exact one-sided permutation p)", "",
              "| effect | rho | p (rho <= observed) | p (rho >= observed) |", "|---|---|---|---|"]
    fl = [floor[s] for s in students]
    for k in ("ptq", "qat", "rescue", "during", "both", "added", "b_vs_a"):
        y = [per_student[s][k] for s in students]
        r_less, p_less = spearman_exact(fl, y, "less")
        _, p_more = spearman_exact(fl, y, "greater")
        lines.append(f"| {k} | {r_less:+.2f} | {p_less:.3f} | {p_more:.3f} |")
    lines.append("")
    lines.append("H3 asks for rho(floor, added) <= -0.71 (one-sided, n = 7).")

    # 4. noise and sign counts
    sig, used = chain_noise(V, QJ.SEED_REPEAT_STUDENTS)
    sig_cal = calib_noise(os.path.join(QJ.REPORTS, "bench", a.tag, "calib_noise.csv"), a.rule) if a.fold == "test" else None
    lines += ["", "## Noise", ""]
    lines.append(f"Seed repeats used (complete chains): {used or 'none'}")
    for k, v in sig.items():
        lines.append(f"- training sd of {k} over seeds 42, 123, 223: {v:.4f}")
    lines.append(f"- calibration sd of the PTQ benefit over 5 calibration seeds (B2): "
                 f"{sig_cal:.4f}" if sig_cal is not None else "- calibration sd: B2 is on the test fold only")
    sd_part2 = sig.get("both")
    sd_ptq = sig_cal if sig_cal is not None else sig.get("ptq")
    sd_f = sig.get("gF", 0.0)
    if sd_part2 is not None and sd_ptq is not None:
        thr_f = max(0.02, 2 * sd_f)
        lines += ["", f"Pairs with an FP32 gain above max(0.02, 2 sd_F) = {thr_f:.3f}: "
                  f"{sum(pairs[p]['gF'] > thr_f for p in pairs)} of {len(pairs)}.", "",
                  "| effect | sd used | kept (> +2 sd) | lost (within 2 sd) | flipped to harm (< -2 sd) | raw positive | raw negative |",
                  "|---|---|---|---|---|---|---|"]
        for k, sd in (("ptq", sd_ptq), ("qat", sd_part2), ("both", sd_part2), ("added", sd_part2),
                      ("during", sd_part2)):
            sel = [p for p in pairs if pairs[p]["gF"] > thr_f] if k != "added" else list(pairs)
            v = [pairs[p][k] for p in sel]
            lines.append(f"| {k} | {sd:.4f}{' (sd of both)' if k in ('qat', 'added', 'during') else ''} | "
                         f"{sum(x > 2 * sd for x in v)} | {sum(abs(x) <= 2 * sd for x in v)} | "
                         f"{sum(x < -2 * sd for x in v)} | {sum(x > 0 for x in v)} | {sum(x < 0 for x in v)} |")
        # McNemar: sign kept under PTQ against under QAT, same pairs (raw signs, pairs with an FP32 gain)
        from scipy.stats import binomtest
        sel = [p for p in pairs if pairs[p]["gF"] > 0]
        kp = [pairs[p]["ptq"] > 0 for p in sel]
        kq = [pairs[p]["qat"] > 0 for p in sel]
        b_ = sum(x and not y for x, y in zip(kp, kq))
        c_ = sum(y and not x for x, y in zip(kp, kq))
        pm = binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else float("nan")
        lines += ["", f"McNemar, sign kept under PTQ against QAT over {len(sel)} pairs with an FP32 gain: kept by "
                  f"PTQ only {b_}, by QAT only {c_}, exact p = {pm:.4f} (pairs are clustered by student; read "
                  f"with the cluster intervals above)."]
        # RQ4 TOST on QAKD(B) - QAKD(A), margin 2 sd of the training noise of a Part II cell
        margin = 2 * sig.get("QAKD_B", sd_part2)
        mu, lo, hi = cluster_ci([per_student[s]["b_vs_a"] for s in students], level=0.90)
        lines += ["", f"RQ4 equivalence (TOST): QAKD(B) - QAKD(A) = {mu:+.4f}, 90% cluster interval "
                  f"[{lo:+.4f}, {hi:+.4f}], margin +/-{margin:.4f} (2 sd of QAKD(B) over seeds): "
                  f"{'equivalent' if -margin < lo and hi < margin else 'not shown equivalent'}."]

    # 5. teacher share and best-teacher agreement (H5)
    lines += ["", "## Teacher's share of variance and best-teacher agreement (H5)", "",
              "| matrix (students x teachers) | teacher share | student share |", "|---|---|---|"]
    for name, k in (("FP32 gain gF", "gF"), ("PTQ benefit", "ptq"), ("QAT benefit", "qat"),
                    ("QAKD(B) benefit (both)", "both"), ("QAKD(A) benefit (during)", "during")):
        M = [[pairs[(s, t)][k] for t in QJ.TEACHERS] for s in students]
        ts, ss = teacher_share(M)
        lines.append(f"| {name} | {ts:.2f} | {ss:.2f} |")
    lines += ["", "| INT8 model | students whose best teacher equals the FP32-best teacher |", "|---|---|"]
    best32 = {s: max(QJ.TEACHERS, key=lambda t: pairs[(s, t)]["B_fp32"]) for s in students}
    for name, cell in (("PTQ(B)", "PTQ_B"), ("QAT(B)", "QAT_B"), ("QAKD(B)", "QAKD_B")):
        agree = [s for s in students if max(QJ.TEACHERS, key=lambda t: pairs[(s, t)][cell]) == best32[s]]
        lines.append(f"| {name} | {len(agree)} of {len(students)} |")

    # 6. the three lowest floors (H2)
    low = sorted(students, key=lambda s: floor[s])[:3]
    lines += ["", f"## H2: the three lowest floors ({', '.join(low)})", ""]
    for k in ("ptq", "qat", "both"):
        v = [pairs[(s, t)][k] for s in low for t in QJ.TEACHERS]
        lines.append(f"- {k}: mean over the {len(v)} pairs {st.mean(v):+.4f}; positive in {sum(x > 0 for x in v)}; "
                     f"per student " + ", ".join(f"{per_student[s][k]:+.3f}" for s in low))

    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    with open(os.path.join(out_dir, "per_student.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    with open(os.path.join(out_dir, "per_pair.csv"), "w", newline="") as f:
        keys = list(next(iter(pairs.values())).keys())
        w = csv.DictWriter(f, fieldnames=["student", "teacher"] + keys)
        w.writeheader()
        for (s, t), e in sorted(pairs.items()):
            w.writerow(dict(student=s, teacher=t, **e))
    print(text)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
