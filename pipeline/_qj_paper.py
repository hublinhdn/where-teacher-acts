"""
where-teacher-acts: every number the paper reports, computed once from the measurement CSVs (read-only).

The paper generators (29_paper_tables.py, 29b_paper_figures.py) call results() and never compute a
number themselves, so a table, a figure and a sentence cannot disagree. The arithmetic is that of
26_analysis.py (the planned analysis), whose functions are reused unchanged:

  floor     PTQ(A), the undistilled student after post-training quantization
  gF        B_FP32 - A_FP32
  ptq       PTQ(B) - PTQ(A)          qat     QAT(B) - QAT(A)          rescue  QAT(A) - PTQ(A)
  during    QAKD(A) - QAT(A)         both    QAKD(B) - QAT(A)         added   QAKD(B) - QAT(B)
  b_vs_a    QAKD(B) - QAKD(A)        interaction  (QAKD(B) - QAKD(A)) - (QAT(B) - QAT(A))

Student means are over the six teachers; pooled means are over the seven students with a cluster
bootstrap over students (the pre-declared interval) and, as a small-sample check, a t interval on the
seven student means. Students are ordered by their MinMax floor (highest first) under both rules, so
rows line up across tables.

Added after the internal review (2026-10-13), all from the same CSVs: the teacher main effect (two-way
ANOVA without replication, teacher against the student by teacher residual); the proportional-damage
reference (A and B losing the same fraction of their FP32 mAP, so the kept share would equal PTQ(A)/A);
one noise model for the sign counts (PTQ benefit: training and calibration sd combined, where it was the
calibration sd alone before; fine-tuned contrasts: the training sd of Both, as before); the H2 McNemar on the three lowest floors, with raw signs and with the plan's rule
(sign changes beyond 2 sd); the validation fold as a replication (val_check); and the FP32 score of
each fine-tuned graph without rounding (fp32_diag), which separates further training from repair.
"""
import os
import sys
import csv
import math
import itertools
import statistics as st
import importlib.util

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
_spec = importlib.util.spec_from_file_location("qj_analysis", os.path.join(THIS, "26_analysis.py"))
AN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AN)      # also moves to the project root, as every pipeline script does

import numpy as np  # noqa: E402
import _qj_common as QJ  # noqa: E402

TAG, FOLD, FT = "main", "test", 40
RULES = {"mm": ("mAP_INT8_minmax", "minmax", "MinMax"), "pc": ("mAP_INT8_pct99.9", "pct", "Percentile 99.9")}
EFFECTS = AN.EFFECTS
CELLS = ("A_fp32", "B_fp32", "PTQ_A", "PTQ_B", "QAT_A", "QAT_B", "QAKD_A", "QAKD_B")

NAME = {"resnet18_tv": "ResNet-18", "squeezenet1_0_tv": "SqueezeNet-1.0", "repvgg_a0_timm": "RepVGG-A0",
        "mobilenet_v3_large_tv": "MobileNetV3-L", "ghostnet_100_timm": "GhostNet",
        "efficientnet_b0_timm": "EfficientNet-B0", "efficientnet_b1_timm": "EfficientNet-B1",
        "efficientnet_b5_timm": "EfficientNet-B5", "seresnext101_32x4d_timm": "SE-ResNeXt-101",
        "convnext_base_timm": "ConvNeXt-B", "convnext_small_timm": "ConvNeXt-S", "resnet101_tv": "ResNet-101",
        "densenet161_tv": "DenseNet-161"}
SHORT = {"resnet18_tv": "R18", "squeezenet1_0_tv": "SqN", "repvgg_a0_timm": "RepVGG",
         "mobilenet_v3_large_tv": "MNv3", "ghostnet_100_timm": "Ghost", "efficientnet_b0_timm": "EffB0",
         "efficientnet_b1_timm": "EffB1"}
# architecture facts for Table 1: depthwise convolutions, squeeze-and-excitation, activation
ARCH = {"resnet18_tv": ("no", "no", "ReLU"), "squeezenet1_0_tv": ("no", "no", "ReLU"),
        "repvgg_a0_timm": ("no", "no", "ReLU"), "mobilenet_v3_large_tv": ("yes", "yes", "h-swish, ReLU"),
        "ghostnet_100_timm": ("yes", "yes", "ReLU"), "efficientnet_b0_timm": ("yes", "yes", "SiLU"),
        "efficientnet_b1_timm": ("yes", "yes", "SiLU"), "efficientnet_b5_timm": ("yes", "yes", "SiLU"),
        "seresnext101_32x4d_timm": ("no", "yes", "ReLU"), "convnext_base_timm": ("yes", "no", "GELU"),
        "convnext_small_timm": ("yes", "no", "GELU"), "resnet101_tv": ("no", "no", "ReLU"),
        "densenet161_tv": ("no", "no", "ReLU")}


def path(*parts):
    return os.path.join(QJ.REPORTS, *parts)


def spearman_two_sided(x, y):
    """Spearman rho with the exact two-sided permutation p (|rho| at least as large)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)

    def rho(a, b):
        a, b = np.asarray(a) - np.mean(a), np.asarray(b) - np.mean(b)
        return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
    r0 = rho(rx, ry)
    perms = [rho(rx, [ry[i] for i in p]) for p in itertools.permutations(range(len(ry)))]
    return r0, sum(abs(v) >= abs(r0) - 1e-12 for v in perms) / len(perms)


def _rule(key, order=None):
    col, rule_arg, _ = RULES[key]
    V = AN.load(path("part2", TAG, f"eval_{FOLD}.csv"), col, FT)
    pairs = {(s, t): AN.pair_effects(V, s, t) for s in QJ.STUDENTS for t in QJ.TEACHERS}
    missing = [k for k, v in pairs.items() if v is None]
    if missing:
        raise RuntimeError(f"incomplete grid under {key}: {missing}")
    floor = {s: V[("A", s, None, QJ.FP32_SEED)]["int8"] for s in QJ.STUDENTS}
    # computations run in this rule's own floor order, exactly as 26_analysis.py does (the cluster
    # bootstrap draws indices, so the order fixes its resamples); `order` only sets the display order
    calc = sorted(QJ.STUDENTS, key=lambda s: -floor[s])
    order = order or calc
    stu = {s: {k: st.mean(pairs[(s, t)][k] for t in QJ.TEACHERS) for k in EFFECTS + CELLS} for s in calc}
    for s in calc:
        stu[s]["floor"] = floor[s]
        for k in ("ptq", "qat", "both", "during", "added"):
            stu[s][f"kept_{k}"] = stu[s][k] / stu[s]["gF"]
        for k in ("ptq", "qat", "both"):
            stu[s][f"beta_{k}"] = AN.beta([pairs[(s, t)][k] for t in QJ.TEACHERS],
                                          [pairs[(s, t)]["gF"] for t in QJ.TEACHERS])
        for k in CELLS[2:]:
            stu[s][f"{k}_min"] = min(pairs[(s, t)][k] for t in QJ.TEACHERS)
            stu[s][f"{k}_max"] = max(pairs[(s, t)][k] for t in QJ.TEACHERS)
    pooled = {}
    for k in EFFECTS + ("kept_ptq", "kept_qat", "kept_both"):
        mu, lo, hi = AN.cluster_ci([stu[s][k] for s in calc])
        pooled[k] = {"mean": mu, "lo": lo, "hi": hi}
    fl = [floor[s] for s in calc]
    rel = [floor[s] / stu[s]["A_fp32"] for s in calc]
    rho = {}
    for k in EFFECTS + ("kept_ptq", "kept_qat", "kept_both"):
        y = [stu[s][k] for s in calc]
        r, p_less = AN.spearman_exact(fl, y, "less")
        _, p_greater = AN.spearman_exact(fl, y, "greater")
        _, p_two = spearman_two_sided(fl, y)
        r_rel, p_rel = spearman_two_sided(rel, y)
        rho[k] = {"rho": r, "p_less": p_less, "p_greater": p_greater, "p_two": p_two,
                  "rho_rel": r_rel, "p_two_rel": p_rel}

    # noise: whole-chain training sd from the seed repeats, calibration sd from B2
    sig, used = AN.chain_noise(V, QJ.SEED_REPEAT_STUDENTS)
    sig_cal = AN.calib_noise(path("bench", TAG, "calib_noise.csv"), rule_arg)
    thr_f = max(0.02, 2 * sig["gF"])
    signs = {}
    # one noise model: the PTQ benefit carries training and calibration noise (combined); the contrasts between
    # fine-tuned models carry the training sd of Both (their calibration noise was not measured)
    sd_ptq = math.hypot(sig["ptq"], sig_cal) if sig_cal is not None else sig["ptq"]
    for k, sd in (("ptq", sd_ptq), ("qat", sig["both"]), ("during", sig["both"]), ("both", sig["both"]),
                  ("added", sig["both"])):
        sel = list(pairs) if k == "added" else [p for p in pairs if pairs[p]["gF"] > thr_f]
        v = [pairs[p][k] for p in sel]
        signs[k] = {"sd": sd, "n": len(v), "kept": sum(x > 2 * sd for x in v), "lost": sum(abs(x) <= 2 * sd for x in v),
                    "flipped": sum(x < -2 * sd for x in v), "pos": sum(x > 0 for x in v), "neg": sum(x < 0 for x in v)}
    from scipy.stats import binomtest
    sel = [p for p in pairs if pairs[p]["gF"] > 0]
    kp = [pairs[p]["ptq"] > 0 for p in sel]
    kq = [pairs[p]["qat"] > 0 for p in sel]
    b_ = sum(x and not y for x, y in zip(kp, kq))
    c_ = sum(y and not x for x, y in zip(kp, kq))
    mcnemar = {"n": len(sel), "ptq_only": b_, "qat_only": c_,
               "p": binomtest(min(b_, c_), b_ + c_, 0.5).pvalue if b_ + c_ else float("nan")}
    margin = 2 * sig["QAKD_B"]
    mu90, lo90, hi90 = AN.cluster_ci([stu[s]["b_vs_a"] for s in calc], level=0.90)
    tost = {"margin": margin, "mean": mu90, "lo90": lo90, "hi90": hi90, "equivalent": -margin < lo90 and hi90 < margin}

    shares = {}
    for k in ("gF", "ptq", "qat", "during", "both", "added"):
        ts, ss = AN.teacher_share([[pairs[(s, t)][k] for t in QJ.TEACHERS] for s in calc])
        shares[k] = {"teacher": ts, "student": ss}
    best32 = {s: max(QJ.TEACHERS, key=lambda t: pairs[(s, t)]["B_fp32"]) for s in calc}
    agree = {}
    for cell in ("PTQ_B", "QAT_B", "QAKD_B"):
        best = {s: max(QJ.TEACHERS, key=lambda t: pairs[(s, t)][cell]) for s in calc}
        agree[cell] = {"n": sum(best[s] == best32[s] for s in calc), "best": best}
    low = sorted(calc, key=lambda s: floor[s])[:3]
    h2 = {k: {"mean": st.mean(pairs[(s, t)][k] for s in low for t in QJ.TEACHERS),
              "pos": sum(pairs[(s, t)][k] > 0 for s in low for t in QJ.TEACHERS), "n": 3 * len(QJ.TEACHERS)}
          for k in ("ptq", "qat", "both")}
    h2_qat_ci = AN.cluster_ci([stu[s]["qat"] for s in low])
    # the H2 McNemar as the plan states it: the three lowest floors, raw signs and sign changes beyond 2 sd
    sel2 = [(s, t) for s in low for t in QJ.TEACHERS]
    raw = [(pairs[p]["ptq"] > 0, pairs[p]["qat"] > 0) for p in sel2]
    b2, c2 = sum(x and not y for x, y in raw), sum(y and not x for x, y in raw)
    flip = [(pairs[p]["ptq"] < -2 * sd_ptq, pairs[p]["qat"] < -2 * sig["both"]) for p in sel2]
    b3, c3 = sum(x and not y for x, y in flip), sum(y and not x for x, y in flip)
    mcnemar_h2 = {"n": len(sel2), "ptq_only": b2, "qat_only": c2,
                  "p": binomtest(min(b2, c2), b2 + c2, 0.5).pvalue if b2 + c2 else float("nan"),
                  "flip_ptq_only": b3, "flip_qat_only": c3,
                  "flip_p": binomtest(min(b3, c3), b3 + c3, 0.5).pvalue if b3 + c3 else float("nan")}
    # small-sample check of the pre-declared cluster bootstrap: t interval on the seven student means
    from scipy.stats import t as tdist, f as fdist
    tci = {}
    for k in EFFECTS:
        x = [stu[s][k] for s in calc]
        h = tdist.ppf(0.975, len(x) - 1) * st.stdev(x) / math.sqrt(len(x))
        tci[k] = {"mean": st.mean(x), "lo": st.mean(x) - h, "hi": st.mean(x) + h}
    # teacher main effect: two-way ANOVA without replication, teacher against the residual (df 5 and 30)
    anova = {}
    for k in ("gF", "ptq", "qat", "during", "both", "added"):
        M = np.array([[pairs[(s, t)][k] for t in QJ.TEACHERS] for s in calc])
        g_, tm, sm = M.mean(), M.mean(axis=0), M.mean(axis=1)
        ss_t = M.shape[0] * ((tm - g_) ** 2).sum()
        ss_e = ((M - sm[:, None] - tm[None, :] + g_) ** 2).sum()
        dft, dfe = M.shape[1] - 1, (M.shape[0] - 1) * (M.shape[1] - 1)
        Fv = (ss_t / dft) / (ss_e / dfe)
        anova[k] = {"F": float(Fv), "p": float(1 - fdist.cdf(Fv, dft, dfe)), "dft": dft, "dfe": dfe,
                    "spread": float(tm.max() - tm.min()),
                    "tmeans": {t: float(v) for t, v in zip(QJ.TEACHERS, tm)}}
    # proportional damage: if A and B lost the same fraction of their FP32 mAP, kept share = PTQ(A)/A
    for s in calc:
        stu[s]["ret_A"] = floor[s] / stu[s]["A_fp32"]
        stu[s]["ret_B"] = st.mean(pairs[(s, t)]["PTQ_B"] / pairs[(s, t)]["B_fp32"] for t in QJ.TEACHERS)
    ret = {"pairs_b_more": sum(pairs[(s, t)]["PTQ_B"] / pairs[(s, t)]["B_fp32"] >
                               pairs[(s, t)]["PTQ_A"] / pairs[(s, t)]["A_fp32"] for s in calc for t in QJ.TEACHERS),
           "stu_b_more": sum(stu[s]["ret_B"] > stu[s]["ret_A"] for s in calc),
           "stu_kept_above_null": sum(stu[s]["kept_ptq"] > stu[s]["ret_A"] for s in calc),
           "rho_null": spearman_two_sided(fl, rel),
           "rho_five": spearman_two_sided([floor[s] for s in calc[:5]], [stu[s]["kept_ptq"] for s in calc[:5]])}
    # leave-one-student-out rho of Added on the floor (H3's trend)
    loo = []
    for i in range(len(calc)):
        xs = [floor[s] for j, s in enumerate(calc) if j != i]
        ys = [stu[s]["added"] for j, s in enumerate(calc) if j != i]
        loo.append(spearman_two_sided(xs, ys)[0])
    # recovered share of the damage, (QAT(A) - PTQ(A)) / (A - PTQ(A)), and QAT(A) itself, against the floor
    recovered = {s: (stu[s]["QAT_A"] - floor[s]) / (stu[s]["A_fp32"] - floor[s]) for s in calc}
    rho_qata = spearman_two_sided(fl, [stu[s]["QAT_A"] for s in calc])
    rho_recovered = spearman_two_sided(fl, [recovered[s] for s in calc])
    # the training sd per seed-repeat student (the pooled sd above averages these variances)
    sd_stu = {}
    t_ = QJ.SEED_REPEAT_TEACHER
    for s in used:
        vals = {"both": [], "QAKD_B": [], "QAT_A": []}
        for seed in (QJ.FP32_SEED,) + tuple(QJ.EXTRA_SEEDS):
            qa, kb = V[("qat_a", s, None, seed)]["int8"], V[("qakd_b", s, t_, seed)]["int8"]
            vals["both"].append(kb - qa)
            vals["QAKD_B"].append(kb)
            vals["QAT_A"].append(qa)
        sd_stu[s] = {k: st.stdev(v) for k, v in vals.items()}
    return {"order": order, "pairs": pairs, "stu": stu, "pooled": pooled, "rho": rho, "sig": sig,
            "sig_cal": sig_cal, "seed_students": used, "thr_f": thr_f, "signs": signs, "mcnemar": mcnemar,
            "tost": tost, "shares": shares, "best32": best32, "agree": agree, "low": low, "h2": h2, "V": V,
            "mcnemar_h2": mcnemar_h2, "h2_qat_ci": h2_qat_ci, "sd_ptq_total": sd_ptq, "tci": tci,
            "anova": anova, "ret": ret, "loo_added": loo, "recovered": recovered, "rho_qata": rho_qata,
            "rho_recovered": rho_recovered, "sd_stu": sd_stu}


def val_check(key):
    """The validation fold as a replication of the test results (checkpoints were selected on it, alike for
    every arm): floors, per-student effects and the H3 trend under one calibration rule."""
    col = RULES[key][0]
    V = AN.load(path("part2", TAG, "eval_val.csv"), col, FT)
    pairs = {(s, t): AN.pair_effects(V, s, t) for s in QJ.STUDENTS for t in QJ.TEACHERS}
    if any(v is None for v in pairs.values()):
        return None
    floor = {s: V[("A", s, None, QJ.FP32_SEED)]["int8"] for s in QJ.STUDENTS}
    calc = sorted(QJ.STUDENTS, key=lambda s: -floor[s])
    stu = {s: {k: st.mean(pairs[(s, t)][k] for t in QJ.TEACHERS) for k in EFFECTS + CELLS} for s in calc}
    r, p_less = AN.spearman_exact([floor[s] for s in calc], [stu[s]["added"] for s in calc], "less")
    return {"floor": floor, "stu": stu, "order": calc, "added_rho": r, "added_pless": p_less,
            "means": {k: st.mean(stu[s][k] for s in calc) for k in EFFECTS}}


def fp32_diag(V):
    """FP32 score of each fine-tuned graph without rounding (Clip nodes kept), against its FP32 start: the
    part of a fine-tuned model's score that comes from further training and not from repairing the quantizer."""
    out = {}
    for s in QJ.STUDENTS:
        a = V[("A", s, None, QJ.FP32_SEED)]["fp32"]
        qa = V[("qat_a", s, None, QJ.FP32_SEED)]["fp32"]
        qb = [V[("qat_b", s, t, QJ.FP32_SEED)]["fp32"] - V[("B", s, t, QJ.FP32_SEED)]["fp32"] for t in QJ.TEACHERS]
        kb = [V[("qakd_b", s, t, QJ.FP32_SEED)]["fp32"] - V[("B", s, t, QJ.FP32_SEED)]["fp32"] for t in QJ.TEACHERS]
        out[s] = {"A": a, "qata_fp32": qa, "qata_gain": qa - a, "qatb_gain": st.mean(qb),
                  "qatb_pos": sum(x > 0 for x in qb), "qakdb_gain": st.mean(kb)}
    return out



def seed_values(R, rule_key):
    """Per seed and student, QAT(A), QAKD(B) (ConvNeXt-S) and their difference, for the seed figure."""
    t = QJ.SEED_REPEAT_TEACHER
    out = {}
    for s in QJ.SEED_REPEAT_STUDENTS:
        for seed in (QJ.FP32_SEED,) + tuple(QJ.EXTRA_SEEDS):
            qa, kb = R["V"].get(("qat_a", s, None, seed)), R["V"].get(("qakd_b", s, t, seed))
            if qa and kb:
                out[(s, seed)] = {"QAT_A": qa["int8"], "QAKD_B": kb["int8"]}
    return out


def anchor(R):
    """The exploratory anchor control (declared 2026-10-13 before running in the dated plan,
    docs/PLAN_AND_LOG.md): QANC(B), the fine-tuned B whose cosine term pulls toward B's own frozen FP32
    model, against QAT(B) and QAKD(B) of the same start (B distilled from ConvNeXt-S), per student, under
    each rule and fold. The declared reading uses Gap = QAKD(B) - QANC(B) with a margin of twice the
    training sd of one QAKD(B) model (as RQ4):
    (a) the 90% t interval of the mean Gap inside +-margin, anchoring matches the teacher; (b) the interval
    above 0 and the mean beyond the margin, the teacher adds more than anchoring; (c) otherwise undecided.
    Returns {} until the measurements exist."""
    from scipy.stats import t as tdist
    out = {}
    tch = QJ.ANCHOR_TEACHER
    for fold in ("test", "val"):
        fa = path("part2", "anchor", f"eval_{fold}.csv")
        if not os.path.exists(fa):
            continue
        for key, (col, _, _) in RULES.items():
            Va = AN.load(fa, col, FT)
            Vm = AN.load(path("part2", TAG, f"eval_{fold}.csv"), col, FT)
            rows = {}
            for s in QJ.STUDENTS:
                a = Va.get(("qanc_b", s, tch, QJ.FP32_SEED))
                qb, kb = Vm.get(("qat_b", s, tch, QJ.FP32_SEED)), Vm.get(("qakd_b", s, tch, QJ.FP32_SEED))
                fl = Vm.get(("A", s, None, QJ.FP32_SEED))
                if None in (a, qb, kb, fl):
                    continue
                rows[s] = {"QANC_B": a["int8"], "QAT_B": qb["int8"], "QAKD_B": kb["int8"], "floor": fl["int8"],
                           "anchor": a["int8"] - qb["int8"], "gap": kb["int8"] - a["int8"],
                           "added": kb["int8"] - qb["int8"], "qanc_fp32": a["fp32"]}
            res = {"rows": rows, "complete": len(rows) == len(QJ.STUDENTS)}
            if res["complete"]:
                order = sorted(rows, key=lambda s: -rows[s]["floor"])
                gaps = [rows[s]["gap"] for s in order]
                n, m = len(gaps), st.mean(gaps)
                h = tdist.ppf(0.95, n - 1) * st.stdev(gaps) / math.sqrt(n)
                sig = R[key]["sig"]["QAKD_B"]
                margin = 2 * sig
                lo, hi = m - h, m + h
                verdict = "a" if (-margin < lo and hi < margin) else ("b" if (lo > 0 and m > margin) else "c")
                anc = [rows[s]["anchor"] for s in order]
                add = [rows[s]["added"] for s in order]
                res.update({"order": order, "gap_mean": m, "gap_lo90": lo, "gap_hi90": hi, "margin": margin,
                            "verdict": verdict, "anchor_mean": st.mean(anc), "added_mean": st.mean(add),
                            "n_beyond": sum(abs(g) > margin for g in gaps),
                            "rho_anchor_floor": spearman_two_sided([rows[s]["floor"] for s in order], anc),
                            "rho_gap_floor": spearman_two_sided([rows[s]["floor"] for s in order], gaps)})
            out[(fold, key)] = res
    return out


def _tci(xs, level):
    """Mean of xs and its t interval at the given two-sided level."""
    from scipy.stats import t as tdist
    n, m = len(xs), st.mean(xs)
    h = tdist.ppf(0.5 + level / 2, n - 1) * st.stdev(xs) / math.sqrt(n)
    return m, m - h, m + h


ED_MARGIN = 0.02     # declared 2026-10-14: about a third of the FP32 gain
ED_KEYS = ("timing", "before_eq", "during_eq", "ft_gain", "ft_ptq", "none_vs_qata", "kd40_gain", "ret",
           "gap_a", "anchor_a", "gap_a_cnxs")


def ed(R):
    """The exploratory equal-dose comparison and the anchor on A (declared 2026-10-14 before running in the
    dated plan, docs/PLAN_AND_LOG.md). Every run starts from A (seed 42) and spends 40 epochs in FP32,
    without (ed_ft, "FT40") or with the teacher ConvNeXt-S (ed_kd, "KD40"), then 40 epochs of
    quantization-aware fine-tuning: ed_none (QAT from FT40), ed_before (QAT from KD40) and ed_during
    (QAKD from FT40). qanc_a is QAT of A for 40 epochs
    with A's own frozen FP32 model in the cosine term. Three fine-tuning seeds; per student the seed mean,
    then the mean over the seven students with t intervals (the student is the unit). Declared readings,
    with a margin of ED_MARGIN mAP:
      (1) Timing = ed_before - ed_during: "timing" if the 95% t interval excludes 0 (its sign says which
          stage), "negligible" if the 90% t interval lies inside the margin, "undecided" otherwise;
      (2) Before_eq = ed_before - ed_none and During_eq = ed_during - ed_none;
      (3) further training: FT40 - A in FP32, PTQ(FT40) - PTQ(A), and ed_none - QAT(A) (grid, seed 42);
      (4) retention at equal length: PTQ(KD40)/KD40 - PTQ(FT40)/FT40 (seed mean of the per-seed difference),
          with the count of positive students and a sign test;
      (5) QANC(A): Gap_A = QAKD(A) (grid, mean over the six teachers) - QANC(A) and Anchor_A = QANC(A) -
          QAT(A) (grid): "anchoring" if the 90% t interval of Gap_A lies inside the margin and the 95% t
          interval of Anchor_A excludes 0, "teacher" if the 95% t interval of Gap_A lies above 0,
          "undecided" otherwise.
    Returns {} until the measurements exist; a fold and rule with missing runs is marked incomplete."""
    from scipy.stats import binomtest
    T, S0 = QJ.ED_TEACHER, QJ.ED_START_SEED
    arms = (("ft", "ed_ft", None), ("kd", "ed_kd", T), ("none", "ed_none", None), ("before", "ed_before", T),
            ("during", "ed_during", T), ("qanc", "qanc_a", None))
    out = {}
    for fold in ("test", "val"):
        fe = path("part2", "ed", f"eval_{fold}.csv")
        if not os.path.exists(fe):
            continue
        for key, (col, _, _) in RULES.items():
            Ve = AN.load(fe, col, FT)
            Vm = AN.load(path("part2", TAG, f"eval_{fold}.csv"), col, FT)
            rows, missing = {}, []
            for s in QJ.STUDENTS:
                per = []
                for seed in QJ.ED_SEEDS:
                    g = {k: Ve.get((arm, s, tt, seed)) for k, arm, tt in arms}
                    if None in g.values() or g["ft"]["fp32"] is None or g["kd"]["fp32"] is None:
                        missing.append((s, seed))
                        continue
                    per.append(g)
                a, qa = Vm.get(("A", s, None, S0)), Vm.get(("qat_a", s, None, S0))
                ka = [Vm.get(("qakd_a", s, t, S0)) for t in QJ.TEACHERS]
                ka_t = Vm.get(("qakd_a", s, T, S0))           # the control's own teacher, ConvNeXt-S
                if len(per) != len(QJ.ED_SEEDS) or None in [a, qa] + ka:
                    continue

                def m(f):
                    return st.mean(f(g) for g in per)
                v = {k: m(lambda g, k=k: g[k]["int8"]) for k, _, _ in arms}
                ft32, kd32 = m(lambda g: g["ft"]["fp32"]), m(lambda g: g["kd"]["fp32"])
                qakd_a = st.mean(x["int8"] for x in ka)
                rows[s] = {
                    "floor": a["int8"], "A_fp32": a["fp32"], "QAT_A": qa["int8"], "QAKD_A": qakd_a,
                    "FT40": ft32, "KD40": kd32, "PTQ_FT40": v["ft"], "PTQ_KD40": v["kd"],
                    "none": v["none"], "before": v["before"], "during": v["during"], "QANC_A": v["qanc"],
                    "timing": v["before"] - v["during"],
                    "timing_sd": st.stdev(g["before"]["int8"] - g["during"]["int8"] for g in per),
                    "before_eq": v["before"] - v["none"], "during_eq": v["during"] - v["none"],
                    "ft_gain": ft32 - a["fp32"], "ft_ptq": v["ft"] - a["int8"], "none_vs_qata": v["none"] - qa["int8"],
                    "kd40_gain": kd32 - ft32,
                    "ret": m(lambda g: g["kd"]["int8"] / g["kd"]["fp32"] - g["ft"]["int8"] / g["ft"]["fp32"]),
                    "gap_a": qakd_a - v["qanc"], "anchor_a": v["qanc"] - qa["int8"],
                    "QAKD_A_cnxs": ka_t["int8"], "gap_a_cnxs": ka_t["int8"] - v["qanc"],
                    # seed variances of the three fine-tuned arms, for the pooled training noise of this design
                    "var_seed": [st.variance(g[k]["int8"] for g in per) for k in ("none", "before", "during")]}
            res = {"rows": rows, "missing": missing, "complete": len(rows) == len(QJ.STUDENTS)}
            if res["complete"]:
                order = sorted(rows, key=lambda s: -rows[s]["floor"])
                summ = {}
                for k in ED_KEYS:
                    xs = [rows[s][k] for s in order]
                    mean, lo95, hi95 = _tci(xs, 0.95)
                    _, lo90, hi90 = _tci(xs, 0.90)
                    npos = sum(x > 0 for x in xs)
                    summ[k] = {"mean": mean, "lo95": lo95, "hi95": hi95, "lo90": lo90, "hi90": hi90,
                               "npos": npos, "p_sign": binomtest(npos, len(xs)).pvalue}
                tm, ga, an = summ["timing"], summ["gap_a"], summ["anchor_a"]
                if tm["lo95"] > 0 or tm["hi95"] < 0:
                    timing_verdict = "timing"
                elif -ED_MARGIN < tm["lo90"] and tm["hi90"] < ED_MARGIN:
                    timing_verdict = "negligible"
                else:
                    timing_verdict = "undecided"
                if -ED_MARGIN < ga["lo90"] and ga["hi90"] < ED_MARGIN and (an["lo95"] > 0 or an["hi95"] < 0):
                    anchor_verdict = "anchoring"
                elif ga["lo95"] > 0:
                    anchor_verdict = "teacher"
                else:
                    anchor_verdict = "undecided"
                res.update({"order": order, "summ": summ, "margin": ED_MARGIN, "timing_verdict": timing_verdict,
                            "anchor_verdict": anchor_verdict,
                            "sd_seed": math.sqrt(st.mean(x for s in order for x in rows[s]["var_seed"])),
                            "rho_timing_floor": spearman_two_sided([rows[s]["floor"] for s in order],
                                                                   [rows[s]["timing"] for s in order])})
            out[(fold, key)] = res
    return out


SENS_TIE = 0.01      # declared 2026-10-16: retention differences below this count as ties
SENS_FLOOR = 0.1     # declared 2026-10-16: a floor below this is collapsed
SENS_MARGIN = 0.02   # declared 2026-10-16: equivalence margin of D1 (ANC40)
SENS_CONFIGS = (("minmax", "MinMax"), ("pct99.9", "P99.9"), ("entropy2048", "Entropy (2048)"),
                ("pct99.99", "P99.99"), ("pct99.999", "P99.999"),
                ("minmax_headfp", "MinMax, FP32 head"), ("pct99.9_headfp", "P99.9, FP32 head"))


def sens(R):
    """The sensitivity analyses declared 2026-10-16 before running (the dated plan, docs/PLAN_AND_LOG.md;
    exploratory), each None until its measurements exist; the teacher is ConvNeXt-S and the student is the
    unit.
      strong  A and B of seed 42 recalibrated with stronger or different post-training settings
              (39_ptq_sensitivity.py --part strong); the two main rules come from the main benchmark. Per
              configuration and fold: the floor PTQ(A), the PTQ benefit PTQ(B) - PTQ(A), the kept shares
              PTQ(X)/X, the number of floors below SENS_FLOOR, and the students whose B keeps a larger share
              than A (differences below SENS_TIE are ties). Declared reading: the two regimes are a property of
              per-tensor MinMax alone if no other configuration leaves a floor below SENS_FLOOR.
      calib   calibration noise of the fine-tuned models QAT(A), QAT(B) and QAKD(B): sd over the five
              calibration seeds (123 from the main benchmark, 223 to 523 from --part calib), pooled as the root
              mean variance; the total noise of a difference between two fine-tuned models treats the training
              sd of Both and the calibration sd as independent.
      anc     ANC40 (ed_anc): 40 FP32 epochs from A with A's own frozen FP32 model in the cosine term, seeds
              42, 123 and 223. With Ret(X) = PTQ(X)/X, D1 = Ret(KD40) - Ret(ANC40) and D0 = Ret(ANC40) -
              Ret(FT40), seed means per student, t intervals over the students: "teacher" if the 95% interval of
              D1 lies above 0, "cosine term" if the 90% interval of D1 lies inside +-SENS_MARGIN and the 95%
              interval of D0 lies above 0, "undecided" otherwise.
      qata    QAT(A) with two more fine-tuning seeds (qat_a_fs, 123 and 223, from A of seed 42): the sd of
              QAT(A) per student over the three seeds, and Rescue, QAT benefit, During and Both recomputed with
              QAT(A) as the mean of the three seeds, against the grid's values (cluster bootstrap and t)."""
    T, S0 = QJ.SEED_REPEAT_TEACHER, QJ.FP32_SEED
    out = {"strong": None, "calib": None, "anc": None, "qata": None}
    Vm = {(fold, key): AN.load(path("part2", TAG, f"eval_{fold}.csv"), RULES[key][0], FT)
          for fold in ("test", "val") for key in RULES}

    # (1) stronger post-training quantization
    f_strong = path("bench", "sens", "ptq_sensitivity.csv")
    if os.path.exists(f_strong):
        rows = [r for r in csv.DictReader(open(f_strong)) if r["status"] == "SUCCESS" and r["part"] == "strong"]
        val = {(r["fold"], r["config"], r["student"], r["checkpoint"]): float(r["mAP_INT8"]) for r in rows}
        res = {}
        for fold in ("test", "val"):
            per_cfg = {}
            for cfg, label in SENS_CONFIGS:
                stu = {}
                for s in QJ.STUDENTS:
                    a32 = Vm[(fold, "mm")][("A", s, None, S0)]["fp32"]
                    b32 = Vm[(fold, "mm")][("B", s, T, S0)]["fp32"]
                    if cfg in ("minmax", "pct99.9"):
                        key = "mm" if cfg == "minmax" else "pc"
                        pa = Vm[(fold, key)][("A", s, None, S0)]["int8"]
                        pb = Vm[(fold, key)][("B", s, T, S0)]["int8"]
                    else:
                        pa, pb = val.get((fold, cfg, s, "A")), val.get((fold, cfg, s, "B"))
                    if pa is None or pb is None:
                        continue
                    stu[s] = {"floor": pa, "ptqb": pb, "benefit": pb - pa, "ret_a": pa / a32, "ret_b": pb / b32,
                              "dret": pb / b32 - pa / a32}
                if len(stu) != len(QJ.STUDENTS):
                    continue
                d = [stu[s]["dret"] for s in QJ.STUDENTS]
                per_cfg[cfg] = {"label": label, "stu": stu,
                                "n_collapsed": sum(stu[s]["floor"] < SENS_FLOOR for s in QJ.STUDENTS),
                                "benefit_mean": st.mean(stu[s]["benefit"] for s in QJ.STUDENTS),
                                "ret_b_more": sum(x >= SENS_TIE for x in d), "ret_tie": sum(abs(x) < SENS_TIE for x in d),
                                "ret_a_more": sum(x <= -SENS_TIE for x in d)}
            if "minmax" in per_cfg and len(per_cfg) == len(SENS_CONFIGS):
                others = [c for c, _ in SENS_CONFIGS if c != "minmax"]
                collapsed_elsewhere = {c: per_cfg[c]["n_collapsed"] for c in others if per_cfg[c]["n_collapsed"]}
                res[fold] = {"cfg": per_cfg, "collapsed_elsewhere": collapsed_elsewhere,
                             "minmax_only": not collapsed_elsewhere}
            elif per_cfg:
                res[fold] = {"cfg": per_cfg, "incomplete": True}
        # reproduction check: the main rule recomputed in the sensitivity run against the main benchmark
        rep = [abs(val[("test", "minmax", s, ck)] - Vm[("test", "mm")][(c, s, tt, S0)]["int8"])
               for s in QJ.STUDENTS for ck, c, tt in (("A", "A", None), ("B", "B", T))
               if ("test", "minmax", s, ck) in val]
        out["strong"] = {"folds": res, "repro_maxdiff": max(rep) if rep else None, "repro_n": len(rep)}

    # (2) calibration noise of the fine-tuned models
    f_cal = path("bench", "sens", "calib_noise_ft.csv")
    if os.path.exists(f_cal):
        rows = [r for r in csv.DictReader(open(f_cal)) if r["status"] == "SUCCESS" and r["part"] == "calib"]
        val = {(r["student"], r["checkpoint"], r["config"]): float(r["mAP_INT8"]) for r in rows}
        res = {}
        for key, cfgname in (("mm", "minmax"), ("pc", "pct99.9")):
            per, variances = {}, []
            for s in QJ.STUDENTS:
                for ck, cell, tt in (("QAT_A", "qat_a", None), ("QAT_B", "qat_b", T), ("QAKD_B", "qakd_b", T)):
                    xs = [Vm[("test", key)][(cell, s, tt, S0)]["int8"]] + [
                        val[(s, ck, f"{cfgname}_seed{c}")] for c in QJ.CALIB_SEEDS[1:] if (s, ck, f"{cfgname}_seed{c}") in val]
                    if len(xs) == len(QJ.CALIB_SEEDS):
                        per[(s, ck)] = {"values": xs, "sd": st.stdev(xs)}
                        variances.append(st.variance(xs))
            if len(per) == 3 * len(QJ.STUDENTS):
                sd_cal = math.sqrt(st.mean(variances))
                sd_both = R[key]["sig"]["both"]
                sd_tot = math.sqrt(sd_both ** 2 + 2 * sd_cal ** 2)
                # the pair counts of the noise table recomputed with the combined noise (descriptive)
                pairs, thr_f = R[key]["pairs"], R[key]["thr_f"]
                signs = {}
                for k in ("qat", "during", "both", "added"):
                    sel = list(pairs) if k == "added" else [p_ for p_ in pairs if pairs[p_]["gF"] > thr_f]
                    v = [pairs[p_][k] for p_ in sel]
                    signs[k] = {"n": len(v), "kept": sum(x > 2 * sd_tot for x in v),
                                "lost": sum(abs(x) <= 2 * sd_tot for x in v), "flipped": sum(x < -2 * sd_tot for x in v)}
                res[key] = {"per": per, "sd_cal": sd_cal, "sd_max": max(v["sd"] for v in per.values()),
                            "sd_max_at": max(per, key=lambda k: per[k]["sd"]),
                            "by_model": {ck: math.sqrt(st.mean(per[(s, ck)]["sd"] ** 2 for s in QJ.STUDENTS))
                                         for ck in ("QAT_A", "QAT_B", "QAKD_B")},
                            "sd_both": sd_both, "sd_total_diff": sd_tot, "signs_total": signs,
                            "sd_ptq_cal": R[key]["sig_cal"]}
        out["calib"] = res or None

    # (3) and (4) from the sensitivity runs
    res_anc, res_qa = {}, {}
    for fold in ("test", "val"):
        fs, fe = path("part2", "sens", f"eval_{fold}.csv"), path("part2", "ed", f"eval_{fold}.csv")
        if not os.path.exists(fs):
            continue
        for key, (col, _, _) in RULES.items():
            Vs, Ve, V = AN.load(fs, col, FT), AN.load(fe, col, FT), Vm[(fold, key)]
            # (3) ANC40
            rows = {}
            for s in QJ.STUDENTS:
                per = []
                for seed in QJ.ED_SEEDS:
                    g = {"ft": Ve.get(("ed_ft", s, None, seed)), "kd": Ve.get(("ed_kd", s, T, seed)),
                         "anc": Vs.get(("ed_anc", s, None, seed))}
                    if None in g.values() or None in [g[k]["fp32"] for k in g]:
                        break
                    per.append({k: g[k]["int8"] / g[k]["fp32"] for k in g} |
                               {f"{k}32": g[k]["fp32"] for k in g} | {f"{k}8": g[k]["int8"] for k in g})
                else:
                    rows[s] = {k: st.mean(x[k] for x in per) for k in per[0]}
                    rows[s]["d1"] = rows[s]["kd"] - rows[s]["anc"]
                    rows[s]["d0"] = rows[s]["anc"] - rows[s]["ft"]
                    rows[s]["anc_gain32"] = rows[s]["anc32"] - rows[s]["ft32"]
            if len(rows) == len(QJ.STUDENTS):
                summ = {}
                for k in ("d1", "d0", "anc_gain32"):
                    xs = [rows[s][k] for s in QJ.STUDENTS]
                    m, lo95, hi95 = _tci(xs, 0.95)
                    _, lo90, hi90 = _tci(xs, 0.90)
                    summ[k] = {"mean": m, "lo95": lo95, "hi95": hi95, "lo90": lo90, "hi90": hi90,
                               "npos": sum(x > 0 for x in xs),
                               # with the study's tie rule: differences below SENS_TIE count as ties
                               "npos_clear": sum(x >= SENS_TIE for x in xs), "nties": sum(abs(x) < SENS_TIE for x in xs),
                               "ties": [s for s, x in zip(QJ.STUDENTS, xs) if abs(x) < SENS_TIE]}
                d1, d0 = summ["d1"], summ["d0"]
                if d1["lo95"] > 0:
                    verdict = "teacher"
                elif -SENS_MARGIN < d1["lo90"] and d1["hi90"] < SENS_MARGIN and d0["lo95"] > 0:
                    verdict = "cosine term"
                else:
                    verdict = "undecided"
                res_anc[(fold, key)] = {"rows": rows, "summ": summ, "verdict": verdict}
            # (4) QAT(A) with three fine-tuning seeds
            qa = {}
            for s in QJ.STUDENTS:
                xs = [V.get(("qat_a", s, None, S0))] + [Vs.get(("qat_a_fs", s, None, seed)) for seed in QJ.FS_SEEDS]
                if None not in xs:
                    qa[s] = [x["int8"] for x in xs]
            if len(qa) == len(QJ.STUDENTS):
                # the students in this rule's floor order, as in _rule(): the cluster bootstrap draws indices,
                # so the same order gives the grid's values the same intervals as the pooled table
                floor = {s: V[("A", s, None, S0)]["int8"] for s in QJ.STUDENTS}
                calc = sorted(QJ.STUDENTS, key=lambda s: -floor[s])
                pe = {s: [AN.pair_effects(V, s, t) for t in QJ.TEACHERS] for s in calc}
                shift = {s: st.mean(qa[s]) - qa[s][0] for s in calc}
                # the sign with which QAT(A) enters each contrast; the Interaction,
                # (QAKD(B) - QAKD(A)) - (QAT(B) - QAT(A)), contains it too (added after the declared reading)
                signs_qa = {"rescue": 1, "qat": -1, "during": -1, "both": -1, "interaction": 1}
                eff = {}
                for name, sg in signs_qa.items():
                    orig = [st.mean(p[name] for p in pe[s]) for s in calc]
                    new = [o + sg * shift[s] for o, s in zip(orig, calc)]
                    rec = {}
                    for tag, xs in (("orig", orig), ("new", new)):
                        m, lo, hi = AN.cluster_ci(xs)
                        _, tlo, thi = _tci(xs, 0.95)
                        rec[tag] = {"mean": m, "lo": lo, "hi": hi, "tlo": tlo, "thi": thi,
                                    "npos": sum(x > 0 for x in xs), "nneg": sum(x < 0 for x in xs),
                                    "per": dict(zip(calc, xs))}
                    eff[name] = rec
                low3 = sorted(QJ.STUDENTS, key=lambda s: V[("A", s, None, S0)]["int8"])[:3]   # H2's three lowest floors
                h2 = [eff["qat"]["new"]["per"][s] for s in low3]
                m3, lo3, hi3 = _tci(h2, 0.95)
                # derived from the QAT benefit: its share of the FP32 gain and that share against the floor,
                # QAT(A) itself against the floor, and the rank of each student's QAT benefit
                fl = [floor[s] for s in calc]
                gf = {s: st.mean(p["gF"] for p in pe[s]) for s in calc}
                kept = {tag: {s: eff["qat"][tag]["per"][s] / gf[s] for s in calc} for tag in ("orig", "new")}
                rho_kept = {tag: spearman_two_sided(fl, [kept[tag][s] for s in calc]) for tag in kept}
                rho_qa = {"orig": spearman_two_sided(fl, [qa[s][0] for s in calc]),
                          "new": spearman_two_sided(fl, [st.mean(qa[s]) for s in calc])}
                rank_qat = {tag: {s: 1 + sum(eff["qat"][tag]["per"][u] > eff["qat"][tag]["per"][s] for u in calc)
                                  for s in calc} for tag in ("orig", "new")}
                # the noise of one fine-tuning run for the students of the chain repeats, beside the chain's own
                rep = [s for s in QJ.SEED_REPEAT_STUDENTS if s in qa]
                sd_rep = math.sqrt(st.mean(st.variance(qa[s]) for s in rep))
                sd_pooled = math.sqrt(st.mean(st.variance(qa[s]) for s in QJ.STUDENTS))
                extra = {"kept": kept, "rho_kept": rho_kept, "rho_qa": rho_qa, "rank_qat": rank_qat,
                         "sd_rep": sd_rep, "rep": rep}
                if fold == "test" and key in R:
                    # H2's sign changes beyond twice the noise of Both, with QAT(A) the mean of three seeds
                    sd_b = R[key]["sig"]["both"]
                    extra["h2_flipqat"] = {tag: sum(AN.pair_effects(V, s, t)["qat"] - (shift[s] if tag == "new" else 0)
                                                    < -2 * sd_b for s in low3 for t in QJ.TEACHERS)
                                           for tag in ("orig", "new")}
                    # RQ4 read with twice this noise as the margin, in place of twice the chain noise of QAKD(B)
                    tost = R[key]["tost"]
                    m2 = 2 * sd_pooled
                    extra["tost_ft"] = {"margin": m2, "equivalent": -m2 < tost["lo90"] and tost["hi90"] < m2}
                res_qa[(fold, key)] = {"qa": qa, "sd": {s: st.stdev(qa[s]) for s in QJ.STUDENTS},
                                       "sd_pooled": sd_pooled, "eff": eff,
                                       # largest move of the four declared contrasts, on the rounded values the
                                       # table prints
                                       "maxshift": max(abs(round(eff[k]["new"]["mean"], 3) - round(eff[k]["orig"]["mean"], 3))
                                                       for k in ("rescue", "qat", "during", "both")),
                                       "h2": {"mean": m3, "lo": lo3, "hi": hi3, "npos": sum(x > 0 for x in h2)}} | extra
    out["anc"] = res_anc or None
    out["qata"] = res_qa or None
    return out


def read_csv(*parts):
    p = path(*parts)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def mech(tag=TAG):
    """The mechanism measurements (36_mech_measure.py, 36b_mech_accum.py, 37_mech_analysis.py), if present:
    per-student summary rows and a loader for the accumulated-noise profile of one deployed model."""
    ps = path("analysis", f"mech_{tag}", "per_student.csv")
    if not os.path.exists(ps):
        return None
    stu = {r["student"]: r for r in csv.DictReader(open(ps))}

    def profile(cell, student, teacher, rule):
        f = path("mech", tag, "accum", f"{cell}__{student}__{teacher or '-'}__{rule}.csv")
        rows = sorted(csv.DictReader(open(f)), key=lambda r: float(r["depth"]))
        bb = [r for r in rows if r["tensor"].startswith("/features/")]
        d = np.array([float(r["depth"]) for r in bb])
        return d / d.max(), np.array([float(r["sqnr_accum_db"]) for r in bb])
    return {"stu": stu, "profile": profile}


def results():
    mm = _rule("mm")
    pc = _rule("pc", order=mm["order"])
    diag = fp32_diag(mm["V"])
    val = {"mm": val_check("mm"), "pc": val_check("pc")}
    params = {r["model"]: r for r in read_csv("bench", TAG, "params.csv")}
    fp32 = {r["model"]: r for r in read_csv("bench", TAG, "fp32_ptq.csv") if r["role"] == "teacher" and r["status"] == "SUCCESS"}
    lat = [r for r in read_csv("bench", TAG, "latency_rr5.csv")] or [r for r in read_csv("bench", TAG, "latency.csv")]
    train = read_csv("part2", "train_summary.csv")
    train_f = read_csv("fp32", "train_summary.csv")
    out = {"mm": mm, "pc": pc, "order": mm["order"], "diag": diag, "val": val, "params": params, "teacher_fp32": fp32, "latency": lat,
            "latency_file": "latency_rr5.csv" if read_csv("bench", TAG, "latency_rr5.csv") else "latency.csv",
            # on-device latency (android/run_android_bench.py), when measured
            "latency_android": read_csv("bench", TAG, "latency_android.csv"),
            "android_device": (__import__("json").load(open(path("bench", TAG, "latency_android_device.json")))
                               if os.path.exists(path("bench", TAG, "latency_android_device.json")) else {}),
            "train": train, "train_fp32": train_f}
    out["anchor"] = anchor(out)
    out["ed"] = ed(out)
    out["sens"] = sens(out)
    return out
