"""
where-teacher-acts: the paper's result figures, drawn from _qj_paper.results() (read-only).

  fig_kept.pdf    share of the FP32 gain kept at INT8 against the INT8 floor, PTQ, QAT and QAKD(B),
                  one panel per calibration rule
  fig_added.pdf   what the teacher adds during quantization on top of a distilled start, QAKD(B) - QAT(B),
                  against the floor; individual pairs and student means; noise band of a difference between
                  two fine-tuned models (the training sd of Both)
  fig_routes.pdf  the INT8 model every route delivers, one panel per student (MinMax): undistilled start A
                  (hollow) and distilled start B (filled, mean of the six teachers, teachers as dots) under
                  PTQ, QAT and QAKD, with the FP32 models as reference lines

Colors: Okabe-Ito blue, orange and bluish green, checked with the dataviz validator for all pairs on a white
surface (CVD and normal-vision separation pass; orange is below 3:1 contrast, so every series also has its
own marker shape and the values are tabulated in the paper). Vector PDF with embedded TrueType fonts, plus
a PNG preview.

    python pipeline/29b_paper_figures.py
"""
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _qj_paper as P  # noqa: E402  (moves to the project root)
import _qj_common as QJ  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

OUT = os.path.join("paper", "figures")
WIDTH = 5.39                     # inches: the 13.7 cm text width of elsarticle preprint
C_PTQ, C_QAT, C_QAKD = "#E69F00", "#009E73", "#0072B2"
INK, INK2, MUTED, GRID, REF_A, REF_B = "#222222", "#4d4d4d", "#8c8c8c", "#ebebeb", "#b8b8b8", "#5c5c5c"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK, "text.color": INK,
})
RULE_TITLE = {"mm": "(a) MinMax calibration", "pc": "(b) Percentile 99.9 calibration"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)
    print(f"   wrote {os.path.join(OUT, name)}.pdf (+ .png preview)")


def spread_positions(xs, min_gap):
    """Where to draw students whose floors are xs: each run of floors closer than min_gap is spread evenly around
    its own mean, min_gap apart, so the shifts stay small and symmetric; every other student stays at its floor."""
    clusters = [[i] for i in sorted(range(len(xs)), key=lambda i: xs[i])]

    def place(cl):
        m = sum(xs[i] for i in cl) / len(cl)
        return [m + (k - (len(cl) - 1) / 2) * min_gap for k in range(len(cl))]
    merged = True
    while merged:
        merged = False
        for j in range(len(clusters) - 1):
            if place(clusters[j + 1])[0] - place(clusters[j])[-1] < min_gap - 1e-12:
                clusters[j:j + 2] = [clusters[j] + clusters[j + 1]]
                merged = True
                break
    out = list(xs)
    for cl in clusters:
        for i, x in zip(cl, place(cl)):
            out[i] = x
    return out


def student_labels(ax, xs, placed, names):
    """Student names along the top edge, at the positions the markers are drawn at; where a student was moved
    away from its floor, a short leader runs from the label to the true floor on the top edge."""
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for x, xx, n in zip(xs, placed, names):
        if abs(xx - x) > 1e-9:
            ax.plot([x, xx], [1.0, 1.035], transform=tr, color=MUTED, lw=0.45, clip_on=False)
        ax.text(xx, 1.02 if abs(xx - x) <= 1e-9 else 1.045, n, transform=tr, rotation=90, ha="center",
                va="bottom", fontsize=6.5, color=INK2)


def fig_kept(R):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.55), sharey=True)
    series = (("kept_ptq", C_PTQ, "o", "PTQ: PTQ(B) $-$ PTQ(A)"), ("kept_qat", C_QAT, "s", "QAT: QAT(B) $-$ QAT(A)"),
              ("kept_both", C_QAKD, "^", "Both: QAKD(B) $-$ QAT(A)"))
    dx = 0.02                      # the three series of a student side by side, in floor units
    for ax, rk in zip(axes, ("mm", "pc")):
        stu, order = R[rk]["stu"], R["order"]
        xs = [stu[s]["floor"] for s in order]
        placed = spread_positions(xs, min_gap=0.064)
        ax.axhline(1.0, color=REF_B, lw=0.7, zorder=1)
        ax.axhline(0.0, color=REF_A, lw=0.7, zorder=1)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        for j, (key, col, mk, lab) in enumerate(series):
            ax.scatter([x + (j - 1) * dx for x in placed], [stu[s][key] for s in order], s=18, color=col, marker=mk,
                       edgecolors="white", linewidths=0.6, zorder=3, label=lab)
        student_labels(ax, xs, placed, [P.SHORT[s] for s in order])
        ax.set_xlim(-0.03, 0.75)
        ax.set_xlabel("INT8 floor, PTQ(A) test mAP")
        ax.set_title(RULE_TITLE[rk], loc="left", pad=34, fontsize=8)
        if rk == "pc":
            ax.text(0.0, 1.03, "all of $g_F$ kept", fontsize=6.5, color=INK2, ha="left", va="bottom")
    axes[0].set_ylabel("INT8 benefit / FP32 gain $g_F$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13), handletextpad=0.3,
               columnspacing=1.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_kept")


def fig_added(R):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45), sharey=True)
    for ax, rk in zip(axes, ("mm", "pc")):
        r = R[rk]
        stu, order, pairs = r["stu"], R["order"], r["pairs"]
        sd = r["sig"]["both"]
        ax.axhspan(-2 * sd, 2 * sd, color="#d9d9d9", alpha=0.45, lw=0, zorder=0)
        ax.axhline(0.0, color=REF_B, lw=0.7, zorder=1)
        xs = [stu[s]["floor"] for s in order]
        placed = spread_positions(xs, min_gap=0.045)
        for f, s in zip(placed, order):
            ax.scatter([f] * len(QJ.TEACHERS), [pairs[(s, t)]["added"] for t in QJ.TEACHERS], s=9, color=C_QAKD,
                       alpha=0.35, linewidths=0, zorder=2)
            ax.scatter([f], [stu[s]["added"]], s=30, color=C_QAKD, edgecolors="white", linewidths=0.9, zorder=3)
        student_labels(ax, xs, placed, [P.SHORT[s] for s in order])
        rh = r["rho"]["added"]
        ax.text(0.98, 0.04, f"Spearman $\\rho = {rh['rho']:+.2f}$, one-sided $p = {rh['p_less']:.3f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK)
        ax.set_xlim(-0.03, 0.75)
        ax.set_xlabel("INT8 floor, PTQ(A) test mAP")
        ax.set_title(RULE_TITLE[rk], loc="left", pad=34, fontsize=8)
    axes[0].set_ylabel("Added: QAKD(B) $-$ QAT(B), mAP")
    handles = [Line2D([], [], marker="o", ls="", color=C_QAKD, ms=5, mec="white", mew=0.8, label="student, mean of six teachers"),
               Line2D([], [], marker="o", ls="", color=C_QAKD, alpha=0.35, ms=3.2, mew=0, label="one student and teacher pair"),
               plt.Rectangle((0, 0), 1, 1, color="#d9d9d9", alpha=0.45, lw=0, label="$\\pm 2\\sigma$ training noise, difference of two fine-tuned models")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13),
               handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_added")


def fig_routes(R):
    r = R["mm"]
    stu, pairs, order = r["stu"], r["pairs"], R["order"]
    regs = (("PTQ", "PTQ_A", "PTQ_B", C_PTQ, "o"), ("QAT", "QAT_A", "QAT_B", C_QAT, "s"),
            ("QAKD", "QAKD_A", "QAKD_B", C_QAKD, "^"))
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.9), sharey=True)
    dx = 0.16
    for ax, s in zip(axes.flat, order):
        u = stu[s]
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.axhline(u["A_fp32"], color=REF_A, lw=0.8, zorder=1)
        ax.axhline(u["B_fp32"], color=REF_B, lw=0.8, zorder=1)
        for i, (_, ca, cb, col, mk) in enumerate(regs):
            ax.plot([i - dx, i + dx], [u[ca], u[cb]], color=col, lw=1.0, alpha=0.7, zorder=2)
            ax.scatter([i + dx] * len(QJ.TEACHERS), [pairs[(s, t)][cb] for t in QJ.TEACHERS], s=5, color=col,
                       alpha=0.4, linewidths=0, zorder=2)
            if ca == "QAKD_A":
                ax.scatter([i - dx] * len(QJ.TEACHERS), [pairs[(s, t)][ca] for t in QJ.TEACHERS], s=5,
                           color=col, alpha=0.4, linewidths=0, zorder=2)
            ax.scatter([i - dx], [u[ca]], s=24, marker=mk, facecolors="white", edgecolors=col, linewidths=1.0, zorder=3)
            ax.scatter([i + dx], [u[cb]], s=24, marker=mk, color=col, edgecolors="white", linewidths=0.7, zorder=3)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["PTQ", "QAT", "QAKD"])
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.02, 0.86)
        ax.set_title(P.NAME[s], fontsize=7.5, pad=3)
    for ax in axes[:, 0]:
        ax.set_ylabel("INT8 test mAP (MinMax)")
    # the eighth panel: legend and the scale of the training noise
    lg = axes.flat[7]
    lg.axis("off")
    sd = r["sig"]["QAKD_B"]   # one fine-tuned model, the quantity each marker shows
    handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK2, mew=1.0, ms=5, label="start A, undistilled FP32"),
               Line2D([], [], marker="o", ls="", color=INK2, mec="white", mew=0.7, ms=5, label="start B, mean of six teachers"),
               Line2D([], [], marker="o", ls="", color=INK2, alpha=0.4, ms=2.6, mew=0, label="individual teachers"),
               Line2D([], [], color=REF_B, lw=0.8, label="FP32 model B"),
               Line2D([], [], color=REF_A, lw=0.8, label="FP32 model A")]
    lg.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.5, handletextpad=0.4,
              borderaxespad=0.0, labelspacing=0.55, bbox_to_anchor=(-0.05, 1.0))
    y0 = 0.08
    lg.plot([0.08, 0.08], [y0, y0 + 2 * sd], color=INK2, lw=1.4, solid_capstyle="butt")
    lg.plot([0.04, 0.12], [y0, y0], color=INK2, lw=0.6)
    lg.plot([0.04, 0.12], [y0 + 2 * sd, y0 + 2 * sd], color=INK2, lw=0.6)
    lg.set_xlim(0, 1)
    lg.text(0.18, y0 + sd, f"$2\\sigma$ training noise of one\nfine-tuned model, chain\nrepeats ({2 * sd:.3f})",
            fontsize=6.5, color=INK2, va="center", ha="left")
    lg.text(0.0, 0.31, "Color and marker shape\ngive the route", fontsize=6.5, color=INK2, va="center")
    fig.subplots_adjust(hspace=0.42, wspace=0.10)
    save(fig, "fig_routes")


def running_median(v, frac=0.08, min_w=3):
    """Running median over a window of about `frac` of the tensors: tensors of different kinds (squeeze-and-
    excitation gates, convolution outputs) alternate along the graph, and the median shows the trend."""
    w = max(min_w, int(round(frac * len(v))) | 1)
    h = w // 2
    return np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])


def fig_mech_accum(R, Mx):
    """Accumulated quantization noise along each student's backbone, PTQ(A) under both rules and QAT(A)."""
    order = R["order"]
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.8), sharey=True, sharex=True)
    series = (("PTQ_A", "mm", C_PTQ, "-", "PTQ(A), MinMax"),
              ("PTQ_A", "pc", C_PTQ, (0, (3, 1.6)), "PTQ(A), percentile 99.9"),
              ("QAT_A", "mm", C_QAT, "-", "QAT(A), learned ranges"))
    for ax, s in zip(axes.flat, order):
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.axhline(0.0, color=REF_B, lw=0.7, zorder=1)
        for cell, rule, col, ls, _ in series:
            d, v = Mx["profile"](cell, s, None, rule)
            ax.plot(d, running_median(v), color=col, lw=1.2, ls=ls, zorder=3)
            # the running median lifts the end of a falling curve: the last tensor itself, unsmoothed
            ax.scatter([d[-1]], [v[-1]], s=11, color=col, edgecolors="white", linewidths=0.4, zorder=4,
                       clip_on=False)
        ax.set_title(f"{P.NAME[s]} ({int(float(Mx['stu'][s]['PTQ_A_mm_n_tensors']))})", fontsize=7.5, pad=3)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-8, 50)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
    for ax in axes[:, 0]:
        ax.set_ylabel("Accumulated SQNR (dB)")
    fig.supxlabel("Relative depth in the backbone (quantized tensors in execution order)", fontsize=8, y=0.01)
    lg = axes.flat[7]
    lg.axis("off")
    handles = [Line2D([], [], color=c, lw=1.4, ls=ls, label=lab) for _, _, c, ls, lab in series]
    handles.append(Line2D([], [], marker="o", ls="", color=INK2, mec="white", mew=0.4, ms=3.6,
                          label="last backbone tensor,\nunsmoothed"))
    handles.append(Line2D([], [], color=REF_B, lw=0.8, label="noise as strong as the signal"))
    lg.legend(handles=handles, loc="center left", frameon=False, fontsize=6.5, handlelength=2.4, borderaxespad=0.0,
              bbox_to_anchor=(-0.05, 0.55))
    fig.subplots_adjust(hspace=0.42, wspace=0.12, bottom=0.12)
    save(fig, "fig_mech_accum")


def fig_mech_geometry(R, Mx):
    """(a) The distilled start's neighbors kept by quantization-aware fine-tuning, against the floor; (b) the
    teacher's neighbors each deployed student keeps. The article cites (a) first."""
    order = R["order"]
    st_ = Mx["stu"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[1]
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    cells = (("np_T_B", "#5c5c5c", "D", "B (FP32)"), ("np_T_PTQ_B", C_PTQ, "o", "PTQ(B)"),
             ("np_T_QAT_B", C_QAT, "s", "QAT(B)"), ("np_T_QAKD_B", C_QAKD, "^", "QAKD(B)"))
    for j, (k, col, mk, lab) in enumerate(cells):
        xs = [i + (j - 1.5) * 0.16 for i in range(len(order))]
        ax.scatter(xs, [float(st_[s][k]) for s in order], s=20, color=col, marker=mk, edgecolors="white",
                   linewidths=0.6, zorder=3, label=lab)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([P.SHORT[s] for s in order], fontsize=6.8, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Teacher's 10 nearest kept (NP@10)")
    ax.set_ylim(0, 0.6)
    ax.set_title("(b) Teacher's neighbors kept", loc="left", fontsize=8, pad=34)
    ax.legend(loc="lower left", frameon=False, fontsize=6.3, ncol=2, handletextpad=0.2, columnspacing=0.8,
              borderaxespad=0.2)
    ax = axes[0]
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    fl = [R["mm"]["stu"][s]["floor"] for s in order]
    placed = spread_positions(fl, min_gap=0.06)
    for j, (k, col, mk, lab) in enumerate((("np_B_QAT_B", C_QAT, "s", "QAT(B)"), ("np_B_QAKD_B", C_QAKD, "^", "QAKD(B)"))):
        ax.scatter([x + (j - 0.5) * 0.028 for x in placed], [float(st_[s][k]) for s in order], s=20, color=col,
                   marker=mk, edgecolors="white", linewidths=0.6, zorder=3, label=lab)
    student_labels(ax, fl, placed, [P.SHORT[s] for s in order])
    rh, p = P.spearman_two_sided(fl, [float(st_[s]["np_B_QAT_B"]) for s in order])
    ax.text(0.98, 0.04, f"QAT(B): $\\rho = {rh:+.2f}$, $p = {p:.3f}$", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.8)
    ax.set_xlim(-0.03, 0.75)
    ax.set_ylim(0.3, 0.9)
    ax.set_xlabel("INT8 floor, PTQ(A) test mAP")
    ax.set_ylabel("Start's 10 nearest kept (NP@10)")
    ax.set_title("(a) Distilled start's neighbors kept", loc="left", fontsize=8, pad=34)
    ax.legend(loc="upper left", frameon=False, fontsize=6.3, handletextpad=0.2, borderaxespad=0.2)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_mech_geometry")


def main():
    R = P.results()
    fig_kept(R)
    fig_added(R)
    fig_routes(R)
    Mx = P.mech()
    if Mx:
        fig_mech_accum(R, Mx)
        fig_mech_geometry(R, Mx)


if __name__ == "__main__":
    main()
