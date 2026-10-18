"""
where-teacher-acts: the graphical abstract (at least 531 x 1328 pixels, readable at 5 x 13 cm). Drawn at
13 x 5 cm and 300 dpi from _qj_paper.results().

    python pipeline/29c_graphical_abstract.py
"""
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import _qj_paper as P  # noqa: E402  (moves to the project root)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

C_PTQ, C_QAT, C_QAKD, INK, INK2, GRID = "#E69F00", "#009E73", "#0072B2", "#222222", "#4d4d4d", "#ebebeb"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8c8c8c", "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.labelsize": 6, "ytick.labelsize": 6})


def box(ax, x, y, w, h, text, fc, ec, fs=6.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight)


def main():
    R = P.results()
    W, H = 13 / 2.54, 5 / 2.54
    fig = plt.figure(figsize=(W, H))
    # left: the message and the design
    ax = fig.add_axes([0.0, 0.0, 0.36, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.93, "Distill first, then quantize", fontsize=9, weight="bold", color=INK, va="top")
    ax.text(0.04, 0.80, "7 compact students \u00d7 6 teachers, pill retrieval,\nmeasured on the deployed 8-bit model",
            fontsize=6.3, color=INK2, va="top", linespacing=1.3)
    cols = (("PTQ", C_PTQ, "post-training\nquantization"), ("QAT", C_QAT, "fine-tune\nno teacher"),
            ("QAKD", C_QAKD, "fine-tune\nwith teacher"))
    x0, y0, cw, ch = 0.34, 0.13, 0.205, 0.15
    for j, (name, col, sub) in enumerate(cols):
        ax.text(x0 + j * cw + cw / 2 - 0.01, y0 + 2 * ch + 0.13, name, ha="center", fontsize=7, weight="bold", color=col)
        ax.text(x0 + j * cw + cw / 2 - 0.01, y0 + 2 * ch + 0.035, sub, ha="center", fontsize=5.6, color=INK2,
                linespacing=1.05)
        for i, (start, fc) in enumerate((("A", "white"), ("B", "#f2f2f2"))):
            box(ax, x0 + j * cw, y0 + (1 - i) * ch, cw - 0.02, ch - 0.02, f"{name}({start})", fc, col, fs=6.0)
    ax.text(0.04, y0 + ch + ch / 2 - 0.01, "A: undistilled", fontsize=6.0, color=INK2, va="center")
    ax.text(0.04, y0 + ch / 2 - 0.01, "B: distilled\n(teacher\nbefore)", fontsize=6.0, color=INK2, va="center",
            linespacing=1.05)
    # middle: the share of the FP32 gain kept after post-training quantization, against the floor
    ax = fig.add_axes([0.43, 0.20, 0.25, 0.62])
    order = R["order"]
    for rk, mk, fc, lab in (("mm", "o", C_PTQ, "MinMax"), ("pc", "o", "white", "percentile")):
        st = R[rk]["stu"]
        ax.scatter([st[s]["floor"] for s in order], [st[s]["kept_ptq"] for s in order], s=14, marker=mk,
                   facecolors=fc, edgecolors=C_PTQ, linewidths=0.9, zorder=3, label=lab)
    ax.axhline(1.0, color="#7f7f7f", lw=0.5, zorder=1)
    ax.axhline(0.0, color="#bfbfbf", lw=0.5, zorder=1)
    ax.set_xlim(-0.03, 0.75)
    ax.set_xlabel("INT8 floor of the student", fontsize=6, color=INK)
    ax.set_ylabel("share of the FP32 gain kept", fontsize=6, color=INK)
    ax.set_title("Post-training quantization", fontsize=6.8, color=INK, pad=3)
    ax.legend(frameon=False, fontsize=6.0, loc="upper left", handletextpad=0.2, borderaxespad=0.1,
              title="calibration", title_fontsize=6.0)
    # right: the take-aways
    ax = fig.add_axes([0.71, 0.0, 0.29, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tips = (("1", "One distilled model serves\nevery route and loses a\nsmaller share to INT8", INK),
            ("2", "Measure the INT8 floor first:\nthe gain survives where it\nholds, vanishes if not", INK),
            ("3", "Low floor: fine-tune the\ndistilled model; a teacher\nduring it adds little", C_QAT))
    y = 0.86
    for num, text, col in tips:
        ax.text(0.02, y, num, fontsize=10, weight="bold", color=col, va="top")
        ax.text(0.16, y - 0.01, text, fontsize=6.4, color=INK, va="top", linespacing=1.25)
        y -= 0.30
    out = os.path.join("paper", "figures")
    fig.savefig(os.path.join(out, "graphical_abstract.pdf"))
    fig.savefig(os.path.join(out, "graphical_abstract.png"), dpi=300)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(os.path.join(out, "graphical_abstract.png")).size
    print(f"   wrote paper/figures/graphical_abstract.pdf and .png ({w} x {h} px)")


if __name__ == "__main__":
    main()
