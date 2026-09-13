"""Deck/site figure for the mixed-population regimes: results/fig8_mix.png.

Reads results/shift_bench_v2.json. This is the left panel of fig8_bench_v2 redrawn
for slides: the three regimes are labelled and the pre-registered 1.5x bar is shown.
Warm template palette (2026-09).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = pathlib.Path(__file__).resolve().parent.parent
R = json.load(open(BASE / "results" / "shift_bench_v2.json"))

plt.rcParams.update({"font.family": "monospace", "font.size": 10.5,
                     "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
                     "savefig.facecolor": "#ffffff", "axes.edgecolor": "#93a7b5",
                     "text.color": "#123f52", "axes.labelcolor": "#123f52",
                     "xtick.color": "#123f52", "ytick.color": "#123f52",
                     "axes.grid": True, "grid.color": "#e4edf3", "grid.linewidth": 0.7})

SIG = [("msp", "model confidence", "#a8443a", "-"),
       ("gated", "confidence gated by OOD", "#123f52", "-"),
       ("ood_knn", "OOD distance", "#3f8ab0", "--"),
       ("sl_fusion", "SL fusion v0.2", "#c8862c", ":")]


def main():
    lams = sorted(float(k) for k in R["mix"])
    fig, ax = plt.subplots(figsize=(7.0, 4.375))
    ax.axvspan(0.15, 0.60, color="#e8f0f6", alpha=0.55, lw=0)
    ax.axhline(1.0, color="#123f52", lw=1, ls=":", label="random referral")
    ax.axhline(1.5, color="#2f7f86", lw=1.2, ls="-.", label="pre-registered bar (1.5x random)")
    for name, label, c, ls in SIG:
        m = [R["mix"][str(l)]["signals"][name]["vs_random"]["mean"] for l in lams]
        s = [R["mix"][str(l)]["signals"][name]["vs_random"]["std"] for l in lams]
        ax.errorbar(lams, m, yerr=s, label=label, color=c, ls=ls, marker="o", ms=4,
                    lw=1.8, capsize=2)
    ax.text(0.01, 2.40, "calm", color="#5d7787", fontsize=10.5)
    ax.text(0.245, 2.40, "shift in progress", color="#3f8ab0", fontsize=10.5)
    ax.text(0.99, 2.40, "complete: tripwire", color="#5d7787", fontsize=10.5, ha="right")
    ax.set_xlabel("share of the live book that has shifted (delayed payers)", fontsize=10)
    ax.set_ylabel("bad approvals caught at 10% referral, vs random", fontsize=9.3)
    ax.set_ylim(0.22, 2.5)
    ax.set_xlim(-0.03, 1.03)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92,
              facecolor="#ffffff", edgecolor="#cddce6")
    fig.tight_layout()
    fig.savefig(BASE / "results" / "fig8_mix.png", dpi=160)
    print("wrote fig8_mix.png")


if __name__ == "__main__":
    main()
