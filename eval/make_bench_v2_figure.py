"""Figure for shift benchmark v2: results/fig8_bench_v2.png (reads results/shift_bench_v2.json)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# RCWARM: template palette
plt.rcParams.update({"font.family": "monospace", "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
                     "savefig.facecolor": "#ffffff", "axes.edgecolor": "#93a7b5", "text.color": "#123f52",
                     "axes.labelcolor": "#123f52", "xtick.color": "#123f52", "ytick.color": "#123f52",
                     "grid.color": "#e4edf3"})

BASE = pathlib.Path(__file__).resolve().parent.parent
R = json.load(open(BASE / "results" / "shift_bench_v2.json"))
SIG = {"msp": ("model confidence", "#a8443a", "-"), "ood_knn": ("OOD distance (kNN)", "#3f8ab0", "-"),
       "gated": ("confidence gated by OOD", "#123f52", "--"), "regime_switch": ("regime switch (martingale)", "#2f7f86", "-"),
       "min_rank": ("rank fusion", "#c8862c", ":"), "sl_fusion": ("SL fusion (v0.2)", "#93a7b5", ":")}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    lams = sorted(float(k) for k in R["mix"])
    for name, (label, c, ls) in SIG.items():
        m = [R["mix"][str(l)]["signals"][name]["vs_random"]["mean"] for l in lams]
        s = [R["mix"][str(l)]["signals"][name]["vs_random"]["std"] for l in lams]
        ax.errorbar(lams, m, yerr=s, label=label, color=c, ls=ls, marker="o", ms=4, capsize=2)
    ax.axhline(1.0, color="k", lw=1, ls=":", label="random referral")
    auc = [R["mix"][str(l)]["auc"]["mean"] for l in lams]
    for l, a in zip(lams, auc):
        ax.annotate(f"AUC {a:.2f}", (l, 0.3), ha="center", fontsize=7.5, color="#4a6472")
    ax.set_xlabel("share of the live population that has shifted (delayed payers)")
    ax.set_ylabel("bad approvals caught at 10% referral, vs random")
    ax.set_title("Mixed populations: who should route to REVIEW?", fontsize=10)
    ax.set_ylim(0.2, None)
    ax.grid(alpha=.25)
    ax.legend(fontsize=7.5, loc="upper right")

    ax = axes[1]
    keys = ["unit_error_0.05", "unit_error_0.10", "unit_error_0.20", "missing_0.10", "gaussian_0.10"]
    labels = ["unit error 5%", "unit error 10%", "unit error 20%", "missing fields 10%", "noise 10%"]
    x = np.arange(len(keys))
    w = 0.38
    for i, (name, lab, c) in enumerate((("msp", "model confidence", "#a8443a"), ("ood_knn", "OOD distance (kNN)", "#3f8ab0"))):
        m = [R["noise"][k]["signals"][name]["corrupted_caught"]["mean"] for k in keys]
        s = [R["noise"][k]["signals"][name]["corrupted_caught"]["std"] for k in keys]
        ax.bar(x + (i - .5) * w, m, w, yerr=s, label=lab, color=c, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("share of corrupted rows inside the 10% referred")
    ax.set_title("Corrupted inputs: does the referral catch them?", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=.25, axis="y")
    ax.legend(fontsize=8)
    fig.suptitle("Shift benchmark v2 (exploratory, Taiwan default, 3 seeds): confidence wins when nothing has moved, "
                 "OOD-gated signals win while a shift is happening, nothing ranks once it is complete", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(BASE / "results" / "fig8_bench_v2.png", dpi=160)
    print("wrote fig8_bench_v2.png")


if __name__ == "__main__":
    main()
