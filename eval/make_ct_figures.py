"""Figures for the calibration-trust experiments (reads results/calibration_trust.json).

  fig5_ct_msweep.png   disbelief (prior shrink removed) vs number of bins M, three
                       synthetic regimes with known ECE: the M -> infinity limit and
                       the debiasing corrections.
  fig6_ct_cells.png    Taiwan, Venn-Abers scorer, seed 0: per (age band x score bin)
                       cell disbelief and cell size; thin cells outlined.
"""
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
R = json.load(open(BASE / "results" / "calibration_trust.json"))
OUT = BASE / "results"
STYLE = {
    "chapter (fixed bins, midpoint)": dict(color="#a8443a", ls="-"),
    "quantile bins, mean rep": dict(color="#3f8ab0", ls="-"),
    "quantile, mean, floor-debiased": dict(color="#2f7f86", ls="--"),
    "quantile, mean, l2-debiased": dict(color="#c8862c", ls="--"),
}


def fig_msweep():
    s1 = R["s1_msweep"]
    Ns = list(next(iter(s1.values()))["N"].keys())
    N = Ns[-1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (name, reg) in zip(axes, s1.items()):
        for est, series in reg["N"][N].items():
            ms = sorted(int(m) for m in series)
            mean = np.array([series[str(m)]["mean"] for m in ms])
            std = np.array([series[str(m)]["std"] for m in ms])
            ax.plot(ms, mean, label=est, **STYLE[est])
            ax.fill_between(ms, mean - std, mean + std, color=STYLE[est]["color"], alpha=0.12)
        ax.axhline(reg["ece_true"], color="k", lw=1, ls=":", label="true ECE (L1)")
        ax.axhline(reg["l1_limit"], color="k", lw=1, ls="-.", label="M -> inf limit  E|Y - q|")
        ax.set_xscale("log")
        ax.set_title(f"{name} (T = {reg['T']}), N = {int(N):,}")
        ax.set_xlabel("number of bins M")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("global disbelief d (prior shrink removed)")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Calibration-trust disbelief vs number of bins: it estimates the binned ECE, "
                 "and converges to E|Y - q|, not to calibration, as M -> infinity", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ct_msweep.png", dpi=160)
    plt.close(fig)


def fig_cells():
    cells = R["s4_taiwan"]["cells_seed0_venn_abers"]
    groups = sorted({k.split("|")[0] for k in cells})
    groups = [g for g in groups if g != "__all__"] + ["__all__"]
    M = max(int(k.split("|")[1]) for k in cells) + 1
    d = np.full((len(groups), M), np.nan)
    n = np.zeros((len(groups), M))
    thin = np.zeros((len(groups), M), bool)
    for k, c in cells.items():
        g, i = k.split("|")
        gi = groups.index(g)
        d[gi, int(i)] = c["d"]
        n[gi, int(i)] = c["n"]
        thin[gi, int(i)] = c["insufficient_evidence"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.imshow(d, cmap="Blues", aspect="auto", vmin=0, vmax=max(np.nanmax(d), 0.05))
    for gi in range(len(groups)):
        for i in range(M):
            if not np.isnan(d[gi, i]):
                ax.text(i, gi, f"{d[gi, i]:.2f}\nn={int(n[gi, i])}", ha="center", va="center", fontsize=7,
                        color="white" if d[gi, i] > 0.6 * np.nanmax(d) else "black")
            if thin[gi, i]:
                ax.add_patch(plt.Rectangle((i - .5, gi - .5), 1, 1, fill=False, ec="#b08a3a", lw=2))
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g if g != "__all__" else "all (marginal)" for g in groups])
    ax.set_xticks(range(M))
    ax.set_xticklabels([f"bin {i}" for i in range(M)], fontsize=8)
    ax.set_xlabel("score bin (quantile bins of the Venn-Abers PD, low to high)")
    ax.set_title("Taiwan default, Venn-Abers scorer: per-cell disbelief (l2-debiased) and cell size; "
                 "amber outline = thin cell (n < 30)", fontsize=9.5)
    fig.colorbar(im, ax=ax, label="disbelief d")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_ct_cells.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_msweep()
    fig_cells()
    print("wrote fig5_ct_msweep.png, fig6_ct_cells.png")
