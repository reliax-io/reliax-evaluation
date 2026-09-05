"""Generate the paper figures from paper/results/results.json."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
R = json.load(open(BASE / "results" / "results.json"))
OUT = BASE / "results"

INK, SLATE, AMBER, ROSE, GREEN, MUTED = "#2a2e31", "#5d7f9b", "#b08a3a", "#8f4b45", "#5f9b7c", "#9a9c98"
plt.rcParams.update({"font.family": "monospace", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                     "ytick.color": INK, "figure.facecolor": "white", "axes.grid": True,
                     "grid.color": "#eae7dc", "grid.linewidth": 0.7})

LABELS = {"msp_tscaled": "max-softmax (T-scaled)", "conformal_pvalue": "conformal p-value",
          "va_width": "Venn-Abers width", "auditor": "error auditor", "composite": "composite score (v1)",
          "sl_fusion": "SL fusion"}
COLORS = {"msp_tscaled": MUTED, "conformal_pvalue": SLATE, "va_width": AMBER,
          "auditor": GREEN, "composite": INK, "sl_fusion": ROSE}

# ---- Figure 1: error capture vs referral rate (Taiwan, seed 0) ------------------
fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=170)
rates = R["_fig"]["curve_rates"]
for sig, curve in R["_fig"]["curves"].items():
    ax.plot(rates, curve, label=LABELS[sig], color=COLORS[sig],
            lw=2 if sig in ("composite", "sl_fusion") else 1.3,
            ls="--" if sig == "msp_tscaled" else "-")
ax.plot([0, 0.5], [0, 0.5], color=MUTED, lw=0.8, ls=":", label="random referral")
ax.axvline(0.10, color=ROSE, lw=0.8, ls=":")
ax.set_xlabel("referral rate (fraction sent to review)")
ax.set_ylabel("fraction of model errors captured")
ax.set_title("Selective prediction, Taiwan default (seed 0)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig1_capture.png"); plt.close(fig)

# ---- Figure 2: per-segment coverage, marginal vs Mondrian (Taiwan) --------------
segs = R["taiwan"]["segments"]
names, marg_m, marg_s, mond_m, mond_s = [], [], [], [], []
for attr in ("sex", "age_band"):
    for name, row in segs[attr].items():
        names.append(f"{name}\n({attr})")
        marg_m.append(row["marginal"]["mean"]); marg_s.append(row["marginal"]["std"])
        mond_m.append(row["mondrian"]["mean"]); mond_s.append(row["mondrian"]["std"])
x = np.arange(len(names)); w = 0.36
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=170)
ax.bar(x - w / 2, marg_m, w, yerr=marg_s, color=SLATE, label="marginal q̂", capsize=2)
ax.bar(x + w / 2, mond_m, w, yerr=mond_s, color=AMBER, label="Mondrian q̂ per segment", capsize=2)
ax.axhline(0.95, color=ROSE, lw=1, ls="--", label="target 1-α = 0.95")
ax.set_xticks(x, names, fontsize=7.5)
ax.set_ylim(0.90, 0.98); ax.set_ylabel("empirical coverage (5 seeds)")
ax.set_title("Per-segment coverage at α = 0.05, Taiwan default", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig2_segments.png"); plt.close(fig)

# ---- Figure 3: martingale trajectories ------------------------------------------
fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=170)
for traj in R["_fig"]["iid_trajs"]:
    ax.plot(traj, color=SLATE, lw=1, alpha=0.8)
for traj in R["_fig"]["shift_trajs"]:
    ax.plot(traj, color=ROSE, lw=1.4, alpha=0.9)
ax.axhline(np.log10(20), color=AMBER, lw=1, ls="--")
ax.axhline(2, color=ROSE, lw=1, ls="--")
ax.set_ylim(-2.7, 3.6)
ax.text(595, np.log10(20) + 0.10, "WATCH  M = 20", fontsize=7, color=AMBER, ha="right")
ax.text(595, 2.10, "ALARM  M = 100  (Ville: P ≤ 0.01)", fontsize=7, color=ROSE, ha="right")
ax.plot([], [], color=SLATE, label="i.i.d. stream (no shift)")
ax.plot([], [], color=ROSE, label="shifted stream (PAY_0 ≥ 1 subgroup)")
ax.set_xlabel("applications processed"); ax.set_ylabel("log₁₀ martingale wealth")
ax.set_title("Exchangeability martingale on real applicants (3 seeds)", fontsize=9)
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig3_martingale.png"); plt.close(fig)

# ---- Figure 4: calibration ------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), dpi=170)
bins = R["_fig"]["reliability_bins"]
for tag, color, label in (("raw", MUTED, "raw model"), ("va", SLATE, "Venn-Abers point")):
    xs = [b[tag]["conf"] for b in bins if b[tag]["n"] > 30]
    ys = [b[tag]["acc"] for b in bins if b[tag]["n"] > 30]
    a1.plot(xs, ys, "o-", color=color, lw=1.2, ms=3, label=label)
a1.plot([0, 1], [0, 1], color=ROSE, lw=0.8, ls=":")
a1.set_xlabel("predicted p(default)"); a1.set_ylabel("observed default rate")
a1.set_title("Reliability diagram (bins with n > 30)", fontsize=8.5)
a1.legend(fontsize=7, frameon=False)
wd = R["_fig"]["width_decile"]
a2.bar(["widest 10% of\nPD intervals", "other 90%"],
       [wd["err_rate_widest_decile"], wd["err_rate_rest"]], color=[ROSE, SLATE], width=0.5)
a2.set_ylabel("model error rate")
a2.set_title("Interval width predicts errors", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUT / "fig4_calibration.png"); plt.close(fig)

print("figures written to", OUT)
