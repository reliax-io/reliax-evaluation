# Reliax whitepaper evaluation

Reproducibility package for the working paper **"Per-Decision Reliability
Envelopes for Credit-Risk Models: An Evaluation on Real Data"** (v0.2,
September 2026). The paper is in [PAPER.md](PAPER.md); a formatted version is
at [reliax.io](https://reliax.io).

Every number in the paper is produced by `eval/run_eval.py` and stored in
`results/results.json`. Nothing is hand-typed without a source there.

## Headline results

| Claim | Measured |
|---|---|
| Conformal coverage at a 0.95 target | 0.9515 +/- 0.0040 (Taiwan), 0.9532 +/- 0.0167 (German) |
| Drift false alarms (5 x 600 i.i.d. real applications) | 0 / 5; real subpopulation shift caught in 4/5 streams |
| Calibration ECE on a miscalibrated model | 0.065 (Venn-Abers) vs 0.083 (T-scaled) vs 0.186 (raw) |
| Best selective-prediction AURC, Taiwan | 0.0955 (subjective-logic fusion) |
| Full envelope latency | 8.0 ms median, 17.5 ms p95 |

Honest negatives are in the paper too: the SL fusion trails plain confidence on
the small German dataset, and the shifted-data regime is untested here (a
pre-registered benchmark is future work).

## Reproduce

```bash
git clone https://github.com/Ouatt-Isma/reliax-evaluation.git && cd reliax-evaluation
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python eval/run_eval.py       # ~2 minutes; rewrites results/results.json
.venv/bin/python eval/make_figures.py   # rewrites results/fig_*.png
```

Splits, seeds and every threshold are fixed in `eval/run_eval.py`.

## Layout

- `reliax_core/` - the envelope's method components (conformal, Mondrian,
  Venn-Abers, test martingale, kNN OOD, PSI, error auditor, SL fusion).
  Methods only: the Reliax product layer is not part of this release.
- `eval/` - dataset loaders, baselines/metrics, experiment runner, figures.
- `data/` - the two real UCI datasets, verbatim, with `PROVENANCE.md`.
- `results/` - `results.json` and the four figures used in the paper.

## License

Code: Apache-2.0. Datasets: CC BY 4.0 (UCI), see `data/PROVENANCE.md`.
