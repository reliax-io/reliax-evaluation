# Dataset provenance

Both datasets are real, public, and stored here verbatim as downloaded on 2026-09-05.

## Taiwan: Default of Credit Card Clients
- File: `default of credit card clients.xls` (5,539,328 bytes)
- Source: UCI Machine Learning Repository, dataset 350
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Citation: I-Cheng Yeh and Che-hui Lien, "The comparisons of data mining
  techniques for the predictive accuracy of probability of default of credit
  card clients", Expert Systems with Applications 36(2), 2009.
- 30,000 rows, 23 attributes, binary label `default payment next month`
  (22.12% positive). License: CC BY 4.0.
- Protected attributes SEX, AGE, MARRIAGE are excluded from model features and
  used only for segment audits (see `../eval/data.py`).

## German Credit (Statlog)
- Files: `german.data`, `german.doc`
- Source: UCI Machine Learning Repository, dataset 144
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- Citation: Hans Hofmann, Statlog (German Credit Data), 1994.
- 1,000 rows, 20 attributes, binary label (30% "bad"). License: CC BY 4.0.
- Attribute 9 (personal status / sex) and attribute 13 (age) are excluded from
  model features and used only for segment audits. Note the known caveat that
  attribute 9 conflates sex and marital status; we map A92/A95 to female and
  A91/A93/A94 to male, following common practice.

## Reproducing the download
```bash
curl -L -o taiwan.zip "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
curl -L -o german.zip "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
```
