# Calibration-trust opinions: what the global opinion is, what happens as M grows, and how to collect the evidence

Companion note to `reliax_core/calibration_trust.py`, `tests/test_calibration_trust.py`
and `eval/run_calibration_trust.py` (results in `results/calibration_trust.json`,
figures `results/fig5_ct_msweep.png`, `results/fig6_ct_cells.png`).

The method is the calibration-based trust assessment of Ouattara, Krontiris,
Dimitrakos and Kargl (FUSION 2025; thesis chapter "Calibration-Based Trust
Assessment of NNs"), adapted to a credit scorer that states a default
probability. Everything below is stated for that adaptation ("rate evidence")
and, where it differs, for the chapter's original ("accuracy evidence").

## 1. Setting and notation

Calibration data: pairs (p_j, y_j), j = 1..N, with p_j in [0, 1] the model's
stated probability of default and y_j in {0, 1} the outcome. Conditional on the
scores the outcomes are independent with E[y_j | p_j] = mu(p_j); the model is
calibrated at p when mu(p) = p. The true (L1) calibration error is
ECE* = E|mu(p) - p|.

A partition B_1..B_M of [0, 1] ("bins", the chapter's "clusters"). For bin i:
n_i points, k_i defaults, observed rate ybar_i = k_i / n_i, representative
RP_i (midpoint of the bin, or the mean stated probability pbar_i). The binned
calibration error with these representatives is

    ECE_M = sum_i (n_i / N) |ybar_i - RP_i|.

Subjective-logic quantification (baseline-prior quantification, prior weight W,
default W = 2): evidence (r, s) gives the opinion

    b = r / (r + s + W),   d = s / (r + s + W),   u = W / (r + s + W).

Rate evidence per bin: s_i = n_i |ybar_i - RP_i|, r_i = n_i - s_i.
Accuracy evidence (chapter): with t_i correct predictions among n_i,
r_i = t_i, s_i = |t_i - n_i RP_i|.

Cumulative fusion of binomial opinions that share W and the base rate is
evidence addition (Josang 2016, ch. 12). The cells are disjoint, so cumulative
fusion is the right operator across bins and across segments; averaging fusion
is the right one across classes that were scored on the same data (the
chapter's second fusion step).

## 2. Proposition 1 (closed form of the global opinion)

With rate evidence and alpha = beta = 1, r_i + s_i = n_i for every bin, hence
R = N - S and S = N ECE_M, and

    d = N / (N + W) * ECE_M,     b = N / (N + W) * (1 - ECE_M),     u = W / (N + W).

With accuracy evidence, R = T (total correct) and S = N ECE_M^acc, so

    d / b = ECE_M^acc / accuracy,    u = W / (T + S + W).

Proof. Evidence addition, then substitute. QED.

What it means. In rate mode the global opinion is a bijection of the pair
(ECE_M, N): disbelief is the binned ECE shrunk by N/(N+W), and uncertainty is
a function of N alone. The global opinion therefore carries exactly the
information of "an ECE together with its sample size", no more. The chapter's
claim that the opinion is "richer than ECE" is true of the per-cell opinions
(the lookup table localises miscalibration and flags thin cells) and false of
the fused global number. In accuracy mode the global opinion mixes accuracy
and calibration through d/b = ECE/acc, which is the chapter's intended
"joint" behaviour; for a credit PD model the rate version is the right one,
because the question is whether the stated default rate is right, not whether
the top class is.

Per cell, u_i = W / (n_i + W) in rate mode: uncertainty is a monotone
transform of cell size. The "insufficient evidence" flag is therefore a cell
size threshold (min_cell_n, default 30) and is reported as such.

## 3. Proposition 2 (the limit M -> infinity is not calibration)

Fix N, use fixed-width bins with midpoint representatives, and assume the
stated probabilities are distinct. Let L1_N = (1/N) sum_j |y_j - p_j|. Then for
every M larger than 1 / min_{j != l} |p_j - p_l|,

    |ECE_M - L1_N| <= 1 / (2M),   hence   lim_{M -> inf} d_M = N / (N + W) * L1_N.

Proof. Beyond that M every bin holds at most one point, so ybar_i = y_j and
|RP_i - p_j| <= 1/(2M) for the point j in bin i; the triangle inequality gives
the bound termwise. QED.

Moreover, for any model,

    E[L1_N] = E|Y - p| = E|mu(p) - p| + 2 E[ min(mu(p), p) (1 - max(mu(p), p)) ]  >=  ECE*,

with equality only if p in {0, 1} almost surely, and for a calibrated model
E[L1_N] = E[2 p (1 - p)].

Proof. E[|Y - p| | p] = mu (1 - p) + (1 - mu) p = mu + p - 2 mu p, and one checks
mu + p - 2 mu p = |mu - p| + 2 min(mu, p) (1 - max(mu, p)) in both orderings
of mu and p. QED.

What it means. The chapter's sweep over M converges, but to the mean absolute
error between outcomes and stated probabilities, which is an upper bound on
the calibration error inflated by a sharpness term. A perfectly calibrated
credit model with default rates around 0.25 converges to disbelief about
0.33 (times N/(N+W)), not 0. This is exactly what the chapter observed and
could not explain: calibrated MNIST (p near 1, so 2p(1-p) near 0) converges
to belief near 1, calibrated CIFAR-10 peaks near M = 100 and then declines.
The chapter's "more clusters yield lower uncertainty" is, in rate mode, false
(u = W/(N+W) does not depend on M) and, in accuracy mode, a by-product of
disbelief inflating with M while the positive evidence T stays fixed. M is a
bias-variance knob for estimating ECE*, not a resolution knob to be maximised.

## 4. Proposition 3 (finite-sample guarantee for a fixed partition)

Condition on the scores (so the bins and the n_i are fixed) and let
mubar_i = (1/n_i) sum_{j in B_i} mu(p_j) and ECE_M* = sum_i (n_i/N) |mubar_i - RP_i|
be the population binned error for the same bins and representatives. Then
for every delta in (0, 1), with probability at least 1 - delta,

    |ECE_M - ECE_M*| <= eps_N(M, delta) := sum_i (n_i / N) sqrt( ln(2M/delta) / (2 n_i) )
                     <= sqrt( M ln(2M/delta) / (2N) ),

and consequently |d - d*| <= N/(N+W) eps_N(M, delta), where d* is the opinion
computed from ECE_M*.

Proof. Hoeffding for the mean of n_i independent [0,1] variables gives
P(|ybar_i - mubar_i| >= t_i) <= 2 exp(-2 n_i t_i^2); with
t_i = sqrt(ln(2M/delta) / (2 n_i)) each bin fails with probability at most
delta/M, and a union bound over the M bins gives simultaneous control. The
reverse triangle inequality | |ybar_i - RP_i| - |mubar_i - RP_i| | <= |ybar_i - mubar_i|
and the weights n_i/N give the first inequality. The second is Cauchy-Schwarz:
sum_i sqrt(n_i) <= sqrt(M N). QED.

Measured (S3 in the results file): N = 3000, M = 10 fixed bins, delta = 0.05,
500 trials; the envelope contains the population value in every trial, with a
mean half-width near 0.07 against a mean absolute error near 0.006. The bound
is conservative by an order of magnitude, as Hoeffding bounds are; a Bernstein
or Wilson version would be tighter and is a straightforward extension.

## 5. Proposition 4 (consistency in the double limit)

Assume mu is L-Lipschitz on [0, 1], fixed-width bins of width 1/M, mean
representatives RP_i = pbar_i. Then with probability at least 1 - 2 delta,

    |ECE_M - ECE*| <= (L + 1) / M + sqrt( M ln(2M/delta) / (2N) ) + sqrt( ln(2/delta) / (2N) ).

Hence ECE_M -> ECE* in probability whenever M -> infinity and M ln M / N -> 0;
the choice M proportional to N^(1/3) balances the two main terms at order
N^(-1/3) up to logarithms.

Proof. Write g(p) = mu(p) - p, which is (L+1)-Lipschitz. With mean
representatives, mubar_i - pbar_i = mean over the bin of g(p_j), so
ECE_M* = sum_i (n_i/N) |mean_{B_i} g|, while (1/N) sum_j |g(p_j)| = sum_i (n_i/N) mean_{B_i} |g|.
Within a bin of width 1/M the values of g differ by at most (L+1)/M, so
|mean g| and mean |g| differ by at most (L+1)/M, giving
|ECE_M* - (1/N) sum_j |g(p_j)|| <= (L+1)/M. Hoeffding on the i.i.d. sample
mean of |g(p_j)| in [0,1] gives the last term, and Proposition 3 the middle
one. QED.

Measured (S2): T = 1.8 (under-confident), true ECE* = 0.0924; with
M = round(N^(1/3)) the quantile-bin mean-representative estimate moves from
0.110 (N = 1e3) to 0.0935 (1e4) to 0.0926 (1e5); the fixed-bin midpoint
estimate of the chapter behaves the same way at these sizes.

## 6. Proposition 5 (small-sample bias and the two corrections)

Suppose bin i is exactly calibrated, mubar_i = RP_i, with n_i points of
common mean RP_i. By the central limit theorem,

    E|ybar_i - RP_i| = sqrt( 2 RP_i (1 - RP_i) / (pi n_i) ) (1 + o(1)),

so a perfectly calibrated model has E[ECE_M] approximately
sum_i (n_i/N) sqrt(2 RP_i (1 - RP_i) / (pi n_i)), of order sqrt(M/N): the
raw disbelief of a calibrated model is positive and grows with M long before
the M -> infinity regime of Proposition 2.

Two corrections are implemented, both applied per bin before the evidence is
formed. Let Z be standard normal and c = sqrt(2/pi) = E|Z|.

  floor:  s_i = n_i ( |ybar_i - RP_i| - sqrt(2 RP_i (1 - RP_i) / (pi n_i)) )_+.
          Under calibration its first-order expectation is
          E[(|Z| - c)_+] / E|Z| = 2 [phi(c) - c (1 - Phi(c))] / c = 0.302 of the raw bias.
          Under a true deviation Delta_i it converges to Delta_i minus the
          floor, i.e. it is biased downward by a vanishing amount.

  l2:     s_i = n_i sqrt( ( (ybar_i - RP_i)^2 - ybar_i (1 - ybar_i) / (n_i - 1) )_+ ).
          The inner quantity is the unbiased estimator of (mubar_i - RP_i)^2
          of Kumar, Liang and Ma (2019): E[(ybar - RP)^2] = (mubar - RP)^2 + Var(ybar)
          and E[ybar (1 - ybar) / (n - 1)] = Var(ybar) for i.i.d. Bernoulli
          outcomes. Under calibration the first-order expectation of the
          square root is E[sqrt((Z^2 - 1)_+)] / E|Z| = 0.429 of the raw bias;
          under a true deviation it is consistent for Delta_i with no
          first-order downward bias.

All three estimators (raw, floor, l2) are consistent in the double limit of
Proposition 4; the corrections change the constant in the sqrt(M/N) term,
not the rate. Monte Carlo on Bin(100, 0.25): raw 0.0344 (theory 0.0345),
floor 0.305 of raw, l2 0.431 of raw. The product default is l2 because a
credit validator cares more about not understating a real deviation than
about the constant under the null.

## 7. Evidence collection: what changed relative to the chapter, and why

1. Representative. The chapter uses the bin midpoint. With the mean stated
   probability in the bin the representative error term (L+1)/M in
   Proposition 4 is the only binning bias left; the midpoint adds a term of
   the same order that does not vanish when the scores are concentrated
   inside a bin (the chapter's own limitation). Default: mean.
2. Bins. Fixed-width bins leave empty or near-empty tails on a credit scorer
   whose scores concentrate below 0.5. Quantile bins give equal-mass cells
   and make the Proposition 3 bound uniform in i. Isotonic bins, the blocks
   of the pool-adjacent-violators fit of outcomes on scores, are the
   data-adaptive monotone partition the chapter proposed in its limitations
   and are the partition Venn-Abers itself works on; the number of cells is
   then data-driven (about 25 on Taiwan, 9 on German). Default: quantile;
   isotonic when the calibration stage already runs Venn-Abers on the same data.
3. Evidence. Rate evidence (defaults versus predicted) instead of
   correctness; alpha/beta weight under- and over-estimation of the default
   rate (capital versus volume). Default alpha = beta = 1; the knob is exposed
   because the customer's loss is asymmetric.
4. Debiasing as in Proposition 5. Default: l2.
5. Segments. Cells are (segment x bin), segments being the customer's
   business segments recorded in the policy (the Mondrian groups), never
   protected classes at inference. Thin cells are flagged at lookup and never
   silently fall back to the marginal cell without saying so.
6. Delayed-outcome re-evaluation. `realized(scores, outcomes)` re-scores the
   fitted partition on later outcomes and reports the gap between the
   disbelief the reference data implied and the disbelief the new outcomes
   realise. This is the calibration half of the certificate's delayed outcome
   verdict.

## 8. What the opinion can and cannot say (limits, unchanged)

- It is a statement about a cell, not about one applicant. The certificate
  line reads "the stated probability carries disbelief d on n calibration
  observations in this cell (signal)", never a per-applicant probability.
- It is fitted on reference outcomes. Under population shift the claimed
  disbelief is wrong until outcomes from the new population arrive; S5 in
  the results file measures the gap on the Taiwan payment-delay split
  (claimed 0.061, realised 0.225; utilisation split 0.054 to 0.111; i.i.d.
  control 0.055 to 0.055). The live input-score
  martingale is the tripwire; `realized` is the repair once labels land.
- As a per-decision ranking signal the cell belief adds nothing over the
  confidence it is built from (S4, accuracy mode: AURC 0.136 against 0.096
  for plain confidence on Taiwan; 0.136 against 0.135 on German). The chapter's "discriminative" finding on
  MNIST and CIFAR-10 does not transfer to credit data and is not claimed.
- The global opinion is a function of (ECE_M, N). Its value to a validator
  is the finite-sample error bar of Proposition 3 and the localisation in
  the cell table, not a number that means more than ECE.

## 9. Measured results (full run, `results/calibration_trust.json`, 170 s)

S1, synthetic, N = 20,000, disbelief with the prior shrink removed (mean over
5 seeds), true ECE* and the M -> infinity limit E|Y - q| for comparison:

| regime | ECE* | E\|Y-q\| | estimator | M=2 | 10 | 30 | 100 | 300 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|---|---|---|
| calibrated (T=1) | 0.000 | 0.333 | chapter (fixed, midpoint) | 0.031 | 0.007 | 0.010 | 0.018 | 0.031 | 0.058 | 0.082 |
| | | | quantile, mean | 0.003 | 0.006 | 0.012 | 0.022 | 0.039 | 0.073 | 0.102 |
| | | | quantile, mean, floor | 0.000 | 0.002 | 0.004 | 0.007 | 0.012 | 0.022 | 0.030 |
| | | | quantile, mean, l2 | 0.000 | 0.002 | 0.005 | 0.009 | 0.018 | 0.033 | 0.049 |
| over-confident (T=0.6) | 0.083 | 0.289 | chapter (fixed, midpoint) | 0.031 | 0.079 | 0.084 | 0.087 | 0.092 | 0.104 | 0.119 |
| | | | quantile, mean | 0.077 | 0.082 | 0.083 | 0.085 | 0.088 | 0.101 | 0.120 |
| | | | quantile, mean, floor | 0.075 | 0.076 | 0.074 | 0.068 | 0.060 | 0.056 | 0.059 |
| | | | quantile, mean, l2 | 0.077 | 0.081 | 0.081 | 0.078 | 0.071 | 0.062 | 0.063 |
| under-confident (T=1.8) | 0.092 | 0.389 | chapter (fixed, midpoint) | 0.031 | 0.089 | 0.092 | 0.093 | 0.097 | 0.104 | 0.115 |
| | | | quantile, mean | 0.088 | 0.091 | 0.092 | 0.094 | 0.098 | 0.115 | 0.136 |
| | | | quantile, mean, floor | 0.084 | 0.083 | 0.078 | 0.069 | 0.058 | 0.048 | 0.047 |
| | | | quantile, mean, l2 | 0.088 | 0.090 | 0.090 | 0.087 | 0.082 | 0.078 | 0.085 |

Reading: the chapter's M = 2 value is the midpoint-representative artefact
(0.031 for a calibrated model); raw estimators drift upward with M toward
E|Y - q| (Propositions 2 and 5); the floor correction is cleanest under the
null and biased downward under real miscalibration at large M; l2 is the
compromise and the product default. Between M = 10 and M = 100 all four
estimators are within 0.01 of ECE* when the model is miscalibrated, which is
the regime a validator will use. At N = 2,000 the same picture appears with
every curve shifted up (the sqrt(M/N) term); see the results file.

S2, consistency with M = round(N^(1/3)), T = 1.8, ECE* = 0.0924: quantile,
mean estimate 0.1097 (N = 1e3), 0.0935 (1e4), 0.0926 (1e5), 0.0924 (1e6);
l2-debiased 0.0976, 0.0904, 0.0919, 0.0922.

S3, Proposition 3 envelope at N = 3000, M = 10, delta = 0.05: holds in 500 of
500 trials; mean half-width 0.073, mean absolute error 0.006.

S4, real credit models (opinions fitted on half of the calibration split,
M = 10 quantile bins, mean representatives, l2, age-band segments; test ECE
with 15 fixed bins for reference):

| dataset | scorer | d | u | binned ECE (fit) | ECE15 (test) | realised d on test | thin cells |
|---|---|---|---|---|---|---|---|
| Taiwan (5 seeds) | raw | 0.004 | 0.0005 | 0.012 | 0.013 | 0.005 | 0 |
| | temperature-scaled | 0.005 | 0.0005 | 0.012 | 0.013 | 0.004 | 0 |
| | Venn-Abers | 0.008 | 0.0005 | 0.015 | 0.014 | 0.007 | 0.8 |
| German (10 seeds) | raw | 0.125 | 0.016 | 0.176 | 0.186 | 0.141 | 30 |
| | temperature-scaled | 0.051 | 0.016 | 0.101 | 0.084 | 0.036 | 30 |
| | Venn-Abers | 0.051 | 0.016 | 0.093 | 0.076 | 0.040 | 22 |

The opinion orders the scorers the way ECE does where there is something to
order (German: raw is the miscalibrated one). On German every cell is thin
(about 125 fitting points across 3 x 10 cells) and the module says so instead
of reporting a confident opinion. The alpha/beta knob on the German raw
scorer: (1,1) 0.125, (2,1) 0.194, (1,2) 0.179. Isotonic (Venn-Abers)
partition: 26 blocks on Taiwan, 9 on German, disbelief 0.004 and 0.109.
Segment view on Taiwan (raw scorer): 21-30 d = 0.008, 31-45 0.007, 46+ 0.011.

S5, Taiwan shift, opinion fitted on the reference population then re-scored
with `realized` on the shifted population: i.i.d. 0.055 -> 0.055, utilisation
0.054 -> 0.111, payment delay 0.061 -> 0.225 (test ECE15 0.059, 0.117, 0.255).

## 9b. Credit-risk held-out results (`eval/run_calibration_credit.py`, `results/calibration_credit.json`)

Opinion fitted on half of the calibration split, then re-scored with the same
cells and stated probabilities on the test applicants (Taiwan 7,500 per seed,
5 seeds; German 250 per seed, 10 seeds). All numbers are out of sample.

| dataset | scorer | clustering | claimed d | realised d | cell MAE | cell Spearman | worst cell in top 3 | thin cells |
|---|---|---|---|---|---|---|---|---|
| Taiwan | raw | chapter (fixed, midpoint) | 0.011 | 0.011 | 0.013 | 0.27 | 3/5 | 0.6 of 9 |
| | raw | quantile, mean | 0.004 | 0.005 | 0.016 | 0.10 | 2/5 | 0 of 10 |
| | raw | isotonic, mean | 0.004 | 0.007 | 0.022 | 0.37 | 1/5 | 6.8 of 26 |
| | Venn-Abers | chapter (fixed, midpoint) | 0.011 | 0.014 | 0.014 | 0.28 | 3/5 | 1.2 of 9 |
| | Venn-Abers | quantile, mean | 0.008 | 0.007 | 0.014 | 0.44 | 4/5 | 0 of 9 |
| | Venn-Abers | isotonic, mean | 0.008 | 0.009 | 0.020 | 0.30 | 1/5 | 3.8 of 19 |
| German | raw | chapter (fixed, midpoint) | 0.137 | 0.132 | 0.107 | 0.10 | 4/10 | 8.7 of 10 |
| | raw | quantile, mean | 0.125 | 0.141 | 0.098 | 0.49 | 4/10 | 10 of 10 |
| | raw | isotonic, mean | 0.109 | 0.145 | 0.078 | 0.58 | 6/10 | 7.4 of 8 |
| | temperature-scaled | quantile, mean | 0.051 | 0.036 | 0.097 | 0.17 | 5/10 | 10 of 10 |
| | Venn-Abers | chapter (fixed, midpoint) | 0.061 | 0.041 | 0.083 | 0.13 | 5/10 | 6.3 of 8 |
| | Venn-Abers | quantile, mean | 0.051 | 0.040 | 0.085 | 0.45 | 8/10 | 7.2 of 8 |
| | Venn-Abers | isotonic, mean | 0.032 | 0.038 | 0.069 | 0.16 | 7/10 | 6.5 of 7 |

Reading, in order of what a validator will ask.

1. The global opinion holds out of sample. On Taiwan the claimed and
   realised disbelief agree within 0.003 for every scorer and clustering; on
   German within 0.02 to 0.04 on 250 test applicants. The Hoeffding envelope
   of Proposition 3 contained the realised value in all 45 (Taiwan) and 90
   (German) cases, but on German it is vacuous (half-width 0.6 to 0.8 with
   cells of 12 points), so the envelope is only informative at Taiwan-size
   calibration sets and above.
2. The calibration stage is visible in the opinion and survives held-out
   scoring: on German the raw scorer realises 0.13 to 0.15 disbelief, the
   Venn-Abers scorer 0.04, and the opinion fitted on the calibration split
   said 0.125 and 0.051 beforehand. This is the number on the deck's hero.
3. Cell-level localisation is only moderately reliable. On Taiwan the model
   is already calibrated, so per-cell deviations are sampling noise around
   zero and rank agreement is weak (Spearman 0.1 to 0.44). On German the
   isotonic partition gives the lowest out-of-sample map error (cell MAE
   0.069 to 0.078 against 0.083 to 0.107 for the chapter's fixed bins) and the
   quantile partition the best rank agreement, at the price of thin cells
   everywhere. The chapter's fixed-width midpoint clustering is never the
   best choice on German and only marginally the best on Taiwan for raw
   scores. Recommended product setting stays quantile (equal-mass cells,
   uniform error bar); isotonic when the calibration set is large enough
   that its blocks are not thin.
4. Segments (quantile, raw scorer): German applicants aged 35 or under
   realise 0.152 disbelief against 0.091 for older applicants, and the
   calibration split predicted the same ordering (0.114 vs 0.094). On Taiwan
   all bands are near 0.01.

## 9c. Does it avoid mistakes? Decision-level results (`eval/run_calibration_decisions.py`, `eval/run_routing_decisions.py`)

The question a lender asks is not "is the ECE lower" but "how many bad
approvals does this stop, at what referral cost". Two experiments answer it
on real data, and both are negative for per-decision mistake avoidance.

**Calibration opinion as a referral rule** (`results/calibration_decisions.json`;
approve if stated PD < 0.5, then REVIEW every approval whose cell the opinion
marks untrustworthy: disbelief above 0.05 with the default rate understated,
or fewer than 30 calibration points):

| dataset | bad approvals / 1,000, model alone | after Venn-Abers bracket (p1 < 0.5) | opinion referral rate | bad approvals caught vs random | vs confidence referral |
|---|---|---|---|---|---|
| Taiwan (5 seeds) | 142 | 139 | 0 to 7% | 0.6 to 1.0x | 0.23 to 0.36x |
| German (10 seeds) | 157 | 177 | 100% (every cell thin) | not measurable | not measurable |

The Venn-Abers bracket does not change which applicants are approved at a
fixed cut-off in any systematic way (isotonic calibration preserves the
ranking; the bracket moves a few near-threshold cases either way). The
opinion's cell referral catches fewer bad approvals than referring at random
on Taiwan, and on German it refers everyone because every cell is thin. The
calibration stage says where the stated probabilities are wrong; it does not
say which applicant will default.

**Conformal set size as the REVIEW rule** (`results/routing_decisions.json`,
Taiwan, 5 seeds; REVIEW when the set holds both labels):

| regime | alpha | coverage | REVIEW rate | bad approvals / 1,000: all approvals -> auto-approved | bad approvals caught by REVIEW | by random referral at that rate | by confidence referral at that rate |
|---|---|---|---|---|---|---|---|
| i.i.d. | 0.05 | 0.952 | 54% | 138 -> 41 | 0.70 | 0.54 | 0.70 |
| i.i.d. | 0.10 | 0.901 | 26% | 138 -> 83 | 0.40 | 0.25 | 0.40 |
| utilisation shift | 0.05 | 0.903 | 32% | 148 -> 85 | 0.42 | 0.33 | 0.42 |
| utilisation shift | 0.10 | 0.856 | 16% | 148 -> 114 | 0.23 | 0.17 | 0.23 |
| payment-delay shift | 0.05 | 0.911 | 78% | 389 -> 87 | 0.78 | 0.78 | 0.78 |
| payment-delay shift | 0.10 | 0.774 | 48% | 389 -> 215 | 0.45 | 0.48 | 0.45 |

Two facts follow. For a binary model, the split-conformal set is ambiguous
exactly when the top-class probability is below 1 - q_hat, so set-size
routing IS a confidence cut, with the cut chosen to carry a coverage
guarantee on the calibration cohort; it cannot catch more than confidence
referral at the same rate, and the table shows it does not. What it adds is
that the threshold is certified rather than hand-picked, and that under
shift the REVIEW rate rises on its own (26% to 48% at alpha 0.10 on the
payment-delay split) because the shifted applicants fall below the certified
cut. Under that severe shift no signal, including this one, catches more
bad approvals than random referral at the same rate: the whitepaper's
negative result restated at the decision level.

What the method is good for, in decision terms, is therefore not a better
per-decision ranking of mistakes. It is (i) a referral threshold with a
stated guarantee, (ii) knowing when that guarantee has stopped holding
(drift 4/5, outcome verdict 0.06 -> 0.23), (iii) knowing where the stated
probabilities are wrong so the model owner fixes them (cells, segments), and
(iv) a record of all of it. Any deck claim of "avoids X% of mistakes versus
the model alone" must be paired with "the same as a confidence cut at that
rate", or it will not survive a validator.

## 10. Open extensions

- Choose W per cell so that u_i equals the Proposition 3 half-width at a
  stated delta; the opinion would then carry its own confidence band instead
  of a fixed prior weight.
- Bernstein or Wilson intervals in place of Hoeffding (tighter by roughly a
  factor 3 at credit default rates).
- Weighted (covariate-shift) re-weighting of the cells when the drift
  martingale alarms, using the same density-ratio weights as the weighted
  conformal repair.
