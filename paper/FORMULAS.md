# Formula spec: DSR, multiple-testing haircuts, and reality-check tests

All equation numbers below are the numbers printed in the primary source itself
(confirmed by extracting text directly from the source PDFs). Notation is kept
as close to the original as legibility allows; `SR_hat` = sample/estimated
Sharpe ratio, `Z` = standard normal CDF, `Z^{-1}` = standard normal inverse CDF
(quantile function).

---

## 1. Probabilistic Sharpe Ratio (PSR)

**Source:** Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient
Frontier," *Journal of Risk* 15(2): 3-44 [`baileylopezdeprado2012psr`],
Eq. (11) (definition), Eq. (8) (variance of `SR_hat`, Mertens 2002 result).
Restated identically as Eq. (2) in Bailey & Lopez de Prado (2014)
[`baileylopezdeprado2014dsr`].

```
PSR(SR*) = Z(  (SR_hat - SR*) * sqrt(n - 1)
              -----------------------------------------------
              sqrt( 1 - gamma3_hat * SR_hat + ((gamma4_hat - 1)/4) * SR_hat^2 ) )
```

Definitions (all computed **in the native sampling frequency of the returns —
no annualization**, per the source's explicit caution that PSR "is invariant
to calendar conventions"):

- `SR_hat` — the sample (point-estimate) Sharpe ratio of the strategy under
  test, `SR_hat = mu_hat / sigma_hat`, in the *original* (non-annualized)
  frequency.
- `SR*` — the benchmark/rejection-threshold Sharpe ratio chosen by the user
  (defaults to 0, i.e. "no investment skill," per the source's footnote 4).
  In the DSR construction (Section 2 below) `SR* = SR0`, the deflated
  benchmark.
- `n` — sample length (number of return observations).
- `gamma3_hat` — sample **skewness** of the returns distribution.
- `gamma4_hat` — sample **kurtosis** of the returns distribution, using the
  **non-excess convention (Normal ⇒ gamma4 = 3)**. This is confirmed by the
  structure of the variance term: for Normal returns (`gamma3=0, gamma4=3`)
  the `(gamma4-1)/4` term reduces to `1/2`, reproducing the standard
  asymptotic result `Var(SR_hat) ≈ (1/n)(1 + SR^2/2)` (Mertens 2002); this
  only holds if `gamma4` is non-excess. **Do not substitute excess kurtosis
  (Normal ⇒ 0) into this formula without first adding 3, or the deflation
  will be wrong.**
- `Z` — CDF of the standard Normal distribution.

`PSR(SR*)` is interpreted as the probability that the strategy's true Sharpe
ratio exceeds the benchmark `SR*`, after correcting for the inflation caused
by short samples and non-Normal (skewed, fat-tailed) returns.

---

## 2. Deflated Sharpe Ratio (DSR)

**Source:** Bailey & Lopez de Prado (2014), *Journal of Portfolio Management*
40(5): 94-107 [`baileylopezdeprado2014dsr`], Eq. (1) (expected max SR under
the null, restated with full derivation as Eq. (5)-(6) in Appendix A.1),
Eq. (2) (DSR itself).

**DSR is PSR evaluated at the deflated benchmark `SR0`:**

```
DSR = PSR(SR0)
```

where `SR0` is the *expected maximum* Sharpe ratio that would be observed by
chance alone, under the null that the true Sharpe ratio of every trial is
zero, after `N` independent trials with trial-SR variance `Var[{SR_n}]`:

```
SR0 = sqrt( Var[{SR_n}] ) * ( (1 - gamma) * Z^{-1}(1 - 1/N)
                             +      gamma  * Z^{-1}(1 - 1/(N*e)) )
```

Definitions:

- `N` — number of **independent** trials (backtests / parameter
  configurations / strategies) considered during the search that produced
  the selected strategy.
- `Var[{SR_n}]` — the **variance across the trials' estimated Sharpe ratios**
  `{SR_1, ..., SR_N}` (i.e., the empirical variance of the SR estimates
  produced by the whole search, *not* the variance of a single strategy's
  returns).
- `gamma ≈ 0.5772156649...` — the **Euler-Mascheroni constant**.
- `e ≈ 2.71828...` — Euler's number.
- `Z^{-1}` — inverse CDF (quantile function) of the standard Normal
  distribution.

This is Eq. (1) in the DSR paper's main text; the paper's Appendix A.1 proves
it by (i) reducing the problem to the expected maximum of `N` iid standard
Normal draws (their Eq. 5, an Extreme Value Theory approximation attributed
to "Bailey et al. [2014a]"), then (ii) rescaling by the trial mean/std (their
Eq. 6, algebraically identical to Eq. 1). The paper's own reference Python
implementation (verified verbatim from the source PDF) is:

```python
def getExpMaxSR(mu, sigma, numTrials):
    # Compute the expected maximum Sharpe ratio (Analytically)
    emc = 0.5772156649  # Euler-Mascheroni constant
    maxZ = (1 - emc) * ss.norm.ppf(1 - 1./numTrials) \
                + emc * ss.norm.ppf(1 - 1./(numTrials * np.e))
    return mu + sigma * maxZ
```

confirming `mu = E[{SR_n}]`, `sigma = sqrt(Var[{SR_n}])`, and that
`ss.norm.ppf` is exactly `Z^{-1}`.

**Then, substituting `SR* = SR0` into the PSR formula (Section 1) gives DSR
(Eq. 2 of the paper):**

```
DSR = Z(  (SR_hat - SR0) * sqrt(T - 1)
         ------------------------------------------------
         sqrt( 1 - gamma3_hat * SR_hat + ((gamma4_hat - 1)/4) * SR_hat^2 ) )
```

using the *selected* strategy's own `SR_hat`, `T` (its sample length), and its
own `gamma3_hat`, `gamma4_hat` (non-excess kurtosis, same convention as
Section 1) — as distinct from the `N`/`Var[{SR_n}]` describing the population
of *all* trials searched.

**Independent-trials caveat (from the source):** the derivation assumes the
`N` trials are independent. The paper's Appendix 3 (referenced but not
reproduced verbatim here — flagged as such) "shows how N can be determined
when the trials are not independent," i.e. an **effective number of trials**
correction is needed when trials are correlated (see Section 5 below for the
general form of this idea as used by Harvey & Liu).

---

## 3. Harvey & Liu (2015) haircut Sharpe ratio

**Source:** Harvey & Liu (2015), "Backtesting," *Journal of Portfolio
Management* 42(1): 13-28 [`harveyliu2015backtesting`]. Page/equation numbers
below are as printed (pp. 13-16).

### 3.1 Single-test p-value → Sharpe ratio link (Eqs. 1-2)

```
t-statistic = mu_hat / (sigma_hat / sqrt(T))                       (1)
SR_hat      = mu_hat / sigma_hat  =  t-statistic / sqrt(T)          (2)
```

### 3.2 Haircut definition

> "We transform the Sharpe ratio into a t-ratio... With this new t-ratio, we
> determine a new Sharpe ratio. **The percentage difference between the
> original Sharpe ratio and the new Sharpe ratio is the haircut.**"
> "The haircut Sharpe ratio... is the Sharpe ratio that would have resulted
> from a single test."

Formally:

```
haircut = 1 - SR_adjusted / SR_original
```

where `SR_adjusted` (their `HSR`, the haircut Sharpe ratio) is obtained by
converting the multiple-testing-adjusted p-value back into a single-test
t-ratio and thence a Sharpe ratio via Eq. (2) run in reverse (their Eq. 5
defines `HSR` implicitly via `p_M = Pr(|t| > HSR-implied t-ratio)`, worked
through numerically in their example, p. 14: 200 trials deflate an annual
`SR = 0.75` to `SR_adj = 0.32`, i.e. haircut `= (0.75-0.32)/0.75 ≈ 60%`).

### 3.3 Multiple-testing p-value adjustments

Let the `M` single-test p-values be ordered `p(1) ≤ p(2) ≤ ... ≤ p(M)`, with
associated hypotheses `H(1), ..., H(M)`.

**Bonferroni (FWER control):**
```
p_i^Bonferroni = min[ M * p(i), 1 ],   i = 1, ..., M
```

**Holm (FWER control, step-down, uniformly more powerful than Bonferroni):**
```
p_i^Holm = min[ max_{j <= i} { (M - j + 1) * p(j) },  1 ],   i = 1, ..., M
```
i.e. built sequentially starting from the smallest p-value; each adjusted
p-value is the running max of `(M-j+1)*p(j)` for `j <= i`, capped at 1.

**Benjamini-Hochberg-Yekutieli / BHY (FDR control, step-up):**
```
p_M^BHY = min[ c(M) * p(M),  1 ]                            if i = M
p_i^BHY = min[ p_{i+1}^BHY,  (M * c(M) / i) * p(i) ]        if i <= M-1
```
where the normalizing constant is
```
c(M) = sum_{j=1}^{M} 1/j
```
NOTE: c(M) MULTIPLIES the adjustment (numerator), it does not divide it — the
Benjamini-Yekutieli (2001) constant makes the procedure MORE conservative under
arbitrary dependence (c(1000) = 7.49). This is what the implementation does
(`raw = p_sorted * M * cM / ranks`); a denominator c(M) would be wrong.
Harvey & Liu explicitly note: Benjamini & Hochberg (1995) originally set
`c(M) = 1` (valid under independence or positive dependence of p-values);
Harvey & Liu instead adopt the **Benjamini & Yekutieli (2001)** choice
`c(M) = sum_{j=1}^{M} 1/j`, which is valid under **arbitrary dependency**
among the test statistics — this is the version that should be used for
correlated backtest trials.

**Ordering of stringency:** for the FDR-vs-FWER criteria in general, BHY controls
a more lenient error rate than Bonferroni/Holm and so rejects more of the BULK of
hypotheses. BUT this ordering does NOT hold for the single most-significant
(rank-1) hypothesis — which is exactly what the best-strategy haircut uses. At
rank 1, Holm coincides with Bonferroni (`(M-1+1) p_(1) = M p_(1)`), and BY
multiplies by `M * c(M) >= M`, so for the top pick BY is the MOST conservative of
the three (largest haircut). This is confirmed by the experiments (haircut_bhy
0.20 > haircut_holm 0.15; null FDR bhy 0.007 < bonferroni 0.057) and by
`test_bhy_numerator_convention`. Do not state "BHY is least conservative" without
this rank-1 caveat.

### 3.4 FWER and FDR definitions used

```
FWER = Pr(N_r >= 1)                     N_r = number of false rejections (false positives)

FDP  = N_r / R   if R > 0,  else 0      R = total number of rejections
FDR  = E[FDP]
```

---

## 4. White's Reality Check (RC) and Hansen's SPA test

**Sources:** White (2000), "A Reality Check for Data Snooping," *Econometrica*
68(5): 1097-1126 [`white2000reality`]; Hansen (2005), "A Test for Superior
Predictive Ability," *Journal of Business & Economic Statistics* 23(4):
365-380 [`hansen2005spa`]. **Flag: the exact in-paper equation numbers for
this section were not independently re-extracted from the primary PDFs in
this pass (paywalled/inaccessible in the environment); the description below
is the standard, widely-reproduced characterization of both tests and should
be cross-checked against the primary text before being treated as a verbatim
quote.**

**Setup common to both:** compare a benchmark ("no-skill"/status-quo) model
against `l = 1, ..., k` candidate models/rules using a loss (or performance)
differential `f_{k,t} = performance of rule k minus benchmark, for t = 1..T`.
Null hypothesis: the best rule is no better than the benchmark, i.e.
`H0: max_k E[f_k] <= 0`.

**Reality Check test statistic:**
```
RC_T = max_{k=1,...,l} ( sqrt(T) * f_bar_k )
```
where `f_bar_k = (1/T) * sum_t f_{k,t}` is the sample-average performance of
rule `k`. White evaluates the (non-standard) sampling distribution of this
max-statistic under the null via the **stationary bootstrap** (Politis &
Romano 1994, Section 5 below): resample `f_{k,t}` with the stationary
bootstrap, **recenter each bootstrapped series by its own full-sample mean
`f_bar_k`** so the resamples satisfy the null by construction, recompute the
max statistic on each bootstrap draw, and take the p-value as the fraction of
bootstrap max-statistics that exceed (or equal) the observed `RC_T`.

**Hansen's SPA test — two modifications relative to RC:**
1. **Studentization:** each rule's statistic is standardized by an estimate
   of its own standard error before taking the max, so that noisy/irrelevant
   alternatives (large variance, no real skill) do not swamp the test the way
   they can under RC's unstandardized max — Hansen's stated motivation is
   that RC "can be quite sensitive to the inclusion of poor and irrelevant
   alternatives."
2. **Sample-dependent (consistent) recentering / null distribution:** rather
   than recentering every rule fully to zero (White's approach, which can be
   overly conservative for rules whose sample mean is negative but not
   significantly so), Hansen recenters using a threshold that shrinks
   toward zero only for rules whose performance is not distinguishable from
   the benchmark, giving upper (SPA_u), lower (SPA_l), and consistent (SPA_c)
   variants, each yielding a bootstrap p-value analogous to RC's.

Both tests report a p-value = the fraction of stationary-bootstrap
replications of the (re-centered) max/studentized-max statistic that meet or
exceed the observed statistic.

---

## 5. Stationary bootstrap

**Source:** Politis & Romano (1994), *Journal of the American Statistical
Association* 89(428): 1303-1313 [`politisromano1994`]. **Flag: exact
in-paper equation numbers not re-verified against primary text in this pass
(same access constraint as Section 4); description below is the standard
characterization.**

Resampling procedure for weakly-dependent stationary time series: blocks of
random length are drawn, with block length distributed geometrically with
mean `1/p` (parameter `p in (0,1]`), and blocks wrap around the sample
(circularly) so the resampled series is itself stationary — unlike the
fixed-block bootstrap. This is the resampling engine underlying both White's
Reality Check and Hansen's SPA test p-values (Section 4), and is the natural
tool for backtest multiple-testing corrections where trial performance series
are serially correlated.

---

## 6. Effective number of trials (correlated trials reduce N)

Appears in two independent forms across the literature reviewed:

1. **Bailey & Lopez de Prado (2014)** [`baileylopezdeprado2014dsr`]: state
   plainly that Eq. (1)/(2)'s `N` must be the number of **independent**
   trials, and reference their own "Appendix 3" for how to determine an
   effective `N` "when the trials are not independent" (not reproduced
   verbatim here — flagged as referenced-but-not-extracted).
2. **Harvey & Liu (2015)** [`harveyliu2015backtesting`]: motivate the BHY
   `c(M) = sum_{j=1}^{M} 1/j` normalizer (Section 3.3 above) precisely
   because it is valid "under arbitrary dependency for the test statistics,"
   i.e. it is their mechanism for not overstating significance when trials
   (factors/strategies) are correlated, as opposed to assuming full
   independence (`c(M)=1`, Benjamini & Hochberg 1995's original setting).
3. **Harvey, Liu & Zhu (2016)** [`harveyliuzhu2016`] independently document
   that correlation among the ~300+ tested factors must be modeled (not just
   counted) to get a correct multiple-testing threshold — their stated
   motivation for a >=300-factor "haircut" t-hurdle of ~3.0 explicitly
   accounts for factor correlation rather than treating each of the 300+
   tests as independent.

**Net implication for experiment design:** neither "correlated trials treated
as independent" (undercorrects — inflates apparent significance) nor "raw
trial count with no correlation adjustment" is defensible; both DSR's
`Appendix 3` and Harvey-Liu's BHY/`c(M)` choice, and HLZ's correlation-aware
threshold, are three independent attempts at the same fix and should agree in
spirit (fewer "effective" independent trials than raw trial count whenever
trials are positively correlated).

---

## Verification notes / what to re-check before relying on this spec

- Sections 1, 2, and 3 (PSR, DSR, Harvey-Liu haircut/Bonferroni/Holm/BHY) were
  extracted **directly from the primary-source PDFs** (`pdftotext` on the
  authors' own posted copies: davidhbailey.com/dhbpapers/{sharpe-frontier,
  deflated-sharpe}.pdf and people.duke.edu/~charvey/Research/Published_Papers/
  P120_Backtesting.PDF) — equation numbers and the Python code snippet in
  Section 2 are verbatim.
- Sections 4 and 5 (White RC, Hansen SPA, stationary bootstrap) were **not**
  extracted from primary full text in this pass (fetch attempts hit a
  certificate mismatch and were not retried against a mirror); the
  descriptions given are standard/consensus characterizations from secondary
  literature and should be spot-checked against Econometrica 68(5) and JBES
  23(4) directly before being treated as verbatim before being cited as exact
  quotes or equation numbers.
- All journal/volume/issue/page metadata in `refs.bib` was cross-checked
  against the CrossRef API (api.crossref.org) in addition to publisher/SSRN
  pages, except Holm (1979), which predates the DOI system and has no
  CrossRef-registered DOI — its bibliographic details are corroborated across
  multiple independent secondary sources but the full text was not directly
  fetched.
