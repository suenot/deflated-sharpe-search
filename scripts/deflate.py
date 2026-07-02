"""Deflation / multiple-testing estimators for backtest selection bias.

Every estimator carries its provenance in the docstring. All Sharpe ratios are
PER-OBSERVATION (not annualised) unless a function says otherwise; annualising is
a monotone rescaling and does not change any significance verdict here.

References (see paper/FORMULAS.md for exact equations):
  PSR  - Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier"
  DSR  - Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
  HL   - Harvey & Liu (2015), "Backtesting" (Bonferroni / Holm / BHY haircuts)
  RC   - White (2000), "A Reality Check for Data Snooping"
  SPA  - Hansen (2005), "A Test for Superior Predictive Ability"
  SB   - Politis & Romano (1994), "The Stationary Bootstrap"
"""
import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329
E = float(np.e)


# --------------------------------------------------------------------------- #
# Sharpe + higher moments
# --------------------------------------------------------------------------- #
def sharpe(returns, ddof=1):
    """Per-observation Sharpe = mean / std (0 if degenerate)."""
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=ddof)
    return float(r.mean() / sd) if sd > 0 else 0.0


def moments(returns):
    """Return (skew g3, NON-excess kurtosis g4). Normal -> (0, 3)."""
    r = np.asarray(returns, dtype=float)
    n = r.size
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd == 0:
        return 0.0, 3.0
    z = (r - mu) / sd
    g3 = float((z ** 3).mean())
    g4 = float((z ** 4).mean())  # non-excess: normal == 3
    return g3, g4


# --------------------------------------------------------------------------- #
# PSR / DSR   (Bailey & Lopez de Prado 2012, 2014)
# --------------------------------------------------------------------------- #
def psr(sr_hat, sr_benchmark, T, skew=0.0, kurt=3.0):
    """Probabilistic Sharpe Ratio: P(true SR > sr_benchmark | data).

    sr_hat, sr_benchmark are per-observation Sharpes; T observations;
    skew g3, kurt g4 NON-excess (normal kurt == 3).
    """
    denom = np.sqrt(max(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * sr_hat ** 2, 1e-12))
    z = (sr_hat - sr_benchmark) * np.sqrt(T - 1.0) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance, N, mean_sr=0.0):
    """E[max of N independent SR estimates ~ N(mean_sr, sr_variance)]
    (Bailey & LdP 2014). Matches the authors' reference getExpMaxSR:

        SR0 = mean_sr + sqrt(V) * ((1-gamma) Z^{-1}(1 - 1/N)
                                   + gamma   Z^{-1}(1 - 1/(N e))).

    Under the null E[SR_n]=0 the mean term vanishes; we carry the empirical
    trial-SR mean to match the reference implementation faithfully.
    """
    if N is None or sr_variance <= 0:
        return float(mean_sr)
    # The EVT expected-maximum is only defined for N >= 2 trials: as N -> 1 the
    # (1-gamma) Z^{-1}(1 - 1/N) term diverges to -inf. A fractional effective N
    # in (1, 2) is a degenerate regime, so we clamp to 2 (the expected max of at
    # least two trials) rather than silently returning the trial mean.
    N = max(float(N), 2.0)
    g = EULER_MASCHERONI
    a = norm.ppf(1.0 - 1.0 / N)
    b = norm.ppf(1.0 - 1.0 / (N * E))
    return float(mean_sr + np.sqrt(sr_variance) * ((1.0 - g) * a + g * b))


def deflated_sharpe(sr_max, sr_estimates, T, skew=0.0, kurt=3.0, N=None):
    """DSR = PSR(sr_max, SR0), where SR0 is the deflated benchmark from the
    mean and dispersion of all trial Sharpes. Returns (dsr, sr0).

    N defaults to the trial count; pass an *effective* N to account for
    correlated trials.
    """
    sr_estimates = np.asarray(sr_estimates, dtype=float)
    v = float(sr_estimates.var(ddof=1))
    m = float(sr_estimates.mean())
    if N is None:
        N = sr_estimates.size
    sr0 = expected_max_sharpe(v, N, mean_sr=m)
    return psr(sr_max, sr0, T, skew, kurt), sr0


def effective_n_trials(returns_matrix):
    """A simple effective-trial count from the average pairwise correlation of
    the trial return streams: N_eff = N / (1 + (N-1) * rho_bar), clipped to
    [1, N]. Crude but monotone and reviewer-legible (correlated trials -> fewer
    independent bets). This is the SMALLEST of the standard estimators; see
    n_eff_estimators() for the full spread."""
    R = np.asarray(returns_matrix, dtype=float)
    N = R.shape[1]
    if N < 2:
        return float(N)
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    iu = np.triu_indices(N, k=1)
    rho_bar = max(float(np.nan_to_num(np.mean(C[iu]), nan=0.0)), 0.0)
    neff = N / (1.0 + (N - 1) * rho_bar)
    return float(min(max(neff, 1.0), N))


def n_eff_estimators(returns_matrix):
    """The spread of standard 'effective number of independent tests' estimators
    for a set of correlated trial return streams. There is no single right
    answer; reporting the spread is the honest move. Returns a dict:
      avg_corr          - N/(1+(N-1) rho_bar)                  (smallest)
      participation     - (sum lam)^2 / sum lam^2  (participation ratio)
      pca_95            - # eigenvalues to reach 95% of variance
      kaiser            - # eigenvalues > 1
      cheverud_nyholt   - 1 + (M-1)(1 - Var(lam)/M)            (largest; known to
                          over-count under strong equicorrelation)
    lam = eigenvalues of the KxK correlation matrix.
    """
    R = np.asarray(returns_matrix, dtype=float)
    K = R.shape[1]
    if K < 2:
        return {k: float(K) for k in
                ("avg_corr", "participation", "pca_95", "kaiser", "cheverud_nyholt")}
    C = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0.0, None)
    total = float(lam.sum())                       # == K for a correlation matrix
    lam_desc = np.sort(lam)[::-1]
    iu = np.triu_indices(K, k=1)
    rho_bar = max(float(np.nan_to_num(np.mean(C[iu]), nan=0.0)), 0.0)
    cum = np.cumsum(lam_desc) / total
    out = dict(
        avg_corr=K / (1.0 + (K - 1) * rho_bar),
        participation=(total ** 2) / float(np.sum(lam ** 2)),
        pca_95=float(int(np.searchsorted(cum, 0.95)) + 1),
        kaiser=float(int(np.sum(lam > 1.0))),
        cheverud_nyholt=1.0 + (K - 1) * (1.0 - float(np.var(lam)) / K),
    )
    return {k: float(min(max(v, 1.0), K)) for k, v in out.items()}


def dsr_neff_crossing(sr_max, sr_estimates, T, skew=0.0, kurt=3.0, alpha=0.05):
    """Largest effective trial count N for which DSR still clears (1 - alpha).
    Returns a float N* (the robustness band is 'DSR retains for N < N*'), or a
    value < 2 if the winner never clears 0.95 for any N >= 2 (rejected for all N).
    """
    K = np.asarray(sr_estimates).size
    lo, hi = 2.0, float(K)
    if deflated_sharpe(sr_max, sr_estimates, T, skew, kurt, N=lo)[0] <= (1 - alpha):
        return lo - 1.0                            # never survives (even at N=2)
    if deflated_sharpe(sr_max, sr_estimates, T, skew, kurt, N=hi)[0] > (1 - alpha):
        return hi                                  # survives even at the raw count
    for _ in range(60):                            # bisection on monotone DSR(N)
        mid = 0.5 * (lo + hi)
        if deflated_sharpe(sr_max, sr_estimates, T, skew, kurt, N=mid)[0] > (1 - alpha):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Harvey-Liu haircut   (Harvey & Liu 2015)
# --------------------------------------------------------------------------- #
def _one_sided_p(t):
    return 1.0 - norm.cdf(t)


def haircut(sr_all, T, method="bonferroni"):
    """Multiple-testing haircut of the BEST strategy's Sharpe.

    method in {bonferroni, holm, bhy}. Returns a dict with the adjusted
    p-value of the best strategy, its haircut Sharpe, and the haircut fraction
    (1 - SR_adj/SR_orig). t-stat = SR * sqrt(T); one-sided p tests SR > 0.
    """
    sr_all = np.asarray(sr_all, dtype=float)
    M = sr_all.size
    t = sr_all * np.sqrt(T)
    p = _one_sided_p(t)
    order = np.argsort(p)          # ascending p (best first)
    p_sorted = p[order]

    if method == "bonferroni":
        p_adj_sorted = np.minimum(1.0, M * p_sorted)
    elif method == "holm":
        factors = (M - np.arange(M)).astype(float)
        p_adj_sorted = np.maximum.accumulate(np.minimum(1.0, factors * p_sorted))
    elif method == "bhy":
        cM = float(np.sum(1.0 / np.arange(1, M + 1)))
        ranks = np.arange(1, M + 1, dtype=float)
        raw = p_sorted * M * cM / ranks
        p_adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
        p_adj_sorted = np.minimum(1.0, p_adj_sorted)
    else:
        raise ValueError(method)

    p_adj = np.empty(M)
    p_adj[order] = p_adj_sorted
    best = int(order[0])
    p_adj_best = float(np.clip(p_adj[best], 1e-16, 1 - 1e-16))
    t_adj = float(norm.ppf(1.0 - p_adj_best))
    sr_orig = float(sr_all[best])
    sr_hc = max(t_adj, 0.0) / np.sqrt(T)
    hc = 1.0 - sr_hc / sr_orig if sr_orig > 0 else 1.0
    return dict(method=method, best=best, p_single=float(p[best]),
                p_adj_best=p_adj_best, sr_original=sr_orig,
                sr_haircut=float(sr_hc), haircut=float(hc))


# --------------------------------------------------------------------------- #
# White Reality Check / Hansen SPA   (stationary bootstrap)
# --------------------------------------------------------------------------- #
def _sb_indices(T, avg_block, n_boot, rng):
    """Politis-Romano stationary bootstrap index matrix (n_boot, T)."""
    p = 1.0 / avg_block
    restart = rng.random((n_boot, T)) < p
    restart[:, 0] = True
    starts = rng.integers(0, T, size=(n_boot, T))
    idx = np.empty((n_boot, T), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, T):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def reality_check(ret_matrix, n_boot=1000, avg_block=10, seed=0, studentized=False):
    """White (2000) Reality Check p-value (studentized=True -> SPA-style).

    ret_matrix: (T, K) strategy returns in EXCESS of the benchmark. Null:
    max_k E[r_k] <= 0. Statistic V = max_k sqrt(T) * mean_k (studentized by
    per-strategy std if requested). Bootstrap recenters by the sample mean.
    """
    R = np.asarray(ret_matrix, dtype=float)
    T, K = R.shape
    rng = np.random.default_rng(seed)
    means = R.mean(0)
    stds = R.std(0, ddof=1)
    scale = np.where(stds > 0, stds, 1.0) if studentized else np.ones(K)
    V_obs = float(np.max(np.sqrt(T) * means / scale))
    idx = _sb_indices(T, avg_block, n_boot, rng)
    # bootstrap means per strategy: (n_boot, K)
    bmeans = R[idx].mean(axis=1)
    stat = np.sqrt(T) * (bmeans - means) / scale
    Vb = stat.max(axis=1)
    return float((np.sum(Vb >= V_obs) + 1) / (n_boot + 1))
