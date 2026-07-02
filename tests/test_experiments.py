"""Deterministic invariants for the deflation estimators and experiments.

These do not pin machine-specific magnitudes; they check the mathematical
properties the paper relies on (calibration, monotone power, PSR/DSR shape).
Run: python -m pytest tests/ -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from deflate import (  # noqa: E402
    psr, expected_max_sharpe, deflated_sharpe, haircut, reality_check, moments,
)
import run_all  # noqa: E402
from deflate import (  # noqa: E402
    effective_n_trials, n_eff_estimators, dsr_neff_crossing,
)


def test_psr_half_at_benchmark():
    assert abs(psr(0.1, 0.1, 1000, 0.0, 3.0) - 0.5) < 1e-9


def test_psr_monotone_in_sr():
    T = 1000
    assert psr(0.15, 0.0, T, 0.0, 3.0) > psr(0.05, 0.0, T, 0.0, 3.0)


def test_expected_max_increases_with_N():
    v = 0.001
    assert expected_max_sharpe(v, 1000) > expected_max_sharpe(v, 10) > 0.0


def test_expected_max_increases_with_variance():
    assert expected_max_sharpe(0.01, 100) > expected_max_sharpe(0.001, 100)


def test_dsr_kills_pure_noise():
    rng = np.random.default_rng(0)
    R = rng.standard_normal((1000, 500))            # all true SR = 0
    srs = R.mean(0) / R.std(0, ddof=1)
    j = int(np.argmax(srs))
    g3, g4 = moments(R[:, j])
    dsr, sr0 = deflated_sharpe(float(srs[j]), srs, 1000, g3, g4)
    assert dsr < 0.95 and sr0 > 0.0


def test_haircut_bounds_and_bonferroni_identity():
    rng = np.random.default_rng(1)
    sr = rng.standard_normal(200) / np.sqrt(500)
    h = haircut(sr, 500, method="bonferroni")
    assert 0.0 <= h["haircut"] <= 1.0
    assert abs(h["p_adj_best"] - min(1.0, 200 * h["p_single"])) < 1e-9


def test_reality_check_range_and_detects_edge():
    rng = np.random.default_rng(2)
    R = rng.standard_normal((500, 50)) * 0.01
    p_null = reality_check(R, n_boot=200, avg_block=10, seed=3)
    assert 0.0 < p_null <= 1.0
    R[:, 0] += 0.01                                  # plant a strong edge
    p_sig = reality_check(R, n_boot=200, avg_block=10, seed=3)
    assert p_sig < p_null


def test_null_calibration_invariants():
    nc = run_all.null_calibration(N=300, T=500, M=200, seed=0)
    assert nc["fdr"]["naive"] > 0.9                  # naive almost always fires
    assert nc["fdr"]["dsr"] <= 0.05                  # DSR controls FDR
    assert nc["fdr"]["hl_bonferroni"] <= 0.10
    assert nc["fdr"]["hl_bhy"] <= 0.05


def test_power_monotone_in_true_sharpe():
    pw = run_all.planted_power(N=300, n_true=10, s_true_list=[0.05, 0.20],
                               T=500, M=150, seed=2)
    powers = [c["power_dsr"] for c in pw["curve"]]
    assert powers[1] > powers[0]                     # stronger edge -> more power
    assert pw["curve"][0]["fp_rate_dsr"] <= 0.05     # few false positives


def test_bhy_numerator_convention():
    """The paper quotes BHY with c(M) = sum 1/j MULTIPLYING the adjustment
    (Benjamini-Yekutieli 2001, arbitrary dependence). With well-separated
    p-values the best trial's adjusted p must equal p * M * c(M)."""
    sr_all = np.array([0.5, 0.0, -0.1])
    T = 100
    h = haircut(sr_all, T, method="bhy")
    cM = 1.0 + 1.0 / 2.0 + 1.0 / 3.0
    assert abs(h["p_adj_best"] - h["p_single"] * 3 * cM) < 1e-12
    # numerator convention is MORE conservative than Bonferroni here
    assert h["p_adj_best"] > 3 * h["p_single"]


def test_effective_n_bounds():
    """N_eff = N / (1 + (N-1) rho_bar): perfectly correlated trials collapse
    to ~1 independent trial; independent trials keep close to N."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal(500)
    dup = np.column_stack([base] * 10)
    assert abs(effective_n_trials(dup) - 1.0) < 1e-6
    indep = rng.standard_normal((2000, 50))
    neff = effective_n_trials(indep)
    assert 20.0 < neff <= 50.0


def test_dsr_monotone_in_trial_count():
    """SR0 grows with N, so DSR falls with N: feeding a smaller effective N
    can only raise DSR (the mechanism behind the effective-N pitfall)."""
    rng = np.random.default_rng(4)
    R = rng.standard_normal((500, 100))
    srs = R.mean(0) / R.std(0, ddof=1)
    s_max = float(srs.max())
    dsr_raw, sr0_raw = deflated_sharpe(s_max, srs, 500, N=100)
    dsr_eff, sr0_eff = deflated_sharpe(s_max, srs, 500, N=5)
    assert sr0_eff < sr0_raw
    assert dsr_eff > dsr_raw


def test_expected_max_clamps_below_two():
    """N in (1,2) is degenerate for the EVT expected-max; the code clamps to 2
    rather than silently returning the trial mean (the bug the review caught)."""
    v, m = 0.01, 0.0
    assert expected_max_sharpe(v, 1.5, m) == expected_max_sharpe(v, 2.0, m)
    assert expected_max_sharpe(v, 1.5, m) > m          # not the mean fallback
    # zero-variance still collapses to the mean (genuinely degenerate)
    assert expected_max_sharpe(0.0, 5, 0.3) == 0.3


def test_n_eff_estimators_spread_and_bounds():
    rng = np.random.default_rng(9)
    base = rng.standard_normal((400, 1))
    R = 0.7 * base + 0.3 * rng.standard_normal((400, 60))   # correlated columns
    est = n_eff_estimators(R)
    assert all(1.0 <= v <= 60.0 for v in est.values())
    # avg-corr is the smallest; cheverud-nyholt the largest (over-counter)
    assert est["avg_corr"] <= est["cheverud_nyholt"]


def test_dsr_neff_crossing_monotone():
    rng = np.random.default_rng(10)
    R = rng.standard_normal((500, 50))
    srs = R.mean(0) / R.std(0, ddof=1)
    strong = dsr_neff_crossing(0.30, srs, 500)          # a huge winner
    weak = dsr_neff_crossing(0.05, srs, 500)            # a modest winner
    assert strong > weak                                # stronger edge survives more trials


def test_determinism():
    a = run_all.null_calibration(N=200, T=400, M=50, seed=7)
    b = run_all.null_calibration(N=200, T=400, M=50, seed=7)
    assert a["best_sr_perobs_mean"] == b["best_sr_perobs_mean"]
    assert a["fdr"] == b["fdr"]
