"""Deflated Sharpe / multiple-testing experiment harness.

Produces results/results.json. Four controlled experiments with KNOWN ground truth:

  1. null_calibration : N iid-noise strategies (true SR = 0), repeated M times.
     A well-calibrated selection test flags a "discovery" at most alpha of the
     time; the naive best-of-N single test flags almost always. This calibrates
     false-discovery control for DSR / Harvey-Liu / Reality Check.
  2. planted_power    : N strategies, a few with a genuine positive SR. Measures
     that the deflated tests still RETAIN real signals (power), not just reject.
  3. search_noise     : a real MA-crossover parameter search on a PURE RANDOM
     WALK (no edge). The winner looks great; deflation must kill it.
  4. search_signal    : the same search on a price with persistent regimes (a
     real trend edge). Deflation must let the winner survive.

Everything is seeded and deterministic. Run: python scripts/run_all.py [--quick]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deflate import (  # noqa: E402
    sharpe, moments, psr, deflated_sharpe, expected_max_sharpe,
    haircut, reality_check, effective_n_trials, n_eff_estimators,
    dsr_neff_crossing,
)

PERIODS_PER_YEAR = 252
ALPHA = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def ann(sr_perobs):
    return sr_perobs * np.sqrt(PERIODS_PER_YEAR)


# --------------------------------------------------------------------------- #
# 1. Null calibration
# --------------------------------------------------------------------------- #
def null_calibration(N, T, M, seed=0):
    rng = np.random.default_rng(seed)
    flags = {k: 0 for k in ["naive", "dsr", "hl_bonferroni", "hl_holm", "hl_bhy"]}
    best_sr, dsr_vals, sr0_vals, naive_p = [], [], [], []
    for _ in range(M):
        R = rng.standard_normal((T, N))              # true SR = 0 everywhere
        srs = R.mean(0) / R.std(0, ddof=1)
        j = int(np.argmax(srs))
        s_max = float(srs[j])
        best_sr.append(s_max)
        # naive single test on the winner (ignores that we searched N of them)
        t = s_max * np.sqrt(T)
        p1 = float(1.0 - _ncdf(t))
        naive_p.append(p1)
        if p1 < ALPHA:
            flags["naive"] += 1
        # DSR
        g3, g4 = moments(R[:, j])
        dsr, sr0 = deflated_sharpe(s_max, srs, T, g3, g4)
        dsr_vals.append(dsr); sr0_vals.append(sr0)
        if dsr > (1.0 - ALPHA):
            flags["dsr"] += 1
        # Harvey-Liu haircuts (discovery if adjusted p < alpha)
        for m in ("bonferroni", "holm", "bhy"):
            h = haircut(srs, T, method=m)
            if h["p_adj_best"] < ALPHA:
                flags["hl_" + m] += 1
    return dict(
        N=N, T=T, M=M, alpha=ALPHA,
        fdr={k: v / M for k, v in flags.items()},
        best_sr_perobs_mean=float(np.mean(best_sr)),
        best_sr_annual_mean=float(ann(np.mean(best_sr))),
        naive_p_median=float(np.median(naive_p)),
        dsr_mean=float(np.mean(dsr_vals)),
        sr0_perobs_mean=float(np.mean(sr0_vals)),
    )


def null_calibration_rc(N, T, M, n_boot, avg_block, seed=1):
    """Reality-Check false-discovery rate on the same null (separate, smaller M
    because the bootstrap is expensive)."""
    rng = np.random.default_rng(seed)
    flags = 0
    for _ in range(M):
        R = rng.standard_normal((T, N))
        p = reality_check(R, n_boot=n_boot, avg_block=avg_block,
                          seed=int(rng.integers(1 << 30)))
        if p < ALPHA:
            flags += 1
    return dict(N=N, T=T, M=M, n_boot=n_boot, avg_block=avg_block,
                reality_check_fdr=flags / M)


def _ncdf(x):
    from scipy.stats import norm
    return float(norm.cdf(x))


# --------------------------------------------------------------------------- #
# 2. Planted power
# --------------------------------------------------------------------------- #
def planted_power(N, n_true, s_true_list, T, M, seed=2):
    """DSR detection power as a function of the planted per-obs Sharpe. The
    bar DSR must clear is the noise-max benchmark SR0 (~annual 1.6 for N=1000),
    NOT zero -- so power is ~0 below it and rises sharply above it."""
    rng = np.random.default_rng(seed)
    curve = []
    for s_true in s_true_list:
        detect = tp = fp = naive_tp = 0
        for _ in range(M):
            R = rng.standard_normal((T, N))
            true_idx = rng.choice(N, size=n_true, replace=False)
            R[:, true_idx] += s_true                   # genuine per-obs edge
            srs = R.mean(0) / R.std(0, ddof=1)
            j = int(np.argmax(srs))
            is_true = j in set(true_idx.tolist())
            s_max = float(srs[j])
            g3, g4 = moments(R[:, j])
            dsr, _ = deflated_sharpe(s_max, srs, T, g3, g4)
            if dsr > (1.0 - ALPHA):
                detect += 1
                tp += int(is_true)
                fp += int(not is_true)
            if (1.0 - _ncdf(s_max * np.sqrt(T))) < ALPHA and is_true:
                naive_tp += 1
        curve.append(dict(
            s_true_perobs=float(s_true), s_true_annual=float(ann(s_true)),
            power_dsr=detect / M, tp_rate_dsr=tp / M, fp_rate_dsr=fp / M,
            naive_truehit=naive_tp / M,
        ))
    return dict(N=N, n_true=n_true, T=T, M=M, curve=curve)


# --------------------------------------------------------------------------- #
# 3-4. Realistic MA-crossover parameter search
# --------------------------------------------------------------------------- #
def gen_returns(T, rng, regime=False, drift=0.0, vol=0.01, switch=0.02):
    eps = rng.normal(0.0, vol, size=T)
    if not regime:
        return eps                                    # pure noise, E[r]=0
    state = 1
    d = np.empty(T)
    for t in range(T):
        if rng.random() < switch:
            state = -state
        d[t] = state * drift
    return d + eps


def ma_crossover_search(returns, fasts, slows, fee=0.0):
    """Causal MA-crossover grid. Position at t (from info <= t) earns r_{t+1}.
    Returns (ret_matrix (T-1, K), labels list of (fast, slow))."""
    logp = np.cumsum(returns)
    T = returns.size
    cols, labels = [], []
    for f in fasts:
        maf = _sma(logp, f)
        for s in slows:
            if s <= f:
                continue
            mas = _sma(logp, s)
            pos = np.sign(maf - mas)                   # +1 long / -1 short / 0
            pos = np.nan_to_num(pos)
            strat = pos[:-1] * returns[1:]             # causal: pos_t * r_{t+1}
            if fee:
                turn = np.abs(np.diff(np.concatenate([[0.0], pos[:-1]])))
                strat = strat - fee * turn
            cols.append(strat)
            labels.append((int(f), int(s)))
    return np.column_stack(cols), labels


def _sma(x, w):
    if w <= 1:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def analyse_search(ret_matrix, labels, T, n_boot, avg_block, seed):
    srs = np.array([sharpe(ret_matrix[:, k]) for k in range(ret_matrix.shape[1])])
    j = int(np.argmax(srs))
    s_max = float(srs[j])
    g3, g4 = moments(ret_matrix[:, j])
    N = ret_matrix.shape[1]
    naive_p = float(1.0 - _ncdf(s_max * np.sqrt(T)))
    psr_vs0 = psr(s_max, 0.0, T, g3, g4)                     # un-deflated significance
    dsr, sr0 = deflated_sharpe(s_max, srs, T, g3, g4, N=N)   # raw trial count

    # The spread of standard effective-#-of-trials estimators + the DSR each
    # implies. There is no single right n_eff; report the whole band.
    est = n_eff_estimators(ret_matrix)
    dsr_by_neff = {}
    for name, ne in est.items():
        d, s0 = deflated_sharpe(s_max, srs, T, g3, g4, N=ne)
        dsr_by_neff[name] = dict(n_eff=float(ne), dsr=float(d),
                                 sr0_annual=float(ann(s0)),
                                 survives=bool(d > (1.0 - ALPHA)))
    # largest n_eff for which DSR still survives (robustness band)
    neff_crossing = float(dsr_neff_crossing(s_max, srs, T, g3, g4, alpha=ALPHA))

    # backward-compatible avg-corr view
    n_eff = effective_n_trials(ret_matrix)
    dsr_eff, sr0_eff = deflated_sharpe(s_max, srs, T, g3, g4, N=n_eff)

    rc = reality_check(ret_matrix, n_boot=n_boot, avg_block=avg_block, seed=seed)
    spa = reality_check(ret_matrix, n_boot=n_boot, avg_block=avg_block,
                        seed=seed + 1, studentized=True)   # studentized RC (SPA-type)
    hcs = {m: haircut(srs, T, method=m) for m in ("bonferroni", "holm", "bhy")}
    return dict(
        K=N, T=T, n_boot=n_boot,
        best_params=[int(labels[j][0]), int(labels[j][1])],
        best_sr_perobs=s_max, best_sr_annual=float(ann(s_max)),
        naive_p=naive_p, psr_vs_zero=float(psr_vs0),
        sr0_perobs=float(sr0), sr0_annual=float(ann(sr0)), dsr=float(dsr),
        n_eff=float(n_eff), sr0_eff_perobs=float(sr0_eff), dsr_eff=float(dsr_eff),
        n_eff_estimators=est, dsr_by_neff=dsr_by_neff,
        neff_survive_max=neff_crossing,
        reality_check_p=float(rc), spa_studentized_p=float(spa),
        haircut_bonferroni=float(hcs["bonferroni"]["haircut"]),
        haircut_holm=float(hcs["holm"]["haircut"]),
        haircut_bhy=float(hcs["bhy"]["haircut"]),
        p_adj_holm=float(hcs["holm"]["p_adj_best"]),
        survives_dsr=bool(dsr > (1.0 - ALPHA)),
        survives_rc=bool(rc < ALPHA),
        mean_pairwise_corr=float(_mean_corr(ret_matrix)),
    )


def _mean_corr(R):
    C = np.corrcoef(R, rowvar=False)
    iu = np.triu_indices(C.shape[0], k=1)
    return float(np.nanmean(C[iu]))


def run_search_case(regime, seed, T, fasts, slows, drift, n_boot, avg_block):
    rng = np.random.default_rng(seed)
    rets = gen_returns(T, rng, regime=regime, drift=drift)
    RM, labels = ma_crossover_search(rets, fasts, slows)
    out = analyse_search(RM, labels, RM.shape[0], n_boot, avg_block, seed + 100)
    out["regime"] = regime
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        cfg = dict(N=200, T=500, M_cal=150, M_rc=60, n_boot=150, n_boot_case=300,
                   M_pow=120, n_true=8, T_search=400)
        s_true_list = [0.08, 0.16]
    else:
        cfg = dict(N=1000, T=1000, M_cal=2000, M_rc=400, n_boot=500, n_boot_case=5000,
                   M_pow=1000, n_true=25, T_search=756)
        s_true_list = [0.05, 0.08, 0.12, 0.16, 0.20]

    # A short (~3y daily) window with a wide grid -> the classic overfit setup:
    # many correlated trials, enough room for a spurious winner to look great.
    fasts = list(range(3, 51, 3))
    slows = list(range(55, 255, 5))
    avg_block = 20

    print("[1/4] null calibration (closed-form tests) ...", flush=True)
    nc = null_calibration(cfg["N"], cfg["T"], cfg["M_cal"], seed=0)
    print("[1b] null calibration (reality check) ...", flush=True)
    ncrc = null_calibration_rc(cfg["N"], cfg["T"], cfg["M_rc"],
                               cfg["n_boot"], avg_block, seed=1)
    nc["fdr"]["reality_check"] = ncrc["reality_check_fdr"]
    nc["rc_reps"] = ncrc["M"]

    print("[2/4] planted power (sweep) ...", flush=True)
    pw = planted_power(cfg["N"], cfg["n_true"], s_true_list, cfg["T"],
                       cfg["M_pow"], seed=2)

    print("[3/4] realistic search on NOISE ...", flush=True)
    noise = run_search_case(False, 10, cfg["T_search"], fasts, slows,
                            0.0, cfg["n_boot_case"], avg_block)
    print("[4/4] realistic search on a REAL edge (regimes) ...", flush=True)
    signal = run_search_case(True, 11, cfg["T_search"], fasts, slows,
                             0.006, cfg["n_boot_case"], avg_block)

    results = dict(
        meta=dict(
            seed=0, alpha=ALPHA, periods_per_year=PERIODS_PER_YEAR,
            quick=bool(args.quick),
            python=sys.version.split()[0], numpy=np.__version__,
            config=cfg, avg_block=avg_block, fasts=fasts, slows=slows,
        ),
        null_calibration=nc,
        planted_power=pw,
        search_noise=noise,
        search_signal=signal,
    )

    out = os.path.join(ROOT, "results",
                       "results_quick.json" if args.quick else "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)

    # headline
    print("\n=== headline ===")
    print(f"null best SR (annual) ~ {nc['best_sr_annual_mean']:.2f}, "
          f"naive p median {nc['naive_p_median']:.3g}")
    print("null FDR:", {k: round(v, 3) for k, v in nc["fdr"].items()})
    print(f"deflated bar SR0 (annual) ~ {ann(nc['sr0_perobs_mean']):.2f}")
    print("planted DSR power vs true annual SR:",
          {round(c["s_true_annual"], 2): round(c["power_dsr"], 2) for c in pw["curve"]})
    for tag, s in [("noise", noise), ("signal", signal)]:
        est = {k: round(v, 1) for k, v in s["n_eff_estimators"].items()}
        print(f"{tag} search: best annual SR {s['best_sr_annual']:.2f}, "
              f"PSR-vs-0 {s['psr_vs_zero']:.3f}, DSR(rawK) {s['dsr']:.3f}, "
              f"RC p {s['reality_check_p']:.4f}, SPA-type p {s['spa_studentized_p']:.4f}")
        print(f"    n_eff spread {est} | DSR survives up to n_eff={s['neff_survive_max']:.1f}")
        print(f"    DSR by n_eff: " +
              ", ".join(f"{k}={v['dsr']:.3f}({'Y' if v['survives'] else 'N'})"
                        for k, v in s["dsr_by_neff"].items()))


if __name__ == "__main__":
    main()
