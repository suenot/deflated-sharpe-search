# How Many Backtest Winners Survive Deflation?

A reproducible, controlled study of **deflating backtest Sharpe ratios under multiple
testing**. A parameter search is a multiple-testing machine: it reports the luckiest of
many trials, so the winner's naive significance is meaningless. We build experiments with
**known ground truth** and measure whether the Deflated Sharpe Ratio (DSR), the Harvey-Liu
haircut, and White's Reality Check restore honesty.

## Headline (from `results/results.json`, seeded, deterministic)

On **1,000 zero-edge strategies** (pure noise), the best annualized Sharpe averages **1.63**
with a single-test p-value of **0.0007** — and the naive "is it significant?" test declares a
discovery **100%** of the time. Principled deflation controls it:

| method | false-discovery rate on pure-noise searches (α = 0.05) |
|---|---|
| naive best-of-N single test | **1.000** |
| DSR | 0.001 |
| Harvey-Liu (Bonferroni / Holm) | 0.057 |
| Harvey-Liu (Benjamini-Yekutieli) | 0.007 |
| White Reality Check | 0.022 |

The deflated DSR bar `SR0` sits at annualized **≈1.63 — exactly the noise ceiling**, not zero.
DSR detection **power** is an S-curve that crosses just above that ceiling: a true edge below
the noise max is indistinguishable from luck; a strong one (annual ≥ 2.5) is retained with
power ≈ 1.

**The correlated-grid trap.** On a real MA-crossover parameter search (640 trials, avg pairwise
correlation ≈ 0.62), feeding DSR the **raw** trial count over-deflates and *false-rejects a genuine
regime edge* (DSR 0.748). The fix is the **effective** number of trials — but there is no single
right value, so we report the whole band of standard estimators and read the verdict across it:

| effective-N estimator | N_eff | DSR bar (annual) | DSR | signal verdict |
|---|---|---|---|---|
| average correlation | 1.6 | 0.25 | 1.000 | retain |
| participation ratio | 2.4 | 0.43 | 1.000 | retain |
| PCA (95% variance) | 16 | 1.85 | 1.000 | retain |
| Kaiser (λ > 1) | 21 | 2.00 | 0.999 | retain |
| Cheverud-Nyholt | 370 | 3.31 | 0.845 | reject |

The genuine edge is retained for **any N_eff < 145** (and at the defensible middle, N_eff = 16–21,
the bar is a *real* annual 1.85–2.00 penalty the edge still clears); the noise winner is rejected at
**every** N_eff. Only a near-independence estimator (Cheverud-Nyholt, which over-counts under
equicorrelation) flips it. White's Reality Check needs no trial count at all and cleanly separates the
two (noise p 0.57 vs signal p 0.0024). Pair the tools: DSR asks *"is the winner special within this
search?"*; the Reality Check asks *"does the best beat cash after data-snooping?"*.

This grew out of a [marketmaker.cc](https://marketmaker.cc) blog post.

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py            # full run -> results/results.json (~4 min)
python scripts/run_all.py --quick    # smoke run -> results/results_quick.json
python scripts/check_paper_numbers.py  # verify every number in the paper vs results.json
python -m pytest tests/ -q           # deterministic invariants (calibration, power, estimators)
tectonic paper/main.tex              # -> paper/main.pdf
```

The synthetic data (iid-Normal / regime-switching) is chosen for **controlled ground truth, not
market realism** — the deliverable is the calibrated method, not a strategy.

## Layout

```
scripts/
  deflate.py             # PSR, DSR (+SR0), Harvey-Liu Bonferroni/Holm/BHY,
                         # White RC / studentized-RC (SPA-type), effective-N estimators
  run_all.py             # 4 controlled experiments -> results.json
  check_paper_numbers.py # verifies every numeric claim in main.tex against results.json
tests/test_experiments.py  # deterministic invariants
results/results.json       # committed representative run
paper/main.tex             # the paper   |   paper/FORMULAS.md  formulas + provenance
```

## License

Code: MIT. Paper text and figures: CC BY 4.0.
