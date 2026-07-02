#!/usr/bin/env python3
"""Verify every numeric claim in paper/main.tex against results/results.json.

Two-way check:
  1. Forward: every claim in CLAIMS must (a) appear in the manuscript body as
     the exact literal (word-boundary matched, at least `min_count` times) and
     (b) agree with the value computed from results.json (or explicitly
     documented derived arithmetic / code constants) within rounding tolerance
     (half a unit in the last quoted decimal).
  2. Reverse: after removing all claim literals and a tiny allowlist of
     structural patterns (equation-number provenance like "Eq.~(11)"), the
     manuscript body must contain NO remaining multi-digit or decimal numeric
     literals. Single digits are exempt (math notation: T-1, kurtosis 3, ...).

Exit code 0 iff everything passes. Run: python3 scripts/check_paper_numbers.py
"""
import json
import math
import os
import re
import sys

from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, "paper", "main.tex")
RESULTS = os.path.join(ROOT, "results", "results.json")

with open(RESULTS) as f:
    R = json.load(f)

SQ = math.sqrt(R["meta"]["periods_per_year"])        # annualization factor
NC = R["null_calibration"]
PW = R["planted_power"]["curve"]
SN = R["search_noise"]
SS = R["search_signal"]
CFG = R["meta"]["config"]
H1000 = sum(1.0 / i for i in range(1, 1001))          # c(1000), harmonic sum
EMC = 0.5772156649015329                              # Euler-Mascheroni


# --------------------------------------------------------------------------- #
# derived quantities (documented arithmetic on results.json entries)
# --------------------------------------------------------------------------- #
def _evt_mult(N):
    """The N-dependent multiplier of sqrt(V) in the SR0 formula (BLdP 2014
    Eq. 1): (1-gamma) Phi^-1(1-1/N) + gamma Phi^-1(1-1/(N e))."""
    return ((1 - EMC) * norm.ppf(1 - 1 / N)
            + EMC * norm.ppf(1 - 1 / (N * math.e)))


def _sqrt_v_perobs(S):
    """Per-observation trial-Sharpe dispersion sqrt(V_SR), recovered from two
    reported (n_eff, SR0) pairs by inverting the SR0 identity: SR0(N) = mean +
    sqrt(V) * mult(N), so sqrt(V) = dSR0/dmult. Pure derived arithmetic."""
    a = S["dsr_by_neff"]["pca_95"]
    b = S["dsr_by_neff"]["cheverud_nyholt"]
    return ((b["sr0_annual"] - a["sr0_annual"])
            / ((_evt_mult(b["n_eff"]) - _evt_mult(a["n_eff"])) * SQ))


def _power50_annual():
    """50%-power point: linear interpolation between the two adjacent grid
    levels of the power curve that bracket power 0.5."""
    lo, hi = PW[1], PW[2]
    frac = (0.5 - lo["power_dsr"]) / (hi["power_dsr"] - lo["power_dsr"])
    return lo["s_true_annual"] + frac * (hi["s_true_annual"] - lo["s_true_annual"])


# --------------------------------------------------------------------------- #
# internal consistency of results.json itself (annualization identities etc.)
# --------------------------------------------------------------------------- #
def _consistency():
    errs = []
    if abs(NC["best_sr_annual_mean"] - NC["best_sr_perobs_mean"] * SQ) > 1e-9:
        errs.append("null: annual best != perobs * sqrt(252)")
    for tag, S in (("noise", SN), ("signal", SS)):
        if abs(S["best_sr_annual"] - S["best_sr_perobs"] * SQ) > 1e-9:
            errs.append(f"{tag}: annual best != perobs * sqrt(252)")
        if abs(S["sr0_annual"] - S["sr0_perobs"] * SQ) > 1e-9:
            errs.append(f"{tag}: sr0_annual != sr0_perobs * sqrt(252)")
        if S["n_boot"] != CFG["n_boot_case"]:
            errs.append(f"{tag}: n_boot != config n_boot_case")
        est = S["n_eff_estimators"]
        if abs(est["avg_corr"] - S["n_eff"]) > 1e-9:
            errs.append(f"{tag}: n_eff_estimators.avg_corr != n_eff")
        for name, ne in est.items():
            if not (1.0 <= ne <= S["K"]):
                errs.append(f"{tag}: n_eff[{name}] outside [1, K]")
            if abs(S["dsr_by_neff"][name]["n_eff"] - ne) > 1e-9:
                errs.append(f"{tag}: dsr_by_neff[{name}] n_eff mismatch")
    for row in PW:
        if abs(row["s_true_annual"] - row["s_true_perobs"] * SQ) > 1e-9:
            errs.append("power: annual != perobs * sqrt(252)")
    for k, v in dict(N=1000, T=1000, M_cal=2000, M_rc=400, n_boot=500,
                     n_boot_case=5000, M_pow=1000, n_true=25,
                     T_search=756).items():
        if CFG[k] != v:
            errs.append(f"config {k} != {v}")
    # the two searches must recover consistent dispersion from either pair of
    # (n_eff, SR0) points -- guards the derived sqrt(V) claims
    for tag, S in (("noise", SN), ("signal", SS)):
        a, b = S["dsr_by_neff"]["participation"], S["dsr_by_neff"]["kaiser"]
        alt = ((b["sr0_annual"] - a["sr0_annual"])
               / ((_evt_mult(b["n_eff"]) - _evt_mult(a["n_eff"])) * SQ))
        if abs(alt - _sqrt_v_perobs(S)) > 1e-9:
            errs.append(f"{tag}: sqrt(V) inversion inconsistent across pairs")
    return errs


# --------------------------------------------------------------------------- #
# code constants quoted in the paper (DGP parameters, Euler-Mascheroni, PCA
# threshold): verified by literal presence in the implementation source.
# --------------------------------------------------------------------------- #
CODE_CONSTANTS = [
    # (needle in source file, file, meaning)
    ("vol=0.01", "scripts/run_all.py", "noise std per observation"),
    ("0.006", "scripts/run_all.py", "regime drift per observation"),
    ("switch=0.02", "scripts/run_all.py", "regime switch probability"),
    ("0.5772156649", "scripts/deflate.py", "Euler-Mascheroni constant"),
    ("0.95", "scripts/deflate.py", "PCA-95 variance threshold"),
]

VOL, DRIFT, SWITCH = 0.01, 0.006, 0.02               # mirror of the needles


def _code_constants():
    errs = []
    for needle, rel, what in CODE_CONSTANTS:
        with open(os.path.join(ROOT, rel)) as f:
            if needle not in f.read():
                errs.append(f"code constant '{needle}' ({what}) not in {rel}")
    return errs


# --------------------------------------------------------------------------- #
# the claims: (label, tex literal, value, min_count)
# tolerance = half a unit in the literal's last decimal place (+eps);
# integer literals must match exactly. Scientific literals ("a\times 10^{b}")
# are compared at the quoted mantissa precision.
# --------------------------------------------------------------------------- #
DN, DS = SN["dsr_by_neff"], SS["dsr_by_neff"]
EN, ES = SN["n_eff_estimators"], SS["n_eff_estimators"]

CLAIMS = [
    # --- configuration / meta ---------------------------------------------
    ("N = T = M_pow", "1000", 1000, 1),
    ("M_cal repetitions", "2000", CFG["M_cal"], 1),
    ("M_rc repetitions", "400", CFG["M_rc"], 1),
    ("null RC bootstrap resamples", "500", CFG["n_boot"], 1),
    ("case-study bootstrap resamples", "5000", CFG["n_boot_case"], 2),
    ("planted strategies", "25", CFG["n_true"], 1),
    ("search price returns", "756", CFG["T_search"], 1),
    ("search strategy returns", "755", SN["T"], 1),
    ("trials K", "640", SN["K"], 1),
    ("periods per year", "252", R["meta"]["periods_per_year"], 1),
    ("alpha / per-obs SR level", "0.05", R["meta"]["alpha"], 1),
    ("expected block length", "20", R["meta"]["avg_block"], 1),
    ("fast grid size", "16", len(R["meta"]["fasts"]), 1),
    ("slow grid size", "40", len(R["meta"]["slows"]), 1),
    ("max fast", "48", max(R["meta"]["fasts"]), 1),
    ("min slow", "55", min(R["meta"]["slows"]), 1),
    ("second slow", "60", R["meta"]["slows"][1], 1),
    ("max slow", "250", max(R["meta"]["slows"]), 1),
    ("DSR threshold (derived 1-alpha)", "0.95", 1 - R["meta"]["alpha"], 2),
    ("BHY constant c(1000) (derived)", "7.49", H1000, 1),
    # DGP constants (verified against source, see CODE_CONSTANTS)
    ("noise std (code)", "0.01", VOL, 1),
    ("regime drift (code)", "0.006", DRIFT, 1),
    ("switch probability (code)", "0.02", SWITCH, 1),
    ("regime length (derived 1/switch)", "50", 1.0 / SWITCH, 2),
    ("50%-power label", "50", 50, 2),
    ("omniscient per-obs SR (derived)", "0.6", DRIFT / VOL, 1),
    ("omniscient annual SR (derived)", "9.5", DRIFT / VOL * SQ, 1),
    ("Euler-Mascheroni (code)", "0.5772", EMC, 1),
    ("PCA variance threshold (code, percent)", "95", 95, 2),
    # --- experiment 1: null calibration ------------------------------------
    ("null FDR naive", "1.000", NC["fdr"]["naive"], 3),
    ("null FDR DSR", "0.001", NC["fdr"]["dsr"], 2),
    ("null FDR Bonferroni", "0.057", NC["fdr"]["hl_bonferroni"], 2),
    ("null FDR Holm", "0.057", NC["fdr"]["hl_holm"], 2),
    ("null FDR BHY", "0.007", NC["fdr"]["hl_bhy"], 2),
    ("BHY theory alpha/c(1000) (derived)", "0.007",
     R["meta"]["alpha"] / H1000, 2),
    ("null FDR Reality Check", "0.0225", NC["fdr"]["reality_check"], 1),
    ("null best annual SR", "1.63", NC["best_sr_annual_mean"], 2),
    ("null SR0 annual (derived)", "1.63", NC["sr0_perobs_mean"] * SQ, 2),
    ("null naive p median", "0.00069", NC["naive_p_median"], 1),
    ("null DSR mean", "0.495", NC["dsr_mean"], 1),
    ("null DSR calibrated target", "0.5", 0.5, 1),
    ("MC standard error (derived)", "0.005",
     math.sqrt(0.05 * 0.95 / CFG["M_cal"]), 2),
    # --- experiment 2: planted power ----------------------------------------
    ("power perobs 1", "0.05", PW[0]["s_true_perobs"], 1),
    ("power perobs 2", "0.08", PW[1]["s_true_perobs"], 1),
    ("power perobs 3", "0.12", PW[2]["s_true_perobs"], 1),
    ("power perobs 4", "0.16", PW[3]["s_true_perobs"], 1),
    ("power perobs 5", "0.20", PW[4]["s_true_perobs"], 1),
    ("power annual 1", "0.79", PW[0]["s_true_annual"], 1),
    ("power annual 2", "1.27", PW[1]["s_true_annual"], 1),
    ("power annual 3", "1.90", PW[2]["s_true_annual"], 1),
    ("power annual 4", "2.54", PW[3]["s_true_annual"], 1),
    ("power annual 5", "3.17", PW[4]["s_true_annual"], 1),
    ("power at level 1", "0.005", PW[0]["power_dsr"], 2),
    ("power at level 2", "0.090", PW[1]["power_dsr"], 1),
    ("power at level 3", "0.651", PW[2]["power_dsr"], 1),
    ("power at level 4", "0.998", PW[3]["power_dsr"], 1),
    ("power at level 5", "1.000", PW[4]["power_dsr"], 3),
    ("power FP rate", "0.000", max(r["fp_rate_dsr"] for r in PW), 1),
    ("naive hit 1", "0.670", PW[0]["naive_truehit"], 1),
    ("naive hit 2", "0.984", PW[1]["naive_truehit"], 1),
    ("naive hit 3-5", "1.000", PW[2]["naive_truehit"], 3),
    ("50%-power point annual (derived)", "1.73", _power50_annual(), 2),
    # --- experiment 3: search on noise --------------------------------------
    ("noise winner params", "(45, 120)", tuple(SN["best_params"]), 1),
    ("noise best annual SR", "0.81", SN["best_sr_annual"], 1),
    ("noise naive p", "0.081", SN["naive_p"], 1),
    ("noise PSR vs zero", "0.918", SN["psr_vs_zero"], 2),
    ("noise mean pairwise corr", "0.61", SN["mean_pairwise_corr"], 2),
    ("noise SR0 annual (raw K)", "0.91", SN["sr0_annual"], 1),
    ("noise DSR raw N", "0.431", SN["dsr"], 1),
    ("noise RC p", "0.570", SN["reality_check_p"], 2),
    ("noise SPA-type p", "0.569", SN["spa_studentized_p"], 2),
    ("noise haircut Bonferroni", "1.00", SN["haircut_bonferroni"], 3),
    ("noise haircut Holm", "1.00", SN["haircut_holm"], 3),
    ("noise haircut BHY", "1.00", SN["haircut_bhy"], 3),
    ("noise Holm-adjusted p", "1.00", SN["p_adj_holm"], 3),
    ("noise sqrt(V) perobs (derived)", "0.014", _sqrt_v_perobs(SN), 2),
    # noise effective-N band
    ("noise n_eff avg-corr", "1.6", EN["avg_corr"], 2),
    ("noise n_eff participation", "2.4", EN["participation"], 2),
    ("noise n_eff PCA-95", "17", EN["pca_95"], 1),
    ("noise n_eff Kaiser", "22", EN["kaiser"], 1),
    ("noise n_eff Cheverud-Nyholt", "379.9", EN["cheverud_nyholt"], 1),
    ("noise bar avg-corr", "0.31", DN["avg_corr"]["sr0_annual"], 1),
    ("noise bar participation", "0.35", DN["participation"]["sr0_annual"], 1),
    ("noise bar PCA-95", "0.61", DN["pca_95"]["sr0_annual"], 2),
    ("noise bar Kaiser", "0.64", DN["kaiser"]["sr0_annual"], 1),
    ("noise bar Cheverud-Nyholt", "0.87", DN["cheverud_nyholt"]["sr0_annual"], 1),
    ("noise DSR avg-corr", "0.805", DN["avg_corr"]["dsr"], 1),
    ("noise DSR participation", "0.785", DN["participation"]["dsr"], 1),
    ("noise DSR PCA-95", "0.633", DN["pca_95"]["dsr"], 1),
    ("noise DSR Kaiser", "0.616", DN["kaiser"]["dsr"], 1),
    ("noise DSR Cheverud-Nyholt", "0.455", DN["cheverud_nyholt"]["dsr"], 1),
    # --- experiment 4: search on a real edge --------------------------------
    ("signal winner params", "(3, 55)", tuple(SS["best_params"]), 1),
    ("signal best annual SR", "3.92", SS["best_sr_annual"], 1),
    ("signal naive p", r"6.1\times 10^{-12}", SS["naive_p"], 1),
    ("signal PSR vs zero", "1.000", SS["psr_vs_zero"], 3),
    ("signal mean pairwise corr", "0.62", SS["mean_pairwise_corr"], 1),
    ("signal SR0 annual (raw K)", "3.51", SS["sr0_annual"], 1),
    ("signal DSR raw N", "0.748", SS["dsr"], 1),
    ("signal RC p", "0.0024", SS["reality_check_p"], 2),
    ("signal SPA-type p", "0.0038", SS["spa_studentized_p"], 2),
    ("signal haircut Bonferroni", "0.148", SS["haircut_bonferroni"], 2),
    ("signal haircut Holm", "0.148", SS["haircut_holm"], 2),
    ("signal haircut BHY", "0.198", SS["haircut_bhy"], 2),
    ("signal Holm-adjusted p", r"3.9\times 10^{-9}", SS["p_adj_holm"], 1),
    ("signal haircut SR Holm annual (derived)", "3.33",
     SS["best_sr_annual"] * (1 - SS["haircut_holm"]), 1),
    ("signal haircut SR BHY annual (derived)", "3.14",
     SS["best_sr_annual"] * (1 - SS["haircut_bhy"]), 1),
    ("signal sqrt(V) perobs (derived)", "0.079", _sqrt_v_perobs(SS), 2),
    # signal effective-N band
    ("signal n_eff avg-corr", "1.6", ES["avg_corr"], 2),
    ("signal n_eff participation", "2.4", ES["participation"], 2),
    ("signal n_eff PCA-95", "16", ES["pca_95"], 1),
    ("signal n_eff Kaiser", "21", ES["kaiser"], 1),
    ("signal n_eff Cheverud-Nyholt", "370.0", ES["cheverud_nyholt"], 2),
    ("signal bar avg-corr", "0.25", DS["avg_corr"]["sr0_annual"], 1),
    ("signal bar participation", "0.43", DS["participation"]["sr0_annual"], 1),
    ("signal bar PCA-95", "1.85", DS["pca_95"]["sr0_annual"], 2),
    ("signal bar Kaiser", "2.00", DS["kaiser"]["sr0_annual"], 2),
    ("signal bar Cheverud-Nyholt", "3.31", DS["cheverud_nyholt"]["sr0_annual"], 1),
    ("signal DSR avg-corr", "1.000", DS["avg_corr"]["dsr"], 3),
    ("signal DSR participation", "1.000", DS["participation"]["dsr"], 3),
    ("signal DSR PCA-95", "1.000", DS["pca_95"]["dsr"], 3),
    ("signal DSR Kaiser", "0.999", DS["kaiser"]["dsr"], 1),
    ("signal DSR Cheverud-Nyholt", "0.845", DS["cheverud_nyholt"]["dsr"], 1),
    ("signal survival crossing N*", "144.8", SS["neff_survive_max"], 2),
    # --- software versions ---------------------------------------------------
    ("python version", "3.14.6", R["meta"]["python"], 1),
    ("numpy version", "2.4.3", R["meta"]["numpy"], 1),
]

# structural patterns removed before the reverse sweep (with justification):
ALLOWLIST_PATTERNS = [
    # equation-number provenance quoted from primary sources, e.g. Eq.~(11)
    r"Eqs?\.~\(\d+\)(?:--\(\d+\))?",
]

SCI_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\\times 10\^\{(-?\d+)\}$")


def _parse_literal(lit):
    """Return (numeric value, tolerance) for a claim literal, or None if the
    literal is non-numeric (version string, parameter tuple)."""
    m = SCI_RE.match(lit)
    if m:
        mant, expo = m.group(1), int(m.group(2))
        dec = len(mant.split(".")[1]) if "." in mant else 0
        return float(mant) * 10.0 ** expo, (0.5 * 10.0 ** -dec + 1e-12) * 10.0 ** expo
    if re.fullmatch(r"-?\d+\.\d+", lit):
        dec = len(lit.split(".")[1])
        return float(lit), 0.5 * 10.0 ** -dec + 1e-9
    if re.fullmatch(r"-?\d+", lit):
        return float(lit), 1e-9
    return None  # tuple / version string


def _body(tex):
    """Manuscript body: comments stripped, preamble and refs commands removed."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, re.S)
    body = m.group(1)
    body = re.sub(
        r"\\(?:eqref|ref|label|pageref|cite[tp]?\*?|bibliographystyle|"
        r"bibliography)\s*(?:\[[^\]]*\])*\{[^}]*\}", " ", body)
    return body


def main():
    failures = []
    failures += [f"[consistency] {e}" for e in _consistency()]
    failures += [f"[code-const] {e}" for e in _code_constants()]

    with open(TEX) as f:
        body = _body(f.read())

    # ---- forward pass: count + remove literals, longest first -------------
    literals = sorted({c[1] for c in CLAIMS}, key=len, reverse=True)
    counts = {}
    text = body
    for lit in literals:
        esc = re.escape(lit)
        if lit[0].isdigit() or lit[0] == "-":
            pat = re.compile(r"(?<![\d.])" + esc + r"(?!\d)")
        else:
            pat = re.compile(esc)
        counts[lit] = len(pat.findall(text))
        text = pat.sub(" ", text)

    for label, lit, value, min_count in CLAIMS:
        if counts[lit] < min_count:
            failures.append(f"[presence] {label}: literal '{lit}' found "
                            f"{counts[lit]}x, need >= {min_count}")
        parsed = _parse_literal(lit)
        if parsed is not None:
            num, tol = parsed
            if abs(num - float(value)) > tol:
                failures.append(f"[value] {label}: literal '{lit}' vs "
                                f"results value {value!r} (tol {tol:g})")
        elif lit.startswith("("):                       # parameter tuple
            want = tuple(int(x) for x in re.findall(r"\d+", lit))
            if want != tuple(value):
                failures.append(f"[value] {label}: '{lit}' vs {value!r}")
        else:                                            # version string
            if lit != str(value):
                failures.append(f"[value] {label}: '{lit}' vs {value!r}")

    # ---- reverse pass: no unexplained numeric literals may remain ----------
    for pat in ALLOWLIST_PATTERNS:
        text = re.sub(pat, " ", text)
    leftovers = []
    for m in re.finditer(r"\d+(?:\.\d+)+|\d{2,}", text):
        ctx = text[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        leftovers.append(f"'{m.group(0)}' near: ...{ctx}...")
    for lo in leftovers:
        failures.append(f"[unexplained number] {lo}")

    if failures:
        print(f"check_paper_numbers: FAIL ({len(failures)} problem(s))")
        for f_ in failures:
            print("  -", f_)
        return 1
    print(f"check_paper_numbers: OK — {len(CLAIMS)} claims verified against "
          f"results.json; no unexplained numeric literals in main.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
