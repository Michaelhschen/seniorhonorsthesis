"""Independent Gaussian/forward-backward reconciliation of V2.1 saved fits.

No EM fitting or hmmlearn import. Scipy Gaussian log densities, explicit
forward/backward recursions, and Hungarian assignment verify the archived
candidate arrays, likelihoods, state-count comparisons and seed experiments.
"""
from pathlib import Path
import ast
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "output/v21_diagnostics"
BASE = ROOT / "package/original_run/outputs"
NAMES = ["full_information", "realtime_information", "nowcast_enhanced"]
CHECKS = 0
MAX_ERROR = 0.


def close(actual, expected, atol=1e-8):
    global CHECKS, MAX_ERROR
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.abs(actual - expected).max())
    MAX_ERROR = max(MAX_ERROR, error)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)
    CHECKS += 1


def main():
    candidates = pd.read_csv(A / "fit_candidates.csv")
    selected_rows = pd.read_csv(A / "state_count_selected.csv")
    spec = json.loads((ROOT / "config/analysis_spec.json").read_text())
    inputs = {n: pd.read_csv(BASE / f"pca_{n}/hmm_pca_input.csv") for n in NAMES}
    x = {n: frame[[f"PCA_{i}" for i in range(1, 6)]].to_numpy() for n, frame in inputs.items()}
    dates = pd.to_datetime(inputs[NAMES[0]].date)
    raw = pd.read_csv(BASE / "full_information.csv", index_col=0, parse_dates=True).loc[dates, spec["anchors"]]
    anchors = ((raw - raw.mean()) / raw.std(ddof=0)).to_numpy()
    weights = np.array(spec["anchor_weights"])
    fit = {}
    for row in candidates.itertuples():
        k, n, s = int(row.states), row.model, int(row.seed)
        arrays = np.load(A / f"K{k}_{n}_seed{s}.npz")
        emission = np.column_stack([
            multivariate_normal.logpdf(x[n], mean=arrays["means"][i], cov=arrays["covars"][i])
            for i in range(k)
        ])
        with np.errstate(divide="ignore"):
            logstart, logtrans = np.log(arrays["startprob"]), np.log(arrays["transmat"])
        alpha = np.empty_like(emission)
        alpha[0] = logstart + emission[0]
        for t in range(1, len(alpha)):
            alpha[t] = logsumexp(alpha[t-1, :, None] + logtrans, axis=0) + emission[t]
        ll = logsumexp(alpha[-1])
        filtered = np.exp(alpha - logsumexp(alpha, axis=1)[:, None])
        beta = np.zeros_like(emission)
        for t in range(len(beta)-2, -1, -1):
            beta[t] = logsumexp(logtrans + emission[t+1] + beta[t+1], axis=1)
        smoothed = np.exp(alpha + beta - ll)
        close(ll, row.log_likelihood)
        close(filtered, arrays["filtered"])
        close(smoothed, arrays["smoothed"])
        close(filtered.sum(1), np.ones(len(dates)))
        close(smoothed.sum(1), np.ones(len(dates)))
        parameters = (k-1) + k*(k-1) + 2*k*5
        close(row.n_parameters, parameters)
        close(row.BIC, -2*ll + parameters*np.log(len(dates)))
        close(row.minimum_posterior_mass, smoothed.sum(0).min())
        close(ast.literal_eval(row.posterior_mass), smoothed.sum(0))
        close(ast.literal_eval(row.hard_occupancy), np.bincount(smoothed.argmax(1), minlength=k))
        fit[k, n, s] = {"smoothed": smoothed, "filtered": filtered, "ll": ll}

    chosen = {}
    for r in selected_rows.itertuples():
        k, n = int(r.states), r.model
        best_seed = max(range(5), key=lambda s: fit[k, n, s]["ll"])
        close(int(r.seed), best_seed)
        chosen[k, n] = fit[k, n, best_seed]
    for r in candidates.itertuples():
        close(r.likelihood_gap_from_selected,
              chosen[int(r.states), r.model]["ll"] - fit[int(r.states), r.model, int(r.seed)]["ll"])

    def verify_comparison(row, fitted):
        profiles = {n: result["smoothed"].T @ anchors / result["smoothed"].sum(0)[:, None]
                    for n, result in fitted.items()}
        losses = []
        for n, mapkey in zip(NAMES[1:], ["LAG_mapping", "NC_mapping"]):
            mapping = ast.literal_eval(getattr(row, mapkey))
            costs = ((profiles[NAMES[0]][:, None] - profiles[n][None, :])**2 * weights).sum(2)
            rows, cols = linear_sum_assignment(costs)
            close(costs[np.arange(len(mapping)), mapping].sum(), costs[rows, cols].sum())
            losses.append(np.abs(fitted[NAMES[0]]["filtered"] - fitted[n]["filtered"][:, mapping]).sum(1).mean())
        lag, nc = losses
        close([row.LAG_L1, row.NC_L1], [lag, nc])
        close(row.absolute_advantage, lag-nc)
        close(row.reduction_percent, 100*(1-nc/lag))

    comparisons = pd.read_csv(A / "state_count_comparison.csv")
    for r in comparisons.itertuples():
        verify_comparison(r, {n: chosen[int(r.states), n] for n in NAMES})
    seeds = pd.read_csv(A / "controlled_initialization.csv")
    for r in seeds.itertuples():
        fitted = {n: fit[4, n, int(r.seed)] for n in NAMES}
        if r.reference_protocol == "fixed_selected_reference":
            fitted[NAMES[0]] = chosen[4, NAMES[0]]
        verify_comparison(r, fitted)
        for short, n in zip(["REV", "LAG", "NC"], NAMES):
            close(getattr(r, short+"_log_likelihood"), fitted[n]["ll"])
            close(getattr(r, short+"_likelihood_gap"), chosen[4, n]["ll"]-fitted[n]["ll"])
    isolated = pd.read_csv(A / "one_pipeline_initialization.csv")
    for r in isolated.itertuples():
        fitted = {n: chosen[4, n] for n in NAMES}
        fitted[r.varied_model] = fit[4, r.varied_model, int(r.seed)]
        verify_comparison(r, fitted)
        close(r.varied_fit_likelihood_gap, chosen[4, r.varied_model]["ll"]-fitted[r.varied_model]["ll"])

    manifest = json.loads((A / "manifest.json").read_text())
    for relative, digest in manifest["input_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((A / name).read_bytes()).hexdigest() == digest
    report = {"candidates": len(candidates), "selected_fits": len(selected_rows),
              "state_count_comparisons": len(comparisons), "controlled_seed_rows": len(seeds),
              "one_pipeline_rows": len(isolated), "numerical_checks": CHECKS,
              "maximum_numerical_discrepancy": MAX_ERROR,
              "scope": "Independent saved-fit reconciliation; not a historical-vintage or economic-taxonomy validation"}
    (ROOT / "audit/v21_diagnostic_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
