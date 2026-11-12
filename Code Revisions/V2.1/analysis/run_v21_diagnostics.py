"""Focused V2.1 diagnostics; preserves the archived primary analysis.

Fits K=2,3,4,5 on each panel's unchanged five-PC observation matrix. Across
seeds, uses one declared procedure throughout: smoothed posterior-weighted
economic profiles for correspondence, followed by filtered probability loss.
These are full-sample retrospective diagnostics, not real-time validation.
"""
from __future__ import annotations

import os
os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
import hashlib
import importlib.metadata
import itertools
import json
import logging
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import hmmModeling as hm

OUT = ROOT / "output/v21_diagnostics"
NAMES = ["full_information", "realtime_information", "nowcast_enhanced"]
SEEDS = list(range(5))
COUNTS = [2, 3, 4, 5]
SPEC = json.loads((ROOT / "config/analysis_spec.json").read_text())
BASE = ROOT / "package/original_run/outputs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def best_assignment(reference, candidate, weights):
    k = len(reference)
    costs = ((reference[:, None] - candidate[None, :]) ** 2 * weights).sum(2)
    permutations = list(itertools.permutations(range(k)))
    totals = [sum(costs[i, p[i]] for i in range(k)) for p in permutations]
    return permutations[int(np.argmin(totals))]


def comparison(fitted, anchor_z, weights):
    """Same smoothed-profile procedure for every seed/state count."""
    profiles = {
        name: result["smoothed"].T @ anchor_z / result["smoothed"].sum(0)[:, None]
        for name, result in fitted.items()
    }
    reference = fitted[NAMES[0]]["filtered"]
    losses = {}
    maps = {}
    for name in NAMES[1:]:
        perm = best_assignment(profiles[NAMES[0]], profiles[name], weights)
        maps[name] = list(perm)
        losses[name] = float(np.abs(reference - fitted[name]["filtered"][:, perm]).sum(1).mean())
    lag, nc = [losses[n] for n in NAMES[1:]]
    return {
        "LAG_L1": lag, "NC_L1": nc, "absolute_advantage": lag - nc,
        "reduction_percent": 100 * (1 - nc / lag) if lag else None,
        "LAG_mapping": maps[NAMES[1]], "NC_mapping": maps[NAMES[2]],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [ROOT / "config/analysis_spec.json", ROOT / "code/hmmModeling.py"]
    x = {}
    dates = None
    for name in NAMES:
        path = BASE / f"pca_{name}/hmm_pca_input.csv"
        inputs.append(path)
        frame = pd.read_csv(path, parse_dates=["date"])
        current_dates = pd.DatetimeIndex(frame.date)
        if dates is None:
            dates = current_dates
        else:
            assert dates.equals(current_dates)
        x[name] = frame[[f"PCA_{i}" for i in range(1, 6)]].to_numpy(float)
        assert x[name].shape == (137, 5) and np.isfinite(x[name]).all()
    raw_path = BASE / "full_information.csv"
    inputs.append(raw_path)
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True).loc[dates, SPEC["anchors"]]
    anchor_z = ((raw - raw.mean()) / raw.std(ddof=0)).to_numpy(float)
    weights = np.array(SPEC["anchor_weights"])
    results = {}
    ledger = []
    logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)
    for k in COUNTS:
        for name in NAMES:
            for seed in SEEDS:
                model = GaussianHMM(
                    n_components=k, covariance_type="diag", n_iter=1000,
                    tol=1e-4, random_state=seed, min_covar=1e-4,
                )
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model.fit(x[name])
                    likelihood = float(model.score(x[name]))
                    smoothed = model.predict_proba(x[name])
                    filtered = hm.filtered_state_probabilities(model, x[name])
                if not np.isfinite(likelihood):
                    raise ValueError(f"Nonfinite fit: {k}/{name}/{seed}")
                assert np.isfinite(smoothed).all() and np.isfinite(filtered).all()
                assert np.allclose(smoothed.sum(1), 1) and np.allclose(filtered.sum(1), 1)
                result = {"model": model, "likelihood": likelihood,
                          "smoothed": smoothed, "filtered": filtered}
                results[k, name, seed] = result
                npar = hm.parameter_count(k, 5, "diag")
                ledger.append({
                    "states": k, "model": name, "seed": seed,
                    "log_likelihood": likelihood, "n_parameters": npar,
                    "BIC": -2 * likelihood + npar * np.log(len(dates)),
                    "AIC": -2 * likelihood + 2 * npar,
                    "iterations": int(model.monitor_.iter),
                    "converged": bool(model.monitor_.converged),
                    "hit_iteration_cap": bool(model.monitor_.iter >= model.n_iter),
                    "minimum_posterior_mass": float(smoothed.sum(0).min()),
                    "hard_occupancy": np.bincount(smoothed.argmax(1), minlength=k).tolist(),
                    "posterior_mass": smoothed.sum(0).tolist(),
                    "warnings": [str(w.message) for w in captured],
                })
        print(f"Fitted K={k}", flush=True)

    selected = {}
    selection_rows = []
    for k in COUNTS:
        selected[k] = {}
        for name in NAMES:
            seed = max(SEEDS, key=lambda s: results[k, name, s]["likelihood"])
            selected[k][name] = results[k, name, seed]
            row = next(r for r in ledger if r["states"] == k and r["model"] == name and r["seed"] == seed)
            selection_rows.append(row)
            for candidate in ledger:
                if candidate["states"] == k and candidate["model"] == name:
                    candidate["likelihood_gap_from_selected"] = row["log_likelihood"] - candidate["log_likelihood"]
                    candidate["selected"] = candidate["seed"] == seed

    # New environment must recover the archived K4 fit before comparison reuse.
    parity_rows = []
    for name in NAMES:
        frame = pd.read_csv(ROOT / f"output/analysis/reproduction_5pc/{name}_smoothed.csv")
        archived = frame[[f"state_{i}_prob" for i in range(4)]].to_numpy(float)
        fresh = selected[4][name]["smoothed"]
        # Compare exact state order as well as probability values, not just LL.
        error = float(abs(archived - fresh).max())
        parity_rows.append({"model": name, "maximum_smoothed_probability_error": error})
        if error > 1e-8:
            raise AssertionError(f"V2.1 fresh K4 failed archived fit parity: {name}: {error}")

    state_rows = []
    for k in COUNTS:
        scores = comparison(selected[k], anchor_z, weights)
        state_rows.append({"states": k, **scores})
    seed_rows = []
    for seed in SEEDS:
        for ref_protocol in ["same_seed_reference", "fixed_selected_reference"]:
            fitted = {n: results[4, n, seed] for n in NAMES}
            if ref_protocol == "fixed_selected_reference":
                fitted[NAMES[0]] = selected[4][NAMES[0]]
            row = {"seed": seed, "reference_protocol": ref_protocol,
                   **comparison(fitted, anchor_z, weights)}
            for short, name in zip(["REV", "LAG", "NC"], NAMES):
                row[f"{short}_log_likelihood"] = fitted[name]["likelihood"]
                row[f"{short}_likelihood_gap"] = selected[4][name]["likelihood"] - fitted[name]["likelihood"]
            row["all_likelihood_gaps_le_2"] = all(row[f"{short}_likelihood_gap"] <= 2 for short in ["REV", "LAG", "NC"])
            seed_rows.append(row)

    one_at_time = []
    for name in NAMES:
        for seed in SEEDS:
            fitted = selected[4].copy()
            fitted[name] = results[4, name, seed]
            gap = selected[4][name]["likelihood"] - fitted[name]["likelihood"]
            one_at_time.append({"varied_model": name, "seed": seed,
                                "varied_fit_log_likelihood": fitted[name]["likelihood"],
                                "varied_fit_likelihood_gap": gap,
                                "likelihood_gap_le_2": gap <= 2,
                                **comparison(fitted, anchor_z, weights)})

    pd.DataFrame(ledger).to_csv(OUT / "fit_candidates.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(OUT / "state_count_selected.csv", index=False)
    pd.DataFrame(state_rows).to_csv(OUT / "state_count_comparison.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(OUT / "controlled_initialization.csv", index=False)
    pd.DataFrame(one_at_time).to_csv(OUT / "one_pipeline_initialization.csv", index=False)
    pd.DataFrame(parity_rows).to_csv(OUT / "baseline_fit_parity.csv", index=False)
    # Save all parameters so this diagnostic can be checked without another fit.
    for (k, name, seed), result in results.items():
        model = result["model"]
        np.savez_compressed(OUT / f"K{k}_{name}_seed{seed}.npz",
                            startprob=model.startprob_, transmat=model.transmat_,
                            means=model.means_, covars=model.covars_,
                            smoothed=result["smoothed"], filtered=result["filtered"])
    manifest = {
        "scope": "Retrospective; immutable archived five-PC inputs; no historical availability claim",
        "sample": [str(dates.min().date()), str(dates.max().date())], "months": len(dates),
        "seeds": SEEDS, "state_counts": COUNTS, "features": 5,
        "correspondence": "Smoothed-posterior weighted common REV outcomes, fixed weights, minimum-cost bijection",
        "loss": "Mean L1 distance between forward-filtered probability vectors",
        "near_best_display_threshold": "LL gap <=2 per fit is descriptive, not an inferential cutoff",
        "python": sys.version,
        "versions": {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "hmmlearn"]},
        "input_sha256": {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        "script_sha256": digest(Path(__file__)),
        "output_sha256": {p.name: digest(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(selection_rows)[["states", "model", "seed", "log_likelihood", "BIC", "minimum_posterior_mass"]].to_string(index=False))
    print(pd.DataFrame(state_rows).to_string(index=False))
    print(pd.DataFrame(seed_rows).to_string(index=False))
    print(pd.DataFrame(one_at_time).to_string(index=False))


if __name__ == "__main__":
    main()
