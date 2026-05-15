"""
Optional thesis validation stage: out-of-sample current-regime classification.

The main thesis comparison is an in-sample information-set comparison. This
script adds a deliberately narrower feasibility check:

    1. Keep the final thesis variable/transformation schema fixed.
    2. Fit scaler, PCA, and HMM only through the training window.
    3. Classify the holdout months recursively with filtered HMM probabilities.
    4. Compare those holdout probabilities with the existing full-sample
       full-information HMM benchmark.

This is not a one-month-ahead trading test. It is an out-of-sample test of
current-regime classification under delayed real-time data versus nowcast-
enhanced data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import dataCleaning as dc
import hmmModeling as hm


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

INFORMATION_SETS = {
    "full_information": OUTPUT_DIR / "full_information.csv",
    "realtime_information": OUTPUT_DIR / "realtime_information.csv",
    "nowcast_enhanced": OUTPUT_DIR / "nowcast_information.csv",
}

REGIME_LABELS = [
    "Expansion / Goldilocks",
    "Overheating / Late Cycle",
    "Contraction / Recession",
    "Recovery / Early Cycle",
]

REGIME_SLUGS = {
    "Expansion / Goldilocks": "expansion_goldilocks",
    "Overheating / Late Cycle": "overheating_late_cycle",
    "Contraction / Recession": "contraction_recession",
    "Recovery / Early Cycle": "recovery_early_cycle",
}

REGIME_PROB_COLS = [f"{REGIME_SLUGS[regime]}_prob" for regime in REGIME_LABELS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an out-of-sample HMM current-regime classification feasibility test."
    )
    parser.add_argument(
        "--holdout-months",
        type=int,
        default=12,
        help="Number of final aligned months to hold out. Default is 12.",
    )
    parser.add_argument(
        "--pca-components",
        default="5",
        help="Comma-separated PCA dimensions to test. Example: 3,5. Default is 5.",
    )
    parser.add_argument(
        "--state-count",
        type=int,
        default=4,
        help="Number of HMM states. Default is the four-regime thesis specification.",
    )
    parser.add_argument(
        "--covariance-type",
        default="diag",
        choices=["diag", "full", "tied", "spherical"],
        help="Gaussian HMM covariance type. Default is diag.",
    )
    parser.add_argument("--n-iter", type=int, default=1000, help="Maximum HMM EM iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="HMM EM convergence tolerance.")
    parser.add_argument(
        "--random-seeds",
        default="0,1,2,3,4",
        help="Comma-separated random seeds; best training likelihood is kept.",
    )
    parser.add_argument(
        "--schema",
        default=str(OUTPUT_DIR / "uniform_cleaning_schema.csv"),
        help="Uniform cleaning schema from hmmModeling.py.",
    )
    parser.add_argument(
        "--benchmark-dir",
        default=str(OUTPUT_DIR / "hmm_models_full_information"),
        help="Directory containing full-sample full-information HMM benchmark exports.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR / "hmm_oos_validation"),
        help="Directory for OOS validation outputs.",
    )
    return parser.parse_args()


def parse_int_list(raw: str, option_name: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"{option_name} cannot be empty.")
    return values


def require_packages() -> None:
    missing = []
    checks = {
        "hmmlearn": "hmmlearn",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
    }
    for import_name, package_name in checks.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        raise SystemExit(
            "Missing package(s): "
            + ", ".join(sorted(missing))
            + "\nInstall dependencies with:\n    "
            + '.\\.venv\\Scripts\\python.exe -m pip install -r "Data Processor\\requirements.txt"'
        )


def read_schema(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Uniform cleaning schema not found: {path}. Run hmmModeling.py --refresh-pca-inputs first."
        )
    schema = pd.read_csv(path)
    required = {"source_variable", "decision", "output_variable"}
    missing = required.difference(schema.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    schema = schema[schema["output_variable"].notna() & (schema["output_variable"].astype(str) != "")]
    if schema.empty:
        raise ValueError(f"{path.name} does not contain any retained variables.")
    return schema.reset_index(drop=True)


def apply_schema(panel: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    transformed = pd.DataFrame(index=panel.index)
    for _, row in schema.iterrows():
        source = str(row["source_variable"])
        decision = str(row["decision"])
        output = str(row["output_variable"])
        if source not in panel.columns:
            raise ValueError(f"Required schema source variable {source!r} is missing from panel.")
        if decision == "keep_level":
            transformed[output] = panel[source]
        elif decision == "keep_difference":
            transformed[output] = panel[source].diff()
        else:
            raise ValueError(f"Unsupported retained schema decision {decision!r} for {source!r}.")
    return transformed.dropna(how="any")


def zscore_with_training_stats(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means = train.mean(axis=0)
    stds = train.std(axis=0, ddof=0)
    zero_std = stds[stds <= 0.0]
    if not zero_std.empty:
        raise ValueError(f"Cannot z-score constant training column(s): {', '.join(zero_std.index)}")
    train_scaled = (train - means) / stds
    holdout_scaled = (holdout - means) / stds
    stats = pd.DataFrame(
        {
            "variable": train.columns,
            "training_mean": means.to_numpy(float),
            "training_std": stds.to_numpy(float),
        }
    )
    return train_scaled, holdout_scaled, stats


def model_log_likelihood(model: Any, x: np.ndarray) -> np.ndarray:
    if not hasattr(model, "_compute_log_likelihood"):
        raise RuntimeError("hmmlearn model does not expose _compute_log_likelihood.")
    return model._compute_log_likelihood(x)


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_value)
    summed = np.sum(stable, axis=axis, keepdims=True)
    out = np.log(summed) + max_value
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def normalize_log_prob(log_prob: np.ndarray) -> np.ndarray:
    return log_prob - logsumexp(log_prob)


def recursive_filtered_probabilities(
    model: Any,
    train_x: np.ndarray,
    holdout_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-300
    log_start = np.log(np.clip(model.startprob_, eps, 1.0))
    log_trans = np.log(np.clip(model.transmat_, eps, 1.0))
    train_log_likelihood = model_log_likelihood(model, train_x)
    holdout_log_likelihood = model_log_likelihood(model, holdout_x)

    log_alpha = normalize_log_prob(log_start + train_log_likelihood[0])
    for row in train_log_likelihood[1:]:
        log_prior = logsumexp(log_alpha[:, None] + log_trans, axis=0)
        log_alpha = normalize_log_prob(log_prior + row)

    filtered_rows = []
    prior_rows = []
    for row in holdout_log_likelihood:
        log_prior = logsumexp(log_alpha[:, None] + log_trans, axis=0)
        prior = np.exp(normalize_log_prob(log_prior))
        log_alpha = normalize_log_prob(log_prior + row)
        filtered = np.exp(log_alpha)
        prior_rows.append(prior)
        filtered_rows.append(filtered)

    return np.vstack(filtered_rows), np.vstack(prior_rows)


def probability_table_from_array(
    dates: pd.Index,
    probabilities: np.ndarray,
    prior_probabilities: np.ndarray,
    model: Any,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    states = probabilities.argmax(axis=1)
    out = pd.DataFrame(probabilities, index=dates, columns=[f"state_{i}_prob" for i in range(model.n_components)])
    for state in range(model.n_components):
        out[f"state_{state}_prior_prob"] = prior_probabilities[:, state]
    out.insert(0, "most_likely_state", states)
    out = out.reset_index().rename(columns={"index": "date"})
    out, _ = hm.attach_regime_labels(out, pd.DataFrame({"state": range(model.n_components)}), labels)
    return out


def add_regime_probability_columns(probabilities: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    out = probabilities.copy()
    for regime in REGIME_LABELS:
        out[f"{REGIME_SLUGS[regime]}_prob"] = 0.0
    for _, row in labels.iterrows():
        state = int(row["state"])
        regime = str(row["regime_label"])
        state_col = f"state_{state}_prob"
        if regime in REGIME_SLUGS and state_col in out.columns:
            out[f"{REGIME_SLUGS[regime]}_prob"] += out[state_col].astype(float)
    total = out[REGIME_PROB_COLS].sum(axis=1).replace(0.0, np.nan)
    out[REGIME_PROB_COLS] = out[REGIME_PROB_COLS].div(total, axis=0).fillna(0.0)
    return out


def benchmark_regime_probabilities(benchmark_dir: Path, n_pca: int) -> pd.DataFrame:
    probability_path = benchmark_dir / f"hmm_state_probabilities_pca{n_pca}.csv"
    labels_path = benchmark_dir / f"hmm_regime_labels_pca{n_pca}.csv"
    if not probability_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing full-information benchmark files for PCA {n_pca}: "
            f"{probability_path.name}, {labels_path.name}"
        )
    probabilities = pd.read_csv(probability_path, parse_dates=["date"])
    labels = pd.read_csv(labels_path)
    return add_regime_probability_columns(probabilities, labels).sort_values("date")


def hard_regime_brier(candidate_probs: np.ndarray, benchmark_regimes: pd.Series) -> float:
    target_indices = np.array([REGIME_LABELS.index(regime) for regime in benchmark_regimes])
    one_hot = np.eye(len(REGIME_LABELS))[target_indices]
    return float(((candidate_probs - one_hot) ** 2).sum(axis=1).mean())


def classification_metrics(
    benchmark: pd.DataFrame,
    candidate: pd.DataFrame,
    model_name: str,
    n_pca: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = benchmark.merge(candidate, on="date", suffixes=("_benchmark", "_candidate"))
    if merged.empty:
        raise ValueError(f"No overlapping holdout dates for {model_name}, PCA {n_pca}.")

    benchmark_probs = merged[[f"{col}_benchmark" for col in REGIME_PROB_COLS]].to_numpy(float)
    candidate_probs = merged[[f"{col}_candidate" for col in REGIME_PROB_COLS]].to_numpy(float)
    gap = np.abs(candidate_probs - benchmark_probs).sum(axis=1)
    hard_agreement = merged["most_likely_regime_benchmark"] == merged["most_likely_regime_candidate"]

    by_month = pd.DataFrame(
        {
            "date": merged["date"],
            "model": model_name,
            "pca_components": n_pca,
            "benchmark_regime": merged["most_likely_regime_benchmark"],
            "candidate_regime": merged["most_likely_regime_candidate"],
            "hard_regime_agreement": hard_agreement.astype(int),
            "information_gap_l1_vs_benchmark": gap,
            "probability_rmse_vs_benchmark": np.sqrt(((candidate_probs - benchmark_probs) ** 2).mean(axis=1)),
        }
    )

    row = {
        "model": model_name,
        "benchmark_model": "full_sample_full_information",
        "pca_components": n_pca,
        "train_start": candidate["train_start"].iloc[0],
        "train_end": candidate["train_end"].iloc[0],
        "holdout_start": merged["date"].min().date().isoformat(),
        "holdout_end": merged["date"].max().date().isoformat(),
        "holdout_months": int(len(merged)),
        "mean_information_gap_l1_vs_benchmark": float(gap.mean()),
        "median_information_gap_l1_vs_benchmark": float(np.median(gap)),
        "probability_rmse_vs_benchmark": float(np.sqrt(((candidate_probs - benchmark_probs) ** 2).mean())),
        "benchmark_regime_brier_score": hard_regime_brier(
            candidate_probs,
            merged["most_likely_regime_benchmark"],
        ),
        "hard_regime_agreement_rate": float(hard_agreement.mean()),
    }
    return row, by_month


def pct_reduction(baseline: float, candidate: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return np.nan
    return float(100.0 * (float(baseline) - float(candidate)) / abs(float(baseline)))


def nowcast_improvement(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_pca, subset in summary.groupby("pca_components"):
        real = subset[subset["model"] == "realtime_information"]
        nowcast = subset[subset["model"] == "nowcast_enhanced"]
        if real.empty or nowcast.empty:
            continue
        real_row = real.iloc[0]
        nowcast_row = nowcast.iloc[0]
        rows.append(
            {
                "pca_components": int(n_pca),
                "realtime_mean_l1_gap": float(real_row["mean_information_gap_l1_vs_benchmark"]),
                "nowcast_mean_l1_gap": float(nowcast_row["mean_information_gap_l1_vs_benchmark"]),
                "l1_gap_reduction": float(
                    real_row["mean_information_gap_l1_vs_benchmark"]
                    - nowcast_row["mean_information_gap_l1_vs_benchmark"]
                ),
                "l1_gap_percent_reduction": pct_reduction(
                    real_row["mean_information_gap_l1_vs_benchmark"],
                    nowcast_row["mean_information_gap_l1_vs_benchmark"],
                ),
                "realtime_probability_rmse": float(real_row["probability_rmse_vs_benchmark"]),
                "nowcast_probability_rmse": float(nowcast_row["probability_rmse_vs_benchmark"]),
                "probability_rmse_reduction": float(
                    real_row["probability_rmse_vs_benchmark"]
                    - nowcast_row["probability_rmse_vs_benchmark"]
                ),
                "probability_rmse_percent_reduction": pct_reduction(
                    real_row["probability_rmse_vs_benchmark"],
                    nowcast_row["probability_rmse_vs_benchmark"],
                ),
                "realtime_brier_score": float(real_row["benchmark_regime_brier_score"]),
                "nowcast_brier_score": float(nowcast_row["benchmark_regime_brier_score"]),
                "brier_score_reduction": float(
                    real_row["benchmark_regime_brier_score"]
                    - nowcast_row["benchmark_regime_brier_score"]
                ),
                "brier_score_percent_reduction": pct_reduction(
                    real_row["benchmark_regime_brier_score"],
                    nowcast_row["benchmark_regime_brier_score"],
                ),
                "realtime_hard_regime_agreement": float(real_row["hard_regime_agreement_rate"]),
                "nowcast_hard_regime_agreement": float(nowcast_row["hard_regime_agreement_rate"]),
                "hard_regime_agreement_change": float(
                    nowcast_row["hard_regime_agreement_rate"]
                    - real_row["hard_regime_agreement_rate"]
                ),
                "nowcast_beats_realtime_all_main_metrics": bool(
                    nowcast_row["mean_information_gap_l1_vs_benchmark"]
                    < real_row["mean_information_gap_l1_vs_benchmark"]
                    and nowcast_row["probability_rmse_vs_benchmark"]
                    < real_row["probability_rmse_vs_benchmark"]
                    and nowcast_row["benchmark_regime_brier_score"]
                    < real_row["benchmark_regime_brier_score"]
                    and nowcast_row["hard_regime_agreement_rate"]
                    >= real_row["hard_regime_agreement_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def month_winner_table(by_month: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_pca, subset in by_month.groupby("pca_components"):
        real = subset[subset["model"] == "realtime_information"]
        nowcast = subset[subset["model"] == "nowcast_enhanced"]
        if real.empty or nowcast.empty:
            continue
        merged = real.merge(nowcast, on="date", suffixes=("_realtime", "_nowcast"))
        realtime_gap = merged["information_gap_l1_vs_benchmark_realtime"]
        nowcast_gap = merged["information_gap_l1_vs_benchmark_nowcast"]
        out = pd.DataFrame(
            {
                "date": merged["date"],
                "pca_components": int(n_pca),
                "benchmark_regime": merged["benchmark_regime_realtime"],
                "realtime_regime": merged["candidate_regime_realtime"],
                "nowcast_regime": merged["candidate_regime_nowcast"],
                "realtime_information_gap_l1": realtime_gap,
                "nowcast_information_gap_l1": nowcast_gap,
                "nowcast_gap_advantage": realtime_gap - nowcast_gap,
                "nowcast_closer_than_realtime": nowcast_gap < realtime_gap,
                "same_distance": np.isclose(nowcast_gap, realtime_gap),
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fit_oos_model(
    model_name: str,
    stationary: pd.DataFrame,
    train_dates: pd.Index,
    holdout_dates: pd.Index,
    n_pca: int,
    args: argparse.Namespace,
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA

    train = stationary.loc[train_dates]
    holdout = stationary.loc[holdout_dates]
    train_scaled, holdout_scaled, normalization_stats = zscore_with_training_stats(train, holdout)

    pca = PCA(n_components=n_pca)
    train_scores_array = pca.fit_transform(train_scaled.to_numpy(float))
    holdout_scores_array = pca.transform(holdout_scaled.to_numpy(float))
    component_names = [f"PCA_{i}" for i in range(1, n_pca + 1)]
    train_scores = pd.DataFrame(train_scores_array, index=train_dates, columns=component_names)

    model, fit_info = hm.fit_best_hmm(
        train_scores_array,
        n_states=args.state_count,
        covariance_type=args.covariance_type,
        seeds=seeds,
        n_iter=args.n_iter,
        tol=args.tol,
    )
    train_probabilities = hm.state_probability_table(train_dates, model, train_scores_array)
    summary = hm.state_summary_table(model, train_scores, train_probabilities)
    labels = hm.regime_label_table(summary)
    summary = hm.attach_regime_labels(train_probabilities, summary, labels)[1]

    filtered, priors = recursive_filtered_probabilities(
        model,
        train_scores_array,
        holdout_scores_array,
    )
    probabilities = probability_table_from_array(holdout_dates, filtered, priors, model, labels)
    probabilities = add_regime_probability_columns(probabilities, labels)
    probabilities.insert(0, "model", model_name)
    probabilities.insert(1, "pca_components", n_pca)
    probabilities["train_start"] = train_dates.min().date().isoformat()
    probabilities["train_end"] = train_dates.max().date().isoformat()
    probabilities["holdout_start"] = holdout_dates.min().date().isoformat()
    probabilities["holdout_end"] = holdout_dates.max().date().isoformat()
    probabilities["random_seed"] = fit_info["random_seed"]
    probabilities["training_log_likelihood"] = fit_info["log_likelihood"]
    probabilities["training_converged"] = fit_info["converged"]
    probabilities["training_iterations"] = fit_info["iterations"]

    variance = pd.DataFrame(
        {
            "model": model_name,
            "pca_components": n_pca,
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    normalization_stats.insert(0, "model", model_name)
    normalization_stats.insert(1, "pca_components", n_pca)
    summary.insert(0, "model", model_name)
    summary.insert(1, "pca_components", n_pca)
    return probabilities, summary, labels.assign(model=model_name, pca_components=n_pca), variance


def main() -> int:
    args = parse_args()
    require_packages()
    if args.holdout_months <= 0:
        raise ValueError("--holdout-months must be positive.")

    pca_components = parse_int_list(args.pca_components, "--pca-components")
    seeds = parse_int_list(args.random_seeds, "--random-seeds")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = read_schema(Path(args.schema))

    panels = {name: dc.read_panel(path) for name, path in INFORMATION_SETS.items()}
    stationary = {name: apply_schema(panel, schema) for name, panel in panels.items()}

    common_dates = None
    for frame in stationary.values():
        dates = pd.Index(frame.dropna(how="any").index)
        common_dates = dates if common_dates is None else common_dates.intersection(dates)
    if common_dates is None or len(common_dates) <= args.holdout_months:
        raise ValueError("Not enough aligned observations for the requested holdout window.")
    common_dates = pd.DatetimeIndex(common_dates).sort_values()

    holdout_dates = common_dates[-args.holdout_months :]
    train_dates = common_dates[: -args.holdout_months]
    if len(train_dates) < max(pca_components) + args.state_count:
        raise ValueError("Training window is too short for the requested PCA/HMM specification.")

    all_probability_rows = []
    all_summary_rows = []
    all_label_rows = []
    all_variance_rows = []
    all_metric_rows = []
    all_by_month_rows = []

    for n_pca in pca_components:
        benchmark = benchmark_regime_probabilities(Path(args.benchmark_dir), n_pca)
        benchmark = benchmark[benchmark["date"].isin(holdout_dates)]
        missing_benchmark_dates = set(holdout_dates).difference(set(benchmark["date"]))
        if missing_benchmark_dates:
            missing_text = ", ".join(pd.Timestamp(date).date().isoformat() for date in sorted(missing_benchmark_dates))
            raise ValueError(f"Benchmark is missing holdout date(s) for PCA {n_pca}: {missing_text}")

        pca_probability_rows = []
        pca_summary_rows = []
        pca_label_rows = []
        pca_variance_rows = []
        pca_metric_rows = []
        pca_by_month_rows = []

        for model_name, frame in stationary.items():
            probabilities, summary, labels, variance = fit_oos_model(
                model_name=model_name,
                stationary=frame.loc[common_dates],
                train_dates=train_dates,
                holdout_dates=holdout_dates,
                n_pca=n_pca,
                args=args,
                seeds=seeds,
            )
            pca_probability_rows.append(probabilities)
            pca_summary_rows.append(summary)
            pca_label_rows.append(labels)
            pca_variance_rows.append(variance)

            metric_row, by_month = classification_metrics(
                benchmark=benchmark,
                candidate=probabilities,
                model_name=model_name,
                n_pca=n_pca,
            )
            pca_metric_rows.append(metric_row)
            pca_by_month_rows.append(by_month)

        pca_probabilities = pd.concat(pca_probability_rows, ignore_index=True)
        pca_summaries = pd.concat(pca_summary_rows, ignore_index=True)
        pca_labels = pd.concat(pca_label_rows, ignore_index=True)
        pca_variance = pd.concat(pca_variance_rows, ignore_index=True)
        pca_metrics = pd.DataFrame(pca_metric_rows)
        pca_by_month = pd.concat(pca_by_month_rows, ignore_index=True)
        pca_month_winners = month_winner_table(pca_by_month)
        pca_improvement = nowcast_improvement(pca_metrics)

        dc.write_dataframe(pca_probabilities, output_dir / f"oos_filtered_probabilities_pca{n_pca}.csv")
        dc.write_dataframe(pca_by_month, output_dir / f"oos_information_gap_by_month_pca{n_pca}.csv")
        dc.write_dataframe(pca_metrics, output_dir / f"oos_summary_metrics_pca{n_pca}.csv")
        dc.write_dataframe(pca_improvement, output_dir / f"oos_nowcast_vs_realtime_improvement_pca{n_pca}.csv")
        dc.write_dataframe(pca_month_winners, output_dir / f"oos_monthly_nowcast_winners_pca{n_pca}.csv")
        dc.write_dataframe(pca_summaries, output_dir / f"oos_training_state_summary_pca{n_pca}.csv")
        dc.write_dataframe(pca_labels, output_dir / f"oos_training_regime_labels_pca{n_pca}.csv")
        dc.write_dataframe(pca_variance, output_dir / f"oos_training_pca_explained_variance_pca{n_pca}.csv")

        all_probability_rows.append(pca_probabilities)
        all_summary_rows.append(pca_summaries)
        all_label_rows.append(pca_labels)
        all_variance_rows.append(pca_variance)
        all_metric_rows.append(pca_metrics)
        all_by_month_rows.append(pca_by_month)

    run_summary = pd.DataFrame(
        [
            {
                "holdout_months": args.holdout_months,
                "train_start": train_dates.min().date().isoformat(),
                "train_end": train_dates.max().date().isoformat(),
                "train_rows": int(len(train_dates)),
                "holdout_start": holdout_dates.min().date().isoformat(),
                "holdout_end": holdout_dates.max().date().isoformat(),
                "holdout_rows": int(len(holdout_dates)),
                "information_sets": ",".join(INFORMATION_SETS),
                "pca_components": ",".join(str(value) for value in pca_components),
                "state_count": args.state_count,
                "covariance_type": args.covariance_type,
                "schema_path": str(Path(args.schema)),
                "benchmark_dir": str(Path(args.benchmark_dir)),
            }
        ]
    )
    dc.write_dataframe(run_summary, output_dir / "oos_run_summary.csv")
    dc.write_dataframe(pd.concat(all_metric_rows, ignore_index=True), output_dir / "oos_summary_metrics_all.csv")
    dc.write_dataframe(
        nowcast_improvement(pd.concat(all_metric_rows, ignore_index=True)),
        output_dir / "oos_nowcast_vs_realtime_improvement_all.csv",
    )
    dc.write_dataframe(pd.concat(all_by_month_rows, ignore_index=True), output_dir / "oos_information_gap_by_month_all.csv")
    dc.write_dataframe(pd.concat(all_probability_rows, ignore_index=True), output_dir / "oos_filtered_probabilities_all.csv")
    dc.write_dataframe(pd.concat(all_summary_rows, ignore_index=True), output_dir / "oos_training_state_summary_all.csv")
    dc.write_dataframe(pd.concat(all_label_rows, ignore_index=True), output_dir / "oos_training_regime_labels_all.csv")
    dc.write_dataframe(pd.concat(all_variance_rows, ignore_index=True), output_dir / "oos_training_pca_explained_variance_all.csv")

    print("Out-of-sample HMM validation complete.")
    print(
        f"Training sample: {train_dates.min().date()} through {train_dates.max().date()} "
        f"({len(train_dates)} rows)"
    )
    print(
        f"Holdout sample: {holdout_dates.min().date()} through {holdout_dates.max().date()} "
        f"({len(holdout_dates)} rows)"
    )
    print(f"Outputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
