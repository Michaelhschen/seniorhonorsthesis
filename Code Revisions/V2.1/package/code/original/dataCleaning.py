"""
Step 2 thesis data pipeline: clean macro panels and prepare PCA factors.

Default input:
    outputs/nowcast_information.csv

Default outputs:
    outputs/adf_results.csv
    outputs/data_cleaning_decisions.csv
    outputs/stationary_inputs.csv
    outputs/normalized_stationary_inputs.csv
    outputs/static_pca_scores.csv
    outputs/static_pca_loadings.csv
    outputs/rolling_pca_scores.csv
    outputs/rolling_pca_loadings.csv
    outputs/rolling_pca_loading_stability.csv
    outputs/pca_explained_variance.csv
    outputs/pca_recommendation.csv
    outputs/hmm_pca_input.csv
    outputs/data_cleaning_pca.xlsx
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_INPUT = OUTPUT_DIR / "nowcast_information.csv"
DEFAULT_DROP_COLUMNS = ["HYSpread", "U6UNRATE", "ISM_NewOrders", "HOUST_YoY", "ICSA_YoY"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean thesis macro data, run ADF tests, and create PCA factors."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input panel CSV with a date column.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for cleaned data and PCA outputs.",
    )
    parser.add_argument(
        "--drop-columns",
        default=",".join(DEFAULT_DROP_COLUMNS),
        help="Comma-separated columns to exclude before cleaning. Default: HYSpread,U6UNRATE,ISM_NewOrders,HOUST_YoY,ICSA_YoY.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Trailing months used for rolling PCA.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of PCA score columns to export.",
    )
    parser.add_argument(
        "--adf-alpha",
        type=float,
        default=0.05,
        help="ADF p-value threshold for stationarity decisions.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.70,
        help="Cumulative explained variance target for PCA diagnostics.",
    )
    parser.add_argument(
        "--loading-stability-threshold",
        type=float,
        default=0.85,
        help="Minimum median sign-adjusted cosine similarity for rolling loadings.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.80,
        help="Absolute stationary-input correlation threshold for diagnostic pairs.",
    )
    parser.add_argument(
        "--no-workbook",
        action="store_true",
        help="Skip writing the summary Excel workbook.",
    )
    return parser.parse_args()


def require_packages() -> None:
    missing = []
    checks = {
        "statsmodels": "statsmodels",
        "sklearn": "scikit-learn",
        "openpyxl": "openpyxl",
    }
    for import_name, package_name in checks.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        install = (
            '.\\.venv\\Scripts\\python.exe -m pip install -r '
            '"Data Processor\\requirements-analysis.txt"'
        )
        raise SystemExit(
            "Missing analysis package(s): "
            + ", ".join(sorted(missing))
            + "\nInstall them from the project root with:\n    "
            + install
        )


def parse_drop_columns(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def read_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input panel not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path.name} must contain a date column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df[~df.index.duplicated(keep="last")]


def adf_result(
    series: pd.Series,
    variable: str,
    stage: str,
    alpha: float,
    transformation: str,
) -> tuple[dict[str, Any], bool]:
    from statsmodels.tsa.stattools import adfuller

    clean = pd.to_numeric(series, errors="coerce").dropna()
    result = {
        "stage": stage,
        "variable": variable,
        "transformation": transformation,
        "nobs_input": int(clean.shape[0]),
        "adf_statistic": np.nan,
        "p_value": np.nan,
        "used_lag": np.nan,
        "nobs_adf": np.nan,
        "critical_1pct": np.nan,
        "critical_5pct": np.nan,
        "critical_10pct": np.nan,
        "alpha": alpha,
        "stationary": False,
        "error": "",
    }

    if clean.shape[0] < 12:
        result["error"] = "too_few_observations"
        return result, False
    if clean.nunique(dropna=True) <= 1:
        result["p_value"] = 0.0
        result["stationary"] = True
        result["error"] = "constant_series_treated_as_stationary"
        return result, True

    try:
        stat, pvalue, used_lag, nobs_adf, critical_values, _ = adfuller(
            clean,
            regression="c",
            autolag="AIC",
        )
    except Exception as exc:
        result["error"] = str(exc)
        return result, False

    result.update(
        {
            "adf_statistic": float(stat),
            "p_value": float(pvalue),
            "used_lag": int(used_lag),
            "nobs_adf": int(nobs_adf),
            "critical_1pct": float(critical_values.get("1%", np.nan)),
            "critical_5pct": float(critical_values.get("5%", np.nan)),
            "critical_10pct": float(critical_values.get("10%", np.nan)),
            "stationary": bool(pvalue <= alpha),
        }
    )
    return result, bool(pvalue <= alpha)


def prepare_stationary_inputs(
    aligned: pd.DataFrame,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adf_rows = []
    transformed = pd.DataFrame(index=aligned.index)

    for column in aligned.columns:
        level_row, level_stationary = adf_result(
            aligned[column],
            variable=column,
            stage="input_level",
            alpha=alpha,
            transformation="level",
        )
        adf_rows.append(level_row)

        if level_stationary:
            transformed[column] = aligned[column]
            adf_rows[-1]["decision"] = "keep_level"
            continue

        level_row["decision"] = "difference_needed"
        differenced = aligned[column].diff()
        diff_row, diff_stationary = adf_result(
            differenced,
            variable=column,
            stage="input_difference",
            alpha=alpha,
            transformation="first_difference",
        )
        diff_row["decision"] = "keep_difference" if diff_stationary else "drop_nonstationary"
        adf_rows.append(diff_row)

        if diff_stationary:
            transformed[f"{column}_diff"] = differenced
        else:
            level_row["decision"] = "drop_nonstationary"

    stationary = transformed.dropna(how="any")
    if stationary.empty:
        raise ValueError("No stationary input rows remain after ADF transformations.")
    return stationary, pd.DataFrame(adf_rows)


def zscore(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0).replace(0.0, np.nan)
    normalized = (df - means) / stds
    normalized = normalized.dropna(axis=1, how="any")
    stats = pd.DataFrame(
        {
            "variable": normalized.columns,
            "mean": [float(means[col]) for col in normalized.columns],
            "std_ddof0": [float(stds[col]) for col in normalized.columns],
        }
    )
    return normalized, stats


def correlation_diagnostics(
    stationary: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = stationary.corr()
    pair_rows = []
    columns = list(matrix.columns)
    for left_idx, left in enumerate(columns):
        for right in columns[left_idx + 1 :]:
            corr = matrix.loc[left, right]
            if pd.isna(corr):
                continue
            if abs(corr) >= threshold:
                pair_rows.append(
                    {
                        "variable_1": left,
                        "variable_2": right,
                        "correlation": float(corr),
                        "abs_correlation": float(abs(corr)),
                        "threshold": threshold,
                    }
                )

    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return matrix, pairs


def component_names(n_components: int) -> list[str]:
    return [f"PCA_{idx}" for idx in range(1, n_components + 1)]


def factor_labels(n_components: int) -> pd.DataFrame:
    labels = {
        "PCA_1": {
            "label": "Growth-Inflation Demand Momentum",
            "hmm_usage": "Included in 5-PC thesis HMM; also interpretable in 3-PC sensitivity checks",
            "interpretation": "Broad demand factor led by ISM activity, GDP, and synchronized CPI/Core PCE inflation changes.",
        },
        "PCA_2": {
            "label": "Labor Slack / Liquidity Stress",
            "hmm_usage": "Included in 5-PC thesis HMM; also interpretable in 3-PC sensitivity checks",
            "interpretation": "Slack and liquidity factor driven by unemployment, money-ratio changes, volatility, and survey weakness.",
        },
        "PCA_3": {
            "label": "Capex-Credit / Term-Spread Conditions",
            "hmm_usage": "Included in 5-PC thesis HMM; also interpretable in 3-PC sensitivity checks",
            "interpretation": "Capital-spending, delinquency-change, dollar, and yield-curve variation.",
        },
        "PCA_4": {
            "label": "Regional Activity / Volatility Shock",
            "hmm_usage": "Included in 5-PC thesis HMM",
            "interpretation": "Regional business outlook, volatility, and fixed-investment residual variation.",
        },
        "PCA_5": {
            "label": "Dollar-Credit Residual Shock",
            "hmm_usage": "Included in 5-PC thesis HMM",
            "interpretation": "Residual dollar, delinquency-change, volatility, and capital-spending variation.",
        },
    }
    rows = []
    for component in component_names(n_components):
        info = labels.get(
            component,
            {
                "label": "Unlabeled",
                "hmm_usage": "Supplemental",
                "interpretation": "Supplemental PCA factor; inspect loadings before interpretation.",
            },
        )
        rows.append({"component": component, **info})
    return pd.DataFrame(rows)


def explained_rows(
    pca_type: str,
    ratios: np.ndarray,
    variance_threshold: float,
    date: pd.Timestamp | None = None,
) -> tuple[list[dict[str, Any]], int]:
    cumulative = np.cumsum(ratios)
    reaches = np.where(cumulative >= variance_threshold)[0]
    recommended = int(reaches[0] + 1) if reaches.size else int(len(ratios))
    rows = []
    for idx, ratio in enumerate(ratios, start=1):
        rows.append(
            {
                "pca_type": pca_type,
                "date": date,
                "component": f"PCA_{idx}",
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance": float(cumulative[idx - 1]),
                "components_to_reach_threshold": recommended,
                "variance_threshold": variance_threshold,
            }
        )
    return rows, recommended


def fit_static_pca(
    stationary: pd.DataFrame,
    n_components: int,
    variance_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    max_components = min(stationary.shape[0], stationary.shape[1])
    if max_components < n_components:
        raise ValueError(
            f"Cannot export {n_components} PCA components from shape {stationary.shape}."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(stationary)

    pca_full = PCA(n_components=max_components)
    pca_full.fit(scaled)
    variance_rows, _ = explained_rows(
        "static",
        pca_full.explained_variance_ratio_,
        variance_threshold,
    )

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)
    score_df = pd.DataFrame(scores, index=stationary.index, columns=component_names(n_components))
    score_df.index.name = "date"

    loading_rows = []
    for comp_idx, component in enumerate(component_names(n_components)):
        for variable, loading in zip(stationary.columns, pca.components_[comp_idx]):
            loading_rows.append(
                {
                    "pca_type": "static",
                    "component": component,
                    "variable": variable,
                    "loading": float(loading),
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[comp_idx]),
                }
            )

    return score_df, pd.DataFrame(loading_rows), pd.DataFrame(variance_rows)


def fit_rolling_pca(
    stationary: pd.DataFrame,
    n_components: int,
    rolling_window: int,
    variance_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if stationary.shape[0] < rolling_window:
        raise ValueError(
            f"Need at least {rolling_window} rows for rolling PCA; got {stationary.shape[0]}."
        )
    if stationary.shape[1] < n_components:
        raise ValueError(
            f"Need at least {n_components} variables for PCA; got {stationary.shape[1]}."
        )

    score_rows = []
    loading_rows = []
    variance_rows = []
    stability_rows = []
    previous_components: np.ndarray | None = None

    for end_pos in range(rolling_window - 1, stationary.shape[0]):
        window = stationary.iloc[end_pos - rolling_window + 1 : end_pos + 1]
        score_date = stationary.index[end_pos]

        scaler = StandardScaler()
        scaled_window = scaler.fit_transform(window)
        max_components = min(window.shape[0], window.shape[1])
        pca_full = PCA(n_components=max_components)
        pca_full.fit(scaled_window)

        components = pca_full.components_[:n_components].copy()
        current_scaled = scaler.transform(stationary.iloc[[end_pos]])
        current_score = pca_full.transform(current_scaled)[0][:n_components]

        similarities: list[float] = []
        if previous_components is not None:
            for comp_idx in range(n_components):
                similarity = float(np.dot(previous_components[comp_idx], components[comp_idx]))
                if similarity < 0:
                    components[comp_idx] *= -1.0
                    current_score[comp_idx] *= -1.0
                    similarity = -similarity
                similarities.append(similarity)
                stability_rows.append(
                    {
                        "date": score_date,
                        "component": f"PCA_{comp_idx + 1}",
                        "sign_adjusted_cosine_similarity": similarity,
                    }
                )

        previous_components = components.copy()

        row = {"date": score_date}
        for comp_idx, component in enumerate(component_names(n_components)):
            row[component] = float(current_score[comp_idx])
            for variable, loading in zip(stationary.columns, components[comp_idx]):
                loading_rows.append(
                    {
                        "date": score_date,
                        "component": component,
                        "variable": variable,
                        "loading": float(loading),
                        "explained_variance_ratio": float(
                            pca_full.explained_variance_ratio_[comp_idx]
                        ),
                    }
                )
        score_rows.append(row)

        rows, _ = explained_rows(
            "rolling",
            pca_full.explained_variance_ratio_,
            variance_threshold,
            date=score_date,
        )
        variance_rows.extend(rows)

    scores = pd.DataFrame(score_rows).set_index("date")
    scores.index.name = "date"
    return (
        scores,
        pd.DataFrame(loading_rows),
        pd.DataFrame(variance_rows),
        pd.DataFrame(stability_rows),
    )


def pca_adf_results(
    scores: pd.DataFrame,
    pca_type: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for column in scores.columns:
        row, _ = adf_result(
            scores[column],
            variable=column,
            stage=f"{pca_type}_pca_score",
            alpha=alpha,
            transformation="score",
        )
        row["decision"] = "stationary" if row["stationary"] else "nonstationary"
        rows.append(row)
    return pd.DataFrame(rows)


def median_loading_stability(
    stability: pd.DataFrame,
    threshold: float,
    n_components: int,
) -> pd.DataFrame:
    rows = []
    for component in component_names(n_components):
        subset = stability[stability["component"] == component]
        median_similarity = (
            float(subset["sign_adjusted_cosine_similarity"].median())
            if not subset.empty
            else np.nan
        )
        rows.append(
            {
                "component": component,
                "median_sign_adjusted_cosine_similarity": median_similarity,
                "loading_stability_threshold": threshold,
                "stable": bool(
                    pd.notna(median_similarity) and median_similarity >= threshold
                ),
            }
        )
    return pd.DataFrame(rows)


def choose_recommended_pca(
    rolling_pca_adf: pd.DataFrame,
    static_pca_adf: pd.DataFrame,
    stability_summary: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    rolling_stationary = bool(rolling_pca_adf["stationary"].all())
    static_stationary = bool(static_pca_adf["stationary"].all())
    rolling_stable = bool(stability_summary["stable"].all())

    if rolling_stable:
        recommendation = "static"
        if static_stationary:
            reason = (
                "rolling PCA loadings are stable, so static PCA is used for HMM input "
                "to preserve the full stationary sample; rolling PCA remains a loading-stability diagnostic"
            )
        else:
            reason = (
                "rolling PCA loadings are stable, so static PCA is used for HMM input "
                "to preserve the full stationary sample; static score ADF diagnostics should be reviewed"
            )
    elif static_stationary:
        recommendation = "static"
        reason = "rolling PCA loading stability failed; static PCA scores are stationary"
    else:
        recommendation = "static"
        reason = (
            "rolling PCA loading stability failed; static PCA is exported as fallback "
            "but has nonstationary score diagnostics to review"
        )

    return pd.DataFrame(
        [
            {
                "recommended_pca_type": recommendation,
                "reason": reason,
                "adf_alpha": alpha,
                "rolling_scores_stationary": rolling_stationary,
                "static_scores_stationary": static_stationary,
                "rolling_loadings_stable": rolling_stable,
            }
        ]
    )


def write_dataframe(df: pd.DataFrame, path: Path, include_index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=include_index)


def write_workbook(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> Path:
    workbook_path = output_dir / "data_cleaning_pca.xlsx"
    try:
        handle = workbook_path.open("a+b")
        handle.close()
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workbook_path = output_dir / f"data_cleaning_pca_{stamp}.xlsx"
        print(f"Workbook is locked; writing Excel output to {workbook_path}")

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, table in tables.items():
            safe_name = sheet_name[:31]
            table.to_excel(writer, sheet_name=safe_name, index=False)
    return workbook_path


def main() -> None:
    require_packages()
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = read_panel(input_path)
    drop_columns = parse_drop_columns(args.drop_columns)
    existing_drop_columns = [column for column in drop_columns if column in panel.columns]
    kept_panel = panel.drop(columns=existing_drop_columns)
    aligned = kept_panel.dropna(how="any")

    if aligned.empty:
        raise ValueError("No complete rows remain after dropping selected columns.")

    stationary, input_adf = prepare_stationary_inputs(aligned, alpha=args.adf_alpha)
    normalized, normalization_stats = zscore(stationary)
    correlation_matrix, high_correlation_pairs = correlation_diagnostics(
        stationary,
        threshold=args.correlation_threshold,
    )

    static_scores, static_loadings, static_variance = fit_static_pca(
        stationary,
        n_components=args.n_components,
        variance_threshold=args.variance_threshold,
    )
    rolling_scores, rolling_loadings, rolling_variance, rolling_stability = fit_rolling_pca(
        stationary,
        n_components=args.n_components,
        rolling_window=args.rolling_window,
        variance_threshold=args.variance_threshold,
    )

    static_pca_adf = pca_adf_results(static_scores, "static", alpha=args.adf_alpha)
    rolling_pca_adf = pca_adf_results(rolling_scores, "rolling", alpha=args.adf_alpha)
    labels = factor_labels(args.n_components)
    adf_results = pd.concat(
        [input_adf, rolling_pca_adf, static_pca_adf],
        ignore_index=True,
        sort=False,
    )

    stability_summary = median_loading_stability(
        rolling_stability,
        threshold=args.loading_stability_threshold,
        n_components=args.n_components,
    )
    recommendation = choose_recommended_pca(
        rolling_pca_adf,
        static_pca_adf,
        stability_summary,
        alpha=args.adf_alpha,
    )

    recommended_type = str(recommendation.loc[0, "recommended_pca_type"])
    hmm_input = rolling_scores if recommended_type == "rolling" else static_scores

    decisions = []
    for column in panel.columns:
        if column in existing_drop_columns:
            reason = "dropped_by_default_redundancy_or_sample_design"
            status = "dropped"
        elif column in kept_panel.columns:
            reason = "kept_for_alignment_and_adf"
            status = "candidate"
        else:
            reason = "not_used"
            status = "not_used"
        series = panel[column]
        decisions.append(
            {
                "variable": column,
                "status": status,
                "reason": reason,
                "missing_count": int(series.isna().sum()),
                "missing_share": float(series.isna().mean()),
                "first_valid": series.first_valid_index(),
                "last_valid": series.last_valid_index(),
            }
        )

    decisions_df = pd.DataFrame(decisions)
    sample_summary = pd.DataFrame(
        [
            {
                "input_file": str(input_path),
                "input_rows": int(panel.shape[0]),
                "input_variables": int(panel.shape[1]),
                "dropped_columns": ", ".join(existing_drop_columns),
                "aligned_rows": int(aligned.shape[0]),
                "aligned_start": aligned.index.min(),
                "aligned_end": aligned.index.max(),
                "stationary_rows": int(stationary.shape[0]),
                "stationary_start": stationary.index.min(),
                "stationary_end": stationary.index.max(),
                "stationary_variables": int(stationary.shape[1]),
                "exported_pca_components": int(args.n_components),
                "rolling_window": int(args.rolling_window),
                "rolling_first_score_date": rolling_scores.index.min(),
                "rolling_last_score_date": rolling_scores.index.max(),
                "recommended_pca_type": recommended_type,
            }
        ]
    )
    decisions_out = pd.concat(
        [
            decisions_df,
            pd.DataFrame(
                [
                    {
                        "variable": "__sample_summary__",
                        "status": "summary",
                        "reason": sample_summary.to_json(orient="records", date_format="iso"),
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    explained_variance = pd.concat(
        [rolling_variance, static_variance],
        ignore_index=True,
        sort=False,
    )

    write_dataframe(adf_results, output_dir / "adf_results.csv")
    write_dataframe(decisions_out, output_dir / "data_cleaning_decisions.csv")
    write_dataframe(sample_summary, output_dir / "data_cleaning_sample_summary.csv")
    write_dataframe(stationary.reset_index(), output_dir / "stationary_inputs.csv")
    write_dataframe(normalized.reset_index(), output_dir / "normalized_stationary_inputs.csv")
    write_dataframe(normalization_stats, output_dir / "normalization_stats.csv")
    write_dataframe(
        correlation_matrix.reset_index().rename(columns={"index": "variable"}),
        output_dir / "stationary_input_correlation_matrix.csv",
    )
    write_dataframe(high_correlation_pairs, output_dir / "stationary_input_high_correlations.csv")
    write_dataframe(rolling_scores.reset_index(), output_dir / "rolling_pca_scores.csv")
    write_dataframe(static_scores.reset_index(), output_dir / "static_pca_scores.csv")
    write_dataframe(rolling_loadings, output_dir / "rolling_pca_loadings.csv")
    write_dataframe(static_loadings, output_dir / "static_pca_loadings.csv")
    write_dataframe(rolling_stability, output_dir / "rolling_pca_loading_stability.csv")
    write_dataframe(stability_summary, output_dir / "rolling_pca_loading_stability_summary.csv")
    write_dataframe(explained_variance, output_dir / "pca_explained_variance.csv")
    write_dataframe(labels, output_dir / "pca_factor_labels.csv")
    write_dataframe(recommendation, output_dir / "pca_recommendation.csv")
    write_dataframe(hmm_input.reset_index(), output_dir / "hmm_pca_input.csv")

    if not args.no_workbook:
        workbook_tables = {
            "Recommendation": recommendation,
            "SampleSummary": sample_summary,
            "CleaningDecisions": decisions_df,
            "ADFResults": adf_results,
            "StationaryInputs": stationary.reset_index(),
            "RollingPCAScores": rolling_scores.reset_index(),
            "StaticPCAScores": static_scores.reset_index(),
            "PCAExplainedVariance": explained_variance,
            "PCAFactorLabels": labels,
            "RollingStability": stability_summary,
            "HighCorrelations": high_correlation_pairs,
            "NormalizationStats": normalization_stats,
        }
        write_workbook(output_dir, workbook_tables)

    print(f"Wrote cleaning and PCA outputs to: {output_dir}")
    print(f"Input panel shape: {panel.shape}")
    print(f"Dropped columns: {existing_drop_columns if existing_drop_columns else 'none'}")
    print(
        "Aligned sample: "
        f"{aligned.index.min().date()} through {aligned.index.max().date()} "
        f"({aligned.shape[0]} rows)"
    )
    print(
        "Stationary sample: "
        f"{stationary.index.min().date()} through {stationary.index.max().date()} "
        f"({stationary.shape[0]} rows, {stationary.shape[1]} variables)"
    )
    print(
        "Rolling PCA sample: "
        f"{rolling_scores.index.min().date()} through {rolling_scores.index.max().date()} "
        f"({rolling_scores.shape[0]} rows)"
    )
    print(f"Recommended HMM PCA input: {recommended_type}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
