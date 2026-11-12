"""
Final thesis pipeline step: compare HMM regime outputs across information sets.

This script does not rebuild PCA or HMM outputs. It only reads existing HMM
outputs and compares:
    - full-information HMM as the internal benchmark
    - real-time HMM as the baseline
    - nowcast-enhanced HMM as the candidate model

Expected default HMM output directories:
    outputs/hmm_models_full_information
    outputs/hmm_models_realtime_information
    outputs/hmm_models_nowcast_enhanced

For backwards compatibility, --nowcast-dir defaults to outputs/hmm_models when
the nowcast-specific directory is not present.

You can also point --hmm-export-root at a parent directory containing HMM export
folders, or pass --exports-manifest as a CSV with model and hmm_dir columns.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

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

MODEL_NAMES = ["full_information", "realtime_information", "nowcast_enhanced"]
REGIME_PROB_COLS = [f"{REGIME_SLUGS[regime]}_prob" for regime in REGIME_LABELS]
REGIME_BY_PROB_COL = {f"{slug}_prob": regime for regime, slug in REGIME_SLUGS.items()}


def parse_args() -> argparse.Namespace:
    default_nowcast = OUTPUT_DIR / "hmm_models_nowcast_enhanced"
    if not default_nowcast.exists() and (OUTPUT_DIR / "hmm_models").exists():
        default_nowcast = OUTPUT_DIR / "hmm_models"

    parser = argparse.ArgumentParser(
        description="Compare full-information, real-time, and nowcast-enhanced HMM outputs."
    )
    parser.add_argument(
        "--full-dir",
        default=str(OUTPUT_DIR / "hmm_models_full_information"),
        help="Existing HMM output directory for the full-information benchmark model.",
    )
    parser.add_argument(
        "--realtime-dir",
        default=str(OUTPUT_DIR / "hmm_models_realtime_information"),
        help="Existing HMM output directory for the real-time baseline model.",
    )
    parser.add_argument(
        "--nowcast-dir",
        default=str(default_nowcast),
        help="Existing HMM output directory for the nowcast-enhanced model.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR / "hmm_model_comparison_across_information_sets"),
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--hmm-export-root",
        default=str(OUTPUT_DIR),
        help=(
            "Parent directory used to auto-discover HMM exports from hmmModeling.py "
            "when the explicit model directories are not present."
        ),
    )
    parser.add_argument(
        "--exports-manifest",
        default="",
        help=(
            "Optional CSV mapping information sets to HMM export directories. "
            "Accepted columns: model or information_set, plus hmm_dir or output_dir."
        ),
    )
    parser.add_argument(
        "--pca-components",
        default="5",
        help="Comma-separated HMM PCA dimensions to compare. Default is the 5-PC thesis baseline.",
    )
    parser.add_argument(
        "--external-benchmark",
        default=str(OUTPUT_DIR / "external_benchmarks.csv"),
        help=(
            "Optional CSV with date plus a recession indicator column "
            "(recession, USREC, nber_recession, or contraction)."
        ),
    )
    parser.add_argument(
        "--nber-cache",
        default=str(OUTPUT_DIR / "external_benchmarks.csv"),
        help="Cache path for the FRED USREC/NBER recession benchmark.",
    )
    parser.add_argument(
        "--skip-nber-download",
        action="store_true",
        help="Do not try to download FRED USREC if the external benchmark file is missing.",
    )
    parser.add_argument(
        "--external-threshold",
        type=float,
        default=0.50,
        help="Probability threshold for NBER false-alarm and detection-lag metrics.",
    )
    parser.add_argument(
        "--external-regime-rule",
        choices=["fixed_contraction", "nber_max"],
        default="fixed_contraction",
        help=(
            "How to choose the regime probability tested against NBER. "
            "fixed_contraction uses the economically labeled Contraction/Recession regime; "
            "nber_max uses the highest average regime probability during NBER months."
        ),
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("--pca-components cannot be empty.")
    return values


def canonical_model_name(raw: str) -> str | None:
    key = (
        str(raw)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
    if key in {"full", "full_information", "full_info", "benchmark"}:
        return "full_information"
    if key in {"realtime", "real_time", "realtime_information", "real_time_information", "baseline"}:
        return "realtime_information"
    if key in {
        "nowcast",
        "nowcast_enhanced",
        "nowcast_information",
        "nowcast_enhanced_information",
        "candidate",
    }:
        return "nowcast_enhanced"
    return None


def infer_model_name_from_path(path: Path) -> str | None:
    parts = [part.lower().replace("-", "_").replace(" ", "_") for part in path.parts]
    joined = "_".join(parts)
    if "full_information" in joined or "full_info" in joined:
        return "full_information"
    if "realtime_information" in joined or "real_time_information" in joined or "realtime" in joined:
        return "realtime_information"
    if "nowcast_enhanced" in joined or "nowcast_information" in joined or "nowcast" in joined:
        return "nowcast_enhanced"
    if path.name.lower() == "hmm_models":
        return "nowcast_enhanced"
    return None


def resolve_manifest_path(raw_path: str, manifest_dir: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return manifest_dir / path


def read_exports_manifest(path: Path) -> dict[str, Path]:
    if not path.exists():
        raise FileNotFoundError(f"HMM exports manifest not found: {path}")

    manifest = pd.read_csv(path)
    model_col = next(
        (column for column in ["model", "information_set", "name"] if column in manifest.columns),
        None,
    )
    dir_col = next(
        (column for column in ["hmm_dir", "output_dir", "export_dir", "path"] if column in manifest.columns),
        None,
    )
    if model_col is None or dir_col is None:
        raise ValueError(
            f"{path} needs a model/information_set column and an hmm_dir/output_dir column."
        )

    out = {}
    for _, row in manifest.iterrows():
        model_name = canonical_model_name(str(row[model_col]))
        if model_name is None:
            continue
        out[model_name] = resolve_manifest_path(str(row[dir_col]), path.parent)
    return out


def discover_hmm_exports(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}

    discovered = {}
    for selected_path in root.rglob("hmm_selected_models.csv"):
        export_dir = selected_path.parent
        model_name = infer_model_name_from_path(export_dir)
        if model_name is not None and model_name not in discovered:
            discovered[model_name] = export_dir
    return discovered


def resolve_model_dirs(args: argparse.Namespace) -> dict[str, Path]:
    explicit_dirs = {
        "full_information": Path(args.full_dir),
        "realtime_information": Path(args.realtime_dir),
        "nowcast_enhanced": Path(args.nowcast_dir),
    }
    manifest_dirs = (
        read_exports_manifest(Path(args.exports_manifest))
        if args.exports_manifest
        else {}
    )
    discovered_dirs = discover_hmm_exports(Path(args.hmm_export_root))

    model_dirs = {}
    for model_name in MODEL_NAMES:
        if model_name in manifest_dirs:
            model_dirs[model_name] = manifest_dirs[model_name]
        elif explicit_dirs[model_name].exists():
            model_dirs[model_name] = explicit_dirs[model_name]
        elif model_name in discovered_dirs:
            model_dirs[model_name] = discovered_dirs[model_name]
        else:
            model_dirs[model_name] = explicit_dirs[model_name]
    return model_dirs


def required_files(model_dir: Path, n_pca: int) -> list[Path]:
    return [
        model_dir / f"hmm_state_probabilities_pca{n_pca}.csv",
        model_dir / f"hmm_regime_labels_pca{n_pca}.csv",
        model_dir / f"hmm_state_summary_pca{n_pca}.csv",
    ]


def validate_inputs(model_dirs: dict[str, Path], pca_components: list[int]) -> pd.DataFrame:
    rows = []
    for model_name, model_dir in model_dirs.items():
        for n_pca in pca_components:
            missing = [str(path) for path in required_files(model_dir, n_pca) if not path.exists()]
            rows.append(
                {
                    "model": model_name,
                    "hmm_dir": str(model_dir),
                    "pca_components": n_pca,
                    "ready": not missing,
                    "missing_files": "; ".join(missing),
                }
            )
    return pd.DataFrame(rows)


def regime_probability_frame(model_dir: Path, n_pca: int) -> pd.DataFrame:
    probabilities = pd.read_csv(
        model_dir / f"hmm_state_probabilities_pca{n_pca}.csv",
        parse_dates=["date"],
    )
    labels = pd.read_csv(model_dir / f"hmm_regime_labels_pca{n_pca}.csv")

    out = probabilities[["date", "most_likely_state", "most_likely_regime"]].copy()
    for regime in REGIME_LABELS:
        slug = REGIME_SLUGS[regime]
        out[f"{slug}_prob"] = 0.0

    for _, row in labels.iterrows():
        state = int(row["state"])
        regime = row["regime_label"]
        source_col = f"state_{state}_prob"
        if regime in REGIME_SLUGS and source_col in probabilities.columns:
            out[f"{REGIME_SLUGS[regime]}_prob"] = probabilities[source_col].astype(float)

    total = out[REGIME_PROB_COLS].sum(axis=1).replace(0.0, np.nan)
    out[REGIME_PROB_COLS] = out[REGIME_PROB_COLS].div(total, axis=0).fillna(0.0)
    return out.sort_values("date")


def entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def benchmark_metrics(
    benchmark: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_name: str,
    n_pca: int,
) -> dict[str, float | int | str]:
    merged = benchmark.merge(candidate, on="date", suffixes=("_benchmark", "_candidate"))
    if merged.empty:
        raise ValueError(f"No overlapping dates for {candidate_name}, PCA {n_pca}.")

    benchmark_probs = merged[[f"{col}_benchmark" for col in REGIME_PROB_COLS]].to_numpy(float)
    candidate_probs = merged[[f"{col}_candidate" for col in REGIME_PROB_COLS]].to_numpy(float)

    benchmark_regimes = merged["most_likely_regime_benchmark"]
    candidate_regimes = merged["most_likely_regime_candidate"]
    target_indices = np.array([REGIME_LABELS.index(regime) for regime in benchmark_regimes])
    candidate_target_probs = np.clip(candidate_probs[np.arange(len(merged)), target_indices], 1e-12, 1.0 - 1e-12)
    one_hot = np.eye(len(REGIME_LABELS))[target_indices]

    contraction_idx = REGIME_LABELS.index("Contraction / Recession")
    benchmark_contraction = (target_indices == contraction_idx).astype(float)
    candidate_contraction = np.clip(candidate_probs[:, contraction_idx], 1e-12, 1.0 - 1e-12)
    information_gap = np.abs(candidate_probs - benchmark_probs).sum(axis=1)

    return {
        "candidate_model": candidate_name,
        "benchmark_model": "full_information",
        "pca_components": n_pca,
        "overlap_start": merged["date"].min().date().isoformat(),
        "overlap_end": merged["date"].max().date().isoformat(),
        "n_overlap_months": int(len(merged)),
        "hard_regime_accuracy": float((benchmark_regimes == candidate_regimes).mean()),
        "mean_information_gap_l1_vs_full": float(information_gap.mean()),
        "median_information_gap_l1_vs_full": float(np.median(information_gap)),
        "benchmark_regime_log_score": float(np.log(candidate_target_probs).mean()),
        "benchmark_regime_brier_score": float(((candidate_probs - one_hot) ** 2).sum(axis=1).mean()),
        "probability_rmse_vs_full": float(np.sqrt(((candidate_probs - benchmark_probs) ** 2).mean())),
        "probability_mae_vs_full": float(np.abs(candidate_probs - benchmark_probs).mean()),
        "mean_candidate_entropy": float(entropy(candidate_probs).mean()),
        "mean_candidate_max_probability": float(candidate_probs.max(axis=1).mean()),
        "full_contraction_brier_score": float(((candidate_contraction - benchmark_contraction) ** 2).mean()),
        "full_contraction_log_score": float(
            (
                benchmark_contraction * np.log(candidate_contraction)
                + (1.0 - benchmark_contraction) * np.log(1.0 - candidate_contraction)
            ).mean()
        ),
    }


def model_fit_rows(model_dirs: dict[str, Path], pca_components: list[int]) -> pd.DataFrame:
    rows = []
    for model_name, model_dir in model_dirs.items():
        selected_path = model_dir / "hmm_selected_models.csv"
        if not selected_path.exists():
            continue
        selected = pd.read_csv(selected_path)
        for n_pca in pca_components:
            subset = selected[selected["pca_components"] == n_pca]
            if subset.empty:
                continue
            row = subset.iloc[0].to_dict()
            row["model"] = model_name
            rows.append(row)
    return pd.DataFrame(rows)


def pct_improvement(baseline: float, improvement: float) -> float:
    if pd.isna(baseline) or pd.isna(improvement):
        return np.nan
    baseline_abs = abs(float(baseline))
    if baseline_abs <= 1e-12:
        return np.nan
    return float(100.0 * improvement / baseline_abs)


def all_positive(values: list[float]) -> bool:
    clean = [float(value) for value in values if not pd.isna(value)]
    return bool(clean) and all(value > 0.0 for value in clean)


def improvement_rows(internal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n_pca, subset in internal.groupby("pca_components"):
        real = subset[subset["candidate_model"] == "realtime_information"]
        nowcast = subset[subset["candidate_model"] == "nowcast_enhanced"]
        if real.empty or nowcast.empty:
            continue
        real_row = real.iloc[0]
        nowcast_row = nowcast.iloc[0]
        accuracy_improvement = float(
            nowcast_row["hard_regime_accuracy"] - real_row["hard_regime_accuracy"]
        )
        log_score_improvement = float(
            nowcast_row["benchmark_regime_log_score"] - real_row["benchmark_regime_log_score"]
        )
        brier_score_improvement = float(
            real_row["benchmark_regime_brier_score"] - nowcast_row["benchmark_regime_brier_score"]
        )
        probability_rmse_improvement = float(
            real_row["probability_rmse_vs_full"] - nowcast_row["probability_rmse_vs_full"]
        )
        probability_mae_improvement = float(
            real_row["probability_mae_vs_full"] - nowcast_row["probability_mae_vs_full"]
        )
        information_gap_improvement = float(
            real_row["mean_information_gap_l1_vs_full"]
            - nowcast_row["mean_information_gap_l1_vs_full"]
        )
        contraction_brier_improvement = float(
            real_row["full_contraction_brier_score"] - nowcast_row["full_contraction_brier_score"]
        )
        positive_metrics = [
            information_gap_improvement,
            probability_rmse_improvement,
            probability_mae_improvement,
            accuracy_improvement,
            log_score_improvement,
            brier_score_improvement,
            contraction_brier_improvement,
        ]
        rows.append(
            {
                "pca_components": n_pca,
                "realtime_information_gap_l1_vs_full": float(
                    real_row["mean_information_gap_l1_vs_full"]
                ),
                "nowcast_information_gap_l1_vs_full": float(
                    nowcast_row["mean_information_gap_l1_vs_full"]
                ),
                "information_gap_l1_reduction": information_gap_improvement,
                "information_gap_l1_percent_reduction": pct_improvement(
                    real_row["mean_information_gap_l1_vs_full"],
                    information_gap_improvement,
                ),
                "realtime_probability_rmse_vs_full": float(real_row["probability_rmse_vs_full"]),
                "nowcast_probability_rmse_vs_full": float(nowcast_row["probability_rmse_vs_full"]),
                "probability_rmse_improvement": probability_rmse_improvement,
                "probability_rmse_percent_improvement": pct_improvement(
                    real_row["probability_rmse_vs_full"],
                    probability_rmse_improvement,
                ),
                "realtime_probability_mae_vs_full": float(real_row["probability_mae_vs_full"]),
                "nowcast_probability_mae_vs_full": float(nowcast_row["probability_mae_vs_full"]),
                "probability_mae_improvement": probability_mae_improvement,
                "probability_mae_percent_improvement": pct_improvement(
                    real_row["probability_mae_vs_full"],
                    probability_mae_improvement,
                ),
                "accuracy_improvement": accuracy_improvement,
                "accuracy_percent_improvement": pct_improvement(
                    real_row["hard_regime_accuracy"],
                    accuracy_improvement,
                ),
                "log_score_improvement": log_score_improvement,
                "log_score_percent_improvement": pct_improvement(
                    real_row["benchmark_regime_log_score"],
                    log_score_improvement,
                ),
                "brier_score_improvement": brier_score_improvement,
                "brier_score_percent_improvement": pct_improvement(
                    real_row["benchmark_regime_brier_score"],
                    brier_score_improvement,
                ),
                "contraction_brier_improvement": contraction_brier_improvement,
                "contraction_brier_percent_improvement": pct_improvement(
                    real_row["full_contraction_brier_score"],
                    contraction_brier_improvement,
                ),
                "positive_values_mean_nowcast_beats_realtime": all_positive(positive_metrics),
            }
        )
    return pd.DataFrame(rows)


def fallback_nber_2020_benchmark(cache_path: Path, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, str]:
    months = pd.DatetimeIndex(pd.to_datetime(dates)).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)
    months = pd.DatetimeIndex(sorted(set(months)))
    out = pd.DataFrame({"date": months})
    periods = out["date"].dt.to_period("M")
    out["external_recession"] = periods.isin(
        [pd.Period("2020-03", freq="M"), pd.Period("2020-04", freq="M")]
    ).astype(int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return (
        out,
        "used local NBER 2020 fallback benchmark: recession months are 2020-03 and 2020-04 "
        "from the NBER February 2020 peak and April 2020 trough chronology",
    )


def download_nber_benchmark(
    cache_path: Path,
    fallback_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame | None, str]:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 thesis-hmm-comparison/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            cache_path.write_bytes(response.read())
    except Exception as exc:
        if fallback_dates is not None and len(fallback_dates) > 0:
            fallback, status = fallback_nber_2020_benchmark(cache_path, fallback_dates)
            return fallback, f"Could not download FRED USREC/NBER benchmark ({exc}); {status}"
        return None, f"Could not download FRED USREC/NBER benchmark: {exc}"

    try:
        raw = pd.read_csv(cache_path)
    except Exception as exc:
        return None, f"Downloaded NBER benchmark but could not read it: {exc}"
    if "observation_date" in raw.columns and "USREC" in raw.columns:
        out = raw.rename(columns={"observation_date": "date"})
        out = out[["date", "USREC"]].rename(columns={"USREC": "external_recession"})
    elif "DATE" in raw.columns and "USREC" in raw.columns:
        out = raw.rename(columns={"DATE": "date"})
        out = out[["date", "USREC"]].rename(columns={"USREC": "external_recession"})
    else:
        return None, f"Downloaded USREC file has unexpected columns: {list(raw.columns)}"
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["external_recession"] = pd.to_numeric(out["external_recession"], errors="coerce")
    out = out.dropna(subset=["date", "external_recession"])
    out["external_recession"] = (out["external_recession"] > 0).astype(int)
    out.to_csv(cache_path, index=False)
    return out, f"downloaded FRED USREC/NBER benchmark to {cache_path}"


def load_external_benchmark(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"External benchmark file not found: {path}"
    df = pd.read_csv(path)
    if "date" not in df.columns:
        return None, f"External benchmark file has no date column: {path}"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    candidates = ["external_recession", "recession", "USREC", "nber_recession", "contraction"]
    indicator = next((column for column in candidates if column in df.columns), None)
    if indicator is None:
        return None, f"External benchmark needs one of these columns: {candidates}"
    out = df[["date", indicator]].copy()
    out = out.rename(columns={indicator: "external_recession"})
    out["external_recession"] = pd.to_numeric(out["external_recession"], errors="coerce")
    out = out.dropna(subset=["external_recession"])
    out["external_recession"] = (out["external_recession"] > 0).astype(int)
    return out, "loaded"


def month_difference(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    later_period = pd.Timestamp(later).to_period("M")
    earlier_period = pd.Timestamp(earlier).to_period("M")
    return int(later_period.ordinal - earlier_period.ordinal)


def external_metrics(
    benchmark: pd.DataFrame,
    model_name: str,
    model_probs: pd.DataFrame,
    n_pca: int,
    threshold: float,
    regime_rule: str,
) -> dict[str, float | int | str] | None:
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    merged = benchmark.merge(model_probs, on="date", how="inner")
    if merged.empty:
        return None

    y = merged["external_recession"].to_numpy(int)
    recession_rows = merged[merged["external_recession"] == 1]
    nber_max_col = (
        f"{REGIME_SLUGS['Contraction / Recession']}_prob"
        if recession_rows.empty
        else recession_rows[REGIME_PROB_COLS].mean().idxmax()
    )
    fixed_col = f"{REGIME_SLUGS['Contraction / Recession']}_prob"
    selected_col = nber_max_col if regime_rule == "nber_max" else fixed_col
    selected_regime = REGIME_BY_PROB_COL[selected_col]
    nber_max_regime = REGIME_BY_PROB_COL[nber_max_col]
    p = np.clip(merged[selected_col].to_numpy(float), 1e-12, 1.0 - 1e-12)
    hard = (p > threshold).astype(int)
    auc = np.nan
    if roc_auc_score is not None and len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))

    recession_start = merged.loc[merged["external_recession"] == 1, "date"].min()
    detected_dates = merged.loc[merged[selected_col] > threshold, "date"]
    if pd.notna(recession_start):
        detected_dates = detected_dates[detected_dates >= recession_start]
    first_detected = detected_dates.min() if not detected_dates.empty else pd.NaT
    detection_lag = (
        month_difference(first_detected, recession_start)
        if pd.notna(recession_start) and pd.notna(first_detected)
        else np.nan
    )
    false_alarm_mask = y == 0
    false_alarm_rate = (
        float((hard[false_alarm_mask] == 1).mean())
        if false_alarm_mask.any()
        else np.nan
    )

    return {
        "model": model_name,
        "pca_components": n_pca,
        "n_overlap_months": int(len(merged)),
        "external_start": merged["date"].min().date().isoformat(),
        "external_end": merged["date"].max().date().isoformat(),
        "recession_months": int(y.sum()),
        "nber_recession_regime": selected_regime,
        "external_regime_rule": regime_rule,
        "nber_max_probability_regime": nber_max_regime,
        "nber_max_differs_from_selected": bool(nber_max_regime != selected_regime),
        "nber_probability_threshold": float(threshold),
        "nber_recession_probability_brier_score": float(((p - y) ** 2).mean()),
        "nber_recession_probability_log_score": float(
            (y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        ),
        "threshold_recession_accuracy": float((hard == y).mean()),
        "nber_false_alarm_rate": false_alarm_rate,
        "nber_detection_lag_months": detection_lag,
        "roc_auc": auc,
        "mean_nber_recession_prob_when_recession": float(p[y == 1].mean()) if y.sum() else np.nan,
        "mean_nber_recession_prob_when_not_recession": float(p[y == 0].mean()) if (1 - y).sum() else np.nan,
    }


def external_improvement_rows(external: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if external.empty:
        return pd.DataFrame(rows)

    for n_pca, subset in external.groupby("pca_components"):
        real = subset[subset["model"] == "realtime_information"]
        nowcast = subset[subset["model"] == "nowcast_enhanced"]
        if real.empty or nowcast.empty:
            continue

        real_row = real.iloc[0]
        nowcast_row = nowcast.iloc[0]
        real_gap = (
            real_row["mean_nber_recession_prob_when_recession"]
            - real_row["mean_nber_recession_prob_when_not_recession"]
        )
        nowcast_gap = (
            nowcast_row["mean_nber_recession_prob_when_recession"]
            - nowcast_row["mean_nber_recession_prob_when_not_recession"]
        )
        brier_improvement = float(
            real_row["nber_recession_probability_brier_score"]
            - nowcast_row["nber_recession_probability_brier_score"]
        )
        log_score_improvement = float(
            nowcast_row["nber_recession_probability_log_score"]
            - real_row["nber_recession_probability_log_score"]
        )
        accuracy_improvement = float(
            nowcast_row["threshold_recession_accuracy"]
            - real_row["threshold_recession_accuracy"]
        )
        auc_improvement = float(nowcast_row["roc_auc"] - real_row["roc_auc"])
        gap_improvement = float(nowcast_gap - real_gap)
        false_alarm_improvement = float(
            real_row["nber_false_alarm_rate"] - nowcast_row["nber_false_alarm_rate"]
        )
        detection_lag_improvement = float(
            real_row["nber_detection_lag_months"]
            - nowcast_row["nber_detection_lag_months"]
        )

        positive_metrics = [
            brier_improvement,
            log_score_improvement,
            accuracy_improvement,
            false_alarm_improvement,
            detection_lag_improvement,
            auc_improvement,
            gap_improvement,
        ]
        rows.append(
            {
                "pca_components": n_pca,
                "realtime_nber_recession_regime": real_row["nber_recession_regime"],
                "nowcast_nber_recession_regime": nowcast_row["nber_recession_regime"],
                "realtime_external_brier_score": float(
                    real_row["nber_recession_probability_brier_score"]
                ),
                "nowcast_external_brier_score": float(
                    nowcast_row["nber_recession_probability_brier_score"]
                ),
                "external_brier_score_improvement": brier_improvement,
                "external_brier_score_percent_improvement": pct_improvement(
                    real_row["nber_recession_probability_brier_score"],
                    brier_improvement,
                ),
                "realtime_external_log_score": float(
                    real_row["nber_recession_probability_log_score"]
                ),
                "nowcast_external_log_score": float(
                    nowcast_row["nber_recession_probability_log_score"]
                ),
                "external_log_score_improvement": log_score_improvement,
                "external_log_score_percent_improvement": pct_improvement(
                    real_row["nber_recession_probability_log_score"],
                    log_score_improvement,
                ),
                "realtime_threshold_recession_accuracy": float(
                    real_row["threshold_recession_accuracy"]
                ),
                "nowcast_threshold_recession_accuracy": float(
                    nowcast_row["threshold_recession_accuracy"]
                ),
                "threshold_recession_accuracy_improvement": accuracy_improvement,
                "threshold_recession_accuracy_percent_improvement": pct_improvement(
                    real_row["threshold_recession_accuracy"],
                    accuracy_improvement,
                ),
                "realtime_false_alarm_rate": float(real_row["nber_false_alarm_rate"]),
                "nowcast_false_alarm_rate": float(nowcast_row["nber_false_alarm_rate"]),
                "false_alarm_rate_reduction": false_alarm_improvement,
                "false_alarm_rate_percent_reduction": pct_improvement(
                    real_row["nber_false_alarm_rate"],
                    false_alarm_improvement,
                ),
                "realtime_detection_lag_months": float(real_row["nber_detection_lag_months"]),
                "nowcast_detection_lag_months": float(nowcast_row["nber_detection_lag_months"]),
                "detection_lag_reduction_months": detection_lag_improvement,
                "realtime_roc_auc": float(real_row["roc_auc"]),
                "nowcast_roc_auc": float(nowcast_row["roc_auc"]),
                "roc_auc_improvement": auc_improvement,
                "roc_auc_percent_improvement": pct_improvement(
                    real_row["roc_auc"],
                    auc_improvement,
                ),
                "realtime_recession_probability_gap": float(real_gap),
                "nowcast_recession_probability_gap": float(nowcast_gap),
                "recession_probability_gap_improvement": gap_improvement,
                "recession_probability_gap_percent_improvement": pct_improvement(
                    real_gap,
                    gap_improvement,
                ),
                "positive_values_mean_nowcast_beats_realtime": all_positive(positive_metrics),
            }
        )
    return pd.DataFrame(rows)


def write_workbook(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    path = output_dir / "hmm_information_set_comparison.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_components = parse_int_list(args.pca_components)
    model_dirs = resolve_model_dirs(args)
    readiness = validate_inputs(model_dirs, pca_components)
    readiness.to_csv(output_dir / "comparison_input_readiness.csv", index=False)

    missing_path = output_dir / "comparison_missing_inputs.csv"
    if not readiness["ready"].all():
        missing = readiness[~readiness["ready"]]
        missing.to_csv(missing_path, index=False)
        raise SystemExit(
            "Comparison inputs are missing. See comparison_missing_inputs.csv in "
            f"{output_dir}"
        )
    readiness.iloc[0:0].to_csv(missing_path, index=False)

    probability_tables: dict[tuple[str, int], pd.DataFrame] = {}
    for model_name, model_dir in model_dirs.items():
        for n_pca in pca_components:
            probability_tables[(model_name, n_pca)] = regime_probability_frame(model_dir, n_pca)

    internal_rows = []
    for n_pca in pca_components:
        benchmark = probability_tables[("full_information", n_pca)]
        for candidate in ["realtime_information", "nowcast_enhanced"]:
            internal_rows.append(
                benchmark_metrics(
                    benchmark,
                    probability_tables[(candidate, n_pca)],
                    candidate_name=candidate,
                    n_pca=n_pca,
                )
            )
    internal = pd.DataFrame(internal_rows)
    improvements = improvement_rows(internal)
    fit = model_fit_rows(model_dirs, pca_components)

    all_probability_dates = pd.DatetimeIndex(
        sorted(
            set().union(
                *[set(table["date"]) for table in probability_tables.values()]
            )
        )
    )

    external_benchmark, external_status = load_external_benchmark(Path(args.external_benchmark))
    if external_benchmark is None and not args.skip_nber_download:
        external_benchmark, external_status = download_nber_benchmark(
            Path(args.nber_cache),
            fallback_dates=all_probability_dates,
        )
    external_rows = []
    if external_benchmark is not None:
        for model_name in model_dirs:
            for n_pca in pca_components:
                row = external_metrics(
                    external_benchmark,
                    model_name,
                    probability_tables[(model_name, n_pca)],
                    n_pca,
                    threshold=args.external_threshold,
                    regime_rule=args.external_regime_rule,
                )
                if row:
                    external_rows.append(row)
    external = pd.DataFrame(external_rows)
    external_improvements = external_improvement_rows(external)
    external_status_df = pd.DataFrame(
        [{"external_benchmark": args.external_benchmark, "status": external_status}]
    )

    internal.to_csv(output_dir / "internal_full_information_benchmark.csv", index=False)
    improvements.to_csv(output_dir / "nowcast_vs_realtime_improvement.csv", index=False)
    fit.to_csv(output_dir / "hmm_fit_statistics_by_information_set.csv", index=False)
    external.to_csv(output_dir / "external_benchmark_comparison.csv", index=False)
    external_improvements.to_csv(
        output_dir / "external_nowcast_vs_realtime_improvement.csv",
        index=False,
    )
    external_status_df.to_csv(output_dir / "external_benchmark_status.csv", index=False)

    write_workbook(
        output_dir,
        {
            "Readiness": readiness,
            "InternalBenchmark": internal,
            "NowcastVsRealtime": improvements,
            "FitStatistics": fit,
            "ExternalBenchmark": external,
            "ExternalImprovement": external_improvements,
            "ExternalStatus": external_status_df,
        },
    )

    print(f"Wrote HMM information-set comparison outputs to: {output_dir}")
    print(f"Compared PCA dimensions: {pca_components}")
    print("Internal benchmark: full_information")
    if external_benchmark is None:
        print(external_status)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
