"""
Step 3 thesis data pipeline: fit HMM regimes on PCA factors.

Default input:
    outputs/full_information.csv
    outputs/realtime_information.csv
    outputs/nowcast_information.csv

Default outputs:
    outputs/pca_full_information/hmm_pca_input.csv
    outputs/pca_realtime_information/hmm_pca_input.csv
    outputs/pca_nowcast_enhanced/hmm_pca_input.csv
    outputs/hmm_models_full_information/*.csv
    outputs/hmm_models_realtime_information/*.csv
    outputs/hmm_models_nowcast_enhanced/*.csv

This script intentionally starts with diagonal covariance because the usable
macro sample is modest relative to a high-dimensional full-covariance HMM.

The PCA cleaning stage exports rolling and static factors. Rolling PCA is used
as a loading-stability diagnostic; when loadings are stable, the HMM uses static
PCA scores to keep the longer stationary sample. The baseline HMM uses five
factors as a parsimonious expanded specification; the PCA diagnostics report
how many factors are needed to reach the 70 percent variance threshold.

For backwards compatibility, passing --input fits a single PCA input and writes
to --output-dir, matching the earlier one-model behavior.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_INPUT = OUTPUT_DIR / "hmm_pca_input.csv"
DEFAULT_HMM_OUTPUT_DIR = OUTPUT_DIR / "hmm_models"

INFORMATION_SETS = {
    "full_information": {
        "panel": OUTPUT_DIR / "full_information.csv",
        "pca_dir": OUTPUT_DIR / "pca_full_information",
        "hmm_dir": OUTPUT_DIR / "hmm_models_full_information",
    },
    "realtime_information": {
        "panel": OUTPUT_DIR / "realtime_information.csv",
        "pca_dir": OUTPUT_DIR / "pca_realtime_information",
        "hmm_dir": OUTPUT_DIR / "hmm_models_realtime_information",
    },
    "nowcast_enhanced": {
        "panel": OUTPUT_DIR / "nowcast_information.csv",
        "pca_dir": OUTPUT_DIR / "pca_nowcast_enhanced",
        "hmm_dir": OUTPUT_DIR / "hmm_models_nowcast_enhanced",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit HMM regime models on PCA factors.")
    parser.add_argument(
        "--input",
        default="",
        help=(
            "Optional single PCA input CSV with a date column. If omitted, the script "
            "fits full-information, real-time, and nowcast-enhanced HMMs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directory for HMM model outputs when --input is supplied. "
            "Defaults to outputs/hmm_models in single-input mode."
        ),
    )
    parser.add_argument(
        "--information-sets",
        default="full_information,realtime_information,nowcast_enhanced",
        help=(
            "Comma-separated information sets to fit in all-model mode. "
            "Options: full_information, realtime_information, nowcast_enhanced."
        ),
    )
    parser.add_argument(
        "--pca-components",
        default="5",
        help="Comma-separated PCA dimensions to fit in the HMM. Default is the 5-PC thesis baseline.",
    )
    parser.add_argument(
        "--pca-output-components",
        type=int,
        default=5,
        help="Number of PCA score columns to create during cleaning. Default keeps PCA 1-5 for diagnostics.",
    )
    parser.add_argument(
        "--state-counts",
        default="4",
        help="Comma-separated hidden-state counts to compare. Default fixes the thesis model at 4 regimes.",
    )
    parser.add_argument(
        "--bic-diagnostic-state-counts",
        default="3,4,5",
        help=(
            "Additional hidden-state counts to fit for BIC diagnostics without changing "
            "the selected thesis state count. Default compares 3, 4, and 5 states."
        ),
    )
    parser.add_argument(
        "--covariance-type",
        default="diag",
        choices=["diag", "full", "tied", "spherical"],
        help="Gaussian HMM covariance type. Diag is preferred for the short PCA sample.",
    )
    parser.add_argument("--n-iter", type=int, default=1000, help="Maximum EM iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="EM convergence tolerance.")
    parser.add_argument(
        "--random-seeds",
        default="0,1,2,3,4",
        help="Comma-separated random seeds; best likelihood is kept for each spec.",
    )
    parser.add_argument(
        "--skip-pca-build",
        action="store_true",
        help="In all-model mode, require existing pca_*/*hmm_pca_input.csv files.",
    )
    parser.add_argument(
        "--refresh-pca-inputs",
        action="store_true",
        help="Rebuild all information-set PCA inputs before fitting HMMs.",
    )
    parser.add_argument(
        "--drop-columns",
        default="HYSpread,U6UNRATE,ISM_NewOrders,HOUST_YoY,ICSA_YoY",
        help="Columns passed through to dataCleaning.py when building PCA inputs.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Rolling PCA window passed through to dataCleaning.py.",
    )
    parser.add_argument(
        "--adf-alpha",
        type=float,
        default=0.05,
        help="ADF alpha passed through to dataCleaning.py.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.70,
        help="PCA variance threshold passed through to dataCleaning.py.",
    )
    parser.add_argument(
        "--loading-stability-threshold",
        type=float,
        default=0.85,
        help="Rolling loading stability threshold passed through to dataCleaning.py.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.80,
        help="High-correlation diagnostic threshold passed through to dataCleaning.py.",
    )
    parser.add_argument(
        "--write-cleaning-workbooks",
        action="store_true",
        help="Allow dataCleaning.py to write Excel workbooks for each information set.",
    )
    parser.add_argument(
        "--independent-cleaning",
        action="store_true",
        help=(
            "Clean each information set separately. By default, all-model mode uses "
            "one shared date sample and ADF transformation schema."
        ),
    )
    return parser.parse_args()


def require_packages() -> None:
    try:
        __import__("hmmlearn")
    except ImportError as exc:
        install = (
            '.\\.venv\\Scripts\\python.exe -m pip install -r '
            '"Data Processor\\requirements-analysis.txt"'
        )
        raise SystemExit(
            "Missing HMM package: hmmlearn\n"
            "Install it from the project root with:\n    "
            + install
        ) from exc


def parse_int_list(raw: str, name: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated integer list: {raw!r}") from exc
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    return values


def parse_information_sets(raw: str) -> list[str]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in INFORMATION_SETS:
            valid = ", ".join(INFORMATION_SETS)
            raise ValueError(f"Unknown information set {item!r}; valid options are: {valid}")
        values.append(item)
    if not values:
        raise ValueError("--information-sets cannot be empty.")
    return values


def read_pca_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"PCA input file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"{path.name} must contain a date column.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="any")
    if df.empty:
        raise ValueError("No complete PCA rows available for HMM fitting.")
    return df


def parameter_count(n_states: int, n_features: int, covariance_type: str) -> int:
    start_params = n_states - 1
    transition_params = n_states * (n_states - 1)
    mean_params = n_states * n_features
    if covariance_type == "diag":
        covariance_params = n_states * n_features
    elif covariance_type == "full":
        covariance_params = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "tied":
        covariance_params = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        covariance_params = n_states
    else:
        raise ValueError(f"Unsupported covariance type: {covariance_type}")
    return start_params + transition_params + mean_params + covariance_params


def fit_best_hmm(
    x: np.ndarray,
    n_states: int,
    covariance_type: str,
    seeds: list[int],
    n_iter: int,
    tol: float,
) -> tuple[Any, dict[str, Any]]:
    from hmmlearn.hmm import GaussianHMM

    best_model = None
    best_info: dict[str, Any] | None = None
    for seed in seeds:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
            min_covar=1e-4,
        )
        try:
            model.fit(x)
            log_likelihood = float(model.score(x))
            converged = bool(model.monitor_.converged)
            iterations = int(model.monitor_.iter)
            error = ""
        except Exception as exc:
            log_likelihood = -np.inf
            converged = False
            iterations = 0
            error = str(exc)

        info = {
            "random_seed": seed,
            "log_likelihood": log_likelihood,
            "converged": converged,
            "iterations": iterations,
            "error": error,
        }
        if best_info is None or log_likelihood > best_info["log_likelihood"]:
            best_model = model if np.isfinite(log_likelihood) else None
            best_info = info

    if best_model is None or best_info is None:
        raise RuntimeError(f"All HMM fits failed for {n_states} states.")
    return best_model, best_info


def state_probability_table(
    dates: pd.Index,
    model: Any,
    x: np.ndarray,
) -> pd.DataFrame:
    probabilities = model.predict_proba(x)
    states = probabilities.argmax(axis=1)
    out = pd.DataFrame(probabilities, index=dates, columns=[f"state_{i}_prob" for i in range(model.n_components)])
    out.insert(0, "most_likely_state", states)
    return out.reset_index().rename(columns={"index": "date"})


def state_summary_table(
    model: Any,
    x: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    state_cols = [col for col in probabilities.columns if col.endswith("_prob")]
    states = probabilities["most_likely_state"].to_numpy()
    rows = []
    for state in range(model.n_components):
        mask = states == state
        row = {
            "state": state,
            "observations": int(mask.sum()),
            "observation_share": float(mask.mean()),
            "self_transition_probability": float(model.transmat_[state, state]),
        }
        for column in x.columns:
            row[f"{column}_mean"] = float(x.loc[mask, column].mean()) if mask.any() else np.nan
        prob_col = f"state_{state}_prob"
        if prob_col in state_cols:
            row["average_posterior_probability"] = float(probabilities[prob_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def regime_label_table(summary: pd.DataFrame) -> pd.DataFrame:
    label_rows = []
    if summary.shape[0] != 4 or "PCA_1_mean" not in summary.columns or "PCA_2_mean" not in summary.columns:
        for state in summary["state"]:
            label_rows.append(
                {
                    "state": int(state),
                    "regime_label": f"Regime {int(state)}",
                    "label_reason": "Generic label because the model is not a 4-state PCA model.",
                }
            )
        return pd.DataFrame(label_rows)

    remaining = set(summary["state"].astype(int).tolist())
    contraction = int(summary.sort_values("PCA_1_mean").iloc[0]["state"])
    remaining.remove(contraction)

    recovery = int(
        summary[summary["state"].isin(remaining)]
        .sort_values(["PCA_1_mean", "PCA_3_mean"], ascending=[False, False])
        .iloc[0]["state"]
    )
    remaining.remove(recovery)

    expansion = int(
        summary[summary["state"].isin(remaining)]
        .sort_values(["PCA_2_mean", "PCA_3_mean"], ascending=[True, False])
        .iloc[0]["state"]
    )
    remaining.remove(expansion)

    overheating = int(next(iter(remaining)))
    labels = {
        contraction: (
            "Contraction / Recession",
            "Lowest PCA_1 growth-momentum mean: negative growth/recession regime.",
            "PCA_1 negative; PCA_2 high stress/slack when the shock is broad; PCA_3 tight or frozen credit.",
        ),
        expansion: (
            "Expansion / Goldilocks",
            "Lowest PCA_2 stress/slack mean after removing contraction and recovery.",
            "PCA_1 high or stable; PCA_2 low stress/slack; PCA_3 neutral or constructive credit.",
        ),
        overheating: (
            "Overheating / Late Cycle",
            "Remaining mature-cycle transition regime after contraction, recovery, and goldilocks are identified.",
            "PCA_1 peaking or decelerating; PCA_2 tight labor or nascent stress; PCA_3 tightening liquidity/credit.",
        ),
        recovery: (
            "Recovery / Early Cycle",
            "Highest PCA_1 growth/rebound mean among non-contraction states.",
            "PCA_1 stabilizing or turning positive; PCA_2 stress/slack still elevated but fading; PCA_3 easing liquidity.",
        ),
    }

    for state in sorted(labels):
        label, reason, target_profile = labels[state]
        state_row = summary[summary["state"] == state].iloc[0]
        label_rows.append(
            {
                "state": state,
                "regime_label": label,
                "label_reason": reason,
                "target_profile": target_profile,
                "PCA_1_mean": float(state_row.get("PCA_1_mean", np.nan)),
                "PCA_2_mean": float(state_row.get("PCA_2_mean", np.nan)),
                "PCA_3_mean": float(state_row.get("PCA_3_mean", np.nan)),
            }
        )
    return pd.DataFrame(label_rows)


def attach_regime_labels(
    probabilities: pd.DataFrame,
    summary: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_map = dict(zip(labels["state"], labels["regime_label"]))
    out_probabilities = probabilities.copy()
    out_probabilities.insert(
        2,
        "most_likely_regime",
        out_probabilities["most_likely_state"].map(label_map),
    )
    label_columns = ["state", "regime_label", "label_reason", "target_profile"]
    out_summary = summary.merge(labels[label_columns], on="state", how="left")
    ordered = ["state", "regime_label", "label_reason", "target_profile"]
    other_cols = [col for col in out_summary.columns if col not in ordered]
    return out_probabilities, out_summary[ordered + other_cols]


def write_hmm_outputs(
    input_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    pca_components: list[int],
    state_counts: list[int],
    seeds: list[int],
    model_name: str,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_suffixes = {f"pca{n_pca}" for n_pca in pca_components}
    for pattern in [
        "hmm_state_probabilities_pca*.csv",
        "hmm_state_summary_pca*.csv",
        "hmm_regime_labels_pca*.csv",
    ]:
        for path in output_dir.glob(pattern):
            if not any(suffix in path.stem for suffix in requested_suffixes):
                path.unlink()

    pca = read_pca_input(input_path)
    comparison_rows = []
    selected_rows = []

    for n_pca in pca_components:
        columns = [f"PCA_{idx}" for idx in range(1, n_pca + 1)]
        missing = [column for column in columns if column not in pca.columns]
        if missing:
            raise ValueError(
                f"Requested {n_pca} PCA components, but input is missing: {missing}"
            )

        x_df = pca[columns].copy()
        x = x_df.to_numpy(dtype=float)
        diagnostic_state_counts = (
            parse_int_list(args.bic_diagnostic_state_counts, "--bic-diagnostic-state-counts")
            if str(args.bic_diagnostic_state_counts).strip()
            else []
        )
        all_state_counts = sorted(set(state_counts + diagnostic_state_counts))
        fitted_specs = []
        for n_states in all_state_counts:
            model, info = fit_best_hmm(
                x,
                n_states=n_states,
                covariance_type=args.covariance_type,
                seeds=seeds,
                n_iter=args.n_iter,
                tol=args.tol,
            )
            n_params = parameter_count(n_states, n_pca, args.covariance_type)
            log_likelihood = info["log_likelihood"]
            aic = 2 * n_params - 2 * log_likelihood
            bic = np.log(x.shape[0]) * n_params - 2 * log_likelihood
            row = {
                "model": model_name,
                "input_path": str(input_path),
                "pca_components": n_pca,
                "n_states": n_states,
                "covariance_type": args.covariance_type,
                "n_observations": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "n_parameters": int(n_params),
                "best_random_seed": int(info["random_seed"]),
                "log_likelihood": log_likelihood,
                "aic": float(aic),
                "bic": float(bic),
                "converged": bool(info["converged"]),
                "iterations": int(info["iterations"]),
                "error": info["error"],
                "selection_candidate": bool(n_states in state_counts),
                "bic_diagnostic_candidate": bool(n_states in diagnostic_state_counts),
            }
            comparison_rows.append(row)
            if n_states in state_counts:
                fitted_specs.append((bic, aic, n_states, model, row))

        _, _, selected_states, selected_model, selected_row = sorted(fitted_specs, key=lambda item: item[0])[0]
        probabilities = state_probability_table(pca.index, selected_model, x)
        summary = state_summary_table(selected_model, x_df, probabilities)
        labels = regime_label_table(summary)
        probabilities, summary = attach_regime_labels(probabilities, summary, labels)

        probabilities.to_csv(output_dir / f"hmm_state_probabilities_pca{n_pca}.csv", index=False)
        summary.to_csv(output_dir / f"hmm_state_summary_pca{n_pca}.csv", index=False)
        labels.to_csv(output_dir / f"hmm_regime_labels_pca{n_pca}.csv", index=False)
        selected_rows.append(
            {
                **selected_row,
                "selection_rule": "minimum_bic_within_requested_state_counts",
                "selected_probability_file": f"hmm_state_probabilities_pca{n_pca}.csv",
                "selected_summary_file": f"hmm_state_summary_pca{n_pca}.csv",
                "selected_label_file": f"hmm_regime_labels_pca{n_pca}.csv",
                "selected_n_states": selected_states,
            }
        )

    pd.DataFrame(comparison_rows).to_csv(output_dir / "hmm_model_comparison.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(output_dir / "hmm_selected_models.csv", index=False)
    return selected_rows


def build_pca_input(
    model_name: str,
    args: argparse.Namespace,
    n_components: int,
) -> Path:
    config = INFORMATION_SETS[model_name]
    panel_path = config["panel"]
    pca_dir = config["pca_dir"]
    pca_input = pca_dir / "hmm_pca_input.csv"

    if not panel_path.exists():
        raise FileNotFoundError(f"Missing {model_name} panel: {panel_path}")

    if args.skip_pca_build:
        if not pca_input.exists():
            raise FileNotFoundError(
                f"Missing PCA input for {model_name}: {pca_input}. "
                "Run without --skip-pca-build to create it."
            )
        return pca_input

    if pca_input.exists() and not args.refresh_pca_inputs:
        return pca_input

    command = [
        sys.executable,
        str(BASE_DIR / "dataCleaning.py"),
        "--input",
        str(panel_path),
        "--output-dir",
        str(pca_dir),
        "--drop-columns",
        args.drop_columns,
        "--rolling-window",
        str(args.rolling_window),
        "--n-components",
        str(n_components),
        "--adf-alpha",
        str(args.adf_alpha),
        "--variance-threshold",
        str(args.variance_threshold),
        "--loading-stability-threshold",
        str(args.loading_stability_threshold),
        "--correlation-threshold",
        str(args.correlation_threshold),
    ]
    if not args.write_cleaning_workbooks:
        command.append("--no-workbook")

    subprocess.run(command, cwd=BASE_DIR.parent, check=True)
    return pca_input


def existing_uniform_pca_ready(model_names: list[str]) -> bool:
    input_frames = []
    stationary_frames = []
    for model_name in model_names:
        pca_dir = INFORMATION_SETS[model_name]["pca_dir"]
        pca_input = pca_dir / "hmm_pca_input.csv"
        stationary_input = pca_dir / "stationary_inputs.csv"
        schema = pca_dir / "uniform_cleaning_schema.csv"
        if not pca_input.exists() or not stationary_input.exists() or not schema.exists():
            return False
        pca_df = pd.read_csv(pca_input, parse_dates=["date"])
        stationary_df = pd.read_csv(stationary_input, parse_dates=["date"])
        input_frames.append((model_name, pca_df))
        stationary_frames.append((model_name, stationary_df))

    first_pca = input_frames[0][1]
    first_stationary = stationary_frames[0][1]
    for _, frame in input_frames[1:]:
        if list(frame.columns) != list(first_pca.columns):
            return False
        if not frame["date"].equals(first_pca["date"]):
            return False
    for _, frame in stationary_frames[1:]:
        if list(frame.columns) != list(first_stationary.columns):
            return False
        if not frame["date"].equals(first_stationary["date"]):
            return False
    return True


def build_uniform_pca_inputs(
    model_names: list[str],
    args: argparse.Namespace,
    n_components: int,
) -> dict[str, Path]:
    import dataCleaning as dc

    pca_inputs = {
        model_name: INFORMATION_SETS[model_name]["pca_dir"] / "hmm_pca_input.csv"
        for model_name in model_names
    }
    if args.skip_pca_build:
        missing = [str(path) for path in pca_inputs.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing PCA inputs with --skip-pca-build: " + "; ".join(missing)
            )
        if not existing_uniform_pca_ready(model_names):
            raise ValueError(
                "Existing PCA inputs are not marked as uniform. "
                "Run without --skip-pca-build to rebuild them."
            )
        return pca_inputs

    if not args.refresh_pca_inputs and existing_uniform_pca_ready(model_names):
        return pca_inputs

    dc.require_packages()
    drop_columns = [item.strip() for item in args.drop_columns.split(",") if item.strip()]

    panels = {}
    kept_panels = {}
    for model_name in model_names:
        panel_path = INFORMATION_SETS[model_name]["panel"]
        if not panel_path.exists():
            raise FileNotFoundError(f"Missing {model_name} panel: {panel_path}")
        panel = dc.read_panel(panel_path)
        panels[model_name] = panel
        existing_drop_columns = [column for column in drop_columns if column in panel.columns]
        kept_panels[model_name] = panel.drop(columns=existing_drop_columns)

    first_model = model_names[0]
    common_columns = [
        column
        for column in kept_panels[first_model].columns
        if all(column in kept_panels[model_name].columns for model_name in model_names)
    ]
    common_index = kept_panels[first_model][common_columns].dropna(how="any").index
    for model_name in model_names[1:]:
        complete_index = kept_panels[model_name][common_columns].dropna(how="any").index
        common_index = common_index.intersection(complete_index)
    common_index = common_index.sort_values()
    if common_index.empty:
        raise ValueError("No common complete date sample remains across information sets.")

    schema_rows = []
    adf_rows = []
    for column in common_columns:
        level_passes = []
        diff_passes = []
        level_rows = []
        diff_rows = []
        for model_name in model_names:
            series = kept_panels[model_name].loc[common_index, column]
            level_row, level_stationary = dc.adf_result(
                series,
                variable=column,
                stage="uniform_input_level",
                alpha=args.adf_alpha,
                transformation="level",
            )
            diff_row, diff_stationary = dc.adf_result(
                series.diff(),
                variable=column,
                stage="uniform_input_difference",
                alpha=args.adf_alpha,
                transformation="first_difference",
            )
            level_row["information_set"] = model_name
            diff_row["information_set"] = model_name
            level_rows.append(level_row)
            diff_rows.append(diff_row)
            level_passes.append(level_stationary)
            diff_passes.append(diff_stationary)

        if all(level_passes):
            decision = "keep_level"
            output_variable = column
        elif all(diff_passes):
            decision = "keep_difference"
            output_variable = f"{column}_diff"
        else:
            decision = "drop_nonstationary_any_information_set"
            output_variable = ""

        for row in level_rows:
            row["decision"] = decision
        for row in diff_rows:
            row["decision"] = decision
        adf_rows.extend(level_rows)
        adf_rows.extend(diff_rows)
        schema_rows.append(
            {
                "source_variable": column,
                "decision": decision,
                "output_variable": output_variable,
                "all_level_stationary": bool(all(level_passes)),
                "all_difference_stationary": bool(all(diff_passes)),
                "information_sets": ", ".join(model_names),
            }
        )

    schema = pd.DataFrame(schema_rows)
    kept_schema = schema[schema["output_variable"] != ""].copy()
    if kept_schema.empty:
        raise ValueError("Uniform cleaning schema dropped every variable.")
    if kept_schema.shape[0] < n_components:
        raise ValueError(
            f"Uniform cleaning kept {kept_schema.shape[0]} variables, "
            f"but {n_components} PCA components were requested."
        )

    validation_rows = []
    pca_summary_rows = []
    for model_name in model_names:
        pca_dir = INFORMATION_SETS[model_name]["pca_dir"]
        pca_dir.mkdir(parents=True, exist_ok=True)
        aligned = kept_panels[model_name].loc[common_index, common_columns]
        stationary = pd.DataFrame(index=common_index)
        for _, row in kept_schema.iterrows():
            source = row["source_variable"]
            output = row["output_variable"]
            if row["decision"] == "keep_level":
                stationary[output] = aligned[source]
            elif row["decision"] == "keep_difference":
                stationary[output] = aligned[source].diff()
        stationary = stationary.dropna(how="any")

        normalized, normalization_stats = dc.zscore(stationary)
        correlation_matrix, high_correlation_pairs = dc.correlation_diagnostics(
            stationary,
            threshold=args.correlation_threshold,
        )
        static_scores, static_loadings, static_variance = dc.fit_static_pca(
            stationary,
            n_components=n_components,
            variance_threshold=args.variance_threshold,
        )
        rolling_scores, rolling_loadings, rolling_variance, rolling_stability = dc.fit_rolling_pca(
            stationary,
            n_components=n_components,
            rolling_window=args.rolling_window,
            variance_threshold=args.variance_threshold,
        )
        static_pca_adf = dc.pca_adf_results(static_scores, "static", alpha=args.adf_alpha)
        rolling_pca_adf = dc.pca_adf_results(rolling_scores, "rolling", alpha=args.adf_alpha)
        stability_summary = dc.median_loading_stability(
            rolling_stability,
            threshold=args.loading_stability_threshold,
            n_components=n_components,
        )
        recommendation = dc.choose_recommended_pca(
            rolling_pca_adf,
            static_pca_adf,
            stability_summary,
            alpha=args.adf_alpha,
        )
        recommended_type = str(recommendation.loc[0, "recommended_pca_type"])
        hmm_input = rolling_scores if recommended_type == "rolling" else static_scores

        model_adf = pd.DataFrame(adf_rows)
        model_adf = model_adf[model_adf["information_set"] == model_name]
        adf_results = pd.concat(
            [model_adf, rolling_pca_adf, static_pca_adf],
            ignore_index=True,
            sort=False,
        )
        explained_variance = pd.concat(
            [rolling_variance, static_variance],
            ignore_index=True,
            sort=False,
        )
        labels = dc.factor_labels(n_components)

        existing_drop_columns = [column for column in drop_columns if column in panels[model_name].columns]
        decisions = []
        for column in panels[model_name].columns:
            if column in existing_drop_columns:
                status = "dropped"
                reason = "uniform_default_drop_column"
            elif column not in common_columns:
                status = "dropped"
                reason = "not_in_common_information_set_columns"
            elif column in set(kept_schema["source_variable"]):
                status = "candidate"
                reason = "kept_by_uniform_cleaning_schema"
            else:
                status = "dropped"
                reason = "dropped_by_uniform_adf_schema"
            series = panels[model_name][column]
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
                    "input_file": str(INFORMATION_SETS[model_name]["panel"]),
                    "input_rows": int(panels[model_name].shape[0]),
                    "input_variables": int(panels[model_name].shape[1]),
                    "dropped_columns": ", ".join(existing_drop_columns),
                    "uniform_information_sets": ", ".join(model_names),
                    "aligned_rows": int(aligned.shape[0]),
                    "aligned_start": aligned.index.min(),
                    "aligned_end": aligned.index.max(),
                    "stationary_rows": int(stationary.shape[0]),
                    "stationary_start": stationary.index.min(),
                    "stationary_end": stationary.index.max(),
                    "stationary_variables": int(stationary.shape[1]),
                    "exported_pca_components": int(n_components),
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

        dc.write_dataframe(adf_results, pca_dir / "adf_results.csv")
        dc.write_dataframe(decisions_out, pca_dir / "data_cleaning_decisions.csv")
        dc.write_dataframe(sample_summary, pca_dir / "data_cleaning_sample_summary.csv")
        dc.write_dataframe(stationary.reset_index(), pca_dir / "stationary_inputs.csv")
        dc.write_dataframe(normalized.reset_index(), pca_dir / "normalized_stationary_inputs.csv")
        dc.write_dataframe(normalization_stats, pca_dir / "normalization_stats.csv")
        dc.write_dataframe(
            correlation_matrix.reset_index().rename(columns={"index": "variable"}),
            pca_dir / "stationary_input_correlation_matrix.csv",
        )
        dc.write_dataframe(high_correlation_pairs, pca_dir / "stationary_input_high_correlations.csv")
        dc.write_dataframe(rolling_scores.reset_index(), pca_dir / "rolling_pca_scores.csv")
        dc.write_dataframe(static_scores.reset_index(), pca_dir / "static_pca_scores.csv")
        dc.write_dataframe(rolling_loadings, pca_dir / "rolling_pca_loadings.csv")
        dc.write_dataframe(static_loadings, pca_dir / "static_pca_loadings.csv")
        dc.write_dataframe(rolling_stability, pca_dir / "rolling_pca_loading_stability.csv")
        dc.write_dataframe(stability_summary, pca_dir / "rolling_pca_loading_stability_summary.csv")
        dc.write_dataframe(explained_variance, pca_dir / "pca_explained_variance.csv")
        dc.write_dataframe(labels, pca_dir / "pca_factor_labels.csv")
        dc.write_dataframe(recommendation, pca_dir / "pca_recommendation.csv")
        dc.write_dataframe(hmm_input.reset_index(), pca_dir / "hmm_pca_input.csv")
        dc.write_dataframe(schema, pca_dir / "uniform_cleaning_schema.csv")

        if args.write_cleaning_workbooks:
            dc.write_workbook(
                pca_dir,
                {
                    "SampleSummary": sample_summary,
                    "UniformSchema": schema,
                    "ADFResults": adf_results,
                    "StationaryInputs": stationary.reset_index(),
                    "RollingPCAScores": rolling_scores.reset_index(),
                    "StaticPCAScores": static_scores.reset_index(),
                    "PCAExplainedVariance": explained_variance,
                    "PCAFactorLabels": labels,
                    "RollingStability": stability_summary,
                    "HighCorrelations": high_correlation_pairs,
                },
            )

        pca_summary_rows.append(
            {
                "model": model_name,
                "pca_dir": str(pca_dir),
                "hmm_pca_input": str(pca_dir / "hmm_pca_input.csv"),
                "aligned_start": aligned.index.min(),
                "aligned_end": aligned.index.max(),
                "aligned_rows": int(aligned.shape[0]),
                "stationary_start": stationary.index.min(),
                "stationary_end": stationary.index.max(),
                "stationary_rows": int(stationary.shape[0]),
                "stationary_variables": int(stationary.shape[1]),
                "hmm_first_score_date": hmm_input.index.min(),
                "hmm_last_score_date": hmm_input.index.max(),
                "hmm_rows": int(hmm_input.shape[0]),
                "recommended_pca_type": recommended_type,
            }
        )

    reference_stationary = pd.read_csv(
        INFORMATION_SETS[model_names[0]]["pca_dir"] / "stationary_inputs.csv",
        parse_dates=["date"],
    )
    reference_pca = pd.read_csv(
        INFORMATION_SETS[model_names[0]]["pca_dir"] / "hmm_pca_input.csv",
        parse_dates=["date"],
    )
    for model_name in model_names:
        stationary = pd.read_csv(
            INFORMATION_SETS[model_name]["pca_dir"] / "stationary_inputs.csv",
            parse_dates=["date"],
        )
        pca_input = pd.read_csv(
            INFORMATION_SETS[model_name]["pca_dir"] / "hmm_pca_input.csv",
            parse_dates=["date"],
        )
        validation_rows.append(
            {
                "model": model_name,
                "same_stationary_columns": list(stationary.columns) == list(reference_stationary.columns),
                "same_stationary_dates": stationary["date"].equals(reference_stationary["date"]),
                "same_hmm_pca_columns": list(pca_input.columns) == list(reference_pca.columns),
                "same_hmm_pca_dates": pca_input["date"].equals(reference_pca["date"]),
                "uniform_ready": (
                    list(stationary.columns) == list(reference_stationary.columns)
                    and stationary["date"].equals(reference_stationary["date"])
                    and list(pca_input.columns) == list(reference_pca.columns)
                    and pca_input["date"].equals(reference_pca["date"])
                ),
            }
        )

    dc.write_dataframe(schema, OUTPUT_DIR / "uniform_cleaning_schema.csv")
    dc.write_dataframe(pd.DataFrame(pca_summary_rows), OUTPUT_DIR / "uniform_pca_input_summary.csv")
    dc.write_dataframe(pd.DataFrame(validation_rows), OUTPUT_DIR / "uniform_pca_input_validation.csv")
    return pca_inputs


def write_manifest(rows: list[dict[str, Any]]) -> None:
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT_DIR / "hmm_model_exports_manifest.csv", index=False)


def main() -> None:
    require_packages()
    args = parse_args()

    pca_components = parse_int_list(args.pca_components, "--pca-components")
    state_counts = parse_int_list(args.state_counts, "--state-counts")
    seeds = parse_int_list(args.random_seeds, "--random-seeds")

    if args.input:
        input_path = Path(args.input)
        output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_HMM_OUTPUT_DIR
        write_hmm_outputs(
            input_path=input_path,
            output_dir=output_dir,
            args=args,
            pca_components=pca_components,
            state_counts=state_counts,
            seeds=seeds,
            model_name="single_input",
        )
        print(f"Wrote HMM outputs to: {output_dir}")
        print(f"Compared PCA dimensions: {pca_components}")
        print(f"Compared state counts: {state_counts}")
        print(f"Covariance type: {args.covariance_type}")
        return

    information_sets = parse_information_sets(args.information_sets)
    max_components = max(max(pca_components), int(args.pca_output_components))
    manifest_rows = []
    selected_all = []
    if args.independent_cleaning:
        pca_inputs = {
            model_name: build_pca_input(model_name, args, n_components=max_components)
            for model_name in information_sets
        }
    else:
        pca_inputs = build_uniform_pca_inputs(
            information_sets,
            args,
            n_components=max_components,
        )

    for model_name in information_sets:
        config = INFORMATION_SETS[model_name]
        pca_input = pca_inputs[model_name]
        output_dir = config["hmm_dir"]
        selected_rows = write_hmm_outputs(
            input_path=pca_input,
            output_dir=output_dir,
            args=args,
            pca_components=pca_components,
            state_counts=state_counts,
            seeds=seeds,
            model_name=model_name,
        )
        selected_all.extend(selected_rows)
        manifest_rows.append(
            {
                "model": model_name,
                "panel_path": str(config["panel"]),
                "pca_dir": str(config["pca_dir"]),
                "hmm_dir": str(output_dir),
                "hmm_pca_input": str(pca_input),
            }
        )

    write_manifest(manifest_rows)
    pd.DataFrame(selected_all).to_csv(
        OUTPUT_DIR / "hmm_selected_models_all_information_sets.csv",
        index=False,
    )

    print("Wrote HMM outputs for information sets:")
    for row in manifest_rows:
        print(f"  {row['model']}: {row['hmm_dir']}")
    print(f"Compared PCA dimensions: {pca_components}")
    print(f"Compared state counts: {state_counts}")
    print(f"Covariance type: {args.covariance_type}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
