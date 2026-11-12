"""Validate archived holdout scores and build its manuscript table, without refitting.

Uses only the Python standard library. All input files are read from the saved
corrected run; the sole output is manuscript/generated/holdout.tex.
"""

from pathlib import Path
import calendar
import csv
import math
import statistics


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "package/corrected_run/outputs"
HOLDOUT = ARCHIVE / "hmm_oos_validation"
REGIMES = (
    "Expansion / Goldilocks",
    "Overheating / Late Cycle",
    "Contraction / Recession",
    "Recovery / Early Cycle",
)
DATES = ["2024-10-31", "2024-11-30", "2024-12-31"] + [
    f"2025-{month:02d}-{calendar.monthrange(2025, month)[1]}"
    for month in range(1, 10)
]
MODELS = ("full_information", "realtime_information", "nowcast_enhanced")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, context):
    require(math.isclose(actual, float(expected), rel_tol=1e-10, abs_tol=1e-11),
            f"Archived holdout mismatch ({context}): {actual} versus {expected}")


def state_map(rows):
    require(len(rows) == 4, "Exactly four original state names are required.")
    mapping = {int(row["state"]): REGIMES.index(row["regime_label"]) for row in rows}
    require(set(mapping) == set(range(4)) and set(mapping.values()) == set(range(4)),
            "Baseline state names must give a bijective four-state map.")
    return mapping


def probability_vector(row, mapping):
    raw = [float(row[f"state_{state}_prob"]) for state in range(4)]
    require(all(math.isfinite(value) and 0 <= value <= 1 + 1e-12 for value in raw),
            f"Invalid archived probability at {row['date']}.")
    close(sum(raw), 1, "probability sum")
    vector = [0.0] * 4
    for state, regime in mapping.items():
        vector[regime] = raw[state]
    # The archived evaluator normalizes mapped state probabilities.
    total = sum(vector)
    return [value / total for value in vector]


def main():
    runs = read_csv(HOLDOUT / "oos_run_summary.csv")
    require(len(runs) == 1, "Expected one archived holdout run.")
    expected = dict(train_start="2014-05-31", train_end="2024-09-30", train_rows="125",
                    holdout_start=DATES[0], holdout_end=DATES[-1], holdout_rows="12",
                    holdout_months="12", pca_components="3,5", state_count="4",
                    covariance_type="diag")
    for field, value in expected.items():
        require(runs[0][field] == value, f"Unexpected archived holdout {field}.")
    require(read_csv(ROOT / "config/reproduction_schema.csv") ==
            read_csv(ARCHIVE / "uniform_cleaning_schema.csv"),
            "Archived holdout schema differs from the baseline reproduction schema.")
    external = read_csv(ROOT / "package/state_audit/external_benchmarks_month_end.csv")
    observed = [row for row in external if row["date"] in DATES]
    require(sorted(row["date"] for row in observed) == DATES and
            all(int(row["external_recession"]) == 0 for row in observed),
            "The table's no-recession observation note is inconsistent with the archive.")
    reference_dir = ARCHIVE / "hmm_models_full_information"
    selected = read_csv(reference_dir / "hmm_selected_models.csv")
    require({row["pca_components"] for row in selected} == {"3", "5"} and
            all(row["probability_method"] == "filtered" and row["n_observations"] == "137"
                for row in selected),
            "The archived reference must be filtered and fitted on all 137 months.")

    table_rows = []
    for pcs in (3, 5):
        reference_map = state_map(read_csv(reference_dir / f"hmm_regime_labels_pca{pcs}.csv"))
        reference = [row for row in read_csv(reference_dir / f"hmm_state_probabilities_pca{pcs}.csv")
                     if row["date"] in DATES]
        require([row["date"] for row in reference] == DATES,
                f"Missing or duplicate {pcs}-PC reference dates.")
        targets = {row["date"]: probability_vector(row, reference_map) for row in reference}
        candidates = read_csv(HOLDOUT / f"oos_filtered_probabilities_pca{pcs}.csv")
        require(len(candidates) == 36 and {row["model"] for row in candidates} == set(MODELS),
                f"Expected all three twelve-month candidate paths for {pcs} PCs.")
        names = read_csv(HOLDOUT / f"oos_training_regime_labels_pca{pcs}.csv")
        summaries = read_csv(HOLDOUT / f"oos_summary_metrics_pca{pcs}.csv")
        require(len(summaries) == 3 and {row["model"] for row in summaries} == set(MODELS),
                "Missing or duplicate archived summary rows.")
        scores = {}
        for model in MODELS:
            mapping = state_map([row for row in names if row["model"] == model])
            rows = [row for row in candidates if row["model"] == model]
            require([row["date"] for row in rows] == DATES,
                    f"Missing or duplicate {pcs}-PC {model} holdout dates.")
            for row in rows:
                require(all(row[field] == value for field, value in expected.items()
                            if field in ("train_start", "train_end", "holdout_start", "holdout_end")),
                        "Candidate training/holdout metadata differs from the archived run.")
            losses = [sum(abs(p - q) for p, q in zip(probability_vector(row, mapping), targets[row["date"]]))
                      for row in rows]
            scores[model] = statistics.mean(losses)
            summary = next(row for row in summaries if row["model"] == model)
            require(summary["holdout_months"] == "12" and summary["pca_components"] == str(pcs),
                    "Summary dimensions differ from the raw probability paths.")
            close(scores[model], summary["mean_information_gap_l1_vs_benchmark"],
                  f"{pcs}-PC {model} raw-probability mean L1")
        rt, nc = scores["realtime_information"], scores["nowcast_enhanced"]
        require(rt > 0, "LAG mean L1 must be positive for a percentage reduction.")
        reduction = 100 * (rt - nc) / rt
        improvement = read_csv(HOLDOUT / f"oos_nowcast_vs_realtime_improvement_pca{pcs}.csv")
        require(len(improvement) == 1, "Expected one archived improvement row per PCA dimension.")
        close(reduction, improvement[0]["l1_gap_percent_reduction"], f"{pcs}-PC reduction")
        table_rows.append(f"{pcs} & {rt:.3f} & {nc:.3f} & {reduction:.2f}" + r" \\")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\caption{Archived twelve-month holdout: internal agreement}",
        r"\label{tab:holdout}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"PCs & LAG $L_1$ & LAG+NC $L_1$ & Reduction (\%) \\",
        r"\midrule",
        *table_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip\begin{minipage}{\textwidth}\footnotesize "
        r"Training: 125 months, May 2014--September 2024; holdout: twelve months, "
        r"October 2024--September 2025. Candidate standardizers, PCA, and HMMs use "
        r"training rows only, followed by recursive filtering. The REV reference is "
        r"filtered with full-sample fitted parameters; the transformation schema was "
        r"chosen using the full sample. Baseline state names are retained. No holdout "
        r"observations are recession months. Scores measure internal agreement with "
        r"the ex post REV reference.\end{minipage}",
        r"\end{table}",
    ]
    destination = ROOT / "manuscript/generated/holdout.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Validated archived 3-PC and 5-PC holdout paths; wrote manuscript/generated/holdout.tex.")


if __name__ == "__main__":
    main()
