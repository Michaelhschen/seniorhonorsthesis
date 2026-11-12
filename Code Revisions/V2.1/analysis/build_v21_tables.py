"""Generate V2.1 focused diagnostic tables from their saved numerical outputs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "output/v21_diagnostics"
T = ROOT / "manuscript/generated"
SHORT = {"full_information": "REV", "realtime_information": "LAG", "nowcast_enhanced": "LAG+NC"}


def table(name, caption, headers, rows, note, fmt):
    lines = [r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\setlength{\tabcolsep}{3.5pt}", r"\renewcommand{\arraystretch}{1.12}",
             r"\caption{" + caption + "}", r"\label{tab:" + name + "}",
             r"\begin{tabular}{" + fmt + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\begin{minipage}{\textwidth}\footnotesize " + note + r"\end{minipage}",
              r"\end{table}"]
    (T / (name + ".tex")).write_text("\n".join(lines), encoding="utf-8")


def main():
    T.mkdir(parents=True, exist_ok=True)
    fits = pd.read_csv(A / "state_count_selected.csv")
    comp = pd.read_csv(A / "state_count_comparison.csv").set_index("states")
    rows = []
    for k, group in fits.groupby("states", sort=True):
        by_model = group.set_index("model")
        rows.append([k, int(group.n_parameters.iloc[0]),
                     *[f"{by_model.loc[n, 'BIC']:.1f}" for n in SHORT],
                     f"{comp.loc[k, 'LAG_L1']:.3f}", f"{comp.loc[k, 'NC_L1']:.3f}",
                     f"{comp.loc[k, 'reduction_percent']:.2f}"])
    table("state_counts", "State-count sensitivity in the fixed five-PC observation space",
          ["States", "Params.", "REV BIC", "LAG BIC", "NC BIC", "LAG $L_1$", "NC $L_1$", r"Red. (\%)"],
          rows, "NC denotes LAG+NC. All fits use 137 months, diagonal covariance, and highest likelihood "
          "among seeds 0--4. BIC is comparable across state counts within a panel's unchanged five-PC space; "
          "it is not comparable across panels or PCA dimensions. Filtered losses use smoothed economic-profile "
          "matching for each state count. The reference and state meanings change across rows, so losses "
          "are not a common accuracy scale. Lower BIC is preferred; positive reductions indicate less "
          "nowcast-panel disagreement. Small state masses make regular BIC an approximate diagnostic.", "rrrrrrrr")

    seed = pd.read_csv(A / "controlled_initialization.csv").query("reference_protocol=='same_seed_reference'")
    rows = [[int(r.seed), f"{r.REV_likelihood_gap:.3f}", f"{r.LAG_likelihood_gap:.3f}",
             f"{r.NC_likelihood_gap:.3f}", f"{r.LAG_L1:.3f}", f"{r.NC_L1:.3f}",
             f"{r.reduction_percent:.2f}"] for r in seed.itertuples()]
    table("controlled_seeds", "Initialization with one state-correspondence procedure",
          ["Seed", r"$\Delta\ell_{REV}$", r"$\Delta\ell_{LAG}$", r"$\Delta\ell_{NC}$", "LAG $L_1$", "NC $L_1$", r"Red. (\%)"],
          rows, "NC denotes LAG+NC. Each row uses the same seed for all three four-state models. "
          "The correspondence procedure always uses smoothed economic profiles with the same anchors and "
          "weights; losses always use filtered probabilities. Each likelihood gap is the selected fit's "
          "log likelihood minus the row fit's log likelihood for that panel. Every row includes a materially "
          "lower-likelihood fit. Thus the sign reversals do not by themselves establish ambiguity among "
          "near-equivalent optima. Independently selected fits give a 58.79 percent profile reduction.", "rrrrrrr")

    rows = [[SHORT[r.model], int(r.states), int(r.n_parameters), int(r.seed),
             f"{r.log_likelihood:.3f}", f"{r.BIC:.1f}", f"{r.minimum_posterior_mass:.2f}"]
            for r in fits.itertuples()]
    table("state_count_details", "Selected state-count fits and the smallest effective state mass",
          ["Panel", "States", "Params.", "Seed", "Log likelihood", "BIC", "Min. mass"], rows,
          "Each panel uses its fixed five-PC input matrix over May 2014--September 2025. "
          "Minimum mass is the smallest smoothed posterior sum across states, measured in expected months. "
          "It is not a count of independent observations. These state-count estimates do not establish "
          "a recurring economic taxonomy. The parameter count excludes scaling and PCA estimation.", "lrrrrrr")

    isolated = pd.read_csv(A / "one_pipeline_initialization.csv")
    rows = [[SHORT[r.varied_model], int(r.seed), f"{r.varied_fit_likelihood_gap:.3f}",
             f"{r.LAG_L1:.3f}", f"{r.NC_L1:.3f}", f"{r.reduction_percent:.2f}"]
            for r in isolated.itertuples()]
    table("initialization_one_at_time", "Varying one fitted pipeline while holding the other two selected",
          ["Varied panel", "Seed", r"$\Delta\ell$", "LAG $L_1$", "NC $L_1$", r"Red. (\%)"], rows,
          "NC denotes LAG+NC. Correspondence always uses smoothed economic profiles; loss uses filtering. "
          "The selected REV/LAG/NC seeds are 2/0/4. With REV and LAG fixed, NC seed 2 is 0.563 log-likelihood "
          "units below the selected solution and gives a 48.72 percent reduction; NC seed 3 is only "
          r"$8.0\times10^{-6}$ below it and gives 58.79 percent. The negative NC-seed-1 result has a "
          "42.051-unit likelihood deficit. A small likelihood deficit is descriptive evidence of a "
          "close fit, not a formal equivalence test.", "lrrrrr")


if __name__ == "__main__":
    main()
