# Senior Honours Thesis Code

Code for Michael H. Chen's *Real-Time Macroeconomic Regime Detection:
Assessing the Role of GDP and Inflation Nowcasting* (working paper V2.1, 11 November 2026).

**Paper has now been updated to V2.2; no material code has changed for data, only analysis."

The revised source, configuration, tests, and aggregate diagnostic results are in
[`Code Revisions/V2.1/`](Code%20Revisions/V2.1/README.md).
Earlier pipeline files remain available in their existing locations and history.

V2.1 adds controlled initialization and state-count comparisons on the same
five-component observation space. It also repairs validation in the optional
holdout and generic-state output paths. The four-state baseline is numerically
recovered, but simpler models reverse the profile-matched comparison. The paper's
claim concerns retrospective model agreement and specification sensitivity;
historical real-time accuracy and portfolio performance remain unestablished.

The V2.1 release excludes manuscripts, PDFs, raw provider caches, input workbooks,
observation panels, credentials, and local environments. It includes selected
aggregate result tables. Its README explains executed checks, historical versus
new results, environment records, and exact-input requirements. Downloading the
code alone does not reproduce the historical estimates without those inputs.

Provide FRED credentials only through `FRED_API_KEY`. Never commit API keys.
