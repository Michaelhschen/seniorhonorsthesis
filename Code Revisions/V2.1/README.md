# Senior Honors Thesis: code revision V2.1

Code accompanying *Public Nowcasts and Macroeconomic Regime Classification:
A Retrospective Assessment*, Michael H. Chen, 11 November 2026.

This is the revised code release. The original public pipeline remains separately
at the repository root and in its preserved Git history. `code/` is the current production
code; `package/code/original/` and `package/code/corrected/` are explicitly
historical comparison fixtures required by the regression tests.
The original fixture records source ancestry at public commit
`6a74d2a810d1d0345f4ccdbd8864c715615084c6`; that hash identifies the historical
fixture, not this revised release. The local checkout also has a separate initial
commit, so publishing must inspect and preserve the current remote history.

## Scope and reproducibility

This public release contains source code, configuration, tests, and selected
aggregate result tables. It excludes the manuscript, PDFs, raw provider caches,
input workbooks, generated observation panels, credentials, and local environments.
It therefore does **not** reproduce the historical estimates from a clean clone
without the separately supplied inputs listed in `DATA_REQUIREMENTS.md`.
Fresh provider downloads can differ from the historical panels because of data
revisions, archive coverage, and transformation choices.

`results/archived/` holds aggregate outputs copied from the completed retrospective
analysis. `results/v21/`, when present, holds the focused V2.1 diagnostics. A saved
result is not evidence that its model was refitted during this release. The final
verification note distinguishes new checks from archived estimates.

## Environment

Use a fresh environment. `requirements-v21-runtime.txt` records the actual Python 3.14.6
V2.1 diagnostic environment; `requirements-lock.txt` is the historical dependency
record. The minimal pipeline dependencies are
`code/requirements.txt` and `code/requirements-analysis.txt`. A dependency record
does not guarantee wheels remain available on every platform or Python version.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-v21-runtime.txt pytest==9.2.0
```

Provide any FRED API key through `FRED_API_KEY`; never put a key in source files.
See `code/README.md` for data-provider and pipeline options. Its historical FI/RT/NC
internal identifiers correspond to the paper's REV/LAG/LAG+NC display labels.

## Checks requiring no thesis workbook

The following tests use synthetic fixtures. They test implementation properties,
not the economic validity of the model:

```powershell
.\.venv\Scripts\python.exe tests/test_v21.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider package/tests/test_filtering.py package/tests/test_collection_corrections.py package/tests/test_availability.py
.\.venv\Scripts\python.exe package/tests/test_state_metrics.py
.\.venv\Scripts\python.exe package/tests/test_external_correction.py
```

The availability suite explicitly skips one integration case when the historical
provider cache is absent. `tests/test_revision.py` also contains data-dependent
checks of fitted arrays, so running every test without the private handoff inputs
is not supported. The six V2.1 unit tests check current production code;
the verification note records the executed suite. Historical fixture tests
intentionally characterize earlier behavior and are distinct from tests of `code/`.

## Historical panel reproduction

After restoring the paths in `DATA_REQUIREMENTS.md`, the archived comparison
runner can be used without downloading replacement panels:

```powershell
.\.venv\Scripts\python.exe package/run_user_panels.py --variant original --work-dir C:/path/to/new/original_run
.\.venv\Scripts\python.exe package/run_user_panels.py --variant corrected --work-dir C:/path/to/new/corrected_run
.\.venv\Scripts\python.exe analysis/run_revision.py
```

The first two commands reproduce historical code fixtures; they are not a route
to execute the latest production code. `analysis/run_revision.py` imports current
`code/` and runs the retrospective specification comparisons from restored panels.
It expects the archived baseline output paths and raw-parser sensitivity panel.
The full analysis remains retrospective: revised values, approximate publication
lags, full-sample preprocessing and estimation, and descriptive state matching.
Filtering conditions on fitted quantities and does not remove those limitations.

Asset-generation scripts are included to disclose how numerical exhibits were
constructed, but the manuscript tree and figure assets are not publicly included.
They require the local paper handoff and restored empirical inputs.

## Integrity and publication

`RELEASE_MANIFEST.json` records SHA-256 hashes and paths of every released file
except the manifest itself. `analysis/prepare_code_release.py` created this
allowlisted directory and did not stage, commit, or push it. A directory on disk
does not establish GitHub publication. Use the verified commit URL in the final
handoff once the remote publication has succeeded.
