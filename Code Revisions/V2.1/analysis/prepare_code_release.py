"""Prepare an allowlisted V2.1 code release without staging or publishing it.

Dry run by default. Run with --write only after the final code checks pass.
The destination must not already exist; no source or existing release is removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DESTINATION = WORKSPACE / "Code Revisions" / "V2.1"

# These scripts describe previous editing/packaging operations, not analysis.
EXCLUDE_ANALYSIS = {
    "package_revision.py",
    "port_corrections.py",
    "revise_manuscript.py",
}
SUMMARY_FILES = {
    "summary.csv",
    "metrics.csv",
    "external_metrics.csv",
    "mapping_diagnostics.csv",
    "conditional_block_bootstrap.csv",
    "state_profiles_original_units.csv",
    "seed_sensitivity.csv",
    "reproduction_parity.csv",
    "nowcast_coverage.csv",
    "pandemic_comparison.csv",
}
SECRET = re.compile(
    rb"(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


README = """# Senior Honors Thesis: code revision V2.1

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
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-v21-runtime.txt pytest==9.2.0
```

Provide any FRED API key through `FRED_API_KEY`; never put a key in source files.
See `code/README.md` for data-provider and pipeline options. Its historical FI/RT/NC
internal identifiers correspond to the paper's REV/LAG/LAG+NC display labels.

## Checks requiring no thesis workbook

The following tests use synthetic fixtures. They test implementation properties,
not the economic validity of the model:

```powershell
.\\.venv\\Scripts\\python.exe tests/test_v21.py
.\\.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider package/tests/test_filtering.py package/tests/test_collection_corrections.py package/tests/test_availability.py
.\\.venv\\Scripts\\python.exe package/tests/test_state_metrics.py
.\\.venv\\Scripts\\python.exe package/tests/test_external_correction.py
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
.\\.venv\\Scripts\\python.exe package/run_user_panels.py --variant original --work-dir C:/path/to/new/original_run
.\\.venv\\Scripts\\python.exe package/run_user_panels.py --variant corrected --work-dir C:/path/to/new/corrected_run
.\\.venv\\Scripts\\python.exe analysis/run_revision.py
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
"""


DATA_REQUIREMENTS = """# Inputs required for historical reproduction

The public code release omits input data and provider caches. The local V2.1
paper handoff preserves them. Obtain the exact historical inputs from the author
or source providers as permitted; newly downloaded values need not match them.

| Expected relative path | Purpose |
| --- | --- |
| `package/source/thesis_data_panels.xlsx` | Historical workbook, sheets `FullInformation`, `RealTime`, `NowcastEnhanced` (141 months, January 2014-September 2025), plus source metadata. |
| `package/source/external_benchmarks.csv` | Historical USREC recession snapshot; the paper uses month-end-aligned dates. |
| `package/original_run/outputs/` | Exported historical panels, cleaning schema, PCA inputs, saved original HMM outputs, and neighboring-state diagnostics. |
| `package/corrected_run/outputs/` | Archived corrected/holdout outputs used by audit and holdout exhibit scripts. |
| `package/comparison_inputs/current_snapshot/` | Alternate snapshot required only for the separate archived snapshot diagnostic. |
| `package/availability_audit/` | Historical nowcast lineage and workbook audit outputs used by exhibit and provenance scripts. |
| `package/snapshot_diagnostics/` | Archived snapshot contrasts used by exhibit scripts. |
| `package/evidence/` and `audit/` | Historical audit outputs and report destinations used by data-dependent verification scripts. |
| `raw_source/` | Historical public-provider cache files needed for the parser audit; no reconstruction of historical availability is implied. |
| `output/raw_archive_audit/nowcast_information.csv` | Parser-only sensitivity panel generated by `analysis/audit_raw_archives.py`. |
| `output/analysis/` | Full saved fitted arrays/probabilities needed by data-dependent verification; public `results/` contains only selected summaries. |
| `manuscript/` | Local paper source and figures required only for exhibit/PDF generation, omitted from the code publication. |

Historical workbook SHA-256:
`7f817abcd6eefa8a6716ec3275eb5996a95c0eb050c85f06a73674f389b9621b`.

The main estimation sample is May 2014-September 2025 (137 observations).
`config/reproduction_schema.csv` fixes the main transformation choices.
`config/analysis_spec.json` records specification contrasts and fit settings.

Do not copy aggregate tables from `results/` into the observation-input paths.
They are descriptive artifacts and cannot substitute for the actual panels,
model parameters, or raw archives. No credentials are required to read the saved
summaries or to run synthetic tests.
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_files() -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for directory in ("code", "analysis", "tests", "config"):
        for path in sorted((ROOT / directory).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.suffix not in {".py", ".txt", ".md", ".csv", ".json"}:
                continue
            if directory == "analysis" and path.name in EXCLUDE_ANALYSIS:
                continue
            if directory == "code" and path.name == "PINNED_COMMIT.txt":
                continue  # This ancestry hash does not identify the revised code.
            files[path.relative_to(ROOT)] = path
    for variant in ("original", "corrected"):
        for path in sorted((ROOT / "package" / "code" / variant).iterdir()):
            if path.is_file() and (path.suffix == ".py" or path.name.startswith("requirements") or path.name == "PINNED_COMMIT.txt"):
                files[path.relative_to(ROOT)] = path
    for path in sorted((ROOT / "package" / "tests").glob("*.py")):
        files[path.relative_to(ROOT)] = path
    for name in ("run_user_panels.py", "run_snapshot_diagnostics.py", "requirements-audit.txt"):
        path = ROOT / "package" / name
        files[path.relative_to(ROOT)] = path
    files[Path("requirements-lock.txt")] = ROOT / "requirements-lock.txt"
    files[Path("requirements-v21-runtime.txt")] = ROOT / "requirements-v21-runtime.txt"
    for name in sorted(SUMMARY_FILES):
        path = ROOT / "output" / "analysis" / name
        if path.is_file():
            files[Path("results/archived") / name] = path
    # V2.1-specific numerical reports: only explicitly aggregate tabular outputs.
    for dirname in ("v21_diagnostics", "v2_1_diagnostics"):
        path = ROOT / "output" / dirname
        if path.is_dir():
            for source in sorted(path.glob("*.csv")):
                files[Path("results/v21") / source.name] = source
    for filename in ("FINAL_REVIEW_V2_1.txt", "FINAL_CHECK_V2_1.txt"):
        path = ROOT / filename
        if path.is_file():
            files[Path("verification") / filename] = path
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create the new allowlisted release directory.")
    args = parser.parse_args()
    files = selected_files()
    if not files or any(not source.is_file() for source in files.values()):
        raise SystemExit("Incomplete source selection; no release written.")
    for relative, source in files.items():
        if SECRET.search(source.read_bytes()):
            raise SystemExit(f"Possible credential in {relative}; no release written.")
    report = {"destination": str(DESTINATION), "source_files": len(files),
              "generated_docs": ["README.md", "DATA_REQUIREMENTS.md"],
              "files": [p.as_posix() for p in sorted(files)], "written": args.write}
    if not args.write:
        print(json.dumps(report, indent=2))
        return
    if DESTINATION.exists():
        raise SystemExit(f"Refusing existing destination: {DESTINATION}")
    if DESTINATION.resolve().parent.parent != WORKSPACE.resolve():
        raise SystemExit("Destination escapes the intended workspace.")
    DESTINATION.mkdir(parents=True)
    for relative, source in files.items():
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (DESTINATION / "README.md").write_text(README, encoding="utf-8")
    (DESTINATION / "DATA_REQUIREMENTS.md").write_text(DATA_REQUIREMENTS, encoding="utf-8")
    hashes = {p.relative_to(DESTINATION).as_posix(): sha256(p)
              for p in sorted(DESTINATION.rglob("*")) if p.is_file()}
    manifest = {"version": "V2.1", "date": "2026-11-11", "scope": "Code and aggregate numerical summaries; input data omitted.",
                "publication_status_at_packaging": "Prepared locally; packaging does not publish.", "files": hashes}
    (DESTINATION / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    for name, digest in hashes.items():
        if sha256(DESTINATION / name) != digest:
            raise SystemExit(f"Release integrity mismatch: {name}")
    report.update(manifest_files=len(hashes), hashes_verified=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

