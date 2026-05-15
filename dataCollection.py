"""
Step 1 thesis data pipeline: build the three information-set panels.

Outputs:
    outputs/full_information.csv
    outputs/realtime_information.csv
    outputs/nowcast_information.csv
    outputs/thesis_data_panels.xlsx
    outputs/source_metadata.csv
    outputs/diagnostics.csv

Later steps should live in separate scripts:
    dataCleaning.py      -> ADF and missing-data diagnostics
    hmmModeling.py    -> HMM regime classification after PCA is settled
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_downloads"
OUTPUT_DIR = BASE_DIR / "outputs"

DEFAULT_START = "2014-01-31"
GDP_NOW_WORKBOOK = RAW_DIR / "gdpnow" / "GDPTrackingModelDataAndForecasts.xlsx"
CLEVELAND_MANUAL_FILE = RAW_DIR / "cleveland_inflation_nowcast_history.csv"


LOCAL_SURVEY_FILES = {
    "ISM_NewOrders": {
        "path": BASE_DIR / "ISM Manufacturing New Orders.xlsx",
        "sheet": "New_Orders",
        "value_column": "Manufacturing New Orders",
        "rt_lag_months": 1,
    },
    "ISM_PMI": {
        "path": BASE_DIR / "ISM Manufacturing PMI.xlsx",
        "sheet": "PMI",
        "value_column": "PMI",
        "rt_lag_months": 1,
    },
    "PhillyFed_BOS": {
        "path": BASE_DIR / "Philly Fed BOS.xlsx",
        "sheet": "BOS",
        "value_column": "BOS",
        "rt_lag_months": 0,
    },
}


GDP_NOW_URLS = [
    "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx",
    "https://www.frbatlanta.org/-/media/Documents/cqer/researchcq/gdpnow/GDPTrackingModelDataAndForecasts.xlsx",
]

CLEVELAND_NOWCAST_URLS = {
    # Public chart-data files loaded by the Cleveland Fed nowcasting page;
    # these are not a documented API.
    "mom": "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json",
    "qoq_saar": "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_quarter.json",
    "yoy": "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_year.json",
}

CLEVELAND_MEASURE_MAP = {
    "CPI Inflation": "CPI",
    "Core CPI Inflation": "Core CPI",
    "PCE Inflation": "PCE",
    "Core PCE Inflation": "Core PCE",
    "Actual CPI Inflation": "CPI",
    "Actual Core CPI Inflation": "Core CPI",
    "Actual PCE Inflation": "PCE",
    "Actual Core PCE Inflation": "Core PCE",
}


@dataclass(frozen=True)
class FredSpec:
    name: str
    series_id: str
    frequency: str
    transform: str
    source: str
    revised: bool
    fallback_lag_months: int
    monthly_aggregation: str = "last"


FRED_SPECS = [
    FredSpec("GDP_QoQAnn", "GDPC1", "quarterly", "qoq_ann", "FRED/ALFRED", True, 3),
    FredSpec(
        "CorpCapex_QoQ",
        "BOGZ1FA105050005Q",
        "quarterly",
        "qoq_pct",
        "FRED/ALFRED Z.1",
        True,
        3,
    ),
    FredSpec(
        "CorpFixedCap_QoQ",
        "BOGZ1FL105015103Q",
        "quarterly",
        "qoq_pct",
        "FRED/ALFRED Z.1",
        True,
        3,
    ),
    FredSpec("Delinq", "DRALACBN", "quarterly", "level", "FRED/ALFRED", True, 3),
    FredSpec("DelinqChange", "DRALACBN", "quarterly", "diff", "FRED/ALFRED", True, 3),
    FredSpec("CPI_YoY", "CPIAUCSL", "monthly", "yoy_pct", "FRED/ALFRED", True, 1),
    FredSpec("CorePCE_YoY", "PCEPILFE", "monthly", "yoy_pct", "FRED/ALFRED", True, 1),
    FredSpec("UNRATE", "UNRATE", "monthly", "level", "FRED/ALFRED", True, 1),
    FredSpec("U6UNRATE", "U6RATE", "monthly", "level", "FRED/ALFRED", True, 1),
    FredSpec(
        "ICSA_YoY",
        "ICSA",
        "weekly",
        "yoy_pct",
        "FRED/ALFRED",
        True,
        1,
        monthly_aggregation="mean",
    ),
    FredSpec("HOUST_YoY", "HOUST", "monthly", "yoy_pct", "FRED/ALFRED", True, 1),
    FredSpec(
        "DXY_YoY",
        "DTWEXBGS",
        "daily",
        "yoy_pct",
        "FRED daily market data",
        False,
        0,
        monthly_aggregation="last",
    ),
    FredSpec(
        "TermSpread_10Y3M",
        "T10Y3MM",
        "monthly",
        "level",
        "FRED interest-rate spread",
        False,
        0,
        monthly_aggregation="last",
    ),
    FredSpec(
        "HYSpread",
        "BAMLH0A0HYM2",
        "daily",
        "level",
        "FRED daily market data",
        False,
        0,
        monthly_aggregation="last",
    ),
    FredSpec(
        "VIX_log",
        "VIXCLS",
        "daily",
        "log",
        "FRED daily market data",
        False,
        0,
        monthly_aggregation="last",
    ),
]


COMPOSITE_SERIES = {
    "M1M2_Ratio": {
        "series_ids": ["M1SL", "M2SL"],
        "frequency": "monthly",
        "source": "FRED/ALFRED",
        "revised": True,
        "fallback_lag_months": 1,
    }
}


def require_packages() -> None:
    missing = []
    for package in ["pandas", "openpyxl"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        install = "python -m pip install -r \"Data Processor\\requirements.txt\""
        raise SystemExit(
            "Missing required package(s): "
            + ", ".join(missing)
            + "\nInstall them from the project root with:\n    "
            + install
        )


def ensure_dirs() -> None:
    for path in [
        RAW_DIR,
        RAW_DIR / "fred_latest",
        RAW_DIR / "fred_vintages",
        RAW_DIR / "gdpnow",
        RAW_DIR / "cleveland_nowcast",
        OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build thesis macro information-set panels.")
    parser.add_argument("--start", default=DEFAULT_START, help="Month-end start date.")
    parser.add_argument(
        "--end",
        default="auto",
        help="Month-end end date, or 'auto' to use the latest local survey date.",
    )
    parser.add_argument(
        "--real-time-mode",
        choices=["auto", "vintage", "lag"],
        default="auto",
        help="Use ALFRED/FRED vintages if possible, or deterministic publication-lag fallback.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh cached downloads instead of reusing raw_downloads files.",
    )
    parser.add_argument(
        "--skip-nowcasts",
        action="store_true",
        help="Build FI and RT panels, then copy RT to RT+NC without nowcast replacement.",
    )
    parser.add_argument(
        "--fred-api-key",
        default="",
        help="Optional FRED API key for this run. Prefer FRED_API_KEY env var for normal use.",
    )
    return parser.parse_args()


def month_end_index(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="ME")


def excel_serial_to_timestamp(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, datetime):
        return pd.Timestamp(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(value), unit="D")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Could not parse date value: {value!r}")
    return parsed


def to_month_end(index: Iterable[object]) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(list(index), errors="coerce")
    return pd.DatetimeIndex(parsed).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)


def read_local_survey_series(name: str, info: dict) -> pd.Series:
    path = info["path"]
    if not path.exists():
        raise FileNotFoundError(f"Missing local survey file for {name}: {path}")

    df = pd.read_excel(path, sheet_name=info["sheet"])
    if "Date" not in df.columns or info["value_column"] not in df.columns:
        raise ValueError(f"{path.name} does not have expected Date/value columns.")

    dates = df["Date"].map(excel_serial_to_timestamp)
    values = pd.to_numeric(df[info["value_column"]], errors="coerce")
    out = pd.Series(values.values, index=to_month_end(dates), name=name)
    out = out[~out.index.isna()].sort_index()
    return out[~out.index.duplicated(keep="last")]


def latest_local_survey_end() -> pd.Timestamp:
    ends = []
    for name, info in LOCAL_SURVEY_FILES.items():
        series = read_local_survey_series(name, info)
        ends.append(series.dropna().index.max())
    return min(ends)


def urlretrieve_cached(url: str, destination: Path, refresh: bool = False) -> Path:
    if destination.exists() and not refresh:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 thesis-data-pipeline/1.0",
        },
    )
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            return destination
        except Exception as exc:
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}") from last_error
    return destination


def read_fred_latest(series_id: str, refresh: bool = False) -> pd.Series:
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    json_destination = RAW_DIR / "fred_latest" / f"{series_id}.json"
    if api_key or (json_destination.exists() and not refresh):
        destination = json_destination
        if not destination.exists() or refresh:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": "1947-01-01",
            }
            url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
            urlretrieve_cached(url, destination, refresh=True)
        data = json.loads(destination.read_text(encoding="utf-8"))
        if "observations" not in data:
            raise ValueError(f"Unexpected FRED API response for {series_id}: {data}")
        dates = pd.to_datetime([item["date"] for item in data["observations"]])
        values = pd.to_numeric([item["value"] for item in data["observations"]], errors="coerce")
        return pd.Series(values, index=dates, name=series_id).sort_index()

    destination = RAW_DIR / "fred_latest" / f"{series_id}.csv"
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    urlretrieve_cached(url, destination, refresh=refresh)
    df = pd.read_csv(destination)
    if "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError(f"Unexpected FRED CSV shape for {series_id}: {list(df.columns)}")
    values = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
    return pd.Series(values.values, index=pd.to_datetime(df["DATE"]), name=series_id).sort_index()


def read_fred_vintage(
    series_id: str,
    vintage_date: pd.Timestamp,
    observation_start: str,
    observation_end: str,
    api_key: str,
    refresh: bool = False,
) -> pd.Series:
    stamp = vintage_date.strftime("%Y-%m-%d")
    destination = RAW_DIR / "fred_vintages" / f"{series_id}_{stamp}.json"
    if not destination.exists() or refresh:
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "realtime_start": stamp,
            "realtime_end": stamp,
            "observation_start": observation_start,
            "observation_end": observation_end,
        }
        url = "https://api.stlouisfed.org/fred/series/observations?" + urllib.parse.urlencode(params)
        urlretrieve_cached(url, destination, refresh=True)

    data = json.loads(destination.read_text(encoding="utf-8"))
    if "observations" not in data:
        raise ValueError(f"Unexpected FRED API response for {series_id} as of {stamp}: {data}")
    obs = data["observations"]
    dates = pd.to_datetime([item["date"] for item in obs])
    values = pd.to_numeric([item["value"] for item in obs], errors="coerce")
    return pd.Series(values, index=dates, name=series_id).sort_index()


def aggregate_to_monthly(raw: pd.Series, aggregation: str) -> pd.Series:
    raw = raw.dropna().sort_index()
    if raw.empty:
        return raw
    if aggregation == "mean":
        return raw.resample("ME").mean()
    if aggregation == "last":
        out = raw.copy()
        out.index = to_month_end(out.index)
        return out.groupby(level=0).last()
    raise ValueError(f"Unknown monthly aggregation: {aggregation}")


def expand_quarterly_to_monthly(q_series: pd.Series) -> pd.Series:
    values = {}
    q_series = q_series.dropna().sort_index()
    for date, value in q_series.items():
        quarter = pd.Timestamp(date).to_period("Q")
        months = pd.period_range(
            quarter.asfreq("M", "start"),
            quarter.asfreq("M", "end"),
            freq="M",
        ).to_timestamp() + pd.offsets.MonthEnd(0)
        for month in months:
            values[month] = value
    return pd.Series(values).sort_index()


def apply_transform(raw: pd.Series, spec: FredSpec) -> pd.Series:
    raw = raw.dropna().sort_index()
    if spec.frequency == "quarterly":
        base = raw.copy()
        base.index = pd.to_datetime(base.index)
        if spec.transform == "qoq_ann":
            transformed = ((base / base.shift(1)) ** 4 - 1.0) * 100.0
        elif spec.transform == "qoq_pct":
            transformed = base.pct_change() * 100.0
        elif spec.transform == "diff":
            transformed = base.diff()
        elif spec.transform == "level":
            transformed = base
        else:
            raise ValueError(f"Unsupported quarterly transform: {spec.transform}")
        transformed.name = spec.name
        return expand_quarterly_to_monthly(transformed)

    monthly = aggregate_to_monthly(raw, spec.monthly_aggregation)
    if spec.transform == "mom_ann":
        transformed = ((monthly / monthly.shift(1)) ** 12 - 1.0) * 100.0
    elif spec.transform == "mom_pct":
        transformed = monthly.pct_change() * 100.0
    elif spec.transform == "yoy_pct":
        transformed = monthly.pct_change(periods=12) * 100.0
    elif spec.transform == "log":
        transformed = np.log(monthly)
    elif spec.transform == "diff":
        transformed = monthly.diff()
    elif spec.transform == "level":
        transformed = monthly
    else:
        raise ValueError(f"Unsupported transform: {spec.transform}")

    transformed.name = spec.name
    return transformed.sort_index()


def build_latest_fred_panel(months: pd.DatetimeIndex, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_cache = {}
    metadata = []

    panel = pd.DataFrame(index=months)
    for spec in FRED_SPECS:
        if spec.series_id not in raw_cache:
            raw_cache[spec.series_id] = read_fred_latest(spec.series_id, refresh=refresh)
        transformed = apply_transform(raw_cache[spec.series_id], spec)
        panel[spec.name] = transformed.reindex(months)
        metadata.append(
            {
                "variable": spec.name,
                "source_id": spec.series_id,
                "source": spec.source,
                "transform": spec.transform,
                "frequency": spec.frequency,
                "monthly_aggregation": spec.monthly_aggregation,
                "revised": spec.revised,
                "fallback_lag_months": spec.fallback_lag_months,
            }
        )

    m1 = aggregate_to_monthly(read_fred_latest("M1SL", refresh=refresh), "last")
    m2 = aggregate_to_monthly(read_fred_latest("M2SL", refresh=refresh), "last")
    ratio = (m1 / m2).rename("M1M2_Ratio")
    panel["M1M2_Ratio"] = ratio.reindex(months)
    metadata.append(
        {
            "variable": "M1M2_Ratio",
            "source_id": "M1SL/M2SL",
            "source": "FRED/ALFRED",
            "transform": "ratio",
            "frequency": "monthly",
            "monthly_aggregation": "last",
            "revised": True,
            "fallback_lag_months": 1,
        }
    )

    return panel, pd.DataFrame(metadata)


def add_local_survey_data(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = []
    out = panel.copy()
    for name, info in LOCAL_SURVEY_FILES.items():
        series = read_local_survey_series(name, info)
        out[name] = series.reindex(out.index)
        metadata.append(
            {
                "variable": name,
                "source_id": info["path"].name,
                "source": "Local Bloomberg export",
                "transform": "level",
                "frequency": "monthly",
                "monthly_aggregation": "month_end",
                "revised": False,
                "fallback_lag_months": info["rt_lag_months"],
            }
        )
    return out, pd.DataFrame(metadata)


def build_lag_fallback_rt(full_panel: pd.DataFrame) -> pd.DataFrame:
    rt = full_panel.copy()
    lags = {spec.name: spec.fallback_lag_months for spec in FRED_SPECS}
    lags["M1M2_Ratio"] = COMPOSITE_SERIES["M1M2_Ratio"]["fallback_lag_months"]
    for name, info in LOCAL_SURVEY_FILES.items():
        lags[name] = info["rt_lag_months"]

    for column, lag in lags.items():
        if column in rt.columns and lag:
            rt[column] = rt[column].shift(lag)
    return rt


def build_vintage_rt_panel(
    months: pd.DatetimeIndex,
    full_panel: pd.DataFrame,
    api_key: str,
    refresh: bool,
) -> pd.DataFrame:
    rt = pd.DataFrame(index=months, columns=full_panel.columns, dtype=float)
    observation_start = "1947-01-01"
    observation_end = (months.max() + pd.offsets.MonthEnd(3)).strftime("%Y-%m-%d")

    non_revised_columns = [spec.name for spec in FRED_SPECS if not spec.revised]
    rt[non_revised_columns] = full_panel[non_revised_columns]
    if "TermSpread_10Y3M" in full_panel.columns:
        rt["TermSpread_10Y3M"] = full_panel["TermSpread_10Y3M"]

    for name, info in LOCAL_SURVEY_FILES.items():
        if info["rt_lag_months"]:
            rt[name] = full_panel[name].shift(info["rt_lag_months"])
        else:
            rt[name] = full_panel[name]

    for model_date in months:
        for spec in [item for item in FRED_SPECS if item.revised]:
            raw = read_fred_vintage(
                spec.series_id,
                model_date,
                observation_start,
                observation_end,
                api_key,
                refresh=refresh,
            )
            transformed = apply_transform(raw, spec)
            rt.loc[model_date, spec.name] = transformed.reindex(months).ffill().loc[model_date]

        m1 = aggregate_to_monthly(
            read_fred_vintage("M1SL", model_date, observation_start, observation_end, api_key, refresh=refresh),
            "last",
        )
        m2 = aggregate_to_monthly(
            read_fred_vintage("M2SL", model_date, observation_start, observation_end, api_key, refresh=refresh),
            "last",
        )
        ratio = (m1 / m2).reindex(months).ffill()
        rt.loc[model_date, "M1M2_Ratio"] = ratio.loc[model_date]

    return rt


def download_gdpnow_workbook(refresh: bool) -> Path | None:
    if GDP_NOW_WORKBOOK.exists() and not refresh:
        return GDP_NOW_WORKBOOK

    for url in GDP_NOW_URLS:
        try:
            return urlretrieve_cached(url, GDP_NOW_WORKBOOK, refresh=refresh)
        except Exception:
            continue
    return GDP_NOW_WORKBOOK if GDP_NOW_WORKBOOK.exists() else None


def download_cleveland_nowcast_archives(refresh: bool) -> dict[str, Path]:
    paths = {}
    for rate_type, url in CLEVELAND_NOWCAST_URLS.items():
        destination = RAW_DIR / "cleveland_nowcast" / f"nowcast_{rate_type}.json"
        paths[rate_type] = urlretrieve_cached(url, destination, refresh=refresh)
    return paths


def parse_gdpnow_history(path: Path) -> pd.DataFrame:
    frames = []
    for sheet in ["TrackingDeepArchives", "TrackingArchives"]:
        try:
            df = pd.read_excel(path, sheet_name=sheet)
        except ValueError:
            continue
        required = {"Forecast Date", "Quarter being forecasted", "GDP Nowcast"}
        if not required.issubset(set(df.columns)):
            continue
        keep = df[["Forecast Date", "Quarter being forecasted", "GDP Nowcast"]].copy()
        keep["source_sheet"] = sheet
        frames.append(keep)

    if not frames:
        return pd.DataFrame(
            columns=["publication_date", "target_quarter", "gdpnow_qoq_ann", "source_sheet"]
        )

    out = pd.concat(frames, ignore_index=True)
    out["publication_date"] = out["Forecast Date"].map(excel_serial_to_timestamp)
    out["target_quarter_date"] = out["Quarter being forecasted"].map(excel_serial_to_timestamp)
    out["target_quarter"] = out["target_quarter_date"].dt.to_period("Q").astype(str)
    out["gdpnow_qoq_ann"] = pd.to_numeric(out["GDP Nowcast"], errors="coerce")
    out = out.dropna(subset=["publication_date", "target_quarter", "gdpnow_qoq_ann"])
    out = out.sort_values(["publication_date", "target_quarter"])
    return out[["publication_date", "target_quarter", "gdpnow_qoq_ann", "source_sheet"]]


def parse_cleveland_manual_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "publication_date",
                "target_period",
                "measure",
                "rate_type",
                "value",
                "source",
            ]
        )

    df = pd.read_csv(path)
    required = {"publication_date", "target_period", "measure", "rate_type", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} is missing required Cleveland nowcast columns: {sorted(missing)}"
        )
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["target_period"] = pd.to_datetime(df["target_period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["source"] = df.get("source", "Manual Cleveland Fed history")
    return df.dropna(subset=["publication_date", "target_period", "measure", "rate_type", "value"])


def parse_cleveland_target_period(label: str, rate_type: str) -> pd.Timestamp:
    clean = str(label).strip()
    if rate_type == "qoq_saar":
        return pd.Period(clean.replace(":", ""), freq="Q").end_time.normalize()
    return pd.Period(clean, freq="M").end_time.normalize()


def parse_cleveland_publication_date(label: object, target_period: pd.Timestamp) -> pd.Timestamp | None:
    text = str(label).strip()
    if not text or not any(char.isdigit() for char in text):
        return None

    parsed = pd.to_datetime(text, format="%m/%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None

    month = int(parsed.month)
    day = int(parsed.day)
    target = pd.Timestamp(target_period)
    for year in [target.year, target.year + 1, target.year - 1]:
        candidate = pd.Timestamp(year=year, month=month, day=day)
        if target - pd.Timedelta(days=15) <= candidate <= target + pd.Timedelta(days=100):
            return candidate
    return pd.Timestamp(year=target.year, month=month, day=day)


def parse_cleveland_nowcast_archives(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for rate_type, path in paths.items():
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        source_url = CLEVELAND_NOWCAST_URLS[rate_type]
        for chart in data:
            target_period = parse_cleveland_target_period(chart["chart"]["subcaption"], rate_type)
            downloaded_at = chart["chart"].get("_comment")
            raw_categories = chart["categories"][0]["category"]
            labels = [item.get("label") for item in raw_categories if not item.get("vline")]
            for dataset in chart["dataset"]:
                series_name = dataset.get("seriesname", "")
                if series_name not in CLEVELAND_MEASURE_MAP or series_name.startswith("Actual "):
                    continue
                measure = CLEVELAND_MEASURE_MAP[series_name]
                for label, item in zip(labels, dataset.get("data", [])):
                    value = pd.to_numeric(item.get("value"), errors="coerce")
                    publication_date = parse_cleveland_publication_date(label, target_period)
                    if pd.isna(value) or publication_date is None:
                        continue
                    rows.append(
                        {
                            "publication_date": publication_date,
                            "target_period": target_period,
                            "measure": measure,
                            "rate_type": rate_type,
                            "value": float(value),
                            "source": "Federal Reserve Bank of Cleveland Inflation Nowcasting",
                            "source_url": source_url,
                            "downloaded_at": downloaded_at,
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "publication_date",
                "target_period",
                "measure",
                "rate_type",
                "value",
                "source",
                "source_url",
                "downloaded_at",
            ]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["rate_type", "target_period", "publication_date", "measure"])
    return out.drop_duplicates(
        subset=["publication_date", "target_period", "measure", "rate_type"],
        keep="last",
    ).reset_index(drop=True)


def load_cleveland_nowcast_history(refresh: bool) -> pd.DataFrame:
    if CLEVELAND_MANUAL_FILE.exists() and not refresh:
        return parse_cleveland_manual_history(CLEVELAND_MANUAL_FILE)

    paths = download_cleveland_nowcast_archives(refresh=refresh)
    history = parse_cleveland_nowcast_archives(paths)
    history.to_csv(CLEVELAND_MANUAL_FILE, index=False)
    return history


def gdpnow_for_month(gdpnow: pd.DataFrame, model_date: pd.Timestamp) -> float | None:
    target_quarter = model_date.to_period("Q").strftime("%YQ%q")
    candidates = gdpnow[
        (gdpnow["publication_date"] <= model_date)
        & (gdpnow["target_quarter"].str.replace("Q", "Q", regex=False) == target_quarter)
    ]
    if candidates.empty:
        # Pandas Period string is YYYYQn; Excel-derived string can be YYYYQn too.
        candidates = gdpnow[
            (gdpnow["publication_date"] <= model_date)
            & (gdpnow["target_quarter"] == str(model_date.to_period("Q")))
        ]
    if candidates.empty:
        return None
    return float(candidates.sort_values("publication_date").iloc[-1]["gdpnow_qoq_ann"])


def cleveland_value_for_month(
    cleveland: pd.DataFrame,
    model_date: pd.Timestamp,
    measure: str,
    rate_type: str,
) -> float | None:
    target_month = pd.Timestamp(model_date).to_period("M").to_timestamp() + pd.offsets.MonthEnd(0)
    candidates = cleveland[
        (cleveland["publication_date"] <= model_date)
        & (
            cleveland["target_period"].dt.to_period("M").dt.to_timestamp()
            + pd.offsets.MonthEnd(0)
            == target_month
        )
        & (cleveland["measure"].str.lower() == measure.lower())
        & (cleveland["rate_type"].str.lower() == rate_type.lower())
    ]
    if candidates.empty:
        return None
    return float(candidates.sort_values("publication_date").iloc[-1]["value"])


def monthly_pct_to_annualized(value: float) -> float:
    return ((1.0 + value / 100.0) ** 12 - 1.0) * 100.0


def build_nowcast_panel(
    rt_panel: pd.DataFrame,
    gdpnow: pd.DataFrame,
    cleveland: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = rt_panel.copy()
    replacements = []

    for model_date in out.index:
        gdp_value = gdpnow_for_month(gdpnow, model_date) if not gdpnow.empty else None
        if gdp_value is not None:
            out.loc[model_date, "GDP_QoQAnn"] = gdp_value
            replacements.append(
                {
                    "date": model_date,
                    "variable": "GDP_QoQAnn",
                    "replacement_source": "Atlanta Fed GDPNow",
                    "replacement_value": gdp_value,
                }
            )

        cpi_value = (
            cleveland_value_for_month(cleveland, model_date, "CPI", "yoy")
            if not cleveland.empty
            else None
        )
        if cpi_value is not None:
            out.loc[model_date, "CPI_YoY"] = cpi_value
            replacements.append(
                {
                    "date": model_date,
                    "variable": "CPI_YoY",
                    "replacement_source": "Cleveland Fed inflation nowcast",
                    "replacement_value": cpi_value,
                }
            )

        core_pce_value = (
            cleveland_value_for_month(cleveland, model_date, "Core PCE", "yoy")
            if not cleveland.empty
            else None
        )
        if core_pce_value is not None:
            out.loc[model_date, "CorePCE_YoY"] = core_pce_value
            replacements.append(
                {
                    "date": model_date,
                    "variable": "CorePCE_YoY",
                    "replacement_source": "Cleveland Fed inflation nowcast",
                    "replacement_value": core_pce_value,
                }
            )

    return out, pd.DataFrame(replacements)


def diagnostics_for_panels(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for panel_name, df in panels.items():
        for column in df.columns:
            series = df[column]
            rows.append(
                {
                    "panel": panel_name,
                    "variable": column,
                    "rows": len(series),
                    "missing": int(series.isna().sum()),
                    "first_valid": series.first_valid_index(),
                    "last_valid": series.last_valid_index(),
                    "mean": series.mean(skipna=True),
                    "std": series.std(skipna=True),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    full_panel: pd.DataFrame,
    rt_panel: pd.DataFrame,
    nowcast_panel: pd.DataFrame,
    metadata: pd.DataFrame,
    diagnostics: pd.DataFrame,
    replacements: pd.DataFrame,
    gdpnow: pd.DataFrame,
    cleveland: pd.DataFrame,
) -> None:
    full_panel.to_csv(OUTPUT_DIR / "full_information.csv", index_label="date")
    rt_panel.to_csv(OUTPUT_DIR / "realtime_information.csv", index_label="date")
    nowcast_panel.to_csv(OUTPUT_DIR / "nowcast_information.csv", index_label="date")
    metadata.to_csv(OUTPUT_DIR / "source_metadata.csv", index=False)
    diagnostics.to_csv(OUTPUT_DIR / "diagnostics.csv", index=False)
    replacements.to_csv(OUTPUT_DIR / "nowcast_replacements.csv", index=False)
    gdpnow.to_csv(OUTPUT_DIR / "gdpnow_history.csv", index=False)
    cleveland.to_csv(OUTPUT_DIR / "cleveland_inflation_nowcast_history_used.csv", index=False)

    workbook_path = OUTPUT_DIR / "thesis_data_panels.xlsx"
    requested_workbook_path = workbook_path
    try:
        handle = workbook_path.open("a+b")
        handle.close()
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workbook_path = OUTPUT_DIR / f"thesis_data_panels_{stamp}.xlsx"
        print(
            f"Workbook is locked: {requested_workbook_path}. "
            f"Writing Excel output to {workbook_path} instead."
        )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        full_panel.to_excel(writer, sheet_name="FullInformation")
        rt_panel.to_excel(writer, sheet_name="RealTime")
        nowcast_panel.to_excel(writer, sheet_name="NowcastEnhanced")
        metadata.to_excel(writer, sheet_name="SourceMetadata", index=False)
        diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)
        replacements.to_excel(writer, sheet_name="NowcastReplacements", index=False)
        gdpnow.to_excel(writer, sheet_name="GDPNowHistory", index=False)
        cleveland.to_excel(writer, sheet_name="ClevelandNowcast", index=False)


def main() -> None:
    require_packages()
    ensure_dirs()
    args = parse_args()
    if args.fred_api_key:
        os.environ["FRED_API_KEY"] = args.fred_api_key

    end = latest_local_survey_end() if args.end == "auto" else pd.Timestamp(args.end)
    months = month_end_index(args.start, end.strftime("%Y-%m-%d"))

    full_fred, fred_metadata = build_latest_fred_panel(months, refresh=args.refresh)
    full_panel, local_metadata = add_local_survey_data(full_fred)
    metadata = pd.concat([fred_metadata, local_metadata], ignore_index=True)

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if args.real_time_mode == "vintage" and not api_key:
        raise SystemExit("real-time-mode=vintage requires FRED_API_KEY in the environment.")

    use_vintage = args.real_time_mode == "vintage" or (args.real_time_mode == "auto" and bool(api_key))
    if use_vintage:
        print("Building real-time panel with FRED/ALFRED vintage snapshots. This can take a while.")
        rt_panel = build_vintage_rt_panel(months, full_panel, api_key, refresh=args.refresh)
        metadata["real_time_method"] = "vintage_as_of_month_end"
    else:
        if args.real_time_mode == "lag":
            print("Using deterministic publication-lag fallback for RT panel by request.")
        else:
            print("FRED_API_KEY not found; using deterministic publication-lag fallback for RT panel.")
        rt_panel = build_lag_fallback_rt(full_panel)
        metadata["real_time_method"] = "fallback_lag"

    gdpnow = pd.DataFrame()
    cleveland = pd.DataFrame()
    replacements = pd.DataFrame()
    if args.skip_nowcasts:
        nowcast_panel = rt_panel.copy()
    else:
        gdpnow_path = download_gdpnow_workbook(refresh=args.refresh)
        if gdpnow_path:
            gdpnow = parse_gdpnow_history(gdpnow_path)
        cleveland = load_cleveland_nowcast_history(refresh=args.refresh)
        nowcast_panel, replacements = build_nowcast_panel(rt_panel, gdpnow, cleveland)

    panels = {
        "full_information": full_panel,
        "realtime_information": rt_panel,
        "nowcast_information": nowcast_panel,
    }
    diagnostics = diagnostics_for_panels(panels)
    write_outputs(full_panel, rt_panel, nowcast_panel, metadata, diagnostics, replacements, gdpnow, cleveland)

    print(f"Wrote outputs to: {OUTPUT_DIR}")
    print(f"Sample: {months.min().date()} through {months.max().date()}")
    print(f"Full panel shape: {full_panel.shape}")
    print(f"RT panel shape: {rt_panel.shape}")
    print(f"Nowcast panel shape: {nowcast_panel.shape}")
    if cleveland.empty:
        print(
            "Note: Cleveland historical nowcasts were not found after checking the "
            "Cleveland Fed JSON archives and the local manual history file."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
