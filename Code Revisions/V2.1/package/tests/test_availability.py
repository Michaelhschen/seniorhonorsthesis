"""Regression audit of pinned original behavior; synthetic fixtures are not thesis data."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('audited_dataCollection', ROOT/'code/original/dataCollection.py')
dc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dc
spec.loader.exec_module(dc)


def test_q1_is_available_from_april_under_three_row_lag():
    raw = pd.Series([10.0, 20.0, 30.0], index=pd.to_datetime(['2024-10-01','2025-01-01','2025-04-01']))
    full = dc.expand_quarterly_to_monthly(raw).to_frame('CorpCapex_QoQ')
    rt = dc.build_lag_fallback_rt(full)
    assert full.loc['2025-01-31','CorpCapex_QoQ'] == 20
    assert rt.loc['2025-03-31','CorpCapex_QoQ'] == 10
    assert rt.loc['2025-04-30','CorpCapex_QoQ'] == 20
    assert rt.loc['2025-05-31','CorpCapex_QoQ'] == 20


def test_lag_panel_moves_revised_values_not_historical_vintages():
    dates = pd.date_range('2025-01-31', periods=3, freq='ME')
    first = pd.DataFrame({'CPI_YoY':[2.0,3.0,4.0]},index=dates)
    revised = first.copy(); revised.iloc[0,0] = 9.0
    assert dc.build_lag_fallback_rt(first).loc['2025-02-28','CPI_YoY'] == 2
    assert dc.build_lag_fallback_rt(revised).loc['2025-02-28','CPI_YoY'] == 9


def test_quarterly_expansion_then_difference_creates_reference_quarter_pulses():
    raw = pd.Series([1.,2.,4.],index=pd.to_datetime(['2024-10-01','2025-01-01','2025-04-01']))
    monthly = dc.expand_quarterly_to_monthly(raw)
    changes = monthly.diff().dropna()
    assert changes.loc['2025-02-28'] == 0
    assert changes.loc['2025-03-31'] == 0
    assert changes.loc['2025-04-30'] == 2


def test_gdpnow_rejects_future_issue_and_wrong_target():
    rows = pd.DataFrame({'publication_date':pd.to_datetime(['2025-01-29','2025-02-01','2025-01-30']),
                         'target_quarter':['2025Q1','2025Q1','2024Q4'],'gdpnow_qoq_ann':[1.,99.,88.]})
    assert dc.gdpnow_for_month(rows,pd.Timestamp('2025-01-31')) == 1


def test_gdpnow_selects_latest_eligible_issue():
    rows = pd.DataFrame({'publication_date':pd.to_datetime(['2025-01-20','2025-01-29']),
                         'target_quarter':['2025Q1','2025Q1'],'gdpnow_qoq_ann':[1.,2.]})
    assert dc.gdpnow_for_month(rows,pd.Timestamp('2025-01-31')) == 2


def test_inflation_selection_requires_correct_target_measure_rate_and_issue():
    rows = pd.DataFrame({'publication_date':pd.to_datetime(['2025-01-29','2025-02-01','2025-01-30','2025-01-30','2025-01-30']),
                         'target_period':pd.to_datetime(['2025-01-31','2025-01-31','2024-12-31','2025-01-31','2025-01-31']),
                         'measure':['CPI','CPI','CPI','Core PCE','CPI'],'rate_type':['yoy','yoy','yoy','yoy','mom'],
                         'value':[2.,99.,88.,77.,66.]})
    assert dc.cleveland_value_for_month(rows,pd.Timestamp('2025-01-31'),'CPI','yoy') == 2


def test_missing_nowcasts_keep_stale_input():
    rt = pd.DataFrame({'GDP_QoQAnn':[1.], 'CPI_YoY':[2.], 'CorePCE_YoY':[3.]}, index=pd.to_datetime(['2025-01-31']))
    result, replacements = dc.build_nowcast_panel(rt,pd.DataFrame(),pd.DataFrame())
    pd.testing.assert_frame_equal(result,rt)
    assert replacements.empty


def test_replacement_log_does_not_retain_issue_date_or_target():
    rt = pd.DataFrame({'GDP_QoQAnn':[1.]}, index=pd.to_datetime(['2025-01-31']))
    nowcast = pd.DataFrame({'publication_date':pd.to_datetime(['2025-01-29']), 'target_quarter':['2025Q1'],'gdpnow_qoq_ann':[2.]})
    result, replacements = dc.build_nowcast_panel(rt,nowcast,pd.DataFrame())
    assert result.iloc[0,0] == 2
    assert list(replacements.columns) == ['date','variable','replacement_source','replacement_value']


def test_cleveland_parser_excludes_actual_series(tmp_path):
    fixture = [{'chart':{'subcaption':'2025-1','_comment':'test fixture'},'categories':[{'category':[{'label':'01/30'}]}],
                'dataset':[{'seriesname':'CPI Inflation','data':[{'value':'2'}]}, {'seriesname':'Actual CPI Inflation','data':[{'value':'99'}]}]}]
    path=tmp_path/'fixture.json';path.write_text(json.dumps(fixture))
    history = dc.parse_cleveland_nowcast_archives({'yoy':path})
    assert history['value'].tolist() == [2.]


@pytest.mark.parametrize('year',[2016,2024])
def test_leap_day_is_dropped_by_original_parser(year):
    assert dc.parse_cleveland_publication_date('02/29',pd.Timestamp(f'{year}-02-29')) is None


def test_cross_year_post_target_issue_is_supported():
    assert dc.parse_cleveland_publication_date('01/03',pd.Timestamp('2024-12-31')) == pd.Timestamp('2025-01-03')


@pytest.mark.skip(reason="Current-provider raw JSON is not part of the supplied historical workbook; prior reconstruction package verified this separately")
def test_leap_day_exists_in_downloaded_yoy_provider_archive():
    path=ROOT/'compat_run/raw_downloads/cleveland_nowcast/nowcast_yoy.json'
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    targets={pd.Period(c['chart']['subcaption'],freq='M'): c for c in data}
    for year in [2016,2024]:
        labels=[x.get('label') for x in targets[pd.Period(f'{year}-02',freq='M')]['categories'][0]['category'] if not x.get('vline')]
        assert '02/29' in labels

