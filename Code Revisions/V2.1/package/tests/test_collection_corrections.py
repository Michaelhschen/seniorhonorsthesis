"""Tests of narrow local collection corrections; no network access."""
import importlib.util
import sys
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('corrected_dc',ROOT/'code/corrected/dataCollection.py')
dc=importlib.util.module_from_spec(spec);sys.modules[spec.name]=dc;spec.loader.exec_module(dc)
spec2=importlib.util.spec_from_file_location('original_dc_compare',ROOT/'code/original/dataCollection.py')
old=importlib.util.module_from_spec(spec2);sys.modules[spec2.name]=old;spec2.loader.exec_module(old)


@pytest.mark.parametrize('header',['DATE','observation_date'])
def test_fred_csv_accepts_both_documented_header_shapes(tmp_path,monkeypatch,header):
    monkeypatch.delenv('FRED_API_KEY',raising=False)
    monkeypatch.setattr(dc,'RAW_DIR',tmp_path)
    folder=tmp_path/'fred_latest';folder.mkdir()
    (folder/'X.csv').write_text(f'{header},X\n2025-01-01,1\n2025-02-01,.\n')
    actual=dc.read_fred_latest('X')
    assert actual.iloc[0] == 1 and np.isnan(actual.iloc[1])
    assert actual.index[0] == pd.Timestamp('2025-01-01')


@pytest.mark.parametrize('year',[2016,2024])
def test_yearless_feb29_resolves_using_valid_target_year(year):
    assert dc.parse_cleveland_publication_date('02/29',pd.Timestamp(f'{year}-02-29')) == pd.Timestamp(f'{year}-02-29')


def test_impossible_month_day_is_rejected():
    assert dc.parse_cleveland_publication_date('02/30',pd.Timestamp('2024-02-29')) is None


def test_cross_year_valid_labels_keep_prior_behavior():
    for target,label in [('2024-12-31','01/03'),('2025-01-31','01/03'),('2025-03-31','01/15'),('2025-01-31','01/31')]:
        assert dc.parse_cleveland_publication_date(label,pd.Timestamp(target)) == old.parse_cleveland_publication_date(label,pd.Timestamp(target))


def make_workbook(path,valid=True):
    data={'Forecast Date':[pd.Timestamp('2025-01-29')], 'Quarter being forecasted':[pd.Timestamp('2025-01-01')], 'GDP Nowcast':[2.]}
    if not valid:data={'unrelated':[1]}
    pd.DataFrame(data).to_excel(path,sheet_name='TrackingArchives',index=False)


def test_html_with_xlsx_suffix_is_rejected(tmp_path):
    path=tmp_path/'bad.xlsx';path.write_text('<html>404 not found</html>')
    with pytest.raises(ValueError,match='not an XLSX'):
        dc.parse_gdpnow_history(path)


def test_cached_bad_workbook_fails_loudly_without_silent_empty_history(tmp_path,monkeypatch):
    path=tmp_path/'bad.xlsx';path.write_text('<html>404</html>')
    monkeypatch.setattr(dc,'GDP_NOW_WORKBOOK',path)
    with pytest.raises(ValueError,match='not an XLSX'):
        dc.download_gdpnow_workbook(False)


def test_required_archive_columns_are_validated(tmp_path):
    path=tmp_path/'bad_columns.xlsx';make_workbook(path,valid=False)
    with pytest.raises(ValueError,match='required history columns'):
        dc.parse_gdpnow_history(path)


def test_required_archive_sheet_is_validated(tmp_path):
    path=tmp_path/'wrong_sheet.xlsx';pd.DataFrame({'x':[1]}).to_excel(path,index=False)
    with pytest.raises(ValueError,match='expected archive sheets'):
        dc.validate_gdpnow_workbook(path)


def test_valid_archive_is_parsed(tmp_path):
    path=tmp_path/'valid.xlsx';make_workbook(path)
    history=dc.parse_gdpnow_history(path)
    assert history['gdpnow_qoq_ann'].tolist()==[2.]
    assert history['target_quarter'].tolist()==['2025Q1']


def test_download_tries_next_url_after_html_and_installs_only_valid_workbook(tmp_path,monkeypatch):
    source=tmp_path/'source.xlsx';make_workbook(source)
    destination=tmp_path/'cache.xlsx'
    monkeypatch.setattr(dc,'GDP_NOW_WORKBOOK',destination)
    monkeypatch.setattr(dc,'GDP_NOW_URLS',['bad','good'])
    seen=[]
    def download(url,path,refresh=False):
        seen.append((url,refresh))
        if url=='bad':path.write_text('<html>404</html>')
        else:shutil.copyfile(source,path)
        return path
    monkeypatch.setattr(dc,'urlretrieve_cached',download)
    assert dc.download_gdpnow_workbook(False)==destination
    assert seen==[('bad',True),('good',True)]
    dc.validate_gdpnow_workbook(destination)
    assert not destination.with_suffix('.download.xlsx').exists()


def test_failed_refresh_preserves_existing_valid_cache_and_raises(tmp_path,monkeypatch):
    destination=tmp_path/'cache.xlsx';make_workbook(destination)
    before=destination.read_bytes()
    monkeypatch.setattr(dc,'GDP_NOW_WORKBOOK',destination)
    monkeypatch.setattr(dc,'GDP_NOW_URLS',['bad'])
    def download(url,path,refresh=False):path.write_text('<html>404</html>');return path
    monkeypatch.setattr(dc,'urlretrieve_cached',download)
    with pytest.raises(RuntimeError,match='cannot silently continue'):
        dc.download_gdpnow_workbook(True)
    assert destination.read_bytes()==before


def test_selection_values_unchanged_and_issue_metadata_exported():
    dates=pd.to_datetime(['2025-01-31','2025-02-28'])
    rt=pd.DataFrame({'GDP_QoQAnn':[1.,2.],'CPI_YoY':[3.,4.],'CorePCE_YoY':[5.,6.]},index=dates)
    gdp=pd.DataFrame({'publication_date':pd.to_datetime(['2025-01-29','2025-02-25']),
                      'target_quarter':['2025Q1','2025Q1'],'gdpnow_qoq_ann':[2.,3.],
                      'source_sheet':['TrackingArchives','TrackingArchives']})
    inflation=pd.DataFrame({'publication_date':pd.to_datetime(['2025-01-30','2025-01-30','2025-02-27']),
                            'target_period':pd.to_datetime(['2025-01-31','2025-01-31','2025-02-28']),
                            'measure':['CPI','Core PCE','CPI'],'rate_type':['yoy']*3,'value':[2.,3.,4.],
                            'source_url':['https://example.test/source']*3})
    expected,_=old.build_nowcast_panel(rt,gdp,inflation)
    actual,ledger=dc.build_nowcast_panel(rt,gdp,inflation)
    pd.testing.assert_frame_equal(actual,expected)
    assert (ledger.publication_date<=ledger.date).all()
    assert {'publication_date','target_period','rate_type','source_sheet','source_url','original_rt_value'}<=set(ledger.columns)
    assert ledger.loc[ledger.variable=='GDP_QoQAnn','original_rt_value'].tolist()==[1.,2.]
    assert actual.loc['2025-02-28','CorePCE_YoY']==6.  # missing nowcast preserves fallback

