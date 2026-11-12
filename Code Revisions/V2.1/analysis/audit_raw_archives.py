"""Reconcile newly located Desktop raw archives without replacing supplied inputs."""
from pathlib import Path
import sys, importlib.util,json,hashlib,warnings
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code'))
import dataCollection as new
spec=importlib.util.spec_from_file_location('raw_original',R/'package/code/original/dataCollection.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
raw=R/'raw_source';out=R/'output/raw_archive_audit';out.mkdir(exist_ok=True)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    oldhist=old.parse_cleveland_nowcast_archives({'yoy':raw/'nowcast_yoy.json'})
newhist=new.parse_cleveland_nowcast_archives({'yoy':raw/'nowcast_yoy.json'})
workbook=R/'package/source/thesis_data_panels.xlsx'
sup=pd.read_excel(workbook,sheet_name='ClevelandNowcast')
sup=sup[sup.rate_type=='yoy'].copy()
keys=['publication_date','target_period','measure','rate_type']
for f in [sup,oldhist,newhist]:
    for c in ['publication_date','target_period']:f[c]=pd.to_datetime(f[c])
comp=sup[keys+['value']].merge(oldhist[keys+['value']],on=keys,how='outer',suffixes=('_supplied','_raw_original'),indicator=True,validate='one_to_one')
both=comp._merge=='both';comp.to_csv(out/'supplied_vs_original_raw_parser.csv',index=False)
newrows=newhist.merge(oldhist[keys],on=keys,how='left',indicator=True,validate='one_to_one').query("_merge=='left_only'").drop(columns='_merge')
newrows.to_csv(out/'recovered_issue_rows.csv',index=False)
rt=pd.read_csv(R/'package/original_run/outputs/realtime_information.csv',index_col=0,parse_dates=True)
nc=pd.read_csv(R/'package/original_run/outputs/nowcast_information.csv',index_col=0,parse_dates=True)
gdp_sup=pd.read_excel(workbook,sheet_name='GDPNowHistory')
gdp_raw=new.parse_gdpnow_history(raw/'GDPTrackingModelDataAndForecasts.xlsx')
gkeys=['publication_date','target_quarter','source_sheet']
for f in [gdp_sup,gdp_raw]:f['publication_date']=pd.to_datetime(f.publication_date)
gcomp=gdp_sup.merge(gdp_raw,on=gkeys,how='outer',suffixes=('_supplied','_raw'),indicator=True,validate='one_to_one')
gcomp.to_csv(out/'supplied_vs_raw_gdp.csv',index=False)
oldpanel,_=new.build_nowcast_panel(rt,gdp_sup,oldhist)
repaired,ledger=new.build_nowcast_panel(rt,gdp_sup,newhist)
repaired.to_csv(out/'nowcast_information.csv')
ledger.to_csv(out/'replacement_lineage.csv',index=False)
changes=[]
for date in nc.index:
    for v in nc.columns:
        if not np.isclose(nc.loc[date,v],repaired.loc[date,v],rtol=0,atol=1e-10,equal_nan=True):
            changes.append(dict(date=str(date.date()),variable=v,supplied=nc.loc[date,v],raw_repaired=repaired.loc[date,v]))
pd.DataFrame(changes,columns=['date','variable','supplied','raw_repaired']).to_csv(out/'changed_panel_cells.csv',index=False)
summary=dict(raw_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()},
    old_history_rows=len(oldhist),new_history_rows=len(newhist),recovered_rows=len(newrows),
    supplied_vs_original_merge_counts=comp._merge.value_counts().to_dict(),
    maximum_common_value_difference=float(abs(comp.loc[both,'value_supplied']-comp.loc[both,'value_raw_original']).max()),
    original_parser_reproduces_supplied_panel=bool(np.allclose(nc,oldpanel,rtol=0,atol=1e-10,equal_nan=True)),
    gdp_merge_counts=gcomp._merge.value_counts().to_dict(),
    maximum_gdp_value_difference=float(abs(gcomp.gdpnow_qoq_ann_supplied-gcomp.gdpnow_qoq_ann_raw).max()),
    changed_panel_cells=changes,
    scope='Parser-only diagnostic using the newly located raw JSON and supplied GDP history; no release-calendar or vintage reconstruction. Provider chart comment is not verified retrieval time.')
(out/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='changed_panel_cells'},indent=2));print(pd.DataFrame(changes).to_string(index=False))
