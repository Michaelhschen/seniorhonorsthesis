"""Rebuild pinned thesis analysis from the user workbook; never collect new panels."""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--work-dir',required=True);ap.add_argument('--variant',choices=['original','corrected'],required=True);args=ap.parse_args()
work=Path(args.work_dir).resolve()
if work.exists():raise SystemExit(f'Refusing existing directory: {work}')
work.mkdir(parents=True)
source=ROOT/'source/thesis_data_panels.xlsx'
srcroot=ROOT/'code'/args.variant
for f in srcroot.iterdir():
    if f.is_file():shutil.copy2(f,work/f.name)
out=work/'outputs';out.mkdir()
mapping={'FullInformation':'full_information.csv','RealTime':'realtime_information.csv','NowcastEnhanced':'nowcast_information.csv'}
inv=[]
for sheet,name in mapping.items():
    frame=pd.read_excel(source,sheet_name=sheet,index_col=0)
    frame.index=pd.DatetimeIndex(frame.index,name='date')
    assert frame.shape==(141,19) and frame.index.is_unique and frame.index.is_monotonic_increasing
    assert frame.index.equals(pd.date_range('2014-01-31','2025-09-30',freq='ME',name='date'))
    assert all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes)
    frame.to_csv(out/name,index=True)
    rt=pd.read_csv(out/name,index_col=0,parse_dates=True)
    np.testing.assert_allclose(rt.to_numpy(),frame.to_numpy(),rtol=1e-14,atol=1e-14,equal_nan=True)
    inv.append({'sheet':sheet,'file':name,'shape':list(frame.shape),'dates':[str(frame.index.min()),str(frame.index.max())],'stage':'economic-transform panel; before stationarity cleaning, scaling or PCA','sha256_csv':hashlib.sha256((out/name).read_bytes()).hexdigest()})
# Separate, explicitly identified benchmark snapshot; no user panel is replaced.
if (ROOT/'source/external_benchmarks.csv').exists():shutil.copy2(ROOT/'source/external_benchmarks.csv',out/'external_benchmarks.csv')
(work/'input_provenance.json').write_text(json.dumps({'source_workbook_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'variant':args.variant,'panels':inv,'collection_stage':'not run; no fresh data downloaded'},indent=2))
env=os.environ.copy();env.pop('FRED_API_KEY',None);env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
bootstrap=r'''
import os,runpy,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));os.environ.pop('FRED_API_KEY',None)
def blocked(*a,**k):raise RuntimeError('NETWORK DISABLED IN USER-PANEL RERUN')
urllib.request.urlopen=blocked
jobs=[('hmmModeling.py',['--refresh-pca-inputs','--pca-components','3,5','--pca-output-components','5','--state-counts','4','--bic-diagnostic-state-counts','3,4,5']),('hmmModelComparison.py',['--pca-components','3,5','--skip-nber-download']),('hmmOutOfSampleValidation.py',['--holdout-months','12','--pca-components','3,5'])]
for script,args in jobs:
 print('RUN',script,*args,flush=True);sys.argv=[script]+args
 try:runpy.run_path(script,run_name='__main__')
 except SystemExit as e:
  if e.code not in (None,0):raise
print('ALL USER-PANEL STAGES COMPLETED',flush=True)
'''
with (work/'execution.log').open('w') as log:subprocess.run([sys.executable,'-c',bootstrap],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print(pd.read_csv(out/'hmm_model_comparison_across_information_sets/internal_full_information_benchmark.csv').to_string(index=False))
