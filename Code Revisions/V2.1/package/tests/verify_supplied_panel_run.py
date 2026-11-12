"""Integrity and numerical checks for this supplied-panel rerun, not real-time validity."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];checks=[]
def check(name,condition):
 if not bool(condition):raise AssertionError(name)
 checks.append(name)
p=ROOT/'source/thesis_data_panels.xlsx'
check('original workbook hash preserved',hashlib.sha256(p.read_bytes()).hexdigest()=='7f817abcd6eefa8a6716ec3275eb5996a95c0eb050c85f06a73674f389b9621b')
mapping={'FullInformation':'full_information.csv','RealTime':'realtime_information.csv','NowcastEnhanced':'nowcast_information.csv'}
for sheet,name in mapping.items():
 x=pd.read_excel(p,sheet_name=sheet,index_col=0)
 check(sheet+' shape',x.shape==(141,19))
 check(sheet+' monthly unique ordered dates',pd.DatetimeIndex(x.index).equals(pd.date_range('2014-01-31','2025-09-30',freq='ME')))
 for variant in ['original_run','corrected_run']:
  y=pd.read_csv(ROOT/variant/'outputs'/name,index_col=0)
  check(variant+'/'+sheet+' columns preserved',list(x.columns)==list(y.columns))
  check(variant+'/'+sheet+' values exported without additional transformation',np.allclose(x,y,rtol=1e-14,atol=1e-14,equal_nan=True))
for model in ['full_information','realtime_information','nowcast_enhanced']:
 a=ROOT/'original_run/outputs';b=ROOT/'corrected_run/outputs'
 x=pd.read_csv(a/f'pca_{model}/hmm_pca_input.csv');y=pd.read_csv(b/f'pca_{model}/hmm_pca_input.csv')
 check(model+' identical PCA inputs',x.equals(y))
 x=pd.read_csv(a/f'hmm_models_{model}/hmm_selected_models.csv');y=pd.read_csv(b/f'hmm_models_{model}/hmm_selected_models.csv')
 for col in ['best_random_seed','log_likelihood','bic','n_observations']:
  check(model+' identical '+col,np.array_equal(x[col].to_numpy(),y[col].to_numpy()))
 for k in [3,5]:
  x=pd.read_csv(a/f'hmm_models_{model}/hmm_regime_labels_pca{k}.csv');y=pd.read_csv(b/f'hmm_models_{model}/hmm_regime_labels_pca{k}.csv')
  check(model+f' PC{k} label mapping unchanged',x[['state','regime_label']].equals(y[['state','regime_label']]))
table=pd.read_csv(ROOT/'original_run/outputs/hmm_model_comparison_across_information_sets/internal_full_information_benchmark.csv').query('pca_components==5').set_index('candidate_model')
expected={'mean_information_gap_l1_vs_full':(1.395,.238),'probability_rmse_vs_full':(.577,.211),'benchmark_regime_brier_score':(1.377,.202),'hard_regime_accuracy':(.307,.891)}
for col,values in expected.items():
 for model,v in zip(['realtime_information','nowcast_enhanced'],values):check('Table5 rounded '+model+' '+col,round(float(table.loc[model,col]),3)==v)
rt,nc=table.loc[['realtime_information','nowcast_enhanced'],'mean_information_gap_l1_vs_full']
check('Table5 rounded improvement 82.9%',round(100*(1-nc/rt),1)==82.9)
summary=pd.read_csv(ROOT/'filter_audit/fixed_parameter_results/summary.csv').set_index('method')
check('forward filter controlled reduction 80.657%',round(summary.loc['filtered','reduction_percent'],3)==80.657)
fits=pd.read_csv(ROOT/'filter_audit/fixed_parameter_results/fit_recovery_checks.csv')
check('controlled selected fits recovered',fits.maximum_recovered_smoothed_probability_error.lt(1e-10).all())
hold=pd.read_csv(ROOT/'original_run/outputs/hmm_oos_validation/oos_run_summary.csv').iloc[0]
check('existing holdout 125 training plus12test',hold.train_rows==125 and hold.holdout_rows==12)
out={'scope':'Supplied-input integrity and reproduction checks; not certification of historical availability or out-of-sample validity','n_passed':len(checks),'checks':checks}
(ROOT/'evidence/supplied_panel_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
