"""Bounded 2x2 input/schema diagnostic; not a model search or validation result."""
from pathlib import Path
import sys,json,itertools,argparse
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'code/original'))
import hmmModeling as hm,hmmModelComparison as mc,dataCleaning as dc
ap=argparse.ArgumentParser();ap.add_argument('--fixed-seeds',action='store_true');args=ap.parse_args()
OUT=ROOT/('snapshot_diagnostics_fixed_seeds' if args.fixed_seeds else 'snapshot_diagnostics');OUT.mkdir(exist_ok=True)
fixed_seeds={'full_information':2,'realtime_information':0,'nowcast_enhanced':4}
DATA={'supplied':ROOT/'original_run/outputs','current':ROOT/'comparison_inputs/current_snapshot'}
SCHEMAS={name:pd.read_csv(path/'uniform_cleaning_schema.csv').fillna('') for name,path in DATA.items()}
names=['full_information','realtime_information','nowcast_enhanced'];files=['full_information.csv','realtime_information.csv','nowcast_information.csv']
summaries=[];details=[];selection=[];loadings=[];perms=[];profiles=[]
for data_name,base in DATA.items():
 panels={n:dc.read_panel(base/f) for n,f in zip(names,files)}
 commoncols=list(SCHEMAS['supplied'].source_variable)
 dates=panels[names[0]][commoncols].dropna().index
 for n in names[1:]:dates=dates.intersection(panels[n][commoncols].dropna().index)
 for schema_name,schema in SCHEMAS.items():
  frames={};tag=f'{data_name}_data__{schema_name}_schema';dest=OUT/tag;dest.mkdir(exist_ok=True)
  for n in names:
   a=panels[n].loc[dates];stationary=pd.DataFrame(index=dates)
   for _,row in schema.iterrows():
    if row.decision=='keep_level':stationary[row.output_variable]=a[row.source_variable]
    elif row.decision=='keep_difference':stationary[row.output_variable]=a[row.source_variable].diff()
   stationary=stationary.dropna();scores,loads,var=dc.fit_static_pca(stationary,n_components=5,variance_threshold=.7)
   xdf=scores[[f'PCA_{i}' for i in range(1,6)]];x=xdf.to_numpy(float)
   model,info=hm.fit_best_hmm(x,4,'diag',[fixed_seeds[n]] if args.fixed_seeds else [0,1,2,3,4],1000,1e-4)
   prob=hm.state_probability_table(xdf.index,model,x);state=hm.state_summary_table(model,xdf,prob);labels=hm.regime_label_table(state)
   prob.to_csv(dest/f'{n}_raw_probabilities.csv',index=False);labels.to_csv(dest/f'{n}_labels.csv',index=False)
   score=prob.copy();score['most_likely_regime']=score.most_likely_state.map(dict(zip(labels.state,labels.regime_label)))
   for _,row in labels.iterrows():score[mc.REGIME_SLUGS[row.regime_label]+'_prob']=prob[f'state_{int(row.state)}_prob']
   frames[n]=score;score.to_csv(dest/f'{n}_regime_probabilities.csv',index=False)
   loads['model']=n;loads['data_snapshot']=data_name;loads['schema_snapshot']=schema_name;loadings.append(loads)
   selection.append({'data_snapshot':data_name,'schema_snapshot':schema_name,'model':n,**info})
   raw=panels[n].reset_index().rename(columns={'index':'date'});raw['date']=pd.to_datetime(raw['date'])
   grouped=score.merge(raw,on='date').groupby(['most_likely_state','most_likely_regime'])
   for (state,label),group in grouped:
    profiles.append({'data_snapshot':data_name,'schema_snapshot':schema_name,'model':n,'state':state,'label':label,'months':len(group),**{v:group[v].mean() for v in ['GDP_QoQAnn','CPI_YoY','CorePCE_YoY','UNRATE','VIX_log','ISM_PMI']}})
  vals={}
  for n in names[1:]:
   row=mc.benchmark_metrics(frames[names[0]],frames[n],n,5);row.update(data_snapshot=data_name,schema_snapshot=schema_name);details.append(row);vals[n]=row['mean_information_gap_l1_vs_full']
   a=frames[names[0]][mc.REGIME_PROB_COLS].to_numpy();b=frames[n][mc.REGIME_PROB_COLS].to_numpy()
   for perm in itertools.permutations(range(4)):perms.append({'data_snapshot':data_name,'schema_snapshot':schema_name,'model':n,'permutation':','.join(map(str,perm)),'mean_L1':np.abs(a-b[:,perm]).sum(1).mean(),'original_mapping':perm==tuple(range(4))})
  summaries.append({'data_snapshot':data_name,'schema_snapshot':schema_name,'RT_L1':vals[names[1]],'NC_L1':vals[names[2]],'reduction_percent':100*(1-vals[names[2]]/vals[names[1]])})
pd.DataFrame(summaries).to_csv(OUT/'factorial_summary.csv',index=False);pd.DataFrame(details).to_csv(OUT/'internal_metrics.csv',index=False);pd.DataFrame(selection).to_csv(OUT/'selected_fits.csv',index=False);pd.concat(loadings).to_csv(OUT/'pca_loadings.csv',index=False);pd.DataFrame(perms).to_csv(OUT/'all_label_permutations.csv',index=False);pd.DataFrame(profiles).to_csv(OUT/'state_profiles.csv',index=False)
print(pd.DataFrame(summaries).to_string(index=False))
