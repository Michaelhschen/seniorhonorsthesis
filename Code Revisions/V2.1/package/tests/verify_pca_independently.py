"""Independent NumPy SVD/standardization check, without fitting another HMM."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];base=ROOT/'original_run/outputs'
schema=pd.read_csv(base/'uniform_cleaning_schema.csv').fillna('')
names=['full_information','realtime_information','nowcast_enhanced'];files=['full_information.csv','realtime_information.csv','nowcast_information.csv']
panels={n:pd.read_csv(base/f,index_col=0,parse_dates=True) for n,f in zip(names,files)}
cols=list(schema.source_variable);dates=panels[names[0]][cols].dropna().index
for n in names[1:]:dates=dates.intersection(panels[n][cols].dropna().index)
rows=[]
for n in names:
 x=pd.DataFrame(index=dates)
 for _,r in schema.iterrows():
  if r.decision=='keep_level':x[r.output_variable]=panels[n].loc[dates,r.source_variable]
  elif r.decision=='keep_difference':x[r.output_variable]=panels[n].loc[dates,r.source_variable].diff()
 x=x.dropna();saved=pd.read_csv(base/f'pca_{n}/stationary_inputs.csv',index_col=0,parse_dates=True)
 assert x.columns.equals(saved.columns) and x.index.equals(saved.index)
 e1=float(np.max(abs(x.to_numpy()-saved.to_numpy())))
 z=(x.to_numpy()-x.to_numpy().mean(0))/x.to_numpy().std(0,ddof=0)
 actual=pd.read_csv(base/f'pca_{n}/normalized_stationary_inputs.csv',index_col=0).to_numpy()
 e2=float(np.max(abs(z-actual)))
 u,s,vt=np.linalg.svd(z-z.mean(0),full_matrices=False);scores=u[:,:5]*s[:5]
 actual=pd.read_csv(base/f'pca_{n}/hmm_pca_input.csv',index_col=0).to_numpy()
 signs=np.where((scores*actual).sum(0)>=0,1,-1);e3=float(np.max(abs(scores*signs-actual)))
 assert x.shape==(137,13) and actual.shape==(137,5)
 assert e1<1e-10 and e2<1e-10 and e3<1e-10
 rows.append({'model':n,'input_shape':list(x.shape),'score_shape':list(actual.shape),'transformed_input_max_error':e1,'standardization_max_error':e2,'independent_SVD_score_max_error':e3,'variance_explained_5PC':float((s[:5]**2).sum()/(s**2).sum())})
(ROOT/'evidence/independent_pca_verification.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
