"""Offline, versioned retrospective diagnostics. Run from any working directory."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
from pathlib import Path
import sys, json, hashlib, itertools, warnings, logging, io, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import hmmModeling as hm
import hmmModelComparison as mc
SPEC = json.loads((ROOT/'config/analysis_spec.json').read_text())
OUT = ROOT/'output/analysis'
OUT.mkdir(parents=True, exist_ok=True)
PKG = ROOT/'package'
BASE = PKG/'original_run/outputs'
NAMES = ['full_information','realtime_information','nowcast_enhanced']
FILES = ['full_information.csv','realtime_information.csv','nowcast_information.csv']
PANELS = {n:pd.read_csv(BASE/f,index_col=0,parse_dates=True) for n,f in zip(NAMES,FILES)}
SCHEMA = pd.read_csv(ROOT/'config/reproduction_schema.csv').fillna('')
# Fix dates once, including initial lag and first difference, for all variants.
aligned = PANELS[NAMES[0]][SCHEMA.source_variable].dropna().index
for frame in PANELS.values():
    assert frame.index.equals(pd.date_range(frame.index.min(),frame.index.max(),freq='ME'))
    aligned = aligned.intersection(frame[SCHEMA.source_variable].dropna().index)
DATES = aligned[1:]
assert len(DATES) == 137
ANCHORS = SPEC['anchors']
WEIGHTS = np.array(SPEC['anchor_weights'])
RAW = PANELS[NAMES[0]].loc[DATES,ANCHORS]
ANCHOR_Z = (RAW - RAW.mean())/RAW.std(ddof=0)
PERMS = list(itertools.permutations(range(4)))
ledger=[]; summaries=[]; profiles=[]; mappings=[]; metrics=[]; uncertainty=[]; parity=[]; seedscores=[]

def transform(frame,schema):
    x=pd.DataFrame(index=aligned)
    for row in schema.itertuples():
        if row.decision=='keep_level': x[row.output_variable]=frame.loc[aligned,row.source_variable]
        elif row.decision=='keep_difference': x[row.output_variable]=frame.loc[aligned,row.source_variable].diff()
        elif not row.decision.startswith('drop_'): raise ValueError(row.decision)
    return x.loc[DATES]

def fit(x,tag,name,seed):
    model=GaussianHMM(n_components=4,covariance_type='diag',n_iter=1000,tol=1e-4,random_state=seed,min_covar=1e-4)
    stream=io.StringIO(); handler=logging.StreamHandler(stream)
    logger=logging.getLogger('hmmlearn.base'); logger.addHandler(handler)
    row=dict(configuration=tag,model=name,seed=seed)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always'); model.fit(x); ll=float(model.score(x))
        hist=list(model.monitor_.history)
        changes=np.diff(hist)
        p=model.predict_proba(x)
        row.update(log_likelihood=ll,iterations=model.monitor_.iter,
            monitor_converged=bool(model.monitor_.converged),
            hit_iteration_cap=model.monitor_.iter>=model.n_iter,
            final_likelihood_change=float(changes[-1]) if len(changes) else None,
            minimum_likelihood_change=float(changes.min()) if len(changes) else None,
            min_diagonal_variance=float(model._covars_.min()),
            max_transition_row_error=float(abs(model.transmat_.sum(1)-1).max()),
            effective_occupancy=p.sum(0).tolist(), hard_occupancy=np.bincount(p.argmax(1),minlength=4).tolist(),
            means=model.means_.tolist(), warnings=[str(w.message) for w in captured],
            likelihood_history=hist, error='')
        row['numerical_review_flag']=bool(not np.isfinite(ll) or (changes < -1e-3).any() or row['hit_iteration_cap'] or model._covars_.min()<1e-8 or p.sum(0).min()<1)
    except Exception as e:
        ll=-np.inf; row.update(error=repr(e),log_likelihood=None,numerical_review_flag=True)
    finally:
        logger.removeHandler(handler)
    row['log_messages']=stream.getvalue(); ledger.append(row)
    return model,ll

def profile(p):
    occupancy=p.sum(0)
    if (occupancy<=0).any(): raise ValueError('Empty state profile')
    return (p.T@ANCHOR_Z.to_numpy())/occupancy[:,None]

def assignment(a,b,weights=WEIGHTS):
    costs=((a[:,None,:]-b[None,:,:])**2*weights).sum(2)
    scores=np.array([sum(costs[i,perm[i]] for i in range(4)) for perm in PERMS])
    order=np.argsort(scores,kind='stable')
    return PERMS[order[0]],float(scores[order[0]]),float(scores[order[1]]-scores[order[0]])

def losses(a,b): return np.abs(a-b).sum(1)

def scores(a,b):
    target=np.eye(4)[a.argmax(1)]
    return dict(L1=float(losses(a,b).mean()),TV=float(losses(a,b).mean()/2),
        RMSE=float(np.sqrt(((a-b)**2).mean())),Brier_hard_FI=float(((target-b)**2).sum(1).mean()),
        hard_agreement=float((a.argmax(1)==b.argmax(1)).mean()))

def mapped(probs,labels):
    bylabel=dict(zip(labels.regime_label,labels.state))
    return probs[:,[int(bylabel[label]) for label in mc.REGIME_LABELS]]

for tag in SPEC['configurations']:
    print('RUN',tag,flush=True)
    dest=OUT/tag;dest.mkdir(exist_ok=True)
    schema=SCHEMA.copy(); panels={n:f.copy() for n,f in PANELS.items()}
    if tag=='raw_parser_repair':
        panels[NAMES[2]]=pd.read_csv(ROOT/'output/raw_archive_audit/nowcast_information.csv',index_col=0,parse_dates=True)
    if tag=='omit_money_ratio': schema.loc[schema.source_variable=='M1M2_Ratio','decision']='drop_ablation'
    if tag in ['fixed_capital_level','quarterly_no_extra_difference']:
        variables=['CorpFixedCap_QoQ'] if tag=='fixed_capital_level' else ['CorpFixedCap_QoQ','CorpCapex_QoQ','DelinqChange']
        schema.loc[schema.source_variable.isin(variables),'decision']='keep_level'
    if tag in ['gdp_only','inflation_only']:
        use=['GDP_QoQAnn'] if tag=='gdp_only' else ['CPI_YoY','CorePCE_YoY']
        panels[NAMES[2]]=PANELS[NAMES[1]].copy()
        panels[NAMES[2]][use]=PANELS[NAMES[2]][use]
    schema['output_variable']=[row.source_variable if row.decision=='keep_level' else row.source_variable+'_diff' if row.decision=='keep_difference' else '' for row in schema.itertuples()]
    schema.to_csv(dest/'schema.csv',index=False)
    r=3 if tag=='three_pc' else 5
    fitted={}; seedfit={}; transforms={}; original_labels={}
    for name in NAMES:
        x=transform(panels[name],schema)
        scaler=StandardScaler(); z=scaler.fit_transform(x)
        pca=PCA(n_components=r); factors=pca.fit_transform(z)
        transforms[name]=(scaler,pca,x)
        pd.DataFrame(pca.components_.T,index=x.columns,columns=[f'PC{i+1}' for i in range(r)]).to_csv(dest/f'{name}_loadings.csv')
        pd.DataFrame({'variable':x.columns,'mean':scaler.mean_,'std_ddof0':scaler.scale_}).to_csv(dest/f'{name}_scaling.csv',index=False)
        pd.DataFrame({'PC':np.arange(1,r+1),'explained_variance_ratio':pca.explained_variance_ratio_}).to_csv(dest/f'{name}_variance.csv',index=False)
        candidates=[fit(factors,tag,name,seed) for seed in SPEC['seeds']]
        eligible=[i for i,(_,ll) in enumerate(candidates) if np.isfinite(ll)]
        if not eligible: raise RuntimeError(f'No finite fit: {tag}/{name}')
        chosen=max(eligible,key=lambda i:candidates[i][1]); model=candidates[chosen][0]
        for row in ledger[-len(candidates):]: row['selected']=row['seed']==SPEC['seeds'][chosen]
        raw={mode:hm.state_probability_table(DATES,model,factors,mode) for mode in ['smoothed','filtered']}
        labels=hm.regime_label_table(hm.state_summary_table(model,pd.DataFrame(factors,index=DATES,columns=[f'PCA_{i+1}' for i in range(r)]),raw['smoothed']))
        labels.to_csv(dest/f'{name}_legacy_labels.csv',index=False)
        original_labels[name]=labels
        fitted[name]={mode:frame[[f'state_{i}_prob' for i in range(4)]].to_numpy() for mode,frame in raw.items()}
        seedfit[name]=[(hm.filtered_state_probabilities(m,factors) if np.isfinite(ll) else None) for m,ll in candidates]
        for mode,frame in raw.items(): frame.to_csv(dest/f'{name}_{mode}.csv',index=False)
        np.savez(dest/f'{name}_parameters.npz',startprob=model.startprob_,transmat=model.transmat_,means=model.means_,covars=model.covars_)
        if tag=='reproduction_5pc':
            old=pd.read_csv(BASE/f'hmm_models_{name}/hmm_state_probabilities_pca5.csv')
            err=float(abs(fitted[name]['smoothed']-old[[f'state_{i}_prob' for i in range(4)]].to_numpy()).max())
            parity.append(dict(model=name,max_probability_error=err,selected_seed=chosen))
            if err>1e-8: raise AssertionError(f'Archived fit mismatch {name}: {err}')
        for state in range(4):
            p=fitted[name]['smoothed'][:,state]; occupancy=p.sum()
            profiles.append(dict(configuration=tag,model=name,state=state,posterior_mass=float(occupancy),
                hard_months=int((fitted[name]['smoothed'].argmax(1)==state).sum()),
                **dict(zip(ANCHORS,(p@RAW.to_numpy()/occupancy).tolist()))))
        if name==NAMES[0]: fi_model=model
    # One common model diagnostic uses the FI scaler/PCA/HMM without refitting.
    if tag=='reproduction_5pc':
        sc,pc,_=transforms[NAMES[0]]
        common={n:hm.filtered_state_probabilities(fi_model,pc.transform(sc.transform(transforms[n][2]))) for n in NAMES}
        for n in NAMES[1:]: metrics.append(dict(configuration='common_classifier',mode='filtered',mapping='shared_FI_state_coordinates',model=n,**scores(common[NAMES[0]],common[n])))
        rt=losses(common[NAMES[0]],common[NAMES[1]]).mean();nc=losses(common[NAMES[0]],common[NAMES[2]]).mean()
        summaries.append(dict(configuration='common_classifier',mode='filtered',mapping='shared_FI_state_coordinates',RT_L1=rt,NC_L1=nc,reduction_percent=100*(1-nc/rt)))
    neutral={n:profile(fitted[n]['smoothed']) for n in NAMES}
    maps={NAMES[0]:tuple(range(4))}
    for name in NAMES[1:]:
        perm,cost,margin=assignment(neutral[NAMES[0]],neutral[name]);maps[name]=perm
        alt=[]
        for j in range(len(ANCHORS)):
            w=WEIGHTS.copy();w[j]=0;w/=w.sum()
            alt.append(assignment(neutral[NAMES[0]],neutral[name],w)[0])
        mappings.append(dict(configuration=tag,model=name,FI_to_candidate=list(perm),cost=cost,second_best_cost_gap=margin,
            leave_one_anchor_out_changed=sum(p!=perm for p in alt),leave_one_anchor_out_maps=alt))
    for mode in ['smoothed','filtered']:
        for mapping in ['legacy_heuristic','economic_profile_descriptive']:
            prob={n:(mapped(fitted[n][mode],original_labels[n]) if mapping=='legacy_heuristic' else fitted[n][mode][:,maps[n]]) for n in NAMES}
            for name in NAMES[1:]: metrics.append(dict(configuration=tag,mode=mode,mapping=mapping,model=name,**scores(prob[NAMES[0]],prob[name])))
            rt=losses(prob[NAMES[0]],prob[NAMES[1]]);nc=losses(prob[NAMES[0]],prob[NAMES[2]])
            summaries.append(dict(configuration=tag,mode=mode,mapping=mapping,RT_L1=rt.mean(),NC_L1=nc.mean(),reduction_percent=100*(1-nc.mean()/rt.mean())))
            pd.DataFrame({'date':DATES,'RT_L1':rt,'NC_L1':nc,'paired_advantage':rt-nc}).to_csv(dest/f'losses_{mode}_{mapping}.csv',index=False)
            if tag=='reproduction_5pc' and mode=='filtered':
                diff=rt-nc
                for block in SPEC['bootstrap']['circular_block_lengths']:
                    rng=np.random.default_rng(SPEC['bootstrap']['seed'])
                    starts=rng.integers(0,len(diff),size=(SPEC['bootstrap']['replications'],int(np.ceil(len(diff)/block))))
                    indices=((starts[:,:,None]+np.arange(block))%len(diff)).reshape(len(starts),-1)[:,:len(diff)]
                    ci=np.quantile(diff[indices].mean(1),[.025,.975])
                    uncertainty.append(dict(mapping=mapping,block_months=block,mean_advantage=diff.mean(),lower95=ci[0],upper95=ci[1],scope='conditional on fitted paths and mapping; no refitting uncertainty'))
    for i,seed in enumerate(SPEC['seeds']):
        p={n:seedfit[n][i] for n in NAMES}
        if any(v is None for v in p.values()): continue
        fp=profile(p[NAMES[0]])
        a={n:(p[n] if n==NAMES[0] else p[n][:,assignment(fp,profile(p[n]))[0]]) for n in NAMES}
        rt=losses(a[NAMES[0]],a[NAMES[1]]).mean();nc=losses(a[NAMES[0]],a[NAMES[2]]).mean()
        seedscores.append(dict(configuration=tag,seed=seed,RT_L1=rt,NC_L1=nc,reduction_percent=100*(1-nc/rt),mapping='filtered_profile_same_seed_descriptive'))

for filename,data in [('summary',summaries),('metrics',metrics),('state_profiles_original_units',profiles),('mapping_diagnostics',mappings),('conditional_block_bootstrap',uncertainty),('reproduction_parity',parity),('seed_sensitivity',seedscores)]:
    pd.DataFrame(data).to_csv(OUT/f'{filename}.csv',index=False)
(OUT/'fit_ledger.json').write_text(json.dumps(ledger,indent=2,allow_nan=False),encoding='utf-8')
pd.DataFrame([{k:v for k,v in row.items() if k not in ['means','likelihood_history','warnings','effective_occupancy','hard_occupancy']} for row in ledger]).to_csv(OUT/'fit_ledger_summary.csv',index=False)
versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','hmmlearn','statsmodels','matplotlib']}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
inputs=[ROOT/'config/analysis_spec.json',ROOT/'config/reproduction_schema.csv',PKG/'source/thesis_data_panels.xlsx',ROOT/'output/raw_archive_audit/nowcast_information.csv',*list((ROOT/'raw_source').glob('*')),*list((ROOT/'code').glob('*.py')),*list((ROOT/'analysis').glob('*.py'))]
manifest=dict(design=SPEC,versions=versions,python=sys.version,dates=[str(DATES.min().date()),str(DATES.max().date())],months=len(DATES),
    inputs={str(p.relative_to(ROOT)):sha(p) for p in inputs},outputs={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'})
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(pd.DataFrame(summaries).to_string(index=False))
