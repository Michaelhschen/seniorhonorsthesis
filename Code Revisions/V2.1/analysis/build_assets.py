from pathlib import Path
import os,sys,json,ast
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'audit/mpl')
sys.path.insert(0,str(ROOT/'code'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hmmModelComparison as mc
from build_holdout import main as build_holdout_table
A=ROOT/'output/analysis'; M=ROOT/'manuscript'; T=M/'generated';T.mkdir(exist_ok=True)
F=M/'figures'
NAMES=mc.MODEL_NAMES; short=dict(zip(NAMES,['REV','LAG','LAG+NC']))
summary=pd.read_csv(A/'summary.csv')
def esc(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')
def table(name,caption,headers,rows,note,fmt=None):
    lines=[r'\begin{table}[htbp]',r'\centering\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.12}',r'\caption{'+caption+'}',r'\label{tab:'+name+'}',r'\begin{tabular}{'+(fmt or 'l'+'r'*(len(headers)-1))+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    lines+=[' & '.join(str(v) for v in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize '+note+r'\end{minipage}',r'\end{table}']
    (T/(name+'.tex')).write_text('\n'.join(lines),encoding='utf-8')

base=summary.query("configuration=='reproduction_5pc'")
rows=[]
for row in base.itertuples():
    rows.append([row.mode.title(),'Baseline rule' if row.mapping=='legacy_heuristic' else 'Economic profiles',f'{row.RT_L1:.3f}',f'{row.NC_L1:.3f}',f'{row.reduction_percent:.2f}'])
table('inference','Inference and state-correspondence diagnostics',['Inference','Mapping','LAG $L_1$','LAG+NC $L_1$','Reduction (\\%)'],rows,'All 137 months; separately fitted full-sample pipelines. The economic-profile assignment is a retrospective sensitivity, not a validated economic taxonomy. Filtering holds fitted parameters and each declared mapping fixed.','llrrr')
metrics=pd.read_csv(A/'metrics.csv'); rows=[]
for mode in ['smoothed','filtered']:
    d=metrics.query("configuration=='reproduction_5pc' and mapping=='legacy_heuristic' and mode==@mode").set_index('model')
    for key,label in [('L1','$L_1$'),('TV','Total variation'),('RMSE','Soft-vector RMSE'),('Brier_hard_FI','Hard-REV Brier'),('hard_agreement','Hard-REV agreement')]:
        rows.append([mode.title(),label,f'{d.loc[NAMES[1],key]:.3f}',f'{d.loc[NAMES[2],key]:.3f}'])
table('internal','Internal scores under the baseline naming rule',['Inference','Metric','LAG','LAG+NC'],rows,'Brier sums over four states against the hard REV state; RMSE averages over months and states against the soft REV vector.','llrr')
external=[]; plotting={}
benchmark,_=mc.load_external_benchmark(ROOT/'package/source/external_benchmarks.csv')
for mode in ['smoothed','filtered']:
    for name in NAMES:
        raw=pd.read_csv(A/'reproduction_5pc'/f'{name}_{mode}.csv',parse_dates=['date'])
        labels=pd.read_csv(A/'reproduction_5pc'/f'{name}_legacy_labels.csv')
        q=raw[['date']].copy()
        for row in labels.itertuples():q[mc.REGIME_SLUGS[row.regime_label]+'_prob']=raw[f'state_{row.state}_prob']
        row=mc.external_metrics(benchmark,name,q,5,.5,'fixed_contraction');row['mode']=mode
        from sklearn.metrics import roc_auc_score
        joined=benchmark.merge(q,on='date',validate='one_to_one')
        row['legacy_clipped_auc']=roc_auc_score(joined.external_recession,np.clip(joined.contraction_recession_prob,1e-12,1-1e-12))
        y=joined.external_recession.to_numpy(); rawp=joined.contraction_recession_prob.to_numpy()
        pair=rawp[y==1][:,None]-rawp[y==0][None,:]
        assert abs(row['roc_auc']-((pair>0).mean()+.5*(pair==0).mean()))<1e-12
        external.append(row)
        plotting[(name,mode)]=q
ext=pd.DataFrame(external);ext.to_csv(A/'external_metrics.csv',index=False)
rows=[]
for r in ext.itertuples():rows.append([r.mode.title(),short[r.model],r.nber_true_positives,r.nber_false_negatives,r.nber_false_positives,r.nber_true_negatives,f'{r.nber_recession_probability_brier_score:.5f}',f'{r.roc_auc:.3f}'])
table('external','Recession diagnostics under the baseline naming rule',['Inference','Model','TP','FN','FP','TN','Brier','AUC'],rows,'Strict probability $>0.50$; USREC defines March--April 2020 as recession months. There are 137 months, including two recession months. Always-zero probability Brier is $2/137=0.01460$ and always-non-recession accuracy is $135/137=98.54\\%$. LAG+NC misses the sole observed episode; LAG detects both months.','llrrrrrr')
labels={'reproduction_5pc':'Baseline schema','three_pc':'Three PCs','omit_money_ratio':'Omit M1/M2','fixed_capital_level':'Fixed-capital growth in levels','quarterly_no_extra_difference':'No extra quarterly differences','gdp_only':'GDP replacements only','inflation_only':'Inflation replacements only','raw_parser_repair':'Recovered leap-day nowcast'}
rows=[]
for tag,label in labels.items():
    a=summary.query("configuration==@tag and mode=='filtered'").set_index('mapping')
    p=a.loc['economic_profile_descriptive'];rows.append([label,f'{a.loc["legacy_heuristic","reduction_percent"]:.2f}',f'{p.RT_L1:.3f}',f'{p.NC_L1:.3f}',f'{p.reduction_percent:.2f}'])
table('sensitivity','Same-sample filtered specification checks',['Specification','Legacy (\\%)','Profile LAG','Profile LAG+NC','Profile (\\%)'],rows,'Five PCs and four states unless indicated. Each row re-estimates the corresponding pipelines, and thus its REV reference can change. Positive percentages indicate lower LAG+NC disagreement. These are separate specification contrasts, not additive effects.','lrrrr')
factor=pd.read_csv(ROOT/'package/snapshot_diagnostics/factorial_summary.csv')
table('snapshot','Archived data-snapshot and schema sensitivity',['Values','Capital treatment','LAG $L_1$','LAG+NC $L_1$','Reduction (\\%)'],[['Supplied' if r.data_snapshot=='supplied' else 'Comparison','Difference' if r.schema_snapshot=='supplied' else 'Level',f'{r.RT_L1:.3f}',f'{r.NC_L1:.3f}',f'{r.reduction_percent:.2f}'] for r in factor.itertuples()],'Smoothed probabilities and baseline names; highest likelihood among seeds 0--4. These archived diagnostics concern two snapshots, not a replacement of the supplied-data estimate.','llrrr')
profile=pd.read_csv(A/'state_profiles_original_units.csv').query("configuration=='reproduction_5pc'")
table('profiles','Posterior-weighted profiles in common REV economic units',['Model/state','Mass','GDP','CPI','Core PCE','U-rate','PMI','Log VIX'],[[f'{short[r.model]} / {r.state+1}',f'{r.posterior_mass:.1f}',*[f'{getattr(r,c):.2f}' for c in ['GDP_QoQAnn','CPI_YoY','CorePCE_YoY','UNRATE','ISM_PMI','VIX_log']]] for r in profile.itertuples()],'Each profile uses the same revised-data REV outcomes weighted by that model\'s smoothed probabilities. GDP is annualized quarterly growth; CPI/core PCE are year-over-year rates; unemployment is a percentage; PMI is an index. State numbers are local to each model before matching. Mass is posterior expected months, not independent observations.','lrrrrrrr')
boot=pd.read_csv(A/'conditional_block_bootstrap.csv')
table('uncertainty','Conditional paired-loss uncertainty',['Mapping','Block (months)','Mean advantage','2.5\\%','97.5\\%'],[['Baseline rule' if r.mapping=='legacy_heuristic' else 'Economic profiles',r.block_months,f'{r.mean_advantage:.3f}',f'{r.lower95:.3f}',f'{r.upper95:.3f}'] for r in boot.itertuples()],'Circular block bootstrap: 2,000 replications, seed 20261102; paired monthly LAG minus LAG+NC $L_1$ losses, filtered baseline. Intervals condition on fitted paths and mapping; they omit estimation, selection, and mapping uncertainty.','lrrrr')
seed=pd.read_csv(A/'seed_sensitivity.csv');seedrows=[]
for tag,group in seed.groupby('configuration',sort=False):seedrows.append([labels[tag],f'{group.RT_L1.min():.3f}',f'{group.RT_L1.max():.3f}',f'{group.reduction_percent.min():.2f}',f'{group.reduction_percent.max():.2f}'])
table('seeds','Same-seed fit sensitivity',['Specification','Min LAG $L_1$','Max LAG $L_1$','Min (\\%)','Max (\\%)'],seedrows,'Each of seeds 0--4 is imposed on all three pipelines, followed by filtered-profile matching. This is distinct from selecting the highest-likelihood seed separately for each model. Negative percentages mean larger LAG+NC disagreement; small LAG denominators can magnify percentages.','lrrrr')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
periodrows=[]
for ax,mapping,title in zip(axes,['legacy_heuristic','economic_profile_descriptive'],['Baseline state names','Descriptive economic-profile mapping']):
    d=pd.read_csv(A/'reproduction_5pc'/f'losses_filtered_{mapping}.csv',parse_dates=['date'])
    ax.plot(d.date,d.RT_L1,label='LAG',color='#9A4F21',lw=1.3);ax.plot(d.date,d.NC_L1,label='LAG+NC',color='#126D83',lw=1.3)
    ax.set_title(title,loc='left');ax.set_ylabel('L1 disagreement');ax.set_ylim(-.06,2.08);ax.legend(loc='upper right',ncols=2)
    for label,mask in [('2020',d.date.dt.year==2020),('Other years',d.date.dt.year!=2020)]:
        q=d.loc[mask];periodrows.append(dict(mapping=mapping,period=label,months=len(q),RT_L1=q.RT_L1.mean(),NC_L1=q.NC_L1.mean()))
fig.tight_layout();fig.savefig(F/'revised_gaps.png',dpi=200);plt.close(fig)
pd.DataFrame(periodrows).to_csv(A/'pandemic_comparison.csv',index=False)
fig,ax=plt.subplots(figsize=(8,3.6))
for name,color in zip(NAMES,['#53575C','#9A4F21','#126D83']):
    q=plotting[(name,'filtered')];q=q[q.date.between('2019-01-01','2021-12-31')]
    ax.plot(q.date,q.contraction_recession_prob,label=short[name],color=color,lw=1.6)
ax.axhline(.5,color='black',ls='--',lw=.8,label='Threshold 0.50')
ax.axvspan(pd.Timestamp('2020-03-01'),pd.Timestamp('2020-04-30'),color='#BEC4CA',alpha=.4,label='USREC recession')
ax.set_ylabel('Baseline contraction probability');ax.set_ylim(-.03,1.05);ax.legend(fontsize=8,ncols=3,loc='upper center');fig.tight_layout();fig.savefig(F/'revised_recession.png',dpi=200);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,5),sharey=True)
for ax,name in zip(axes,NAMES):
    d=pd.read_csv(A/'reproduction_5pc'/f'{name}_loadings.csv',index_col=0)
    im=ax.imshow(d.to_numpy(),vmin=-.75,vmax=.75,cmap='RdBu_r',aspect='auto')
    ax.set_title(short[name]);ax.set_xticks(range(5),d.columns);ax.set_yticks(range(len(d)),[s.replace('DXY_YoY_diff','Broad dollar YoY diff.').replace('_',' ') for s in d.index],fontsize=8)
fig.subplots_adjust(left=.22,right=.87,wspace=.1)
cax=fig.add_axes([.90,.20,.018,.60]);fig.colorbar(im,cax=cax,label='Loading')
fig.savefig(F/'revised_loadings.png',dpi=200);plt.close(fig)
# Coverage and source metadata are explicitly distinct from historical release verification.
lineage=pd.read_csv(ROOT/'package/availability_audit/selected_nowcast_lineage.csv',parse_dates=['origin','issue_date'])
coverage=[]
for sample,dates in [('Input',pd.date_range('2014-01-31','2025-09-30',freq='ME')),('Estimation',pd.date_range('2014-05-31','2025-09-30',freq='ME'))]:
    for variable in ['GDP_QoQAnn','CPI_YoY','CorePCE_YoY']:
        q=lineage[(lineage.variable==variable)&lineage.origin.isin(dates)]
        coverage.append(dict(sample=sample,variable=variable,months=len(dates),replacements=len(q),fallback=len(dates)-len(q),mean_issue_age_days=q.issue_age_days.mean(),max_issue_age_days=q.issue_age_days.max()))
pd.DataFrame(coverage).to_csv(A/'nowcast_coverage.csv',index=False)
table('coverage','Supplied-history nowcast coverage',['Sample','Series','Months','Replaced','Fallback','Mean age','Max age'],[[r['sample'],{'GDP_QoQAnn':'GDP','CPI_YoY':'CPI','CorePCE_YoY':'Core PCE'}[r['variable']],r['months'],r['replacements'],r['fallback'],f"{r['mean_issue_age_days']:.1f}",r['max_issue_age_days']] for r in coverage],'Age is calendar days since the recorded issue date. Eligibility is verified against supplied histories, not independently archived historical releases.','llrrrrr')
meta=pd.read_excel(ROOT/'package/source/thesis_data_panels.xlsx',sheet_name='SourceMetadata')
meta.to_csv(A/'source_metadata_as_supplied.csv',index=False)
print(ext[['mode','model','nber_true_positives','nber_false_positives','nber_recession_probability_brier_score','roc_auc']].to_string(index=False))
print(pd.DataFrame(coverage).to_string(index=False))
build_holdout_table()


# V2.1: identify the distinct comparisons before interpreting their magnitudes.
principal = [
    ('Separate; smoothed; baseline labels', 'reproduction_5pc', 'smoothed', 'legacy_heuristic'),
    ('Separate; filtered; baseline labels', 'reproduction_5pc', 'filtered', 'legacy_heuristic'),
    ('Separate; filtered; economic profiles', 'reproduction_5pc', 'filtered', 'economic_profile_descriptive'),
    ('Common REV classifier; filtered', 'common_classifier', 'filtered', 'shared_FI_state_coordinates'),
]
rows=[]
for label,configuration,mode,mapping in principal:
    selected=summary[(summary.configuration==configuration)&(summary['mode']==mode)&(summary.mapping==mapping)]
    if len(selected)!=1: raise ValueError(f'Expected one comparison: {label}')
    row=selected.iloc[0]
    rows.append([label,f'{row.RT_L1:.3f}',f'{row.NC_L1:.3f}',f'{row.RT_L1-row.NC_L1:.3f}',f'{row.reduction_percent:.2f}'])
table('principal_comparisons','Principal retrospective comparisons',
      ['Comparison','LAG','LAG+NC','Change','Reduction (\\%)'],rows,
      'Mean $L_1$ probability-vector disagreement with the revised-data reference, '
      'May 2014--September 2025 (137 months). Change is LAG minus LAG+NC loss; '
      'a positive value means closer agreement. All fits use the full sample. '
      'The common-classifier row freezes the REV scaler, PCA, and HMM for every panel; '
      'the other rows refit each panel. These are different conditional comparisons, '
      'not interchangeable estimates or an additive decomposition.',
      '@{}p{0.49\\textwidth}rrrr@{}')

compact=[]
for row in profile.itertuples():
    compact.append([f'{short[row.model]} / S{row.state+1}',f'{row.posterior_mass:.1f}',
                    f'{row.GDP_QoQAnn:.2f}',f'{row.CPI_YoY:.2f}',
                    f'{row.UNRATE:.2f}',f'{row.VIX_log:.2f}'])
table('profiles_compact','Economic profiles and state occupancy',
      ['Panel/state','Mass','GDP','CPI','U-rate','Log VIX'],compact,
      'Common revised-data outcomes weighted by each panel\'s smoothed state probabilities. '
      'Mass is posterior expected months out of 137. GDP is annualized quarterly growth; '
      'CPI is year-over-year inflation; unemployment is a percentage; VIX is in logs. '
      'S1--S4 are local identifiers before matching, not shared economic labels. '
      'Full profiles including core PCE inflation and PMI appear in Appendix C.',
      'lrrrrr')
