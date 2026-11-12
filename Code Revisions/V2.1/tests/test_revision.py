from pathlib import Path
import sys, itertools
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
import hmmModelComparison as mc

def fixture(tmp_path):
    labels=pd.DataFrame({'state':range(4),'regime_label':mc.REGIME_LABELS})
    p=pd.DataFrame({'date':['2020-01-31','2020-02-29'],'most_likely_state':[0,1],
        'most_likely_regime':mc.REGIME_LABELS[:2],**{f'state_{i}_prob':v for i,v in enumerate([[.7,.1],[.1,.7],[.1,.1],[.1,.1]])}})
    return labels,p

@pytest.mark.parametrize('defect',['missing','duplicate','fractional','unknown_label','duplicate_label','missing_prob','extra_prob','negative','nan','sum','date','hard_state','hard_label'])
def test_malformed_mapping_fails(tmp_path,defect):
    labels,p=fixture(tmp_path)
    if defect=='missing': labels=labels.iloc[:3]
    elif defect=='duplicate': labels.loc[3,'state']=2
    elif defect=='fractional': labels['state']=labels.state.astype(float);labels.loc[3,'state']=2.5
    elif defect=='unknown_label': labels.loc[3,'regime_label']='Unknown'
    elif defect=='duplicate_label': labels.loc[3,'regime_label']=labels.loc[0,'regime_label']
    elif defect=='missing_prob': p=p.drop(columns='state_3_prob')
    elif defect=='extra_prob': p['state_4_prob']=0
    elif defect=='negative': p.loc[0,'state_3_prob']=-.1
    elif defect=='nan': p.loc[0,'state_3_prob']=np.nan
    elif defect=='sum': p.loc[0,'state_3_prob']=.2
    elif defect=='date': p.loc[1,'date']=p.loc[0,'date']
    elif defect=='hard_state': p.loc[0,'most_likely_state']=3
    elif defect=='hard_label': p.loc[0,'most_likely_regime']='Unknown'
    labels.to_csv(tmp_path/'hmm_regime_labels_pca5.csv',index=False)
    p.to_csv(tmp_path/'hmm_state_probabilities_pca5.csv',index=False)
    with pytest.raises(ValueError): mc.regime_probability_frame(tmp_path,5)

def test_valid_mapping_preserves_mass(tmp_path):
    labels,p=fixture(tmp_path)
    labels.to_csv(tmp_path/'hmm_regime_labels_pca5.csv',index=False)
    p.to_csv(tmp_path/'hmm_state_probabilities_pca5.csv',index=False)
    q=mc.regime_probability_frame(tmp_path,5)
    np.testing.assert_array_equal(q[mc.REGIME_PROB_COLS],p[[f'state_{i}_prob' for i in range(4)]])

@pytest.mark.parametrize('name',mc.MODEL_NAMES)
def test_independent_gaussian_forward_recursion(name):
    base=ROOT/'output/analysis/reproduction_5pc'
    a=np.load(base/f'{name}_parameters.npz')
    x=pd.read_csv(ROOT/'package/original_run/outputs'/f'pca_{name}/hmm_pca_input.csv')[[f'PCA_{i+1}' for i in range(5)]].to_numpy()
    ll=np.column_stack([multivariate_normal.logpdf(x,mean=a['means'][k],cov=a['covars'][k]) for k in range(4)])
    with np.errstate(divide='ignore'): start=np.log(a['startprob']);transition=np.log(a['transmat'])
    rows=[]
    for i in range(len(x)):
        alpha=(start if i==0 else logsumexp(alpha[:,None]+transition,axis=0))+ll[i]
        alpha-=logsumexp(alpha); rows.append(np.exp(alpha))
    actual=pd.read_csv(base/f'{name}_filtered.csv')[[f'state_{i}_prob' for i in range(4)]].to_numpy()
    np.testing.assert_allclose(rows,actual,atol=1e-10,rtol=1e-10)

def test_scores_preserve_extreme_probability_ranking():
    q=pd.DataFrame({'date':pd.date_range('2020-01-31',periods=2,freq='ME')})
    for c in mc.REGIME_PROB_COLS:q[c]=0.
    q['contraction_recession_prob']=[1e-15,2e-15]
    q['expansion_goldilocks_prob']=1-q.contraction_recession_prob
    b=q[['date']].copy();b['external_recession']=[0,1]
    a=mc.external_metrics(b,'test',q,5,.5,'fixed_contraction')
    assert a['roc_auc']==1.0
    assert a['nber_recession_probability_brier_score']==np.mean((np.array([1e-15,2e-15])-[0,1])**2)
