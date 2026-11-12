"""Synthetic regression tests only; not estimates from the thesis data."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module
fixed=load('corrected_hmm',ROOT/'code/corrected/hmmModeling.py')
original=load('original_hmm',ROOT/'code/original/hmmModeling.py')

def model(covariance_type='diag'):
    m=GaussianHMM(n_components=2,covariance_type=covariance_type,init_params='')
    m.n_features=2;m.startprob_=np.array([.5,.5]);m.transmat_=np.array([[.97,.03],[.03,.97]])
    m.means_=np.array([[-1.,-.4],[1.,.4]])
    m.covars_={'diag':np.full((2,2),.6),'spherical':np.full(2,.6),
               'full':np.repeat((np.eye(2)*.6)[None],2,axis=0),'tied':np.eye(2)*.6}[covariance_type]
    return m

@pytest.mark.parametrize('kind',['diag','full','tied','spherical'])
def test_forward_recursion_matches_terminal_prefix_posteriors(kind):
    m=model(kind);x=np.random.default_rng(41).normal(size=(17,2))
    actual=fixed.filtered_state_probabilities(m,x)
    expected=np.array([m.predict_proba(x[:t+1])[-1] for t in range(len(x))])
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)

@pytest.mark.parametrize('kind',['diag','full','tied','spherical'])
def test_future_invariance_and_normalization(kind):
    m=model(kind);x=np.random.default_rng(40).normal(size=(20,2));y=x.copy();y[8:]=12
    a=fixed.filtered_state_probabilities(m,x);b=fixed.filtered_state_probabilities(m,y)
    np.testing.assert_array_equal(a[:8],b[:8]);np.testing.assert_allclose(a.sum(1),1,atol=1e-12)
    assert np.isfinite(a).all() and (a>=0).all()

def test_original_probabilities_change_with_future_emissions():
    m=model();x=np.array([[0.,0.],[2.,1.],[2.,1.]])
    a=original.state_probability_table(pd.RangeIndex(1),m,x[:1]).filter(regex='_prob$').to_numpy()
    b=original.state_probability_table(pd.RangeIndex(3),m,x).filter(regex='_prob$').to_numpy()
    assert abs(a[0,0]-b[0,0])>.4
    c=fixed.filtered_state_probabilities(m,x)
    np.testing.assert_allclose(c[0],a[0],atol=1e-12)

def test_explicit_smoothed_option_matches_legacy():
    m=model();x=np.random.default_rng(3).normal(size=(10,2));dates=pd.RangeIndex(10)
    a=original.state_probability_table(dates,m,x)
    b=fixed.state_probability_table(dates,m,x,'smoothed')
    pd.testing.assert_frame_equal(a,b)

def test_exact_zero_probabilities_remain_zero():
    m=model();m.startprob_=np.array([1.,0.]);m.transmat_=np.eye(2)
    x=np.full((10,2),1e3);p=fixed.filtered_state_probabilities(m,x)
    np.testing.assert_array_equal(p,np.tile([1.,0.],(10,1)))

@pytest.mark.parametrize('x',[np.empty((0,2)),np.zeros(2),np.array([[np.nan,0.]]),np.array([[np.inf,0.]])])
def test_invalid_observations_rejected(x):
    with pytest.raises(ValueError):fixed.filtered_state_probabilities(model(),x)

def test_invalid_method_rejected():
    with pytest.raises(ValueError):fixed.state_probability_table(pd.RangeIndex(2),model(),np.zeros((2,2)),'invalid')

if __name__=='__main__':
    m=model();x=np.array([[0.,0.],[2.,1.],[2.,1.]])
    report={'scope':'synthetic fixed-parameter demonstration, not thesis estimates',
            'original_first_probability_one_observation':m.predict_proba(x[:1])[0].tolist(),
            'original_first_probability_with_future_observations':m.predict_proba(x)[0].tolist(),
            'corrected_first_probability_with_future_observations':fixed.filtered_state_probabilities(m,x)[0].tolist()}
    (ROOT/'filter_audit/filtering_evidence.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
