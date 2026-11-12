"""Offline structural tests of the unchanged published thesis code.

Synthetic inputs below are NOT empirical replications of the manuscript.
Run: python3 test_state_metrics.py --repo ../original
"""
from __future__ import annotations
import argparse
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ap = argparse.ArgumentParser()
ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1] / 'code/original'))
args, extra = ap.parse_known_args()
REPO = Path(args.repo).resolve()
# Keep historical characterization tests isolated when pytest has already
# imported the current production modules while collecting another test file.
fixture_names = ['dataCollection', 'dataCleaning', 'hmmModeling',
                 'hmmModelComparison', 'hmmOutOfSampleValidation']
previous_modules = {name: sys.modules.get(name) for name in fixture_names}
previous_path = sys.path.copy()
try:
    sys.path.insert(0, str(REPO))
    for name in fixture_names:
        sys.modules.pop(name, None)
    hm = importlib.import_module('hmmModeling')
    mc = importlib.import_module('hmmModelComparison')
    oos = importlib.import_module('hmmOutOfSampleValidation')
finally:
    sys.path[:] = previous_path
    for name, module in previous_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module
EVIDENCE = {'scope': 'Offline structural tests using synthetic inputs; not manuscript data or a Table 5 replication',
            'commit': ((REPO/'PINNED_COMMIT.txt').read_text().strip() if (REPO/'PINNED_COMMIT.txt').exists() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()),
            'results': {}}

def summary():
    return pd.DataFrame({'state':[0,1,2,3], 'PCA_1_mean':[-3.,-1.,1.,3.],
      'PCA_2_mean':[2.,-2.,1.,0.], 'PCA_3_mean':[0.,1.,0.,-1.],
      'self_transition_probability':[.8,.8,.8,.8]})

def label_map(s):
    return hm.regime_label_table(s).set_index('state')['regime_label'].to_dict()

def probs_frame(dates, contraction):
    contraction = np.asarray(contraction, dtype=float)
    f = pd.DataFrame({'date': dates})
    for c in mc.REGIME_PROB_COLS: f[c] = 0.
    f['contraction_recession_prob'] = contraction
    f['expansion_goldilocks_prob'] = 1-contraction
    return f

class StateMetricAudit(unittest.TestCase):
    def test_01_pc1_sign_changes_semantic_label(self):
        s=summary(); before=label_map(s); s['PCA_1_mean'] *= -1; after=label_map(s)
        self.assertNotEqual(before,after)
        self.assertEqual(before[0],'Contraction / Recession')
        self.assertEqual(after[3],'Contraction / Recession')
        EVIDENCE['results']['pca1_sign_flip']={'original_labels':before,'flipped_labels':after,
          'meaning':'Changing an arbitrary coordinate sign changes the contraction event without changing latent geometry'}

    def test_02_pc2_sign_changes_semantic_label(self):
        s=summary(); before=label_map(s); s['PCA_2_mean'] *= -1; after=label_map(s)
        self.assertEqual(before[1],'Expansion / Goldilocks')
        self.assertEqual(after[2],'Expansion / Goldilocks')
        EVIDENCE['results']['pca2_sign_flip']={'original_expansion_state':1,'flipped_expansion_state':2}

    def test_03_label_rule_ignores_transition_summary(self):
        s=summary(); before=label_map(s)
        s['self_transition_probability']=[.01,.99,.1,.2]
        self.assertEqual(before,label_map(s))
        EVIDENCE['results']['transition_summary_ignored']=True

    def test_04_unique_mean_state_ids_are_permutation_equivariant(self):
        s=summary(); before=label_map(s); permutation={0:2,1:0,2:3,3:1}
        s['state']=s['state'].map(permutation)
        self.assertEqual({permutation[k]:v for k,v in before.items()},label_map(s))
        EVIDENCE['results']['unique_mean_raw_id_permutation_test']='passed; sign dependence is separate from arbitrary numeric IDs'

    def test_05_summary_is_hard_assignment_sample_mean(self):
        model=SimpleNamespace(n_components=2,transmat_=np.eye(2),means_=np.array([[-50.],[50.]]))
        x=pd.DataFrame({'PCA_1':[2.,4.,8.]})
        p=pd.DataFrame({'most_likely_state':[0,0,1], 'state_0_prob':[.8,.7,.2], 'state_1_prob':[.2,.3,.8]})
        s=hm.state_summary_table(model,x,p)
        np.testing.assert_allclose(s['PCA_1_mean'],[3.,8.])
        EVIDENCE['results']['state_summary_means']={'hard_assignment_means':[3.,8.],'model_emission_means':[-50.,50.]}

    def test_06_detection_lag_counts_post_episode_false_alarm(self):
        dates=pd.date_range('2020-01-31',periods=6,freq='ME')
        y=np.array([0,0,1,1,0,0]); p=np.array([.1,.1,.1,.1,.8,.1])
        r=mc.external_metrics(pd.DataFrame({'date':dates,'external_recession':y}), 'synthetic', probs_frame(dates,p),5,.5,'fixed_contraction')
        self.assertEqual(r['nber_detection_lag_months'],2)
        self.assertEqual(int(((p>.5)&(y==1)).sum()),0)
        self.assertEqual(int(((p>.5)&(y==0)).sum()),1)
        EVIDENCE['results']['post_recession_detection']={'reported_lag_months':r['nber_detection_lag_months'],
          'TP':0,'FN':2,'FP':1,'first_crossing':'2020-05-31','recession_end':'2020-04-30',
          'correct_episode_result':'missed episode, not a successful two-month-lag detection'}

    def test_07_threshold_is_strictly_greater(self):
        dates=pd.date_range('2020-03-31',periods=2,freq='ME'); y=np.array([1,0])
        r=mc.external_metrics(pd.DataFrame({'date':dates,'external_recession':y}), 'synthetic', probs_frame(dates,[.5,.1]),5,.5,'fixed_contraction')
        self.assertEqual(r['threshold_recession_accuracy'],.5)
        self.assertTrue(math.isnan(r['nber_detection_lag_months']))
        EVIDENCE['results']['threshold_rule']='p > threshold (0.5 exactly is classified negative)'

    def test_08_published_rounded_metrics_uniquely_determine_counts(self):
        derived={}
        for name,acc,fpr in [('RT',.927,.074),('NC',.971,.015)]:
            matches=[]
            for tp in range(3):
                for fp in range(136):
                    tn=135-fp; fn=2-tp
                    if round((tp+tn)/137,3)==acc and round(fp/135,3)==fpr:
                        matches.append({'TP':tp,'FN':fn,'FP':fp,'TN':tn,'recall':tp/2,
                          'balanced_accuracy':.5*(tp/2+tn/135)})
            self.assertEqual(len(matches),1);derived[name]=matches[0]
        self.assertEqual(derived['NC']['TP'],0)
        EVIDENCE['results']['Table5_derived_counts']={'conditional_on':'137 months, 2 positive months, repository FPR definition, nearest rounding to 3 decimals',
          'not_original_prediction_exports':True,'counts':derived}

    def test_09_external_zero_probability_baseline(self):
        dates=pd.date_range('2014-05-31',periods=137,freq='ME');y=np.zeros(137,dtype=int);y[70:72]=1
        r=mc.external_metrics(pd.DataFrame({'date':dates,'external_recession':y}), 'synthetic', probs_frame(dates,np.zeros(137)),5,.5,'fixed_contraction')
        self.assertAlmostEqual(r['threshold_recession_accuracy'],135/137)
        self.assertAlmostEqual(r['nber_recession_probability_brier_score'],2/137,places=11)
        EVIDENCE['results']['trivial_baseline']={'accuracy':135/137,'Brier':2/137}

    def test_10_internal_metric_bounds_and_hard_target_brier(self):
        date=pd.to_datetime(['2020-01-31']);b=pd.DataFrame({'date':date,'most_likely_regime':['Expansion / Goldilocks']})
        c=pd.DataFrame({'date':date,'most_likely_regime':['Contraction / Recession']})
        for col,val_b,val_c in zip(mc.REGIME_PROB_COLS,[1,0,0,0],[0,0,1,0]):b[col]=val_b;c[col]=val_c
        r=mc.benchmark_metrics(b,c,'synthetic',5)
        self.assertEqual(r['mean_information_gap_l1_vs_full'],2.)
        self.assertEqual(r['benchmark_regime_brier_score'],2.)
        self.assertAlmostEqual(r['probability_rmse_vs_full'],math.sqrt(.5))
        EVIDENCE['results']['internal_scores']='L1 and sum-over-states hard-target Brier attain 2 correctly; Brier above 1 is not an error'

    def test_11_training_scaler_does_not_use_holdout(self):
        train=pd.DataFrame({'a':[0.,2.,4.],'b':[3.,6.,9.]}); hold=pd.DataFrame({'a':[6.,8.],'b':[12.,15.]})
        t1,h1,s1=oos.zscore_with_training_stats(train,hold)
        t2,h2,s2=oos.zscore_with_training_stats(train,hold+1000.)
        pd.testing.assert_frame_equal(t1,t2);pd.testing.assert_frame_equal(s1,s2)
        np.testing.assert_allclose(s1['training_mean'],[2.,6.])
        EVIDENCE['results']['existing_holdout_scaler']='passed training-statistics invariance test'

    def test_12_existing_recursive_filter_and_prior_match_reference(self):
        class M:
            startprob_=np.array([.6,.4]);transmat_=np.array([[.8,.2],[.3,.7]])
            def _compute_log_likelihood(self,x):return np.log(np.asarray(x))
        m=M(); train=np.array([[.2,.7],[.8,.1]]); hold=np.array([[.3,.5],[.9,.2],[.4,.8]])
        f,pri=oos.recursive_filtered_probabilities(m,train,hold)
        alpha=m.startprob_*train[0];alpha/=alpha.sum()
        for emission in train[1:]:alpha=(alpha@m.transmat_)*emission;alpha/=alpha.sum()
        expected_f=[];expected_p=[]
        for emission in hold:
            prior=alpha@m.transmat_;expected_p.append(prior.copy());alpha=prior*emission;alpha/=alpha.sum();expected_f.append(alpha.copy())
        np.testing.assert_allclose(f,expected_f,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(pri,expected_p,rtol=1e-13,atol=1e-13)
        altered=hold.copy();altered[-1]=[.99,.01]
        f2,p2=oos.recursive_filtered_probabilities(m,train,altered)
        np.testing.assert_allclose(f[:2],f2[:2],atol=1e-13)
        EVIDENCE['results']['existing_holdout_filter']={'reference_recursion_max_abs_error':float(np.max(np.abs(f-np.asarray(expected_f)))),
          'prior_max_abs_error':float(np.max(np.abs(pri-np.asarray(expected_p)))),'future_observation_invariance':'passed'}

    def test_13_parameter_count(self):
        self.assertEqual(hm.parameter_count(4,5,'diag'),55)
        self.assertEqual(hm.parameter_count(4,5,'full'),95)
        EVIDENCE['results']['parameter_count']={'diag_4_states_5_features':55,'full_4_states_5_features':95}

    def test_14_optional_three_state_label_attachment_is_incomplete(self):
        s=summary().iloc[:3].copy(); labels=hm.regime_label_table(s)
        with self.assertRaises(KeyError):
            hm.attach_regime_labels(pd.DataFrame({'date':[pd.Timestamp('2020-01-31')], 'most_likely_state':[0]}),s,labels)
        EVIDENCE['results']['non_four_state_output_limitation']='Generic labels omit target_profile, but attach_regime_labels requires it; this is an optional non-four-state path, not the baseline 4-state result'

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(StateMetricAudit)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    EVIDENCE['tests_run']=result.testsRun;EVIDENCE['failures']=len(result.failures);EVIDENCE['errors']=len(result.errors)
    EVIDENCE['all_passed']=result.wasSuccessful()
    out=Path(__file__).with_name('state_metrics_evidence.json')
    out.write_text(json.dumps(EVIDENCE,indent=2,allow_nan=False)+'\n')
    print('Evidence written to',out)
    sys.exit(0 if result.wasSuccessful() else 1)
