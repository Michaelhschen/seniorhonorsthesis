"""Tests for the isolated external-validation correction; all inputs synthetic."""
from __future__ import annotations
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
original=load('comparison_original',ROOT/'code/original/hmmModelComparison.py')
corrected=load('comparison_corrected',ROOT/'code/corrected/hmmModelComparison.py')
EVIDENCE={'scope':'Synthetic regression tests only; no new thesis empirical estimates','cases':{}}

def frames(y,p):
    dates=pd.date_range('2020-01-31',periods=len(y),freq='ME')
    b=pd.DataFrame({'date':dates,'external_recession':y})
    m=pd.DataFrame({'date':dates})
    for c in corrected.REGIME_PROB_COLS:m[c]=0.
    m['contraction_recession_prob']=p;m['expansion_goldilocks_prob']=1-np.asarray(p)
    return b,m

def metric(b,m,module=corrected,threshold=.5,rule='fixed_contraction'):
    return module.external_metrics(b,'synthetic',m,5,threshold,rule)

class ExternalCorrectionTests(unittest.TestCase):
    def test_01_post_episode_crossing_is_not_detection(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.1,.1,.8,.1])
        old=metric(b,m,original);new=metric(b,m)
        self.assertEqual(old['nber_detection_lag_months'],2)
        self.assertTrue(math.isnan(new['nber_detection_lag_months']))
        self.assertEqual(new['nber_detection_status'],'missed')
        self.assertEqual(new['nber_missed_episodes'],1)
        self.assertEqual(new['nber_true_positives'],0)
        self.assertEqual(new['nber_false_negatives'],2)
        self.assertEqual(new['nber_false_positives'],1)
        self.assertEqual(new['nber_true_negatives'],3)
        EVIDENCE['cases']['post_episode_false_alarm']={'old_lag':2,'corrected_lag':None,'corrected_status':new['nber_detection_status'],'TP':0,'FN':2,'FP':1,'TN':3}

    def test_02_same_month_detection_and_balanced_accuracy(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.6,.2,.1,.1]);r=metric(b,m)
        self.assertEqual(r['nber_detection_lag_months'],0)
        self.assertEqual(r['nber_recall'],.5)
        self.assertEqual(r['nber_specificity'],1.)
        self.assertEqual(r['nber_balanced_accuracy'],.75)
        self.assertEqual(r['no_recession_accuracy'],4/6)
        self.assertEqual(r['no_recession_brier_score'],2/6)
        self.assertEqual(r['nber_detected_episodes'],1)

    def test_03_detection_during_second_recession_month(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.2,.7,.1,.1]);r=metric(b,m)
        self.assertEqual(r['nber_detection_lag_months'],1)
        detail=json.loads(r['nber_episode_details'])[0]
        self.assertEqual(detail['first_observed_crossing_month'],'2020-04')
        self.assertTrue(detail['complete_episode'])

    def test_04_multiple_episodes_remain_separate(self):
        b,m=frames([0,1,1,0,0,1,1,0,0],[.1,.1,.1,.8,.1,.1,.8,.1,.1]);r=metric(b,m)
        self.assertEqual(r['nber_episode_count'],2)
        self.assertEqual(r['nber_missed_episodes'],1)
        self.assertEqual(r['nber_detected_episodes'],1)
        self.assertTrue(math.isnan(r['nber_detection_lag_months']))
        details=json.loads(r['nber_episode_details'])
        self.assertEqual([x['status'] for x in details],['missed','detected'])
        self.assertEqual(details[1]['detection_lag_months'],1)
        EVIDENCE['cases']['multiple_episodes']=details

    def test_05_probability_equal_threshold_is_negative(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.5,.5,.1,.1]);r=metric(b,m)
        self.assertEqual(r['nber_true_positives'],0)
        self.assertEqual(r['nber_detection_status'],'missed')
        self.assertEqual(r['nber_threshold_rule'],'probability > threshold')

    def test_06_nonrecession_only_sample(self):
        b,m=frames([0,0,0],[.1,.8,.1]);r=metric(b,m)
        self.assertEqual(r['nber_episode_count'],0)
        self.assertEqual(r['nber_detection_status'],'no_recession_episode')
        self.assertTrue(math.isnan(r['nber_recall']))
        self.assertTrue(math.isnan(r['nber_balanced_accuracy']))
        self.assertTrue(math.isnan(r['roc_auc']))
        self.assertEqual(r['nber_specificity'],2/3)
        self.assertEqual(r['no_recession_accuracy'],1.)
        self.assertEqual(r['no_recession_brier_score'],0.)

    def test_07_recession_only_sample_is_censored(self):
        b,m=frames([1,1,1],[.1,.8,.1]);r=metric(b,m)
        self.assertTrue(math.isnan(r['nber_specificity']))
        self.assertTrue(math.isnan(r['nber_balanced_accuracy']))
        self.assertTrue(math.isnan(r['nber_detection_lag_months']))
        self.assertEqual(r['nber_detection_status'],'detected')
        self.assertEqual(r['nber_incomplete_episodes'],1)
        self.assertEqual(r['no_recession_accuracy'],0.)
        self.assertEqual(r['no_recession_brier_score'],1.)

    def test_08_missing_model_onset_not_relabelled_as_zero_lag(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.1,.8,.1,.1]);m=m.iloc[3:].copy();r=metric(b,m)
        self.assertEqual(r['n_overlap_months'],3)
        self.assertEqual(r['n_benchmark_months_without_model'],3)
        self.assertTrue(math.isnan(r['nber_detection_lag_months']))
        self.assertEqual(r['nber_detection_status'],'detected')
        self.assertEqual(r['nber_incomplete_episodes'],1)
        detail=json.loads(r['nber_episode_details'])[0]
        self.assertEqual(detail['start_month'],'2020-03')
        self.assertEqual(detail['evaluated_recession_months'],1)

    def test_09_incomplete_episode_without_crossing_is_not_definite_miss(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.1,.1,.1,.1]);m=m.drop(index=3);r=metric(b,m)
        self.assertEqual(r['nber_detection_status'],'not_detected_in_observed_part')
        self.assertEqual(r['nber_missed_episodes'],0)
        self.assertEqual(r['nber_incomplete_episodes'],1)
        self.assertEqual(r['n_overlap_months'],5)

    def test_10_nan_probability_fails_without_row_dropping(self):
        b,m=frames([0,1,0],[.1,.1,.1]);m.loc[1,'contraction_recession_prob']=np.nan
        with self.assertRaisesRegex(ValueError,'no evaluation rows were dropped'):metric(b,m)

    def test_11_duplicates_and_nonbinary_indicators_fail(self):
        b,m=frames([0,1,0],[.1,.1,.1])
        with self.assertRaisesRegex(ValueError,'Duplicate month'):metric(b,pd.concat([m,m.iloc[:1]]))
        b.loc[1,'external_recession']=2
        with self.assertRaisesRegex(ValueError,'binary'):metric(b,m)

    def test_12_existing_scores_and_keys_preserved_on_valid_case(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.7,.8,.2,.1]);old=metric(b,m,original);new=metric(b,m)
        self.assertTrue(set(old).issubset(new))
        for key,value in old.items():
            if isinstance(value,float):self.assertAlmostEqual(value,new[key],places=13,msg=key)
            else:self.assertEqual(value,new[key],key)
        counts=sum(new[k] for k in ['nber_true_positives','nber_false_negatives','nber_false_positives','nber_true_negatives'])
        self.assertEqual(counts,new['n_overlap_months'])

    def test_13_unsorted_inputs_have_same_episode_results(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.7,.8,.2,.1]);a=metric(b,m);r=metric(b.iloc[::-1],m.iloc[::-1])
        self.assertEqual(a,r)

    def test_14_empty_overlap_retains_none_return(self):
        b,m=frames([0,1,0],[.1,.1,.1]);m['date']=m['date']+pd.offsets.YearEnd(2)
        # Preserve distinct monthly dates while moving outside the benchmark.
        m['date']=pd.date_range('2022-01-31',periods=3,freq='ME')
        self.assertIsNone(metric(b,m))

    def test_15_invalid_threshold_and_rule_fail_explicitly(self):
        b,m=frames([0,1,0],[.1,.1,.1])
        for t in [-.1,1.1,np.nan]:
            with self.assertRaises(ValueError):metric(b,m,threshold=t)
        with self.assertRaises(ValueError):metric(b,m,rule='unknown')

    def test_16_invalid_dates_and_out_of_range_probability_fail(self):
        b,m=frames([0,1,0],[.1,.1,.1]);bad=b.copy();bad['date']=bad['date'].astype(str);bad.loc[1,'date']='not a date'
        with self.assertRaisesRegex(ValueError,'Invalid date'):metric(bad,m)
        m.loc[1,'contraction_recession_prob']=1.1
        with self.assertRaisesRegex(ValueError,'within'):metric(b,m)

    def test_17_zero_endpoint_does_not_create_positive_via_clipping(self):
        b,m=frames([0,1,0],[0.,0.,0.]);r=metric(b,m,threshold=0.)
        self.assertEqual(r['nber_true_positives'],0)
        self.assertEqual(r['nber_false_positives'],0)

    def test_18_nber_max_diagnostic_remains_available(self):
        b,m=frames([0,1,0],[.1,.2,.1]);r=metric(b,m,rule='nber_max')
        self.assertEqual(r['nber_recession_regime'],'Expansion / Goldilocks')
        self.assertEqual(r['nber_true_positives'],1)
        self.assertEqual(r['external_regime_rule'],'nber_max')

    def test_19_month_start_benchmark_matches_month_end_predictions(self):
        b,m=frames([0,0,1,1,0,0],[.1,.1,.7,.8,.2,.1])
        b['date']=b['date'].dt.to_period('M').dt.to_timestamp()
        self.assertIsNone(metric(b,m,original))
        r=metric(b,m)
        self.assertEqual(r['n_overlap_months'],6)
        self.assertEqual(r['nber_detection_lag_months'],0)
        self.assertEqual(r['external_date_alignment_rule'],'calendar month-end')
        EVIDENCE['cases']['month_start_end_alignment']={'original_overlap':0,'corrected_overlap':6,'observations_dropped':0}

    def test_20_loader_accepts_raw_fred_dates_without_dropping(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path=Path(d)/'fred.csv'
            path.write_text('observation_date,USREC\n2020-01-01,0\n2020-02-01,0\n2020-03-01,1\n')
            result,status=corrected.load_external_benchmark(path)
            self.assertEqual(len(result),3)
            self.assertEqual(result['date'].dt.strftime('%Y-%m-%d').tolist(),['2020-01-31','2020-02-29','2020-03-31'])
            path.write_text('date,external_recession\n2020-01-01,0\n2020-02-01,\n')
            with self.assertRaisesRegex(ValueError,'non-missing'):corrected.load_external_benchmark(path)

    def test_21_downloader_normalizes_and_caches_month_end(self):
        content=b'observation_date,USREC\n2020-01-01,0\n2020-02-01,0\n2020-03-01,1\n'
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            path=Path(d)/'fred.csv'
            with patch.object(corrected.urllib.request,'urlopen',return_value=io.BytesIO(content)):
                result,status=corrected.download_nber_benchmark(path)
            cached=pd.read_csv(path)
            self.assertEqual(cached['date'].tolist(),['2020-01-31','2020-02-29','2020-03-31'])
            self.assertEqual(len(result),3)

    def test_22_start_and_end_dates_in_same_month_are_duplicate(self):
        b,m=frames([0,1,0],[.1,.7,.1])
        extra=b.iloc[:1].copy();extra['date']=pd.Timestamp('2020-01-01')
        with self.assertRaisesRegex(ValueError,'Duplicate month'):metric(pd.concat([b,extra]),m)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExternalCorrectionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    EVIDENCE.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),all_passed=result.wasSuccessful())
    Path(__file__).with_name('external_correction_evidence.json').write_text(json.dumps(EVIDENCE,indent=2,allow_nan=False)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
