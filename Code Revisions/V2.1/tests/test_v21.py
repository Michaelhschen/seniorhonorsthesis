"""Synthetic regressions for V2.1 code changes; run with Python unittest."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import hmmModeling as hm
import hmmModelComparison as mc
import hmmOutOfSampleValidation as oos


def fixture():
    labels = pd.DataFrame({"state": range(4), "regime_label": mc.REGIME_LABELS})
    p = pd.DataFrame({"date": pd.date_range("2020-01-31", periods=2, freq="ME"),
                      "most_likely_state": [0, 1], "most_likely_regime": mc.REGIME_LABELS[:2],
                      **{f"state_{i}_prob": v for i, v in enumerate([[.7, .1], [.1, .7], [.1, .1], [.1, .1]])}})
    return p, labels


class ExtremeModel:
    """Zero-supported alternative has much higher emissions; cannot enter it."""
    n_components = 2
    startprob_ = np.array([1., 0.])
    transmat_ = np.eye(2)

    def _compute_log_likelihood(self, x):
        return np.tile([-1000., 0.], (len(x), 1))


class Revision21Tests(unittest.TestCase):
    def test_generic_state_labels_complete_output_schema(self):
        for states, columns in [(2, []), (3, ["PCA_1_mean"]), (4, ["PCA_1_mean", "PCA_2_mean"])]:
            with self.subTest(states=states):
                summary = pd.DataFrame({"state": range(states), **{c: np.arange(states) for c in columns}})
                labels = hm.regime_label_table(summary)
                probs = pd.DataFrame({"date": ["2020-01-31"], "most_likely_state": [0]})
                p, s = hm.attach_regime_labels(probs, summary, labels)
                self.assertEqual(p.most_likely_regime.iloc[0], "Regime 0")
                self.assertTrue(s.target_profile.notna().all())

    def test_oos_mapping_preserves_mass_and_metadata(self):
        p, labels = fixture()
        p["train_end"] = "2019-12-31"
        for state in range(4):
            p[f"state_{state}_prior_prob"] = .25
        out = oos.add_regime_probability_columns(p, labels)
        np.testing.assert_array_equal(out[mc.REGIME_PROB_COLS], p[[f"state_{s}_prob" for s in range(4)]])
        self.assertEqual(out.train_end.tolist(), p.train_end.tolist())
        np.testing.assert_array_equal(out[[f"state_{s}_prior_prob" for s in range(4)]], .25 * np.ones((2, 4)))

    def test_oos_rejects_malformed_mapping_and_probability_exports(self):
        for defect in ["missing", "duplicate", "fractional", "unknown_label", "duplicate_label",
                       "missing_prob", "extra_prob", "negative", "nan", "sum", "date", "hard_state", "hard_label"]:
            with self.subTest(defect=defect):
                p, labels = fixture()
                if defect == "missing": labels = labels.iloc[:3]
                elif defect == "duplicate": labels.loc[3, "state"] = 2
                elif defect == "fractional":
                    labels["state"] = labels.state.astype(float)
                    labels.loc[3, "state"] = 2.5
                elif defect == "unknown_label": labels.loc[3, "regime_label"] = "Unknown"
                elif defect == "duplicate_label": labels.loc[3, "regime_label"] = labels.loc[0, "regime_label"]
                elif defect == "missing_prob": p = p.drop(columns="state_3_prob")
                elif defect == "extra_prob": p["state_4_prob"] = 0.
                elif defect == "negative": p.loc[0, "state_3_prob"] = -.1
                elif defect == "nan": p.loc[0, "state_3_prob"] = np.nan
                elif defect == "sum": p.loc[0, "state_3_prob"] = .2
                elif defect == "date": p.loc[1, "date"] = p.loc[0, "date"]
                elif defect == "hard_state": p.loc[0, "most_likely_state"] = 3
                elif defect == "hard_label": p.loc[0, "most_likely_regime"] = "Unknown"
                with self.assertRaises(ValueError):
                    oos.add_regime_probability_columns(p, labels)

    def test_oos_filter_preserves_impossible_transition_support(self):
        filtered, priors = oos.recursive_filtered_probabilities(ExtremeModel(), np.zeros((2, 1)), np.zeros((3, 1)))
        np.testing.assert_array_equal(filtered, np.tile([1., 0.], (3, 1)))
        np.testing.assert_array_equal(priors, np.tile([1., 0.], (3, 1)))

    def test_oos_filter_continues_common_forward_recursion(self):
        from hmmlearn.hmm import GaussianHMM
        model = GaussianHMM(n_components=2, covariance_type="diag", init_params="")
        model.n_features = 1
        model.startprob_ = np.array([.4, .6])
        model.transmat_ = np.array([[.95, .05], [.1, .9]])
        model.means_ = np.array([[-1.], [1.]])
        model.covars_ = np.array([[.5], [.5]])
        x = np.array([[-1.], [0.], [1.], [2.], [-2.]])
        actual, prior = oos.recursive_filtered_probabilities(model, x[:2], x[2:])
        expected = np.array([model.predict_proba(x[:t+1])[-1] for t in range(len(x))])
        np.testing.assert_allclose(actual, expected[2:], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(prior, expected[1:-1] @ model.transmat_, atol=1e-12, rtol=1e-12)

    def test_oos_invalid_observations_fail(self):
        for train, holdout in [(np.empty((0, 1)), np.zeros((2, 1))),
                               (np.zeros((2, 1)), np.empty((0, 1))),
                               (np.zeros((2, 1)), np.zeros((2, 2))),
                               (np.zeros((2, 1)), np.array([[np.inf]]))]:
            with self.subTest(train=train.shape, holdout=holdout.shape):
                with self.assertRaises(ValueError):
                    oos.recursive_filtered_probabilities(ExtremeModel(), train, holdout)


if __name__ == "__main__":
    unittest.main(testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=2))
