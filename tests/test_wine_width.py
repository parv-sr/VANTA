"""Fast, network-independent tests for wine audit and latent-width methodology."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"experiments"/"datasets_ex03"),
               str(ROOT/"experiments"/"rn_interlude")]
import datasets as ex03
from benchmark_datasets import load_task
from wine_audit import feature_groups, make_group_split, overlap_summary, training_descriptors
from latent_dimension_sweep import make_splits, preprocess_fixed_split, BASELINE_WIDTH, DEFAULT_WIDTHS


class TestWineFeatureAudit(unittest.TestCase):
    def synthetic_bundle(self, root, *, rows=180):
        rng=np.random.default_rng(2026)
        groups=np.repeat(np.arange(rows//3),3)
        # Identical predictors in each group and varying quality targets.
        values=np.stack([5+groups*.01, 9+groups*.03, 3+groups%7],axis=1)
        path=Path(root)/"wine_red"
        path.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(dict(
            alcohol=values[:,0],acidity=values[:,1],sulphates=values[:,2],
            quality=(groups%3)+5)).to_csv(path/"winequality-red.csv",index=False)
        return load_task("wine_red",data_root=Path(root))

    def test_group_disjoint_split_keeps_duplicates_together(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle=self.synthetic_bundle(temp)
            labels=feature_groups(bundle)
            self.assertEqual(len(set(labels)),60)
            for seed in [42,43,44]:
                split=make_group_split(bundle,seed)
                sets=[set(labels[split[key]]) for key in ["train","val","test"]]
                self.assertFalse(sets[0]&sets[1])
                self.assertFalse(sets[0]&sets[2])
                self.assertFalse(sets[1]&sets[2])
                self.assertEqual(sum(map(len,split.values())),len(labels))
                row=overlap_summary(split,labels,dataset="wine_red",
                                    split_seed=seed,split_method="feature_group_disjoint")
                self.assertEqual(row["test_rows_with_duplicate_in_train"],0.)
                self.assertEqual(row["val_rows_with_duplicate_in_train"],0.)

    def test_audit_detects_row_split_duplicate_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle=self.synthetic_bundle(temp)
            labels=feature_groups(bundle)
            split=ex03.split_indices(bundle,42)
            result=overlap_summary(split,labels,dataset="wine_red",
                                   split_seed=42,split_method="row_random")
            self.assertGreater(result["test_rows_with_duplicate_in_train"],0.)

    def test_fixed_split_preprocessing_uses_exact_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle=self.synthetic_bundle(temp)
            idx=make_group_split(bundle,111)
            old=ex03.split_indices
            data=preprocess_fixed_split(bundle,idx)
            self.assertIs(ex03.split_indices,old)
            for k in idx:
                np.testing.assert_array_equal(data["splits"][k]["indices"],idx[k])
            means=data["splits"]["train"]["numeric"].mean(axis=0)
            np.testing.assert_allclose(means,np.zeros_like(means),atol=1e-5)
            train_rows,features,pairs=training_descriptors(bundle,idx)
            self.assertEqual(train_rows["train_rows"],len(idx["train"]))
            self.assertEqual(train_rows["num_numeric"],3)
            self.assertEqual(len(features),3)
            self.assertEqual(len(pairs),3)

    def test_reference_width_always_in_sweep(self):
        self.assertIn(BASELINE_WIDTH,DEFAULT_WIDTHS)
        self.assertEqual(sorted(DEFAULT_WIDTHS),list(DEFAULT_WIDTHS))


if __name__=="__main__":
    unittest.main()
