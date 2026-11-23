"""Network-free protocol tests and real-model synthetic integration tests for EX06."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_test_ex06_runner", ROOT / "experiments/ex06/run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)
protocol = experiment.protocol


def original_split(bundle, seed):
    train, rest = train_test_split(np.arange(len(bundle["frame"])), test_size=.2, random_state=seed)
    val, test = train_test_split(rest, test_size=.5, random_state=seed)
    return dict(train=train, val=val, test=test)


split_indices = original_split


def fixture_preprocess(bundle, seed):
    indices = split_indices(bundle, seed)
    frame = bundle["frame"]
    scaler = StandardScaler().fit(frame.iloc[indices["train"]][bundle["numeric"]])
    ytrain = frame.iloc[indices["train"]].target.to_numpy(float)
    vocab = {c: {v: i+1 for i, v in enumerate(sorted(frame.iloc[indices["train"]][c].unique()))}
             for c in bundle["categorical"]}
    data = dict(numeric=bundle["numeric"], categorical=bundle["categorical"], features=bundle["features"],
        cardinalities=[len(vocab[c])+1 for c in bundle["categorical"]], vocabularies=vocab,
        target_mean=float(ytrain.mean()), target_std=float(ytrain.std()), splits={})
    for name, idx in indices.items():
        x = scaler.transform(frame.iloc[idx][bundle["numeric"]]).astype(np.float32)
        y = frame.iloc[idx].target.to_numpy(float)
        cat = np.column_stack([frame.iloc[idx][c].map(vocab[c]).fillna(0).to_numpy("int64")
                               for c in bundle["categorical"]]) if vocab else np.empty((len(idx),0), dtype="int64")
        target = ((y-data["target_mean"])/data["target_std"]).astype(np.float32)
        data["splits"][name] = dict(indices=idx, numeric=x, actual=y,
            tensor=TensorDataset(torch.from_numpy(x), torch.from_numpy(cat), torch.from_numpy(target)))
    return data


def bundle():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(120,3))
    frame = pd.DataFrame(x, columns=["a","b","c"])
    frame["category"] = np.where(np.arange(120)%3, "one", "two")
    frame["target"] = x[:,0]-.5*x[:,1]**2+.1*rng.normal(size=len(x))
    return dict(frame=frame, features=["a","b","c","category"], numeric=["a","b","c"],
        categorical=["category"], target="target", source_sha256="synthetic-v1", source_slug="synthetic/test")


def config():
    return dict(datasets=["fixture"], widths=[4,24], search_splits=[701], search_inits=[801],
        confirm_splits=[703], confirm_inits=[803], batch_size=32, epochs=2, patience=2,
        eval_every=1, lr=.001, weight_decay=.0001, compact_tolerance=.01, data_root=None)


def search_rows():
    return pd.DataFrame([dict(run_id=f"{f}-{d}",phase="search",dataset="fixture",family=f,d=d,
        split_seed=701,init_seed=801,val_NRMSE=v,test_RMSE=999 if d==4 else -999)
        for f,d,v in [("rn",4,1.005),("rn",24,1.),("rnx_gelu2",4,.95),("rnx_gelu2",24,1.1)]])


class TestExperiment6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def data(self):
        b=bundle()
        return protocol.prepare_data(b,original_split(b,701),fixture_preprocess)

    def test_accuracy_selection_not_compactness_or_test_scores(self):
        chosen=protocol.select_widths(search_rows(),config(),"fixture")
        self.assertEqual(chosen["rn"]["accuracy_d"],24)
        self.assertEqual(chosen["rn"]["compact_d"],4)
        self.assertEqual(chosen["rnx_gelu2"]["accuracy_d"],4)
        altered=search_rows();altered["test_RMSE"]=0
        self.assertEqual(chosen,protocol.select_widths(altered,config(),"fixture"))

    def test_search_must_be_balanced_and_complete(self):
        rows=search_rows()
        for broken in (rows.iloc[:-1],pd.concat([rows,rows.iloc[:1]])):
            with self.assertRaises(ValueError):
                protocol.select_widths(broken,config(),"fixture")

    def test_nonfinite_selection_rejected(self):
        rows=search_rows();rows.loc[0,"val_NRMSE"]=np.nan
        with self.assertRaises(ValueError):protocol.select_widths(rows,config(),"fixture")

    def test_confirmation_fixed_and_independently_tuned_pairs(self):
        chosen={"rn":{"accuracy_d":8},"rnx_gelu2":{"accuracy_d":32}}
        self.assertEqual(set(protocol.confirmation_plan(chosen)),
                         {("rn",24),("rn",8),("rnx_gelu2",8),("rnx_gelu2",32)})
        chosen={m:{"accuracy_d":24} for m in protocol.FAMILIES}
        self.assertEqual(len(protocol.confirmation_plan(chosen)),2)

    def test_wine_duplicate_groups_are_predictor_only(self):
        b=bundle();repeated=pd.concat([b["frame"].iloc[:40]]*3,ignore_index=True)
        repeated.loc[40:,"target"]+=5;b["frame"]=repeated
        parts=protocol.split_indices(b,"wine_white",701,original_split)
        keys={k:set(map(tuple,repeated.iloc[v][b["features"]].to_numpy())) for k,v in parts.items()}
        for a,c in (("train","val"),("train","test"),("val","test")):self.assertFalse(keys[a]&keys[c])
        self.assertEqual(sum(map(len,parts.values())),len(repeated))

    def test_nonwine_preserves_original_split_backend(self):
        sentinel={"train":np.arange(3)}
        self.assertIs(protocol.split_indices(bundle(),"airbnb",1,lambda *_:sentinel),sentinel)

    def test_preprocessing_has_no_global_mutation(self):
        b=bundle();parts=original_split(b,77)
        original=fixture_preprocess.__globals__["split_indices"]
        data=protocol.prepare_data(b,parts,fixture_preprocess)
        self.assertIs(fixture_preprocess.__globals__["split_indices"],original)
        np.testing.assert_allclose(data["splits"]["train"]["numeric"].mean(0),0,atol=1e-6)
        for k in parts:np.testing.assert_array_equal(parts[k],data["splits"][k]["indices"])

    def test_initial_outputs_shared_parameters_and_head(self):
        data=self.data()
        for d in (4,24):
            experiment.seed_all(801);rn=protocol.make_model("rn",d,data)
            experiment.seed_all(801);gelu=protocol.make_model("rnx_gelu2",d,data)
            for k,v in rn.state_dict().items():torch.testing.assert_close(v,gelu.state_dict()[k],rtol=0,atol=0)
            x,c,_=data["splits"]["train"]["tensor"].tensors
            torch.testing.assert_close(rn(x,c),gelu(x,c),rtol=0,atol=0)
            self.assertIsInstance(gelu.readout,torch.nn.Linear)
            self.assertEqual(sum(p.numel() for p in gelu.parameters())-sum(p.numel() for p in rn.parameters()),18)

    def test_generic_model_import_cannot_shadow_rnx(self):
        with patch.dict(sys.modules,{"model":types.SimpleNamespace(wrong=True)}):
            protocol.model_classes.cache_clear()
            rn,rnx=protocol.model_classes()
            self.assertEqual(rn.__name__,"NormalizedFeatureAttentionRegressor")
            self.assertEqual(rnx.__name__,"NonlinearFeatureAttentionRegressor")

    def test_search_never_evaluates_test(self):
        data=self.data();data["splits"]["test"]=object()
        identity=dict(run_id="isolation",phase="search",dataset="fixture",family="rn",d=4,split_seed=701,init_seed=801)
        with contextlib.redirect_stdout(io.StringIO()):
            row,curves=experiment.fit_one(data,identity,config(),torch.device("cpu"))
        self.assertFalse(any(k.startswith("test_") for k in row))
        self.assertIn("train_RMSE",row)
        self.assertTrue(curves and "train_RMSE" in curves[-1])

    def test_paired_sd_uses_partition_means(self):
        rows=[]
        for split,delta in [(1,2.),(2,-4.)]:
            for init in [1,2]:
                for family,error in [("rn",10.),("rnx_gelu2",10.+delta)]:
                    rows.append(dict(phase="confirm",dataset="fixture",family=family,d=24,split_seed=split,
                        init_seed=init,test_RMSE=error,test_MAE=error/2,test_NRMSE=error/10,test_R2=1-error/10))
        chosen={"fixture":{f:{"accuracy_d":24} for f in protocol.FAMILIES}}
        summary=protocol.build_summary(pd.DataFrame(rows),chosen)
        r=summary[(summary.record_type=="paired")&(summary.contrast=="B_fixed_width")].iloc[0]
        self.assertEqual(r.n_runs,4);self.assertEqual(r.n_splits,2)
        self.assertEqual(r.RMSE_delta_mean,-1.)
        self.assertAlmostEqual(r.RMSE_delta_sd_split_means,np.std([2.,-4.],ddof=1))

    def test_store_locks_rejects_mismatch_and_duplicate_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with protocol.Store(root,config(),{"source":"a"}) as store:
                with self.assertRaises(RuntimeError):protocol.Store(root,config(),{"source":"a"})
                row=search_rows().iloc[0].to_dict();curve=[dict(run_id=row["run_id"],epoch=1)]
                store.commit(row,curve)
                with self.assertRaises(ValueError):store.commit(row,curve)
            with self.assertRaises(ValueError):protocol.Store(root,dict(config(),lr=.02),{"source":"a"})
            with protocol.Store(root,config(),{"source":"a"}) as store:self.assertTrue(store.completed(row["run_id"]))

    def test_orphan_curves_removed_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with protocol.Store(root,config(),{}):pass
            pd.DataFrame([dict(run_id="interrupted",epoch=2)]).to_csv(root/"learning_curves.csv",index=False)
            with protocol.Store(root,config(),{}) as store:self.assertTrue(store.curves.empty)

    def test_confirmation_requires_completed_search(self):
        backend=types.SimpleNamespace(split_indices=original_split,preprocess=fixture_preprocess)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"Complete validation"):
                experiment.execute(config(),Path(tmp),"confirm",torch.device("cpu"),{},
                                   loader=lambda _:bundle(),data_module=backend)

    def test_end_to_end_four_files_and_resumption(self):
        backend=types.SimpleNamespace(split_indices=original_split,preprocess=fixture_preprocess)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            experiment.execute(config(),root,"all",torch.device("cpu"),{"test":True},
                               loader=lambda _:bundle(),data_module=backend)
            self.assertEqual({p.name for p in root.iterdir()},
                             {"manifest.json","results.csv","learning_curves.csv","summary.csv"})
            result=pd.read_csv(root/"results.csv")
            search=result[result.phase=="search"]
            self.assertEqual(len(search),4);self.assertTrue(search.test_RMSE.isna().all())
            self.assertTrue(result[result.phase=="confirm"].test_RMSE.notna().all())
            with patch.object(experiment,"fit_one",side_effect=AssertionError("Completed run repeated")):
                experiment.execute(config(),root,"all",torch.device("cpu"),{"test":True},
                                   loader=lambda _:bundle(),data_module=backend)
            pd.testing.assert_frame_equal(result,pd.read_csv(root/"results.csv"))

    def test_cli_rejects_invalid_settings(self):
        for args in (["--search-splits","1","--confirm-splits","1"],["--widths","4","8"],
                     ["--widths","4","24","24"],["--lr","nan"],["--epochs","0"]):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):experiment.arguments(args)


if __name__=="__main__":
    unittest.main()
