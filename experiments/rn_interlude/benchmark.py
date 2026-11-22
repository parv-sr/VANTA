"""15-task reproducible tabular regression benchmark, VANTA-RN by default.

  uv run python experiments/rn_interlude/benchmark.py --datasets all --seeds 42 43 44
  uv run python experiments/rn_interlude/benchmark.py --models rn rnx --baselines mlp xgboost catboost --device cuda
  uv run python experiments/rn_interlude/benchmark.py --datasets ames insurance --inspect

All candidate models share split indices and training-fitted preprocessing.
The benchmark is an exploratory random-split comparison, not time forecasting.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "datasets_ex03"), str(HERE)]
from datasets import preprocess
from benchmark_datasets import TASKS, load_task, profile_train
import run_exp as ex03

NEURAL = ("rn", "r", "rnx")
BASELINES = ("mlp", "xgboost", "catboost")


def load_file_class(path: Path, symbol: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def make_model(name, data, args):
    f = len(data["features"])
    n = len(data["numeric"])
    keyword = dict(num_features=f, numeric_positions=list(range(n)),
        categorical_positions=list(range(n, f)),
        categorical_cardinalities=data["cardinalities"],
        latent_dim=args.latent_dim, gamma_init=.1, learnable_gamma=True)
    if name == "rn":
        cls = load_file_class(ROOT/"VANTA_RN"/"model.py",
            "NormalizedFeatureAttentionRegressor", "_benchmark_rn")
    elif name == "r":
        cls = load_file_class(ROOT/"VANTA_R"/"model.py",
            "ResidualFeatureAttentionRegressor", "_benchmark_r")
    elif name == "rnx":
        cls = load_file_class(ROOT/"VANTA_RNX"/"model.py",
            "NonlinearFeatureAttentionRegressor", "_benchmark_rnx")
        keyword.update(kernel_map=args.rnx_kernel, value_map=args.rnx_value)
    elif name == args.plugin_name and args.plugin:
        path, symbol = args.plugin.rsplit(":", 1)
        cls = load_file_class(Path(path).expanduser().resolve(), symbol, "_benchmark_plugin")
        keyword.update(json.loads(args.plugin_kwargs))
    else:
        raise ValueError(f"Unknown model {name}")
    return cls(**keyword)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y, prediction, train_std):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    error = p - y
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    denom = float(np.sum((y-y.mean())**2))
    r2 = 1-float(np.sum(error**2))/denom if denom > 0 else float("nan")
    return dict(RMSE=rmse, MAE=mae, R2=r2, NRMSE=rmse/max(float(train_std),1e-12))


@torch.inference_mode()
def neural_predict(model, ds, data, device):
    model.eval()
    chunks = []
    for xn, xc, _ in DataLoader(ds, batch_size=2048):
        out = model(xn.to(device), xc.to(device))
        chunks.append(out.detach().cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)*data["target_std"] + data["target_mean"]


@torch.inference_mode()
def val_mse(model, ds, device):
    model.eval()
    total, count = 0., 0
    for xn, xc, y in DataLoader(ds, batch_size=2048):
        err = model(xn.to(device),xc.to(device)) - y.to(device)
        total += float(err.square().sum().item())
        count += len(y)
    return total/max(count, 1)


def train_neural(name, data, args, seed, folder, ref_parameters):
    seed_everything(seed)
    if name == "mlp":
        model, hidden = ex03.build_mlp(data, ref_parameters)
    else:
        model, hidden = make_model(name,data,args), None
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = model.to(args.torch_device)
    loader = DataLoader(data["splits"]["train"]["tensor"],batch_size=args.batch_size,
        shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    val_ds = data["splits"]["val"]["tensor"]
    best, best_epoch, best_state = float("inf"), 0, None
    updates = 0
    for epoch in range(1,args.epochs+1):
        model.train()
        for xn,xc,y in loader:
            opt.zero_grad(set_to_none=True)
            err = model(xn.to(args.torch_device),xc.to(args.torch_device))-y.to(args.torch_device)
            loss = err.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss: {name} seed {seed} epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.,error_if_nonfinite=True)
            opt.step()
            updates += 1
        if epoch == 1 or epoch%args.eval_every == 0 or epoch == args.epochs:
            current=val_mse(model,val_ds,args.torch_device)
            if current < best-1e-7:
                best, best_epoch = current, epoch
                best_state = copy.deepcopy(model.state_dict())
            if epoch-best_epoch >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    if args.save_checkpoints:
        torch.save(dict(model=name,state_dict=best_state,seed=seed,best_epoch=best_epoch,
            features=data["features"],cardinalities=data["cardinalities"],
            vocabularies=data["vocabularies"],target_mean=data["target_mean"],
            target_std=data["target_std"],latent_dim=args.latent_dim,
            hidden_dim=hidden,parameters=parameters), folder/"checkpoint.pt")
    return (lambda part: neural_predict(model,data["splits"][part]["tensor"],data,args.torch_device)), dict(
        parameters=parameters,mlp_hidden_width=hidden,
        mlp_reference_parameters=ref_parameters if name=="mlp" else None,
        mlp_parameter_ratio=parameters/max(ref_parameters,1) if name=="mlp" else None,
        best_epoch=best_epoch,updates=updates)


def train_tree(name, data, args, seed, folder):
    train,val=data["splits"]["train"],data["splits"]["val"]
    if name == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(objective="reg:squarederror", n_estimators=args.trees,
            max_depth=6, learning_rate=.04, min_child_weight=5, reg_lambda=5.,
            subsample=.9,colsample_bytree=.9,tree_method="hist",
            device="cuda" if args.tree_gpu else "cpu",
            random_state=seed,n_jobs=args.threads,eval_metric="rmse",
            early_stopping_rounds=min(100,max(20,args.trees//5)))
        model.fit(train["classical"],train["actual"],eval_set=[(
            val["classical"],val["actual"])],verbose=False)
        predict=lambda part: model.predict(data["splits"][part]["classical"])
        extra=dict(best_iteration=int(model.best_iteration))
        if args.save_checkpoints:
            model.save_model(str(folder/"xgboost.json"))
    else:
        from catboost import CatBoostRegressor
        model=CatBoostRegressor(iterations=args.trees,depth=7,
            learning_rate=.04,l2_leaf_reg=4.,loss_function="RMSE",
            cat_features=data["categorical"],random_seed=seed,
            task_type="GPU" if args.tree_gpu else "CPU",
            thread_count=args.threads,allow_writing_files=False,verbose=False)
        model.fit(train["catboost"],train["actual"],
            eval_set=(val["catboost"],val["actual"]),
            early_stopping_rounds=min(100,max(20,args.trees//5)),use_best_model=True,
            verbose=False)
        predict=lambda part: model.predict(data["splits"][part]["catboost"])
        extra=dict(best_iteration=int(model.best_iteration_))
        if args.save_checkpoints:
            model.save_model(str(folder/"catboost.cbm"))
    return predict,extra


def config_snapshot(args, models, seed, dataset, source_hash):
    return dict(dataset=dataset,seed=seed,source_sha256=source_hash,
        models=list(models),latent_dim=args.latent_dim,
        batch_size=args.batch_size,lr=args.lr,epochs=args.epochs,
        patience=args.patience,eval_every=args.eval_every,
        trees=args.trees,weight_decay=args.weight_decay,
        rnx_kernel=args.rnx_kernel,rnx_value=args.rnx_value,
        device=str(args.torch_device),tree_gpu=args.tree_gpu,
        plugin=args.plugin,plugin_name=args.plugin_name,
        plugin_kwargs=args.plugin_kwargs,
        mlp_match=args.mlp_match,threads=args.threads,
        save_checkpoints=args.save_checkpoints)


def run_model(dataset, data, profile, name, args, seed, ref_parameters):
    folder=args.output/dataset/f"seed_{seed}"/name
    folder.mkdir(parents=True,exist_ok=True)
    record=folder/"result.json"
    conf=config_snapshot(args,args.models+args.baselines,seed,dataset,profile["source_sha256"])
    conf["name"]=name
    conf["reference_params"]=ref_parameters if name=="mlp" else None
    if record.exists():
        payload=json.loads(record.read_text())
        if payload["configuration"] != conf:
            raise RuntimeError(f"Existing result for different settings: {folder}. Use --output for new studies.")
        return payload["metrics"]
    start=time.perf_counter()
    if name in BASELINES and name!="mlp":
        predict,info=train_tree(name,data,args,seed,folder)
    else:
        predict,info=train_neural(name,data,args,seed,folder,ref_parameters)
    yval = data["splits"]["val"]["actual"]
    ytest = data["splits"]["test"]["actual"]
    pval,ptext=predict("val"),predict("test")
    row=dict(dataset=dataset,seed=seed,model=name,
        source_slug=data.get("source_slug", ""),
        duration_seconds=round(time.perf_counter()-start,3),
        **info,
        **{f"val_{k}":v for k,v in metrics(yval,pval,data["target_std"]).items()},
        **{f"test_{k}":v for k,v in metrics(ytest,ptext,data["target_std"]).items()},
        **profile)
    # Scores are diagnostic; no hyperparameter selection on the test partition.
    if args.save_predictions:
        pd.DataFrame(dict(row_index=data["splits"]["test"]["indices"],
            actual=ytest,prediction=ptext)).to_csv(folder/"test_predictions.csv",index=False)
    record.write_text(json.dumps(dict(configuration=conf,metrics=row),indent=2,default=str))
    return row


def summary_tables(all_rows,output,reference_model):
    if not all_rows: return
    df=pd.DataFrame(all_rows).drop_duplicates(["dataset","seed","model"])
    output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/"per_seed.csv",index=False)
    grouped=[]
    for (dataset,model),g in df.groupby(["dataset","model"]):
        row=dict(dataset=dataset,model=model,n=len(g),
            domain=g["domain"].iloc[0],characteristic=g["characteristic"].iloc[0],
            rows=int(g["rows"].iloc[0]),features=int(g["features"].iloc[0]),
            categorical_fraction=float(g["categorical_fraction"].mean()),
            missing_fraction_train=float(g["missing_fraction_train"].mean()),
            extreme_numeric_fraction_train=float(g["extreme_numeric_fraction_train"].mean()),
            target_skew_train=float(g["target_skew_train"].mean()))
        for m in ["R2","RMSE","MAE","NRMSE"]:
            vals=g[f"test_{m}"].to_numpy(float)
            row[f"{m}_mean"]=float(np.mean(vals))
            row[f"{m}_sd"]=float(np.std(vals,ddof=1)) if len(vals)>1 else float("nan")
        grouped.append(row)
    summary=pd.DataFrame(grouped)
    summary.to_csv(output/"summary.csv",index=False)
    ref=df[df.model==reference_model]
    paired=[]
    for model in df.model.unique():
        if model==reference_model:continue
        other=df[df.model==model]
        p=ref.merge(other,on=["dataset","seed"],suffixes=("_ref","_other"))
        for dataset,g in p.groupby("dataset"):
            delta=g.test_NRMSE_other-g.test_NRMSE_ref
            paired.append(dict(dataset=dataset,reference=reference_model,competitor=model,
                paired_seeds=len(g),reference_wins=int((delta>0).sum()),
                mean_relative_nrmse_advantage=float(delta.mean()),
                sd_relative_nrmse_advantage=float(delta.std(ddof=1)) if len(g)>1 else float("nan")))
    pd.DataFrame(paired,columns=["dataset","reference","competitor","paired_seeds",
        "reference_wins","mean_relative_nrmse_advantage","sd_relative_nrmse_advantage"]).to_csv(
            output/"paired.csv",index=False)
    if summary.model.nunique()>1:
        ranks=summary.copy()
        ranks["rank"]=ranks.groupby("dataset").NRMSE_mean.rank(method="average")
        ranks.groupby("model").agg(mean_rank=("rank","mean"),tasks=("dataset","nunique")).to_csv(
            output/"ranks.csv")
    if reference_model in df.model.unique():
        prof=summary[summary.model==reference_model].copy()
        # Data-regime tags use dataset-level TRAIN descriptors only.
        prof["size_regime"]=pd.cut(prof["rows"],bins=[0,1500,10000,float("inf")],
            labels=["small_<=1500","medium_1501_10000","large_>10000"])
        prof["feature_mix"]=np.where(prof["categorical_fraction"]>=.3,
            "categorical_heavy",np.where(prof["categorical_fraction"]>0,
            "mixed","numeric_only"))
        prof["input_tail_regime"]=np.where(
            prof["extreme_numeric_fraction_train"]>=.005,
            "many_>5sigma","few_>5sigma")
        prof["target_skew_regime"]=np.where(
            np.abs(prof["target_skew_train"])>=2,
            "high_abs_skew","lower_abs_skew")
        prof=prof.sort_values("NRMSE_mean")
        prof.to_csv(output/"reference_strengths.csv",index=False)
        regimes=[]
        for facet in ["domain","size_regime","feature_mix","input_tail_regime","target_skew_regime"]:
            for label,group in prof.groupby(facet,observed=True):
                regimes.append(dict(facet=facet,regime=str(label),
                    datasets=len(group),mean_R2=float(group["R2_mean"].mean()),
                    mean_NRMSE=float(group["NRMSE_mean"].mean()),
                    median_NRMSE=float(group["NRMSE_mean"].median())))
        pd.DataFrame(regimes).to_csv(output/"regime_summary.csv",index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",nargs="+",default=["all"],
        help="Task names from --list-datasets or 'all'")
    p.add_argument("--list-datasets",action="store_true")
    p.add_argument("--inspect",action="store_true",help="Download/load schemas but do not train")
    p.add_argument("--models",nargs="+",default=["rn"],help="rn r rnx or --plugin-name")
    p.add_argument("--baselines",nargs="*",default=[],
        help="Optional: mlp xgboost catboost; no baselines by default")
    p.add_argument("--plugin",default="",help="Future model source: /abs/path/model.py:ClassName")
    p.add_argument("--plugin-name",default="plugin",help="CLI model ID for --plugin")
    p.add_argument("--plugin-kwargs",default="{}",help="JSON options forwarded to plugin class")
    p.add_argument("--rnx-kernel",choices=("identity","tanh","cubic"),default="identity")
    p.add_argument("--rnx-value",choices=("identity","asinh","learned_asinh","gelu2"),default="identity")
    p.add_argument("--mlp-match",default="rn",
        help="Parameter budget reference: rn, r, rnx or plugin name")
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--tree-gpu",action="store_true",
        help="Use XGBoost GPU and CatBoost task_type=GPU; off by default")
    p.add_argument("--epochs",type=int,default=300)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--latent-dim",type=int,default=24)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--patience",type=int,default=100)
    p.add_argument("--eval-every",type=int,default=5)
    p.add_argument("--trees",type=int,default=2000)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--data-root",type=Path,default=None)
    p.add_argument("--output",type=Path,default=HERE/"results"/"benchmark15")
    p.add_argument("--save-checkpoints",action="store_true")
    p.add_argument("--save-predictions",action="store_true")
    p.add_argument("--continue-on-error",action="store_true")
    args=p.parse_args()
    if args.list_datasets:
        for name,t in TASKS.items():
            print(f"{name:16} {t.domain:19} {t.slug} ({t.filename})")
        return
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA selected but torch.cuda.is_available() is false")
    args.torch_device=torch.device("cuda" if args.device=="cuda" or
        (args.device=="auto" and torch.cuda.is_available()) else "cpu")
    all_tasks=list(TASKS) if args.datasets==["all"] else args.datasets
    invalid=set(all_tasks)-set(TASKS)
    if invalid:p.error(f"Unknown dataset(s): {sorted(invalid)}")
    if not args.seeds or len(set(args.seeds))!=len(args.seeds) or not 3<=len(args.seeds)<=5:
        p.error("Provide between three and five distinct --seeds")
    if min(args.epochs,args.batch_size,args.latent_dim,args.patience,args.eval_every,args.trees,args.threads)<=0:
        p.error("Budgets and dimensions must be positive")
    if len(set(args.models))!=len(args.models) or len(set(args.baselines))!=len(args.baselines):
        p.error("Duplicate model identifiers")
    allowed=set(NEURAL)|({args.plugin_name} if args.plugin else set())
    if not args.models or set(args.models)-allowed:p.error("Unknown model. Choose rn/r/rnx or provide a plugin.")
    if set(args.baselines)-set(BASELINES):p.error("Unknown baseline; choose mlp/xgboost/catboost")
    if set(args.models)&set(args.baselines):p.error("Duplicate model in both lists")
    if args.mlp_match not in allowed:p.error("Unknown --mlp-match reference")
    try: json.loads(args.plugin_kwargs)
    except json.JSONDecodeError:p.error("Invalid plugin JSON")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True,exist_ok=True)
    info=dict(datasets=all_tasks,models=args.models,baselines=args.baselines,seeds=args.seeds,
        device=str(args.torch_device),tree_gpu=args.tree_gpu,
        train=dict(epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,
        weight_decay=args.weight_decay,patience=args.patience,eval_every=args.eval_every),
        model_config=dict(latent_dim=args.latent_dim,rnx_kernel=args.rnx_kernel,
        rnx_value=args.rnx_value,plugin=args.plugin,plugin_kwargs=args.plugin_kwargs),
        environment=dict(torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        tree_iterations=args.trees)
    (args.output/"run_settings.json").write_text(json.dumps(info,indent=2))
    all_rows=[]
    for name in all_tasks:
        try:
            print(f"\n=== {name}: loading ===",flush=True)
            bundle=load_task(name,data_root=args.data_root)
            print(f"{name} rows={len(bundle['frame'])} "
                  f"features={len(bundle['features'])} "
                  f"numeric={len(bundle['numeric'])} "
                  f"categorical={len(bundle['categorical'])}",flush=True)
            if args.inspect:continue
            for seed in args.seeds:
                data=preprocess(bundle,seed)
                profile=profile_train(data,bundle,seed)
                reference=make_model(args.mlp_match,data,args)
                ref_parameters=sum(p.numel() for p in reference.parameters() if p.requires_grad)
                del reference
                for model in [*args.models,*args.baselines]:
                    print(f"{name} seed={seed} model={model}",flush=True)
                    try:
                        row=run_model(name,data,profile,model,args,seed,ref_parameters)
                    except Exception:
                        if not args.continue_on_error: raise
                        trace=traceback.format_exc()
                        with (args.output/"failures.txt").open("a") as fp:
                            fp.write(f"\n{name} seed={seed} model={model}\n{trace}\n")
                        print(f"FAILED {name} seed={seed} model={model}: {trace}",flush=True)
                        continue
                    all_rows.append(row)
                    summary_tables(all_rows,args.output,args.models[0])
                    print(f"test R²={row['test_R2']:.4f} NRMSE={row['test_NRMSE']:.4f}",flush=True)
        except Exception:
            if not args.continue_on_error: raise
            trace=traceback.format_exc()
            with (args.output/"failures.txt").open("a") as fp:
                fp.write(f"\n{name}\n{trace}\n")
            print(f"FAILED {name}: {trace}",flush=True)
    if not args.inspect:
        summary_tables(all_rows,args.output,args.models[0])
        print(f"Saved benchmark tables under {args.output}")


if __name__=="__main__":
    main()
