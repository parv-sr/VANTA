"""RN latent-width sweep with separate partition and initialization seeds.

Search uses validation RMSE exclusively. Confirmation tests selected d versus
the frozen d=24 reference on unseen split/initialization combinations.

Examples from repository root:
  uv run python experiments/rn_interlude/latent_dimension_sweep.py --stage search
  uv run python experiments/rn_interlude/latent_dimension_sweep.py --stage confirm

Feature-group-disjoint wine partitions are the default. For historical
row-level comparison pass --split-mode row_random and a separate --output.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/"experiments"/"datasets_ex03"),str(HERE)]
import datasets as ex03
from benchmark_datasets import TASKS,load_task
from wine_audit import make_group_split,index_sha256
from benchmark import metrics, load_file_class

DEFAULT_WIDTHS = (4,8,16,24,32,48,64)
BASELINE_WIDTH=24


def make_splits(bundle, partition_seed: int, mode: str):
    if mode == "feature_group_disjoint":
        return make_group_split(bundle,partition_seed)
    if mode == "row_random":
        return ex03.split_indices(bundle,partition_seed)
    raise ValueError(f"Unknown split mode: {mode}")


def preprocess_fixed_split(bundle, indices):
    """Apply original EX03 training-fitted transforms on exactly these rows."""
    original=ex03.split_indices
    try:
        ex03.split_indices=lambda _bundle,_seed:indices
        return ex03.preprocess(bundle,0)
    finally:
        ex03.split_indices=original


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_mse(model, dataset, device):
    model.eval()
    total,count=0.,0
    for n,c,y in DataLoader(dataset,batch_size=2048,shuffle=False):
        diff=model(n.to(device),c.to(device))-y.to(device)
        total+=float(diff.square().sum().item())
        count+=int(len(y))
    return total/max(count,1)


@torch.no_grad()
def predict(model, dataset, data, device):
    model.eval()
    outputs=[]
    for n,c,_ in DataLoader(dataset,batch_size=2048,shuffle=False):
        outputs.append(model(n.to(device),c.to(device)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)*data["target_std"]+data["target_mean"]


def train(data, width, init_seed, args, directory, confirm):
    seed_everything(init_seed)
    cls=load_file_class(ROOT/"VANTA_RN"/"model.py",
        "NormalizedFeatureAttentionRegressor","_width_sweep_rn")
    numeric=len(data["numeric"]);total=len(data["features"])
    model=cls(num_features=total,numeric_positions=list(range(numeric)),
        categorical_positions=list(range(numeric,total)),
        categorical_cardinalities=data["cardinalities"],
        latent_dim=width,gamma_init=.1,learnable_gamma=True).to(args.torch_device)
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    loader=DataLoader(data["splits"]["train"]["tensor"],
        batch_size=args.batch_size,shuffle=True,
        generator=torch.Generator().manual_seed(init_seed))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,
                          weight_decay=args.weight_decay)
    best=float("inf");best_state=None;best_epoch=0;updates=0
    started=time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train()
        for xn,xc,y in loader:
            opt.zero_grad(set_to_none=True)
            error=model(xn.to(args.torch_device),xc.to(args.torch_device))-y.to(args.torch_device)
            loss=error.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at d={width}, epoch={epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.,
                error_if_nonfinite=True)
            opt.step()
            updates+=1
        if epoch==1 or epoch%args.eval_every==0 or epoch==args.epochs:
            vmse=eval_mse(model,data["splits"]["val"]["tensor"],args.torch_device)
            if vmse < best-1e-7:
                best=vmse;best_epoch=epoch
                best_state=copy.deepcopy(model.state_dict())
            if epoch-best_epoch >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(best_state)
    pred_val=predict(model,data["splits"]["val"]["tensor"],data,args.torch_device)
    val=metrics(data["splits"]["val"]["actual"],pred_val,data["target_std"])
    report=dict(d=width,init_seed=init_seed,parameters=parameters,
        best_epoch=best_epoch,updates=updates,
        train_seconds=round(time.perf_counter()-started,3),
        gamma=float(model.gamma.detach().cpu()),
        **{f"val_{k}":v for k,v in val.items()})
    # Training-fitted raw Q/K norm distributions and effective rank diagnostics
    # use a deterministic training subset, never validation or test statistics.
    sample=data["splits"]["train"]["tensor"]
    n,cat,_=sample.tensors
    n=n[:min(128,len(n))].to(args.torch_device)
    cat=cat[:min(128,len(cat))].to(args.torch_device)
    with torch.no_grad():
        raw_q,raw_k,_=model.project_raw(n,cat)
        q,k,_=model.project(n,cat)
        report["train_raw_q_norm_p95"]=float(torch.quantile(raw_q.norm(dim=-1).float(),.95).cpu())
        report["train_raw_k_norm_p95"]=float(torch.quantile(raw_k.norm(dim=-1).float(),.95).cpu())
        report["train_normalized_q_norm_p95"]=float(torch.quantile(q.norm(dim=-1).float(),.95).cpu())
        report["train_normalized_k_norm_p95"]=float(torch.quantile(k.norm(dim=-1).float(),.95).cpu())
        # Small diagnostic-only attention matrices; no F² operation in training.
        a=q[:min(16,len(q))]@k[:min(16,len(k))].transpose(-1,-2)/math.sqrt(total)
        singular=torch.linalg.svdvals(a.float())
        weights=singular.square()
        rank=weights.sum(dim=-1).square()/(weights.square().sum(dim=-1)+1e-12)
        report["train_attention_effective_rank_mean"]=float(rank.mean().cpu())
    if confirm:
        yp=predict(model,data["splits"]["test"]["tensor"],data,args.torch_device)
        tst=metrics(data["splits"]["test"]["actual"],yp,data["target_std"])
        report.update({f"test_{k}":v for k,v in tst.items()})
        if args.save_predictions:
            pd.DataFrame(dict(row_index=data["splits"]["test"]["indices"],
              actual=data["splits"]["test"]["actual"],prediction=yp)).to_csv(
              directory/"test_predictions.csv",index=False)
    if args.save_checkpoints:
        torch.save(dict(model="VANTA-RN-v0.1",latent_dim=width,
          state_dict=best_state,parameters=parameters,init_seed=init_seed,
          best_epoch=best_epoch,features=data["features"],
          cardinalities=data["cardinalities"],
          target_mean=data["target_mean"],target_std=data["target_std"]),
          directory/"checkpoint.pt")
    return report


def sha_code():
    h=hashlib.sha256()
    for path in [Path(__file__),HERE/"wine_audit.py",HERE/"benchmark_datasets.py",
        ROOT/"experiments"/"datasets_ex03"/"datasets.py",
        ROOT/"VANTA_R"/"model.py",ROOT/"VANTA_RN"/"model.py"]:
        h.update(path.read_bytes())
    return h.hexdigest()


def run_one(dataset,bundle,split_seed,init_seed,width,args,confirm):
    path=args.output/dataset/args.split_mode/("confirm" if confirm else "search")/(
        f"split_{split_seed}")/(f"init_{init_seed}")/f"d_{width}"
    path.mkdir(parents=True,exist_ok=True)
    output=path/"metrics.json"
    indices=make_splits(bundle,split_seed,args.split_mode)
    config=dict(dataset=dataset,split_seed=split_seed,init_seed=init_seed,
        d=width,split_mode=args.split_mode,
        source_sha256=bundle["source_sha256"],
        split_indices_sha256=index_sha256(indices),
        code_sha256=args.code_sha,
        epochs=args.epochs,batch_size=args.batch_size,lr=args.lr,
        weight_decay=args.weight_decay,patience=args.patience,
        eval_every=args.eval_every,device=str(args.torch_device),
        stage="confirm" if confirm else "search")
    manifest=path/"config.json"
    if manifest.exists() and json.loads(manifest.read_text())!=config:
        raise RuntimeError(f"Existing run settings differ: {path}")
    if output.exists():
        return json.loads(output.read_text())
    manifest.write_text(json.dumps(config,indent=2))
    data=preprocess_fixed_split(bundle,indices)
    results=train(data,width,init_seed,args,path,confirm)
    results.update(dataset=dataset,split_seed=split_seed,
        init_seed=init_seed,split_mode=args.split_mode,
        source_sha256=bundle["source_sha256"],
        split_indices_sha256=config["split_indices_sha256"])
    output.write_text(json.dumps(results,indent=2))
    print(f"{dataset} {config['stage']} split={split_seed} init={init_seed} "
          f"d={width} val_RMSE={results['val_RMSE']:.5f}",flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",required=True,choices=("search","confirm"))
    p.add_argument("--datasets",nargs="+",choices=list(TASKS),
        default=["wine_red","wine_white"])
    p.add_argument("--split-mode",choices=("row_random","feature_group_disjoint"),
        default="feature_group_disjoint")
    p.add_argument("--widths",type=int,nargs="+",default=list(DEFAULT_WIDTHS))
    p.add_argument("--search-split-seeds",type=int,nargs="+",default=[501,502])
    p.add_argument("--search-init-seeds",type=int,nargs="+",default=[601,602])
    p.add_argument("--confirm-split-seeds",type=int,nargs="+",default=[503,504,505])
    p.add_argument("--confirm-init-seeds",type=int,nargs="+",default=[603,604,605])
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--epochs",type=int,default=650)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--patience",type=int,default=120)
    p.add_argument("--eval-every",type=int,default=5)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--selection-tolerance",type=float,default=.01,
        help="Choose smallest d within this relative validation-RMSE tolerance of the winner")
    p.add_argument("--save-predictions",action="store_true")
    p.add_argument("--save-checkpoints",action="store_true")
    p.add_argument("--data-root",type=Path,default=None)
    p.add_argument("--output",type=Path,default=HERE/"results"/"latent_dimension")
    args=p.parse_args()
    if min([*args.widths,args.batch_size,args.epochs,args.patience,
            args.eval_every,args.threads])<=0:
        p.error("Width and training settings must be positive")
    if len(set(args.widths))!=len(args.widths):
        p.error("Widths must be unique")
    if BASELINE_WIDTH not in args.widths:
        p.error("Include frozen baseline width d=24 in --widths")
    for name in ["search_split_seeds","search_init_seeds","confirm_split_seeds","confirm_init_seeds"]:
        values=getattr(args,name)
        if not values or len(set(values))!=len(values):
            p.error(f"Seeds in --{name.replace('_','-')} must be present and distinct")
    if set(args.search_split_seeds)&set(args.confirm_split_seeds):
        p.error("Search and confirmation partition seeds must be disjoint")
    if set(args.search_init_seeds)&set(args.confirm_init_seeds):
        p.error("Search and confirmation initialization seeds must be disjoint")
    if not 0<=args.selection_tolerance<.5:
        p.error("Selection tolerance must be between 0 and 0.5")
    if args.split_mode=="feature_group_disjoint" and any(
        ds not in ("wine_red","wine_white") for ds in args.datasets):
        p.error("Feature-disjoint mode is restricted to wine tasks; use --split-mode row_random for other datasets")
    if args.device=="cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    args.torch_device=torch.device("cuda" if args.device=="cuda" or (
        args.device=="auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(args.threads)
    args.code_sha=sha_code()
    for dataset in args.datasets:
        bundle=load_task(dataset,data_root=args.data_root)
        folder=args.output/dataset/args.split_mode
        folder.mkdir(parents=True,exist_ok=True)
        chosen=folder/"selection.json"
        specification=dict(widths=args.widths,split_mode=args.split_mode,
          source_sha256=bundle["source_sha256"],
          code_sha256=args.code_sha,
          search_split_seeds=args.search_split_seeds,
          search_init_seeds=args.search_init_seeds,
          confirm_split_seeds=args.confirm_split_seeds,
          confirm_init_seeds=args.confirm_init_seeds,
          batch_size=args.batch_size,epochs=args.epochs,lr=args.lr,
          weight_decay=args.weight_decay,patience=args.patience,
          eval_every=args.eval_every,selection_tolerance=args.selection_tolerance)
        if chosen.exists() and json.loads(chosen.read_text())["specification"]!=specification:
            p.error(f"Existing selection has different configuration: {chosen}")
        rows=[]
        if args.stage=="search":
            for split in args.search_split_seeds:
                for init in args.search_init_seeds:
                    for width in args.widths:
                        rows.append(run_one(dataset,bundle,split,init,width,args,False))
            df=pd.DataFrame(rows)
            df.to_csv(folder/"search_metrics.csv",index=False)
            ranking=df.groupby("d")["val_NRMSE"].agg(["mean","std","count"]).sort_values("mean")
            optimum=float(ranking["mean"].min())
            acceptable=ranking.index[ranking["mean"]<=optimum*(1+args.selection_tolerance)]
            winner=int(min(acceptable))
            ranking.to_csv(folder/"validation_width_ranking.csv")
            chosen.write_text(json.dumps(dict(specification=specification,
                selected_width=winner,
                best_validation_NRMSE=optimum,
                selection_method="smallest d within tolerance of best mean validation NRMSE",
                search_run_count=len(df)),indent=2))
            print(f"{dataset}: selected d={winner}; validation rankings:\n{ranking.to_string()}",flush=True)
        else:
            if not chosen.exists():
                p.error(f"Run search for {dataset} first")
            selection=json.loads(chosen.read_text())
            winner=int(selection["selected_width"])
            for split in args.confirm_split_seeds:
                for init in args.confirm_init_seeds:
                    for width in sorted(set([BASELINE_WIDTH,winner])):
                        rows.append(run_one(dataset,bundle,split,init,width,args,True))
            df=pd.DataFrame(rows)
            df.to_csv(folder/"confirmation_metrics.csv",index=False)
            metrics_to_summarize=["test_R2","test_RMSE","test_MAE","test_NRMSE",
                "test_worst1_sse_share","test_worst5_sse_share"]
            summary=df.groupby("d")[metrics_to_summarize].agg(["mean","std","count"])
            summary.to_csv(folder/"confirmation_summary.csv")
            print(f"{dataset}: confirm selected d={winner} vs d=24:\n{summary.to_string()}",flush=True)
            if winner!=BASELINE_WIDTH:
                paired=df.pivot_table(index=["split_seed","init_seed"],columns="d",values="test_NRMSE")
                delta=paired[winner]-paired[BASELINE_WIDTH]
                delta.rename("selected_minus_d24_test_NRMSE").to_csv(folder/"paired_test_delta.csv")
                print(f"Selected minus reference mean test NRMSE: {float(delta.mean()):.6f}",flush=True)
    print(f"Outputs: {args.output}")


if __name__=="__main__":
    main()
