"""EX05-A: nonlinear query/key maps and value-map ablations.

Run: uv run python experiments/ex05/run_feature_maps.py --dataset ames --stage search
Then: uv run python experiments/ex05/run_feature_maps.py --dataset ames --stage confirm

Requires the existing EX03 preprocessing and RN training-budget helper.
"""
from __future__ import annotations
import argparse
import functools
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'experiments'/'datasets_ex03'),
                 str(ROOT/'experiments'/'rn_interlude'), str(ROOT/'VANTA_RNX')]
from datasets import load_dataset, preprocess
import training_budget_test as budget
from model import NonlinearFeatureAttentionRegressor, ReferenceRN

CANDIDATES = {
    'baseline': ('identity', 'identity'),
    'qk_tanh': ('tanh', 'identity'),
    'qk_cubic': ('cubic', 'identity'),
    'value_asinh': ('identity', 'asinh'),
    'value_learned_asinh': ('identity', 'learned_asinh'),
    'value_gelu2': ('identity', 'gelu2'),
}
BUDGETS = {'ames': 650, 'california': 450, 'concrete': 1000, 'airbnb': 350}
LR = {'ames': .001, 'california': .002, 'concrete': .001, 'airbnb': .002}
ORIGINAL = budget.NormalizedFeatureAttentionRegressor


def execute(data, seed, label, pair, args, output, confirm):
    folder = output/('confirm' if confirm else 'search')/f'seed_{seed}'/label
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(batch_size=128, lr=LR[args.dataset], latent_dim=24, gamma_init=.1)
    metadata = dict(seed=seed, label=label, maps=list(pair), config=config,
        epochs=args.epochs, patience=args.patience, eval_every=args.eval_every,
        weight_decay=args.weight_decay, stage='confirm' if confirm else 'search')
    marker, metrics = folder/'config.json', folder/'metrics.json'
    if marker.exists() and json.loads(marker.read_text()) != metadata:
        raise RuntimeError(f'Incompatible existing run: {folder}')
    if metrics.exists():
        return json.loads(metrics.read_text())
    marker.write_text(json.dumps(metadata, indent=2))
    training_args = argparse.Namespace(device=args.device, epochs=args.epochs,
        patience=args.patience, eval_every=args.eval_every, weight_decay=args.weight_decay)
    factory = ORIGINAL if pair == ('identity', 'identity') else functools.partial(
        NonlinearFeatureAttentionRegressor, kernel_map=pair[0], value_map=pair[1])
    try:
        budget.NormalizedFeatureAttentionRegressor = factory
        predict, details = budget.train_neural(
            data, 'VANTA-RN', config, seed, training_args, folder, 0)
    finally:
        budget.NormalizedFeatureAttentionRegressor = ORIGINAL
    if pair != ('identity', 'identity'):
        checkpoint_path = folder/'checkpoint.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        checkpoint['version'] = 'VANTA-RNX-v0.1'
        checkpoint['kernel_map'], checkpoint['value_map'] = pair
        torch.save(checkpoint, checkpoint_path)
    val = budget.evaluate_prediction(data['splits']['val']['actual'], predict('val'))
    row = dict(variant=label, seed=seed, parameters=details['parameters'],
        best_epoch=details['best_epoch'], updates=details['total_updates'],
        **{f'val_{k}': v for k,v in val.items()})
    if confirm:
        test = data['splits']['test']
        yp = predict('test')
        scores = budget.evaluate_prediction(test['actual'], yp)
        row.update({f'test_{k}': v for k,v in scores.items()})
        pd.DataFrame(dict(row_index=test['indices'], actual=test['actual'],
            prediction=yp)).to_csv(folder/'test_predictions.csv', index=False)
    metrics.write_text(json.dumps(row, indent=2))
    print(f'{label} / seed {seed}: validation RMSE {val["RMSE"]:.3f}', flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=sorted(BUDGETS))
    parser.add_argument('--stage', required=True, choices=('search', 'confirm'))
    parser.add_argument('--search-seeds', nargs='+', type=int, default=[401, 402])
    parser.add_argument('--confirm-seeds', nargs='+', type=int, default=[403, 404, 405])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=120)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    args = parser.parse_args()
    args.epochs = args.epochs or BUDGETS[args.dataset]
    if set(args.search_seeds) & set(args.confirm_seeds):
        parser.error('Search/confirmation seeds must not overlap')
    if len(args.search_seeds) != len(set(args.search_seeds)) or len(args.confirm_seeds) != len(set(args.confirm_seeds)):
        parser.error('Duplicate seeds')
    if min(args.epochs, args.patience, args.eval_every, args.threads) <= 0:
        parser.error('Training settings must be positive')
    torch.set_num_threads(args.threads)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = args.output/args.dataset
    output.mkdir(parents=True, exist_ok=True)
    selection_file = output/'selection.json'
    specification = dict(dataset=args.dataset, search_seeds=args.search_seeds,
        confirm_seeds=args.confirm_seeds, epochs=args.epochs,
        patience=args.patience, eval_every=args.eval_every,
        weight_decay=args.weight_decay,
        variants={k:list(v) for k,v in CANDIDATES.items()})
    if selection_file.exists() and json.loads(selection_file.read_text())['spec'] != specification:
        parser.error('Selection metadata differs from requested study')
    bundle = load_dataset(args.dataset)
    if args.stage == 'search':
        rows = []
        for seed in args.search_seeds:
            data = preprocess(bundle, seed)
            for name,maps in CANDIDATES.items():
                rows.append(execute(data, seed, name, maps, args, output, False))
        dataframe = pd.DataFrame(rows)
        ranked = dataframe.groupby('variant').val_RMSE.mean()
        qk = min(('baseline','qk_tanh','qk_cubic'), key=ranked.get)
        values = min(('baseline','value_asinh','value_learned_asinh','value_gelu2'), key=ranked.get)
        combination = (CANDIDATES[qk][0], CANDIDATES[values][1])
        if combination not in CANDIDATES.values():
            for seed in args.search_seeds:
                data = preprocess(bundle, seed)
                rows.append(execute(data, seed, 'combined', combination, args, output, False))
        dataframe = pd.DataFrame(rows)
        dataframe.to_csv(output/'search_metrics.csv', index=False)
        selection_file.write_text(json.dumps(dict(
            spec=specification, qk=qk, value=values, combined=list(combination)), indent=2))
        print('Validation ranking:\n', dataframe.groupby('variant').val_RMSE.mean().sort_values().to_string())
    else:
        if not selection_file.is_file():
            parser.error('Run search first')
        selected = json.loads(selection_file.read_text())
        chosen = {'baseline': CANDIDATES['baseline'],
                  selected['qk']: CANDIDATES[selected['qk']],
                  selected['value']: CANDIDATES[selected['value']]}
        combo = tuple(selected['combined'])
        if combo not in chosen.values():
            chosen['combined'] = combo
        unique, used = {}, set()
        for name,maps in chosen.items():
            if maps not in used:
                unique[name] = maps
                used.add(maps)
        rows = []
        for seed in args.confirm_seeds:
            data = preprocess(bundle, seed)
            for name,maps in unique.items():
                rows.append(execute(data, seed, name, maps, args, output, True))
        dataframe = pd.DataFrame(rows)
        dataframe.to_csv(output/'confirmation_metrics.csv', index=False)
        summary = dataframe.groupby('variant').agg(
            n=('seed','count'), R2=('test_R2','mean'), R2_sd=('test_R2','std'),
            RMSE=('test_RMSE','mean'), RMSE_sd=('test_RMSE','std'),
            MAE=('test_MAE','mean'), MAE_sd=('test_MAE','std'))
        summary.to_csv(output/'confirmation_summary.csv')
        print('Confirmation (no test selection):\n', summary.to_string())
    print('Results:', output)


if __name__ == '__main__':
    main()
