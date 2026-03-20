#!/usr/bin/env python
"""
Run and compare tuning experiments.

Usage:
    # Run a single experiment (baseline)
    python scripts/experiment.py --trials 100 --desc "baseline 100 trials"

    # Run with specific models only
    python scripts/experiment.py --trials 50 --models goals assists --desc "goals+assists only"

    # Show experiment history
    python scripts/experiment.py --history

    # Show best runs per model
    python scripts/experiment.py --best

    # Compare last N runs
    python scripts/experiment.py --compare 3
"""
import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FPLPipeline
from src.experiment_log import get_history, get_best_run


def run_experiment(args):
    """Run a tuning experiment."""
    pipeline = FPLPipeline('data')
    pipeline.load_data()
    pipeline.compute_features()
    pipeline.tune(
        models=args.models,
        n_iter=args.trials,
        test_size=args.test_size,
        use_subprocess=args.subprocess,
        description=args.desc,
    )


def show_history(args):
    """Show experiment history."""
    df = get_history('data', model=args.model)
    if df.empty:
        print("No experiments logged yet.")
        return

    # Pivot to show one row per run with all models
    print("\n=== Experiment History ===\n")
    for run_id in df['run_id'].unique():
        run = df[df['run_id'] == run_id]
        row = run.iloc[0]
        print(f"Run: {row['run_id']}")
        print(f"  Time: {row['timestamp'][:19]}  |  Trials: {row['n_iter']}  |  Test: {row['test_size']:.0%}")
        if row.get('description'):
            print(f"  Desc: {row['description']}")
        if row.get('fpl_points_mae') is not None:
            bonus_str = f"  |  inc-bonus: {row['fpl_points_mae_inc_bonus']:.4f}" if row.get('fpl_points_mae_inc_bonus') else ""
            print(f"  FPL Points MAE (ex-bonus): {row['fpl_points_mae']:.4f}{bonus_str}")

        print(f"  {'Model':<15} {'Metric':<15} {'CV':<10} {'Test':<10} {'MAE':<10} {'#Feat'}")
        print(f"  {'-'*70}")
        for _, m in run.iterrows():
            cv = f"{m['cv_score']:.4f}" if m['cv_score'] is not None else "N/A"
            test = f"{m['test_score']:.4f}" if m['test_score'] is not None else "N/A"
            mae = f"{m['mae']:.4f}" if m['mae'] is not None else "N/A"
            nf = str(int(m['n_features'])) if pd.notna(m['n_features']) else "N/A"
            print(f"  {m['model_name']:<15} {m['metric_name'] or '':<15} {cv:<10} {test:<10} {mae:<10} {nf}")
        print()


def show_best(args):
    """Show best run per model."""
    models = ['goals', 'assists', 'minutes', 'defcon', 'clean_sheet', 'saves']
    print("\n=== Best Runs Per Model ===\n")
    print(f"{'Model':<15} {'Best Test':<12} {'MAE':<12} {'#Feat':<8} {'Run ID'}")
    print("-" * 65)
    for model_name in models:
        best = get_best_run('data', model_name)
        if best is not None:
            print(f"{model_name:<15} {best['test_score']:<12.4f} {best['mae']:<12.4f} {int(best['n_features']):<8} {best['run_id'][:20]}")
        else:
            print(f"{model_name:<15} {'--':<12} {'--':<12} {'--':<8} no runs")


def compare_runs(args):
    """Compare last N runs side by side."""
    df = get_history('data')
    if df.empty:
        print("No experiments logged yet.")
        return

    run_ids = df['run_id'].unique()[:args.compare]
    models = ['goals', 'assists', 'minutes', 'defcon', 'clean_sheet', 'saves']

    print(f"\n=== Comparing Last {len(run_ids)} Runs ===\n")

    # Header
    header = f"{'Model':<15}"
    for rid in run_ids:
        header += f" {rid[:15]:<17}"
    print(header)
    print("-" * (15 + 17 * len(run_ids)))

    for model_name in models:
        row = f"{model_name:<15}"
        for rid in run_ids:
            match = df[(df['run_id'] == rid) & (df['model_name'] == model_name)]
            if not match.empty:
                val = match.iloc[0]['test_score']
                row += f" {val:<17.4f}" if val is not None else f" {'N/A':<17}"
            else:
                row += f" {'--':<17}"
        print(row)

    # FPL points row
    row = f"{'FPL pts MAE':<15}"
    for rid in run_ids:
        match = df[df['run_id'] == rid]
        if not match.empty:
            val = match.iloc[0].get('fpl_points_mae')
            row += f" {val:<17.4f}" if val is not None else f" {'N/A':<17}"
        else:
            row += f" {'--':<17}"
    print(row)


def main():
    parser = argparse.ArgumentParser(description='FPL Experiment Runner')
    parser.add_argument('--trials', '-t', type=int, default=100, help='Optuna trials per model')
    parser.add_argument('--models', '-m', nargs='+', default=None, help='Models to tune')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--desc', '-d', type=str, default='', help='Experiment description')
    parser.add_argument('--subprocess', action='store_true', help='Use subprocess for tuning')
    parser.add_argument('--history', action='store_true', help='Show experiment history')
    parser.add_argument('--best', action='store_true', help='Show best runs per model')
    parser.add_argument('--compare', type=int, default=0, help='Compare last N runs')
    parser.add_argument('--model', type=str, default=None, help='Filter history to model')
    args = parser.parse_args()

    if args.history:
        show_history(args)
    elif args.best:
        show_best(args)
    elif args.compare > 0:
        compare_runs(args)
    else:
        run_experiment(args)


if __name__ == '__main__':
    main()
