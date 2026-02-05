#!/usr/bin/env python
"""Hyperparameter tuning for FPL models."""
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import uniform, randint, loguniform
import xgboost as xgb

from src.data_loader import load_player_stats, load_fixtures, merge_fixtures
from src.features import compute_rolling_features
from src.models import GoalsModel, AssistsModel, MinutesModel, DefconModel


# Search spaces for each model
SEARCH_SPACES = {
    'goals': {
        'n_estimators': randint(100, 400),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    },
    'assists': {
        'n_estimators': randint(100, 400),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'min_child_weight': randint(1, 10),
    },
    'minutes': {
        'n_estimators': randint(100, 400),
        'max_depth': randint(3, 8),
        'learning_rate': loguniform(0.01, 0.3),
        'min_child_weight': randint(1, 15),
    },
    'defcon': {
        'n_estimators': randint(100, 400),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'min_child_weight': randint(1, 10),
    },
}

MODEL_CLASSES = {
    'goals': GoalsModel,
    'assists': AssistsModel,
    'minutes': MinutesModel,
    'defcon': DefconModel,
}


def tune_model(model_name: str, df: pd.DataFrame, n_iter: int = 50, verbose: bool = True):
    """Tune a single model using RandomizedSearchCV."""
    if model_name not in MODEL_CLASSES:
        print(f"Unknown model: {model_name}")
        return None
    
    model_class = MODEL_CLASSES[model_name]
    search_space = SEARCH_SPACES.get(model_name, {})
    
    # Filter data
    df_train = df[df['minutes'] >= 1].copy()
    
    # Prepare features and target
    model_instance = model_class()
    features = model_instance.FEATURES
    target = model_instance.TARGET
    
    available_features = [f for f in features if f in df_train.columns]
    for f in available_features:
        df_train[f] = df_train[f].fillna(0)
    
    X = df_train[available_features].values
    y = df_train[target].fillna(0).values
    
    if verbose:
        print(f"\nTuning {model_name.upper()} ({n_iter} iterations)...")
        print(f"  Features: {len(available_features)}, Samples: {len(X):,}")
    
    # Base model
    base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
    
    # Run RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=n_iter,
        cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42,
        n_jobs=1,
        verbose=1 if verbose else 0,
    )
    
    search.fit(X, y)
    
    best_params = search.best_params_
    best_score = -search.best_score_
    
    if verbose:
        print(f"  Best RMSE: {best_score:.4f}")
        print(f"  Best params: {best_params}")
    
    return {
        'model_name': model_name,
        'best_params': best_params,
        'best_rmse': best_score,
        'features': available_features,
    }


def main():
    parser = argparse.ArgumentParser(description='Tune FPL models')
    parser.add_argument('--trials', '-t', type=int, default=50, help='Number of trials per model')
    parser.add_argument('--models', '-m', nargs='+', default=['goals', 'assists', 'minutes', 'defcon'],
                        help='Models to tune')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FPL MODEL TUNING")
    print("=" * 60)
    
    # Load data
    data_dir = Path('data')
    df = load_player_stats(data_dir)
    fixtures = load_fixtures(data_dir)
    df = df.merge(fixtures[['match_id', 'gameweek']],
                  on='match_id', how='left')
    df = merge_fixtures(df, fixtures)
    
    # Filter to 2020/21 and beyond
    MIN_SEASON = '2020/2021'
    valid_seasons = [s for s in sorted(df['season'].unique()) if s >= MIN_SEASON]
    df = df[df['season'].isin(valid_seasons)].copy()
    print(f"Using seasons: {valid_seasons}")
    
    df = compute_rolling_features(df, verbose=True)
    
    # Tune each model
    results = {}
    output_dir = data_dir / 'tuning_results'
    output_dir.mkdir(exist_ok=True)
    
    for model_name in args.models:
        result = tune_model(model_name, df, n_iter=args.trials)
        if result:
            results[model_name] = result
            
            # Save to JSON
            output_file = output_dir / f'{model_name}_tuned.json'
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Saved: {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<12} {'RMSE':<10}")
    print("-" * 22)
    for name, res in results.items():
        print(f"{name.upper():<12} {res['best_rmse']:.4f}")


if __name__ == '__main__':
    main()
