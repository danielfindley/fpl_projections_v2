"""Pre-computed feature rankings for Optuna-integrated feature selection.

Rankings are computed once before tuning starts, then each Optuna trial
picks a ranking method + number of features as hyperparameters.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings


def compute_feature_rankings(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task: str = 'regression',
    xgb_params: dict = None,
) -> Dict[str, List[str]]:
    """Pre-compute feature rankings using multiple methods.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array
        feature_names: List of feature names matching X columns
        task: 'regression' or 'classification'
        xgb_params: Optional XGBoost params (used for tree-based rankings)

    Returns:
        Dict mapping method name -> list of feature names ranked best-to-worst
    """
    rankings = {}

    rankings['xgb_gain'] = _rank_xgb(X, y, feature_names, 'gain', task, xgb_params)
    rankings['xgb_cover'] = _rank_xgb(X, y, feature_names, 'cover', task, xgb_params)
    rankings['lgbm'] = _rank_lgbm(X, y, feature_names, task)
    rankings['permutation'] = _rank_permutation(X, y, feature_names, task, xgb_params)
    rankings['mutual_info'] = _rank_mutual_info(X, y, feature_names, task)

    return rankings


def _rank_xgb(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    importance_type: str, task: str, xgb_params: dict = None,
) -> List[str]:
    """Rank features by XGBoost built-in importance (gain or cover)."""
    import xgboost as xgb

    params = {
        'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
    }
    if xgb_params:
        params.update({k: v for k, v in xgb_params.items()
                       if k not in ('selected_features',)})

    if task == 'classification':
        model = xgb.XGBClassifier(**params)
    else:
        model = xgb.XGBRegressor(**params)

    model.fit(X, y)

    booster = model.get_booster()
    scores = booster.get_score(importance_type=importance_type)
    # Map f0, f1, ... back to feature names
    importance = np.zeros(len(feature_names))
    for key, val in scores.items():
        idx = int(key[1:])  # 'f0' -> 0
        importance[idx] = val

    ranked_idx = np.argsort(-importance)
    return [feature_names[i] for i in ranked_idx]


def _rank_lgbm(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], task: str,
) -> List[str]:
    """Rank features by LightGBM split-based importance."""
    import lightgbm as lgb

    if task == 'classification':
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=-1, n_jobs=-1,
        )

    model.fit(X, y)
    importance = model.feature_importances_
    ranked_idx = np.argsort(-importance)
    return [feature_names[i] for i in ranked_idx]


def _rank_permutation(
    X: np.ndarray, y: np.ndarray, feature_names: List[str],
    task: str, xgb_params: dict = None,
) -> List[str]:
    """Rank features by permutation importance (model-agnostic)."""
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import TimeSeriesSplit
    import xgboost as xgb

    params = {
        'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
    }
    if xgb_params:
        params.update({k: v for k, v in xgb_params.items()
                       if k not in ('selected_features',)})

    if task == 'classification':
        model = xgb.XGBClassifier(**params)
        scoring = 'neg_log_loss'
    else:
        model = xgb.XGBRegressor(**params)
        scoring = 'neg_mean_absolute_error'

    # Use last fold of TimeSeriesSplit for eval
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]

    model.fit(X[train_idx], y[train_idx])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = permutation_importance(
            model, X[val_idx], y[val_idx],
            n_repeats=5, random_state=42, scoring=scoring, n_jobs=-1,
        )

    importance = result.importances_mean
    ranked_idx = np.argsort(-importance)
    return [feature_names[i] for i in ranked_idx]


def _rank_mutual_info(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], task: str,
) -> List[str]:
    """Rank features by mutual information (non-parametric, model-free)."""
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

    if task == 'classification':
        mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=5)

    ranked_idx = np.argsort(-mi_scores)
    return [feature_names[i] for i in ranked_idx]


def select_features(
    rankings: Dict[str, List[str]],
    method: str,
    n_features: int,
    protected_features: List[str] = None,
) -> List[str]:
    """Select top-N features from a specific ranking method.

    Protected features (e.g. pred_minutes, pred_team_goals) are always included
    and don't count toward n_features.

    Args:
        rankings: Pre-computed rankings dict
        method: Ranking method name (key in rankings)
        n_features: Number of features to select (excluding protected)
        protected_features: Features that must always be included

    Returns:
        List of selected feature names
    """
    protected = set(protected_features or [])
    ranked = rankings[method]

    selected = []
    for feat in ranked:
        if feat in protected:
            continue
        selected.append(feat)
        if len(selected) >= n_features:
            break

    # Add protected features
    return selected + [f for f in protected if f in ranked]
