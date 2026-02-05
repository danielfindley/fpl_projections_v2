"""
FPL Prediction Pipeline

Main pipeline for loading data, computing features, training models, and generating predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .data_loader import load_player_stats, load_fixtures, merge_fixtures, get_fpl_positions, map_fpl_position, get_fpl_availability
from .features import compute_rolling_features
from .models import GoalsModel, AssistsModel, MinutesModel, DefconModel, CleanSheetModel, BonusModel


# FPL point values
FPL_POINTS = {
    'goal': {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4},
    'assist': 3,
    'clean_sheet': {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0},
    'defcon': 2,
    'appearance_60': 2,
    'appearance_1': 1,
}

# Team name normalization (FPL API name -> canonical name)
TEAM_NAME_MAP = {
    'man city': 'manchester city',
    'man utd': 'manchester united',
    'spurs': 'tottenham hotspur',
    'tottenham': 'tottenham hotspur',
    "nott'm forest": 'nottingham forest',
    'nottm forest': 'nottingham forest',
    'wolves': 'wolverhampton wanderers',
    'wolverhampton': 'wolverhampton wanderers',
    'brighton': 'brighton and hove albion',
    'brighton & hove albion': 'brighton and hove albion',
    'west ham': 'west ham united',
    'newcastle': 'newcastle united',
    'leicester': 'leicester city',
    'leeds': 'leeds united',
}


def normalize_team_name(name: str) -> str:
    """Normalize team name to canonical form for matching."""
    if pd.isna(name):
        return ''
    name_lower = str(name).lower().strip()
    # Check direct mapping
    if name_lower in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name_lower]
    # Return lowercased name
    return name_lower


class FPLPipeline:
    """Complete FPL prediction pipeline."""
    
    # Only use seasons from 2020/21 onwards for training
    MIN_SEASON = '2020/2021'
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.df = None
        self.models: Dict = {}
        self.tuned_params: Dict = {}  # Store tuned hyperparameters
        self.fpl_positions = {}
        self.current_season_players = set()  # Track players in current season
    
    def load_data(self, verbose: bool = True) -> 'FPLPipeline':
        """Load and merge all data."""
        if verbose:
            print("=" * 60)
            print("LOADING DATA")
            print("=" * 60)
        
        self.df = load_player_stats(self.data_dir, verbose)
        fixtures = load_fixtures(self.data_dir, verbose)
        
        # Merge to get gameweek from fixtures (already renamed in load_fixtures)
        self.df = self.df.merge(
            fixtures[['match_id', 'gameweek']],
            on='match_id', how='left'
        )
        
        # Merge to get opponent and is_home
        self.df = merge_fixtures(self.df, fixtures)
        
        # CRITICAL: Final deduplication after all merges
        n_before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['match_id', 'player_id'], keep='first')
        n_dupes = n_before - len(self.df)
        if verbose and n_dupes > 0:
            print(f"Removed {n_dupes:,} duplicate rows after merges")
        
        # Filter to 2020/21 and beyond for training
        all_seasons = sorted(self.df['season'].unique())
        valid_seasons = [s for s in all_seasons if s >= self.MIN_SEASON]
        self.df = self.df[self.df['season'].isin(valid_seasons)].copy()
        
        # Track current season players (most recent season)
        current_season = max(valid_seasons)
        self.current_season_players = set(
            self.df[self.df['season'] == current_season]['player_id'].unique()
        )
        
        if verbose:
            print(f"Filtered to seasons: {valid_seasons}")
            print(f"Current season ({current_season}): {len(self.current_season_players)} active players")
            print(f"Final dataset: {len(self.df):,} records")
        
        return self
    
    def compute_features(self, verbose: bool = True) -> 'FPLPipeline':
        """Compute all rolling features."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if verbose:
            print("\n" + "=" * 60)
            print("COMPUTING FEATURES")
            print("=" * 60)
        
        self.df = compute_rolling_features(self.df, verbose)
        return self
    
    def train(self, verbose: bool = True) -> 'FPLPipeline':
        """Train all models on full dataset using tuned params if available.
        
        Call this after tune() to train final models on ALL data before prediction.
        """
        if self.df is None:
            raise ValueError("Data not loaded")
        
        if verbose:
            print("\n" + "=" * 60)
            print("TRAINING FINAL MODELS ON ALL DATA")
            if self.tuned_params:
                print("(using tuned hyperparameters)")
            else:
                print("(using default hyperparameters)")
            print("=" * 60)
        
        # Minutes model
        mins_params = self.tuned_params.get('minutes', {})
        self.models['minutes'] = MinutesModel(**mins_params)
        self.models['minutes'].fit(self.df, verbose)
        
        # Goals model
        goals_params = self.tuned_params.get('goals', {})
        self.models['goals'] = GoalsModel(**goals_params)
        self.models['goals'].fit(self.df, verbose)
        
        # Assists model
        assists_params = self.tuned_params.get('assists', {})
        self.models['assists'] = AssistsModel(**assists_params)
        self.models['assists'].fit(self.df, verbose)
        
        # Defcon model
        defcon_params = self.tuned_params.get('defcon', {})
        self.models['defcon'] = DefconModel(**defcon_params)
        self.models['defcon'].fit(self.df, verbose)
        
        # Clean sheet model (uses tuned params if available)
        cs_params = self.tuned_params.get('clean_sheet', {})
        self.models['clean_sheet'] = CleanSheetModel(**cs_params)
        self.models['clean_sheet'].fit(self.df, verbose)
        
        # Bonus model (not tuned - uses Monte Carlo simulation)
        self.models['bonus'] = BonusModel()
        self.models['bonus'].fit(self.df, verbose)
        
        return self
    
    def tune(self, models: list = None, n_iter: int = 100, test_size: float = 0.2,
             verbose: bool = True, use_subprocess: bool = False) -> 'FPLPipeline':
        """Tune model hyperparameters using Optuna with holdout test set evaluation.
        
        This method:
        1. Splits data into train/test (temporal split - most recent data held out)
        2. Tunes hyperparameters using 5-fold CV on training set
        3. Evaluates best models on held-out test set
        4. Stores tuned params (does NOT train final models - call train() for that)
        
        Args:
            models: List of model names to tune. Defaults to ['goals', 'assists', 'minutes', 'defcon', 'clean_sheet']
            n_iter: Number of Optuna trials per model (default 100)
            test_size: Fraction of data to hold out for testing (default 0.2)
            verbose: Print progress
            use_subprocess: If True, run each model's tuning in a separate subprocess to save RAM
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() and compute_features() first.")
        
        # Default to tuning these models (includes clean_sheet classifier)
        if models is None:
            models = ['goals', 'assists', 'minutes', 'defcon', 'clean_sheet']
        
        if verbose:
            print("\n" + "=" * 60)
            print("TUNING HYPERPARAMETERS WITH HOLDOUT TEST SET")
            print("=" * 60)
        
        # Create temporal train/test split
        # Sort by season and gameweek, hold out most recent data
        df_sorted = self.df[self.df['minutes'] >= 1].copy()
        df_sorted = df_sorted.sort_values(['season', 'gameweek'])
        
        n_total = len(df_sorted)
        n_test = int(n_total * test_size)
        n_train = n_total - n_test
        
        df_train = df_sorted.iloc[:n_train].copy()
        df_test = df_sorted.iloc[n_train:].copy()
        
        if verbose:
            print(f"\nData split (temporal):")
            print(f"  Train: {len(df_train):,} samples")
            print(f"  Test:  {len(df_test):,} samples (most recent {test_size:.0%})")
            
            # Show what's in test set
            test_seasons = df_test['season'].unique()
            test_gws = df_test.groupby('season')['gameweek'].agg(['min', 'max'])
            print(f"  Test set spans: {list(test_seasons)}")
        
        if verbose:
            print("\n" + "-" * 60)
            print("PHASE 1: Hyperparameter Tuning (5-fold CV on train set)")
            print("-" * 60)
        
        if use_subprocess:
            self.tuned_params = self._tune_with_subprocess(models, n_iter, verbose, df_train)
        else:
            self.tuned_params = self._tune_in_process(models, n_iter, verbose, df_train)
        
        # Evaluate on held-out test set
        if verbose:
            print("\n" + "-" * 60)
            print("PHASE 2: Evaluation on Held-Out Test Set")
            print("-" * 60)
        
        self._evaluate_on_test_set(models, df_train, df_test, verbose)
        
        if verbose:
            print("\n" + "=" * 60)
            print("TUNING COMPLETE")
            print("Tuned params stored. Call train() to train final models on ALL data.")
            print("=" * 60)
        
        return self
    
    def _tune_in_process(self, models: list, n_iter: int, verbose: bool, df_train: pd.DataFrame) -> Dict:
        """Run tuning in the current process using Optuna with 5-fold CV.
        
        Uses appropriate loss functions:
        - Goals, Assists, Defcon: Poisson deviance (good for count/rate data)
        - Minutes: Huber loss (robust to outliers)
        - Clean Sheet: Log loss (binary classification)
        """
        import optuna
        from sklearn.model_selection import cross_val_score, KFold
        from sklearn.metrics import make_scorer, mean_poisson_deviance
        import xgboost as xgb
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Custom Huber loss scorer
        def huber_loss(y_true, y_pred, delta=10.0):
            """Huber loss - robust to outliers. Delta=10 for minutes prediction."""
            residual = np.abs(y_true - y_pred)
            quadratic = np.minimum(residual, delta)
            linear = residual - quadratic
            return np.mean(0.5 * quadratic**2 + delta * linear)
        
        huber_scorer = make_scorer(huber_loss, greater_is_better=False)
        
        # Poisson deviance scorer (need to handle zeros)
        def safe_poisson_deviance(y_true, y_pred):
            """Poisson deviance that handles edge cases."""
            y_pred = np.clip(y_pred, 1e-8, None)
            y_true = np.clip(y_true, 0, None)
            return mean_poisson_deviance(y_true, y_pred)
        
        poisson_scorer = make_scorer(safe_poisson_deviance, greater_is_better=False)
        
        # Model-specific scoring (regressors)
        SCORING = {
            'goals': ('Poisson Deviance', poisson_scorer),
            'assists': ('Poisson Deviance', poisson_scorer),
            'defcon': ('Poisson Deviance', poisson_scorer),
            'minutes': ('Huber Loss', huber_scorer),
        }
        
        SEARCH_SPACES = {
            'goals': {
                'n_estimators': (100, 400),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
            },
            'assists': {
                'n_estimators': (100, 400),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
            },
            'minutes': {
                'n_estimators': (100, 400),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 15),
            },
            'defcon': {
                'n_estimators': (100, 400),
                'max_depth': (3, 10),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
            },
            'clean_sheet': {
                'n_estimators': (50, 300),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
            },
        }
        
        MODEL_CLASSES = {
            'goals': GoalsModel,
            'assists': AssistsModel,
            'minutes': MinutesModel,
            'defcon': DefconModel,
            'clean_sheet': CleanSheetModel,
        }
        
        tuned_params = {}
        
        for model_name in models:
            if model_name not in MODEL_CLASSES:
                continue
            
            # Handle clean_sheet classifier separately
            if model_name == 'clean_sheet':
                tuned_params['clean_sheet'] = self._tune_clean_sheet_in_process(
                    n_iter, verbose, df_train, SEARCH_SPACES['clean_sheet']
                )
                continue
                
            model_class = MODEL_CLASSES[model_name]
            space = SEARCH_SPACES.get(model_name, {})
            score_name, scorer = SCORING.get(model_name, ('RMSE', 'neg_root_mean_squared_error'))
            
            model_instance = model_class()
            features = [f for f in model_instance.FEATURES if f in df_train.columns]
            target = model_instance.TARGET
            
            X = df_train[features].fillna(0).values
            y = df_train[target].fillna(0).values
            
            if verbose:
                print(f"\nTuning {model_name.upper()} ({n_iter} trials, 5-fold CV, {score_name})...")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                    'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
                    'random_state': 42,
                    'verbosity': 0,
                    'n_jobs': -1,
                }
                
                model = xgb.XGBRegressor(**params)
                scores = cross_val_score(model, X, y, cv=5, scoring=scorer, n_jobs=-1)
                return -scores.mean()  # Optuna minimizes, scorer returns negative
            
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_iter, show_progress_bar=verbose)
            
            tuned_params[model_name] = study.best_params
            if verbose:
                print(f"  Best CV {score_name}: {study.best_value:.4f}")
                print(f"  Params: {study.best_params}")
        
        return tuned_params
    
    def _tune_clean_sheet_in_process(self, n_iter: int, verbose: bool, 
                                      df_train: pd.DataFrame, space: dict) -> Dict:
        """Tune CleanSheetModel (classifier) using Brier score."""
        import optuna
        from sklearn.model_selection import cross_val_score
        import xgboost as xgb
        
        # Prepare team-level data
        cs_model = CleanSheetModel()
        team_df = cs_model.prepare_team_features(df_train)
        team_df = team_df.dropna(subset=['team_conceded_roll5'])
        
        features = [f for f in cs_model.FEATURES if f in team_df.columns]
        X = team_df[features].fillna(0).values
        y = team_df['clean_sheet'].values
        
        if verbose:
            print(f"\nTuning CLEAN_SHEET ({n_iter} trials, 5-fold CV, Brier Score)...")
            print(f"  Team-matches: {len(X)}, CS rate: {y.mean():.1%}")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
                'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1,
                'eval_metric': 'logloss',
            }
            
            model = xgb.XGBClassifier(**params)
            # neg_brier_score is the negative Brier score (higher/less negative is better)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_brier_score', n_jobs=-1)
            return -scores.mean()  # Optuna minimizes, return positive Brier score
        
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_iter, show_progress_bar=verbose)
        
        if verbose:
            print(f"  Best CV Brier Score: {study.best_value:.4f}")
            print(f"  Params: {study.best_params}")
        
        return study.best_params
    
    def _tune_with_subprocess(self, models: list, n_iter: int, verbose: bool, df_train: pd.DataFrame) -> Dict:
        """Run each model's tuning in a separate subprocess to save RAM using Optuna with 5-fold CV.
        
        Uses appropriate loss functions:
        - Goals, Assists, Defcon: Poisson deviance (good for count/rate data)
        - Minutes: Huber loss (robust to outliers)
        - Clean Sheet: Log loss (binary classification)
        """
        import subprocess
        import sys
        import json
        import tempfile
        import os
        
        # Scoring type for each model
        SCORING_NAMES = {
            'goals': 'Poisson Deviance',
            'assists': 'Poisson Deviance',
            'defcon': 'Poisson Deviance',
            'minutes': 'Huber Loss',
            'clean_sheet': 'Brier Score',
        }
        
        SEARCH_SPACES = {
            'clean_sheet': {
                'n_estimators': (50, 300),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'min_child_weight': (1, 10),
            },
        }
        
        tuned_params = {}
        
        # Save training data to temp file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            temp_data_path = f.name
            df_train.to_csv(f, index=False)
        
        try:
            for model_name in models:
                # Handle clean_sheet specially (needs team-level aggregation)
                if model_name == 'clean_sheet':
                    tuned_params['clean_sheet'] = self._tune_clean_sheet_in_process(
                        n_iter, verbose, df_train, SEARCH_SPACES['clean_sheet']
                    )
                    continue
                
                score_name = SCORING_NAMES.get(model_name, 'RMSE')
                if verbose:
                    print(f"\nTuning {model_name.upper()} ({n_iter} trials, 5-fold CV, {score_name}) in subprocess...")
                
                # Create temp file for results
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    temp_result_path = f.name
                
                # Python script to run in subprocess
                tune_script = f'''
import pandas as pd
import numpy as np
import json
import sys
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_poisson_deviance
import xgboost as xgb

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add parent dir to path
sys.path.insert(0, r"{self.data_dir.parent}")

from src.models import GoalsModel, AssistsModel, MinutesModel, DefconModel

# Custom Huber loss scorer
def huber_loss(y_true, y_pred, delta=10.0):
    """Huber loss - robust to outliers. Delta=10 for minutes prediction."""
    residual = np.abs(y_true - y_pred)
    quadratic = np.minimum(residual, delta)
    linear = residual - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)

huber_scorer = make_scorer(huber_loss, greater_is_better=False)

# Poisson deviance scorer (need to handle zeros)
def safe_poisson_deviance(y_true, y_pred):
    """Poisson deviance that handles edge cases."""
    y_pred = np.clip(y_pred, 1e-8, None)
    y_true = np.clip(y_true, 0, None)
    return mean_poisson_deviance(y_true, y_pred)

poisson_scorer = make_scorer(safe_poisson_deviance, greater_is_better=False)

# Model-specific scoring
SCORING = {{
    'goals': poisson_scorer,
    'assists': poisson_scorer,
    'defcon': poisson_scorer,
    'minutes': huber_scorer,
}}

MODEL_CLASSES = {{
    'goals': GoalsModel,
    'assists': AssistsModel,
    'minutes': MinutesModel,
    'defcon': DefconModel,
}}

SEARCH_SPACES = {{
    'goals': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
    }},
    'assists': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
    }},
    'minutes': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 8),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 15),
    }},
    'defcon': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 10),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
    }},
}}

model_name = "{model_name}"
n_iter = {n_iter}

df_train = pd.read_csv(r"{temp_data_path}", encoding='utf-8')
model_class = MODEL_CLASSES[model_name]
space = SEARCH_SPACES[model_name]
scorer = SCORING[model_name]

model_instance = model_class()
features = [f for f in model_instance.FEATURES if f in df_train.columns]
target = model_instance.TARGET

X = df_train[features].fillna(0).values
y = df_train[target].fillna(0).values

def objective(trial):
    params = {{
        'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
        'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
        'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }}
    
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring=scorer, n_jobs=-1)
    return -scores.mean()  # Optuna minimizes, scorer returns negative

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

result = {{
    'best_params': study.best_params,
    'best_score': study.best_value
}}

with open(r"{temp_result_path}", 'w') as f:
    json.dump(result, f)

print(f"Best CV {score_name}: {{study.best_value:.4f}}")
print(f"Params: {{study.best_params}}")
'''
                
                # Run in subprocess
                result = subprocess.run(
                    [sys.executable, '-c', tune_script],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    if verbose:
                        print(f"  ERROR: {result.stderr}")
                    continue
                
                # Print subprocess output
                if verbose and result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        print(f"  {line}")
                
                # Read results
                try:
                    with open(temp_result_path, 'r') as f:
                        tune_result = json.load(f)
                    tuned_params[model_name] = tune_result['best_params']
                except Exception as e:
                    if verbose:
                        print(f"  Failed to read results: {e}")
                finally:
                    # Clean up result file
                    try:
                        os.unlink(temp_result_path)
                    except:
                        pass
        finally:
            # Clean up data file
            try:
                os.unlink(temp_data_path)
            except:
                pass
        
        return tuned_params
    
    def _evaluate_on_test_set(self, models: list, df_train: pd.DataFrame, 
                               df_test: pd.DataFrame, verbose: bool):
        """Train models on train set with tuned params and evaluate on test set.
        
        Reports metrics matching the tuning loss functions:
        - Goals, Assists, Defcon: Poisson deviance + MAE
        - Minutes: Huber loss + MAE
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
        import xgboost as xgb
        
        # Custom Huber loss
        def huber_loss(y_true, y_pred, delta=10.0):
            residual = np.abs(y_true - y_pred)
            quadratic = np.minimum(residual, delta)
            linear = residual - quadratic
            return np.mean(0.5 * quadratic**2 + delta * linear)
        
        # Safe Poisson deviance
        def safe_poisson_deviance(y_true, y_pred):
            y_pred = np.clip(y_pred, 1e-8, None)
            y_true = np.clip(y_true, 0, None)
            return mean_poisson_deviance(y_true, y_pred)
        
        from sklearn.metrics import roc_auc_score
        
        MODEL_CLASSES = {
            'goals': GoalsModel,
            'assists': AssistsModel,
            'minutes': MinutesModel,
            'defcon': DefconModel,
        }
        
        # Model-specific primary metrics (for regressors)
        PRIMARY_METRICS = {
            'goals': ('Poisson Dev', safe_poisson_deviance),
            'assists': ('Poisson Dev', safe_poisson_deviance),
            'defcon': ('Poisson Dev', safe_poisson_deviance),
            'minutes': ('Huber Loss', huber_loss),
        }
        
        # Store test metrics
        test_metrics = {}
        
        # Handle regression models
        for model_name in models:
            if model_name == 'clean_sheet':
                continue  # Handle separately below
            if model_name not in MODEL_CLASSES or model_name not in self.tuned_params:
                continue
            
            model_class = MODEL_CLASSES[model_name]
            params = self.tuned_params[model_name]
            metric_name, metric_fn = PRIMARY_METRICS[model_name]
            
            # Create model with tuned params
            model_instance = model_class(**params)
            features = [f for f in model_instance.FEATURES if f in df_train.columns]
            target = model_instance.TARGET
            
            # Prepare train data
            X_train = df_train[features].fillna(0).values
            y_train = df_train[target].fillna(0).values
            
            # Prepare test data
            X_test = df_test[features].fillna(0).values
            y_test = df_test[target].fillna(0).values
            
            # Train on train set
            xgb_model = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            
            # Predict on test set
            y_pred = xgb_model.predict(X_test)
            
            # Calculate metrics
            primary_metric = metric_fn(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            test_metrics[model_name] = {
                'metric_name': metric_name,
                'primary': primary_metric,
                'MAE': mae
            }
        
        # Handle clean_sheet classifier separately (team-level data)
        if 'clean_sheet' in models and 'clean_sheet' in self.tuned_params:
            from sklearn.metrics import brier_score_loss
            
            params = self.tuned_params['clean_sheet']
            cs_model = CleanSheetModel()
            
            # Prepare team-level train data
            team_train = cs_model.prepare_team_features(df_train)
            team_train = team_train.dropna(subset=['team_conceded_roll5'])
            
            # Prepare team-level test data
            team_test = cs_model.prepare_team_features(df_test)
            team_test = team_test.dropna(subset=['team_conceded_roll5'])
            
            features = [f for f in cs_model.FEATURES if f in team_train.columns]
            
            X_train = team_train[features].fillna(0).values
            y_train = team_train['clean_sheet'].values
            X_test = team_test[features].fillna(0).values
            y_test = team_test['clean_sheet'].values
            
            # Train classifier
            xgb_model = xgb.XGBClassifier(**params, random_state=42, verbosity=0, n_jobs=-1, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics - Brier score (lower is better) and AUC (higher is better)
            brier = brier_score_loss(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            test_metrics['clean_sheet'] = {
                'metric_name': 'Brier Score',
                'primary': brier,
                'MAE': auc  # Use AUC as secondary metric for classifiers
            }
        
        # Print test set performance
        if verbose:
            print(f"\n{'Model':<12} {'Metric':<12} {'Test Value':<12} {'Secondary':<12}")
            print("-" * 48)
            for model_name, metrics in test_metrics.items():
                secondary_label = 'AUC' if model_name == 'clean_sheet' else 'MAE'
                print(f"{model_name.upper():<12} {metrics['metric_name']:<12} {metrics['primary']:<12.4f} {metrics['MAE']:<12.4f}")
            
            # Add context for interpretation
            print("\n(Lower is better for all metrics except AUC which should be higher)")
            print("Poisson Dev: count/rate data | Huber: robust to outliers | Brier: probability calibration")
    
    def predict(self, gameweek: int, season: str = '2025/2026', 
                verbose: bool = True) -> pd.DataFrame:
        """Generate predictions for a target gameweek."""
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"PREDICTING GW{gameweek} ({season})")
            print("=" * 60)
        
        # Get test data - latest features for players
        test_df = self._build_test_set(gameweek, season, verbose)
        
        if len(test_df) == 0:
            print("WARNING: No players found for prediction")
            return pd.DataFrame()
        
        # Generate predictions
        test_df['pred_minutes'] = self.models['minutes'].predict(test_df)
        test_df['pred_goals_per90'] = self.models['goals'].predict(test_df)
        test_df['pred_exp_goals'] = self.models['goals'].predict_expected(test_df, test_df['pred_minutes'])
        test_df['pred_assists_per90'] = self.models['assists'].predict(test_df)
        test_df['pred_exp_assists'] = self.models['assists'].predict_expected(test_df, test_df['pred_minutes'])
        test_df['pred_defcon_prob'] = self.models['defcon'].predict_threshold_prob(test_df, test_df['pred_minutes'])
        
        # Clean sheet - need team-level features
        test_df['pred_cs_prob'] = self._predict_clean_sheet(test_df, gameweek, season, verbose)
        
        # FPL positions (needed for bonus)
        self.fpl_positions = get_fpl_positions()
        test_df['fpl_position'] = test_df.apply(
            lambda r: map_fpl_position(r.get('position'), r.get('player_name'), self.fpl_positions),
            axis=1
        )
        
        # Bonus - pass all predictions for proper simulation
        test_df['pred_bonus'] = self.models['bonus'].predict(
            test_df,
            pred_goals=test_df['pred_exp_goals'].values,
            pred_assists=test_df['pred_exp_assists'].values,
            pred_cs_prob=test_df['pred_cs_prob'].values,
            pred_minutes=test_df['pred_minutes'].values,
            fpl_positions=test_df['fpl_position'].values
        )
        
        # Calculate expected points
        test_df = self._calculate_expected_points(test_df)
        
        # Save predictions
        output_path = self.data_dir / 'predictions' / f'gw{gameweek}_{season.replace("/", "-")}.csv'
        output_path.parent.mkdir(exist_ok=True)
        test_df.to_csv(output_path, index=False)
        
        if verbose:
            print(f"\nSaved predictions to: {output_path}")
            print(f"Total players: {len(test_df)}")
        
        return test_df
    
    def _build_test_set(self, gameweek: int, season: str, verbose: bool) -> pd.DataFrame:
        """Build test set for prediction from latest available data."""
        # Get data before target gameweek
        prior_data = self.df[
            ~((self.df['season'] == season) & (self.df['gameweek'] >= gameweek))
        ].copy()
        
        if len(prior_data) == 0:
            prior_data = self.df.copy()
        
        # Get latest features per player
        latest = prior_data.sort_values(['player_id', 'season', 'gameweek']).groupby('player_id').last().reset_index()
        
        # Filter to only players active in current season
        latest = latest[latest['player_id'].isin(self.current_season_players)].copy()
        
        if verbose:
            print(f"Found {len(latest)} active players with historical data")
        
        # Get FPL availability and filter out unavailable players
        fpl_availability = get_fpl_availability()
        if fpl_availability:
            def get_availability(name):
                if pd.isna(name):
                    return 100
                name_lower = str(name).lower()
                if name_lower in fpl_availability:
                    chance = fpl_availability[name_lower].get('chance_of_playing')
                    status = fpl_availability[name_lower].get('status', 'a')
                    # If status is injured/suspended/unavailable with 0% chance, exclude
                    if status in ['i', 's', 'u'] and chance == 0:
                        return 0
                    return chance if chance is not None else 100
                # Try last name
                parts = str(name).split()
                if len(parts) > 1:
                    last = parts[-1].lower()
                    if last in fpl_availability:
                        chance = fpl_availability[last].get('chance_of_playing')
                        status = fpl_availability[last].get('status', 'a')
                        if status in ['i', 's', 'u'] and chance == 0:
                            return 0
                        return chance if chance is not None else 100
                return 100  # Default to available
            
            latest['fpl_chance_of_playing'] = latest['player_name'].apply(get_availability)
            
            # Filter: exclude players with 0% chance (injured/suspended)
            n_before = len(latest)
            latest = latest[latest['fpl_chance_of_playing'] > 0].copy()
            n_filtered = n_before - len(latest)
            
            if verbose and n_filtered > 0:
                print(f"  Filtered out {n_filtered} unavailable players (injured/suspended)")
        
        # Get fixtures for target gameweek
        fixtures = self._get_gw_fixtures(gameweek, season)
        
        if len(fixtures) == 0:
            if verbose:
                print("No fixtures found - using latest data as-is")
            return latest
        
        # Build lookup of latest team stats for updating opponent features
        team_stats_lookup = self._build_team_stats_lookup()
        
        # Map players to fixtures using normalized team names
        test_rows = []
        
        # Pre-normalize fixture team names
        fixture_teams = []
        for _, fix in fixtures.iterrows():
            home_norm = normalize_team_name(fix['home_team'])
            away_norm = normalize_team_name(fix['away_team'])
            fixture_teams.append({
                'home_norm': home_norm,
                'away_norm': away_norm,
                'home_team': fix['home_team'],
                'away_team': fix['away_team'],
            })
        
        for _, player in latest.iterrows():
            player_team = normalize_team_name(player.get('team', ''))
            
            # Find matching fixture
            for fix in fixture_teams:
                # Check if player's team matches home or away team
                if player_team == fix['home_norm'] or player_team in fix['home_norm'] or fix['home_norm'] in player_team:
                    row = player.to_dict()
                    row['opponent'] = fix['away_team']
                    row['is_home'] = 1
                    row['gameweek'] = gameweek
                    row['season'] = season
                    # Update team features to latest (fixes stale stats for players who missed games)
                    self._update_team_features(row, player.get('team', ''), team_stats_lookup)
                    # Update opponent features based on actual opponent
                    self._update_opponent_features(row, fix['away_team'], team_stats_lookup)
                    test_rows.append(row)
                    break
                elif player_team == fix['away_norm'] or player_team in fix['away_norm'] or fix['away_norm'] in player_team:
                    row = player.to_dict()
                    row['opponent'] = fix['home_team']
                    row['is_home'] = 0
                    row['gameweek'] = gameweek
                    row['season'] = season
                    # Update team features to latest (fixes stale stats for players who missed games)
                    self._update_team_features(row, player.get('team', ''), team_stats_lookup)
                    # Update opponent features based on actual opponent
                    self._update_opponent_features(row, fix['home_team'], team_stats_lookup)
                    test_rows.append(row)
                    break
        
        if verbose:
            print(f"Matched {len(test_rows)} players to GW{gameweek} fixtures")
        
        return pd.DataFrame(test_rows) if test_rows else latest
    
    def _build_team_stats_lookup(self) -> dict:
        """Build lookup of latest team stats (offensive + defensive) for feature updates."""
        # Get latest stats per team - include ALL team-level features
        team_cols = ['team', 'season', 'gameweek', 
                     # Offensive stats
                     'team_goals_roll5', 'team_goals_roll10', 'team_xg_roll5', 'team_xg_roll10',
                     # Defensive stats
                     'team_conceded_roll5', 'team_conceded_roll10', 'team_xga_roll5', 'team_xga_roll10',
                     # Clean sheet rates
                     'team_cs_rate_roll5', 'team_cs_rate_roll10']
        
        available_cols = [c for c in team_cols if c in self.df.columns]
        
        team_df = self.df[available_cols].drop_duplicates(['team', 'season', 'gameweek'])
        latest_team = team_df.sort_values(['team', 'season', 'gameweek']).groupby('team').last().reset_index()
        
        # Build lookup by normalized team name
        lookup = {}
        for _, row in latest_team.iterrows():
            team_norm = normalize_team_name(row['team'])
            lookup[team_norm] = row.to_dict()
        
        return lookup
    
    def _update_opponent_features(self, row: dict, opponent_name: str, team_stats_lookup: dict):
        """Update opponent features (opp_goals_roll5, etc.) based on actual opponent."""
        opp_norm = normalize_team_name(opponent_name)
        
        # Find opponent in lookup
        opp_stats = None
        for team_key, stats in team_stats_lookup.items():
            if opp_norm == team_key or opp_norm in team_key or team_key in opp_norm:
                opp_stats = stats
                break
        
        if opp_stats:
            # Map team offensive stats to opponent features
            # Opponent's goals scored = their team_goals_roll
            if 'team_goals_roll5' in opp_stats:
                row['opp_goals_roll5'] = opp_stats['team_goals_roll5']
            if 'team_goals_roll10' in opp_stats:
                row['opp_goals_roll10'] = opp_stats['team_goals_roll10']
            if 'team_xg_roll5' in opp_stats:
                row['opp_xg_roll5'] = opp_stats['team_xg_roll5']
            if 'team_xg_roll10' in opp_stats:
                row['opp_xg_roll10'] = opp_stats['team_xg_roll10']
            # Opponent's defensive weakness = their team_conceded_roll
            if 'team_conceded_roll5' in opp_stats:
                row['opp_conceded_roll5'] = opp_stats['team_conceded_roll5']
            if 'team_conceded_roll10' in opp_stats:
                row['opp_conceded_roll10'] = opp_stats['team_conceded_roll10']
            if 'team_xga_roll5' in opp_stats:
                row['opp_xga_roll5'] = opp_stats['team_xga_roll5']
            if 'team_xga_roll10' in opp_stats:
                row['opp_xga_roll10'] = opp_stats['team_xga_roll10']
    
    def _update_team_features(self, row: dict, team_name: str, team_stats_lookup: dict):
        """Update team features to latest values (fixes stale stats for players who missed games)."""
        team_norm = normalize_team_name(team_name)
        
        # Find team in lookup
        team_stats = None
        for team_key, stats in team_stats_lookup.items():
            if team_norm == team_key or team_norm in team_key or team_key in team_norm:
                team_stats = stats
                break
        
        if team_stats:
            # Update team defensive stats
            if 'team_conceded_roll5' in team_stats:
                row['team_conceded_roll5'] = team_stats['team_conceded_roll5']
            if 'team_conceded_roll10' in team_stats:
                row['team_conceded_roll10'] = team_stats['team_conceded_roll10']
            if 'team_xga_roll5' in team_stats:
                row['team_xga_roll5'] = team_stats['team_xga_roll5']
            if 'team_xga_roll10' in team_stats:
                row['team_xga_roll10'] = team_stats['team_xga_roll10']
            # Update clean sheet rates
            if 'team_cs_rate_roll5' in team_stats:
                row['team_cs_rate_roll5'] = team_stats['team_cs_rate_roll5']
            if 'team_cs_rate_roll10' in team_stats:
                row['team_cs_rate_roll10'] = team_stats['team_cs_rate_roll10']
            # Update team offensive stats too
            if 'team_goals_roll5' in team_stats:
                row['team_goals_roll5'] = team_stats['team_goals_roll5']
            if 'team_goals_roll10' in team_stats:
                row['team_goals_roll10'] = team_stats['team_goals_roll10']
            if 'team_xg_roll5' in team_stats:
                row['team_xg_roll5'] = team_stats['team_xg_roll5']
            if 'team_xg_roll10' in team_stats:
                row['team_xg_roll10'] = team_stats['team_xg_roll10']
    
    def _get_gw_fixtures(self, gameweek: int, season: str) -> pd.DataFrame:
        """Get fixtures for a gameweek from FPL API or local data."""
        import requests
        try:
            bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
            teams = {t['id']: t['name'] for t in bootstrap['teams']}
            
            fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=10).json()
            gw_fixtures = [f for f in fixtures if f.get('event') == gameweek]
            
            if gw_fixtures:
                return pd.DataFrame([{
                    'home_team': teams.get(f['team_h'], 'Unknown'),
                    'away_team': teams.get(f['team_a'], 'Unknown'),
                } for f in gw_fixtures])
        except:
            pass
        
        # Fallback to local fixtures
        fixtures_file = self.data_dir / 'all_fixtures_8_seasons.csv'
        if fixtures_file.exists():
            fixtures = pd.read_csv(fixtures_file)
            return fixtures[(fixtures['season'] == season) & (fixtures['round'] == gameweek)][['home_team', 'away_team']]
        
        return pd.DataFrame()
    
    def _predict_clean_sheet(self, test_df: pd.DataFrame, gameweek: int, 
                             season: str, verbose: bool) -> np.ndarray:
        """Predict clean sheet probability for each player's team."""
        # Get team-level features
        team_features = self.models['clean_sheet'].prepare_team_features(self.df)
        
        # Get latest team features
        latest_team = team_features.sort_values(['team', 'season', 'gameweek']).groupby('team').last().reset_index()
        
        # Map to players
        cs_probs = []
        for _, row in test_df.iterrows():
            team = str(row.get('team', '')).lower()
            
            # Find matching team
            team_row = None
            for _, t in latest_team.iterrows():
                if team in str(t['team']).lower() or str(t['team']).lower() in team:
                    team_row = t
                    break
            
            if team_row is not None:
                # Create single-row df for prediction
                team_pred = pd.DataFrame([team_row])
                team_pred['is_home'] = row.get('is_home', 0)
                cs_prob = self.models['clean_sheet'].predict_proba(team_pred)[0]
            else:
                cs_prob = 0.25  # Default
            
            cs_probs.append(cs_prob)
        
        return np.array(cs_probs)
    
    def _calculate_expected_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected FPL points."""
        df = df.copy()
        
        def calc_points(row):
            pos = row.get('fpl_position', 'MID')
            mins = row.get('pred_minutes', 0)
            
            # Appearance
            if mins >= 60:
                app_pts = FPL_POINTS['appearance_60']
            elif mins >= 1:
                app_pts = FPL_POINTS['appearance_1']
            else:
                app_pts = 0
            
            # Goals
            goal_pts = row.get('pred_exp_goals', 0) * FPL_POINTS['goal'].get(pos, 5)
            
            # Assists
            assist_pts = row.get('pred_exp_assists', 0) * FPL_POINTS['assist']
            
            # Clean sheet (only if 60+ mins)
            cs_pts = row.get('pred_cs_prob', 0) * FPL_POINTS['clean_sheet'].get(pos, 0) if mins >= 60 else 0
            
            # Defcon (DEF/MID only, 60+ mins)
            defcon_pts = 0
            if pos in ['DEF', 'MID'] and mins >= 60:
                defcon_pts = row.get('pred_defcon_prob', 0) * FPL_POINTS['defcon']
            
            # Bonus
            bonus_pts = row.get('pred_bonus', 0)
            
            return pd.Series({
                'exp_goals_pts': goal_pts,
                'exp_assists_pts': assist_pts,
                'exp_cs_pts': cs_pts,
                'exp_defcon_pts': defcon_pts,
                'exp_bonus_pts': bonus_pts,
                'exp_appearance_pts': app_pts,
                'exp_total_pts': app_pts + goal_pts + assist_pts + cs_pts + defcon_pts + bonus_pts
            })
        
        points_df = df.apply(calc_points, axis=1)
        return pd.concat([df, points_df], axis=1)
    
    def get_top_players(self, predictions: pd.DataFrame, n: int = 30) -> pd.DataFrame:
        """Get top N players by expected points."""
        cols = ['player_name', 'team', 'fpl_position', 'opponent', 'is_home',
                'pred_minutes', 'pred_exp_goals', 'pred_exp_assists', 
                'pred_cs_prob', 'pred_defcon_prob', 'pred_bonus', 'exp_total_pts']
        available_cols = [c for c in cols if c in predictions.columns]
        return predictions.nlargest(n, 'exp_total_pts')[available_cols]
