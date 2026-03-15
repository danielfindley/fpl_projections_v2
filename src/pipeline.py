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
from .models import GoalsModel, AssistsModel, MinutesModel, DefconModel, CleanSheetModel, BonusModel, CardsModel, SavesModel
from .models.minutes import StarterClassifier, StarterMinutesModel, SubMinutesModel, ALL_FEATURES as MINUTES_ALL_FEATURES, STARTER_FEATURES, SUB_FEATURES
from .experiment_log import log_experiment, get_history, get_best_run, log_predictions, get_predictions


# FPL point values
FPL_POINTS = {
    'goal': {'GK': 6, 'DEF': 6, 'MID': 5, 'FWD': 4},
    'assist': 3,
    'clean_sheet': {'GK': 4, 'DEF': 4, 'MID': 1, 'FWD': 0},
    'goals_conceded_2': {'GK': -1, 'DEF': -1, 'MID': 0, 'FWD': 0},  # Per 2 goals conceded
    'saves_per_3': 1,  # 1 point per 3 saves (GK only)
    'defcon': 2,
    'yellow_card': -1,
    'red_card': -3,
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
        Training order matters: CleanSheetModel first (generates pred_team_goals feature),
        then Goals/Assists models use that feature.
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
        
        # Minutes model (independent)
        mins_params = self.tuned_params.get('minutes', {})
        self.models['minutes'] = MinutesModel(**mins_params)
        self.models['minutes'].fit(self.df, verbose)
        
        # Clean sheet model FIRST (needed for pred_team_goals feature)
        cs_params = self.tuned_params.get('clean_sheet', {})
        self.models['clean_sheet'] = CleanSheetModel(**cs_params)
        self.models['clean_sheet'].fit(self.df, verbose)
        
        # Generate leak-free OOF predicted team goals for training data
        self._generate_oof_team_goals(verbose)
        
        # Goals model (uses pred_team_goals)
        goals_params = self.tuned_params.get('goals', {})
        # Ensure pred_team_goals is included in selected features if feature selection was done
        if goals_params.get('selected_features') and 'pred_team_goals' not in goals_params['selected_features']:
            goals_params['selected_features'] = goals_params['selected_features'] + ['pred_team_goals']
        self.models['goals'] = GoalsModel(**goals_params)
        self.models['goals'].fit(self.df, verbose)
        
        # Assists model (uses pred_team_goals)
        assists_params = self.tuned_params.get('assists', {})
        if assists_params.get('selected_features') and 'pred_team_goals' not in assists_params['selected_features']:
            assists_params['selected_features'] = assists_params['selected_features'] + ['pred_team_goals']
        self.models['assists'] = AssistsModel(**assists_params)
        self.models['assists'].fit(self.df, verbose)
        
        # Defcon model (independent)
        defcon_params = self.tuned_params.get('defcon', {})
        self.models['defcon'] = DefconModel(**defcon_params)
        self.models['defcon'].fit(self.df, verbose)
        
        # Bonus model (not tuned - uses Monte Carlo simulation)
        self.models['bonus'] = BonusModel()
        self.models['bonus'].fit(self.df, verbose)

        # Cards model (fouls -> yellow/red card expectations)
        cards_params = self.tuned_params.get('cards', {})
        self.models['cards'] = CardsModel(**cards_params)
        self.models['cards'].fit(self.df, verbose)

        # Saves model (GK only - saves per 90)
        saves_params = self.tuned_params.get('saves', {})
        self.models['saves'] = SavesModel(**saves_params)
        self.models['saves'].fit(self.df, verbose)

        return self
    
    def _generate_oof_team_goals(self, verbose: bool = True):
        """Generate out-of-fold predicted team goals using CleanSheetModel.

        For each match, predicts how many goals the opponent will concede
        (= how many goals this team will score). Uses TimeSeriesSplit CV on the
        CleanSheetModel to avoid data leakage.

        The prediction is mapped to every player row as 'pred_team_goals'.
        """
        from sklearn.model_selection import TimeSeriesSplit
        import xgboost as xgb

        if verbose:
            print("\nGenerating leak-free predicted team goals (TimeSeriesSplit OOF)...")

        cs_model = CleanSheetModel()
        team_df = cs_model.prepare_team_features(self.df)
        team_df = team_df.dropna(subset=['team_conceded_roll5', 'goals_conceded'])

        # Ensure temporal ordering for TimeSeriesSplit
        team_df = team_df.sort_values(['season', 'gameweek']).reset_index(drop=True)

        features = [f for f in cs_model.features_to_use if f in team_df.columns]
        X = team_df[features].fillna(0).values
        y = team_df['goals_conceded'].fillna(0).values

        # Get XGB params from tuned clean_sheet params (strip non-XGB keys)
        cs_params = {k: v for k, v in self.tuned_params.get('clean_sheet', {}).items()
                     if k not in ('selected_features',)}
        cs_params.setdefault('objective', 'count:poisson')
        cs_params.setdefault('random_state', 42)
        cs_params.setdefault('verbosity', 0)

        # TimeSeriesSplit OOF predictions at team-match level
        oof_preds = np.full(len(y), np.nan)
        tscv = TimeSeriesSplit(n_splits=5)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            model = xgb.XGBRegressor(**cs_params)
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = np.clip(model.predict(X[val_idx]), 1e-6, 10.0)
        
        # Now we have predicted goals_conceded for each team-match (OOF)
        # For a player on team A vs opponent B: 
        #   pred_team_goals = predicted goals conceded by B (= goals scored by A)
        # So we need to look up the OPPONENT's predicted goals_conceded
        
        team_df['oof_goals_conceded'] = oof_preds
        
        # Build lookup: for each (opponent, season, gameweek) -> their predicted goals_conceded
        # This is: how many goals did the opponent concede in this match = how many team scored
        opp_conceded_lookup = team_df[['team', 'season', 'gameweek', 'oof_goals_conceded']].copy()
        opp_conceded_lookup = opp_conceded_lookup.rename(columns={
            'team': 'opponent',
            'oof_goals_conceded': 'pred_team_goals'
        })
        
        # Normalize opponent names for matching
        def normalize_name(name):
            if pd.isna(name):
                return ''
            return str(name).lower().replace(' ', '_').replace("'", "").strip()
        
        opp_conceded_lookup['opponent_norm'] = opp_conceded_lookup['opponent'].apply(normalize_name)
        self.df['opponent_norm'] = self.df['opponent'].apply(normalize_name)
        
        # Deduplicate lookup to avoid row multiplication during merge
        opp_conceded_lookup = opp_conceded_lookup.drop_duplicates(
            subset=['opponent_norm', 'season', 'gameweek'], keep='first'
        )
        
        # Drop existing pred_team_goals if present (from previous run)
        if 'pred_team_goals' in self.df.columns:
            self.df = self.df.drop(columns=['pred_team_goals'])
        
        n_before = len(self.df)
        
        # Merge into player-level data
        self.df = self.df.merge(
            opp_conceded_lookup[['opponent_norm', 'season', 'gameweek', 'pred_team_goals']],
            on=['opponent_norm', 'season', 'gameweek'],
            how='left'
        )
        
        # Safety check: merge should not create duplicate rows
        if len(self.df) != n_before:
            self.df = self.df.drop_duplicates(subset=['player_id', 'season', 'gameweek'], keep='first')
            if verbose:
                print(f"  WARNING: Merge created duplicates, deduplicated {len(self.df)} rows")
        
        # Fill NaN with league average (~1.3 goals/game)
        self.df['pred_team_goals'] = self.df['pred_team_goals'].fillna(1.3)
        
        # Clean up
        self.df = self.df.drop(columns=['opponent_norm'], errors='ignore')
        
        if verbose:
            valid = self.df['pred_team_goals'].notna().sum()
            print(f"  Mapped pred_team_goals to {valid:,}/{len(self.df):,} player rows")
            print(f"  Mean pred_team_goals: {self.df['pred_team_goals'].mean():.3f}")
    
    def tune(self, models: list = None, n_iter: int = 100, test_size: float = 0.2,
             verbose: bool = True, use_subprocess: bool = False,
             description: str = '') -> 'FPLPipeline':
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
            models = ['goals', 'assists', 'minutes', 'defcon', 'clean_sheet', 'saves']
        
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
            self.tuned_params, cv_scores = self._tune_with_subprocess(models, n_iter, verbose, df_train)
        else:
            self.tuned_params, cv_scores = self._tune_in_process(models, n_iter, verbose, df_train)

        # Evaluate on held-out test set
        if verbose:
            print("\n" + "-" * 60)
            print("PHASE 2: Evaluation on Held-Out Test Set")
            print("-" * 60)

        test_metrics = self._evaluate_on_test_set(models, df_train, df_test, verbose)

        # Log experiment to SQLite
        try:
            fpl_mae_dict = test_metrics.pop('_fpl_points_mae', None)
            fpl_ex = fpl_mae_dict['mae_ex_bonus'] if isinstance(fpl_mae_dict, dict) else fpl_mae_dict
            fpl_inc = fpl_mae_dict['mae_inc_bonus'] if isinstance(fpl_mae_dict, dict) else None
            run_id = log_experiment(
                data_dir=str(self.data_dir),
                n_iter=n_iter,
                test_size=test_size,
                tuned_params=self.tuned_params,
                cv_scores=cv_scores,
                test_metrics=test_metrics,
                description=description,
                fpl_points_mae=fpl_ex,
                fpl_points_mae_inc_bonus=fpl_inc,
            )
            if verbose:
                print(f"\nExperiment logged: {run_id}")
        except Exception as e:
            if verbose:
                print(f"\nWarning: Failed to log experiment: {e}")

        if verbose:
            print("\n" + "=" * 60)
            print("TUNING COMPLETE")
            print("Tuned params stored. Call train() to train final models on ALL data.")
            print("=" * 60)

        return self
    
    def _tune_in_process(self, models: list, n_iter: int, verbose: bool, df_train: pd.DataFrame) -> Dict:
        """Run tuning in the current process using Optuna with TimeSeriesSplit CV.

        Two-phase approach:
        1. Optuna tunes hyperparams (including regularization) using all features
        2. RFECV with best hyperparams finds the optimal feature subset

        Uses TimeSeriesSplit for time-aware cross-validation.
        """
        import optuna
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.feature_selection import RFECV
        from sklearn.metrics import make_scorer, mean_poisson_deviance
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        tscv = TimeSeriesSplit(n_splits=5)

        def huber_loss(y_true, y_pred, delta=10.0):
            residual = np.abs(y_true - y_pred)
            quadratic = np.minimum(residual, delta)
            linear = residual - quadratic
            return np.mean(0.5 * quadratic**2 + delta * linear)

        huber_scorer = make_scorer(huber_loss, greater_is_better=False)

        def safe_poisson_deviance(y_true, y_pred):
            y_pred = np.clip(y_pred, 1e-8, None)
            y_true = np.clip(y_true, 0, None)
            return mean_poisson_deviance(y_true, y_pred)

        poisson_scorer = make_scorer(safe_poisson_deviance, greater_is_better=False)

        mae_scorer = 'neg_mean_absolute_error'
        rmse_scorer = 'neg_root_mean_squared_error'
        SCORING = {
            'goals': ('Poisson Deviance', poisson_scorer),
            'assists': ('Poisson Deviance', poisson_scorer),
            'defcon': ('RMSE', rmse_scorer),
            'minutes': ('Huber Loss', huber_scorer),
            'saves': ('MAE', mae_scorer),
        }

        SEARCH_SPACES = {
            'goals': {
                'n_estimators': (100, 400),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.6, 1.0),
                'reg_alpha': (1e-3, 10.0, 'log'),
                'reg_lambda': (1e-3, 10.0, 'log'),
            },
            'assists': {
                'n_estimators': (100, 400),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.6, 1.0),
                'reg_alpha': (1e-3, 10.0, 'log'),
                'reg_lambda': (1e-3, 10.0, 'log'),
            },
            'minutes': {
                'n_estimators': (100, 400),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 15),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.6, 1.0),
                'reg_alpha': (1e-3, 10.0, 'log'),
                'reg_lambda': (1e-3, 10.0, 'log'),
            },
            'defcon': {
                'n_estimators': (100, 400),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.6, 1.0),
                'reg_alpha': (1e-3, 10.0, 'log'),
                'reg_lambda': (1e-3, 10.0, 'log'),
            },
            'clean_sheet': {
                'n_estimators': (50, 300),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.6, 1.0),
                'reg_alpha': (1e-3, 10.0, 'log'),
                'reg_lambda': (1e-3, 10.0, 'log'),
            },
            'saves': {
                'n_estimators': (100, 400),
                'max_depth': (3, 7),
                'learning_rate': (0.01, 0.3, 'log'),
                'min_child_weight': (1, 10),
                'colsample_bytree': (0.4, 1.0),
                'subsample': (0.6, 1.0),
                'reg_alpha': (1e-3, 10.0, 'log'),
                'reg_lambda': (1e-3, 10.0, 'log'),
            },
        }

        MODEL_CLASSES = {
            'goals': GoalsModel,
            'assists': AssistsModel,
            'minutes': MinutesModel,
            'defcon': DefconModel,
            'clean_sheet': CleanSheetModel,
            'saves': SavesModel,
        }

        tuned_params = {}
        cv_scores = {}

        for model_name in models:
            if model_name not in MODEL_CLASSES:
                continue

            if model_name == 'clean_sheet':
                cs_result = self._tune_clean_sheet_in_process(
                    n_iter, verbose, df_train, SEARCH_SPACES['clean_sheet']
                )
                tuned_params['clean_sheet'] = cs_result['params']
                cv_scores['clean_sheet'] = cs_result['cv_score']
                continue

            if model_name == 'minutes':
                mins_result = self._tune_minutes_in_process(
                    n_iter, verbose, df_train, SEARCH_SPACES['minutes']
                )
                tuned_params['minutes'] = mins_result['params']
                cv_scores['minutes'] = mins_result['cv_score']
                continue

            model_class = MODEL_CLASSES[model_name]
            space = SEARCH_SPACES.get(model_name, {})
            score_name, scorer = SCORING.get(model_name, ('RMSE', 'neg_root_mean_squared_error'))

            model_instance = model_class()
            all_features = [f for f in model_instance.FEATURES if f in df_train.columns]
            target = model_instance.TARGET
            n_total_features = len(all_features)

            # Filter to GKs only for saves model
            tune_df = df_train
            if model_name == 'saves':
                tune_df = df_train[df_train['is_gk'] == 1].copy()

            X_full = tune_df[all_features].fillna(0).values
            y = tune_df[target].fillna(0).values

            if verbose:
                print(f"\nTuning {model_name.upper()} ({n_iter} trials, TimeSeriesSplit CV, {score_name})...")
                print(f"  Phase 1: Hyperparameter search ({n_total_features} features)")

            # Phase 1: Optuna tunes hyperparams using all features
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                    'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree'][0], space['colsample_bytree'][1]),
                    'subsample': trial.suggest_float('subsample', space['subsample'][0], space['subsample'][1]),
                    'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha'][0], space['reg_alpha'][1], log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda'][0], space['reg_lambda'][1], log=True),
                    'random_state': 42,
                    'verbosity': 0,
                    'n_jobs': -1,
                }

                # Use Poisson objective for goals model
                if model_name == 'goals':
                    params['objective'] = 'count:poisson'

                model = xgb.XGBRegressor(**params)
                scores = cross_val_score(model, X_full, y, cv=tscv, scoring=scorer, n_jobs=1)
                return -scores.mean()

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_iter, show_progress_bar=verbose)

            best_params = study.best_params.copy()
            if model_name == 'goals':
                best_params['objective'] = 'count:poisson'

            if verbose:
                print(f"  Best CV {score_name}: {study.best_value:.4f}")
                print(f"  Phase 2: RFECV feature selection with best hyperparams...")

            # Phase 2: RFECV with best hyperparams to find optimal feature subset
            min_feats = 15 if model_name == 'goals' else 5
            rfecv_model = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-1)
            rfecv = RFECV(
                estimator=rfecv_model,
                step=1,
                cv=tscv,
                scoring=scorer,
                min_features_to_select=min_feats,
                n_jobs=1,
            )
            rfecv.fit(X_full, y)

            selected_mask = rfecv.support_
            selected_features = [all_features[i] for i in range(n_total_features) if selected_mask[i]]

            tuned_params[model_name] = {
                **best_params,
                'selected_features': selected_features,
            }
            cv_scores[model_name] = study.best_value

            if verbose:
                print(f"  RFECV selected {len(selected_features)}/{n_total_features} features")
                print(f"  Params: {best_params}")

        return tuned_params, cv_scores

    def _tune_clean_sheet_in_process(self, n_iter: int, verbose: bool,
                                      df_train: pd.DataFrame, space: dict) -> Dict:
        """Tune CleanSheetModel (Poisson regression for goals against) with RFECV feature selection.

        Two-phase approach:
        1. Optuna tunes hyperparams (including regularization) using all features
        2. RFECV with best hyperparams finds the optimal feature subset

        Uses TimeSeriesSplit for time-aware cross-validation.
        """
        import optuna
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.feature_selection import RFECV
        from sklearn.metrics import make_scorer, mean_poisson_deviance
        import xgboost as xgb

        def safe_poisson_deviance(y_true, y_pred):
            y_pred = np.clip(y_pred, 1e-8, None)
            y_true = np.clip(y_true, 0, None)
            return mean_poisson_deviance(y_true, y_pred)
        poisson_scorer = make_scorer(safe_poisson_deviance, greater_is_better=False)

        tscv = TimeSeriesSplit(n_splits=5)

        # Prepare team-level data (already sorted temporally by prepare_team_features)
        cs_model = CleanSheetModel()
        team_df = cs_model.prepare_team_features(df_train)
        team_df = team_df.dropna(subset=['team_conceded_roll5', 'goals_conceded'])

        all_features = [f for f in cs_model.FEATURES if f in team_df.columns]
        n_total_features = len(all_features)
        X_full = team_df[all_features].fillna(0).values
        y = team_df['goals_conceded'].fillna(0).values

        if verbose:
            print(f"\nTuning GOALS_AGAINST ({n_iter} trials, TimeSeriesSplit CV, Poisson Deviance)...")
            print(f"  Phase 1: Hyperparameter search ({n_total_features} features)")
            print(f"  Team-matches: {len(X_full)}, Avg conceded: {y.mean():.3f}")

        # Phase 1: Optuna tunes hyperparams using all features
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
                'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree'][0], space['colsample_bytree'][1]),
                'subsample': trial.suggest_float('subsample', space['subsample'][0], space['subsample'][1]),
                'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha'][0], space['reg_alpha'][1], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda'][0], space['reg_lambda'][1], log=True),
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1,
                'objective': 'count:poisson',
            }

            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_full, y, cv=tscv, scoring=poisson_scorer, n_jobs=1)
            return -scores.mean()

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_iter, show_progress_bar=verbose)

        best_params = study.best_params.copy()

        if verbose:
            print(f"  Best CV Poisson Deviance: {study.best_value:.4f}")
            print(f"  Phase 2: RFECV feature selection with best hyperparams...")

        # Phase 2: RFECV with best hyperparams to find optimal feature subset
        rfecv_model = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-1, objective='count:poisson')
        rfecv = RFECV(
            estimator=rfecv_model,
            step=1,
            cv=tscv,
            scoring=poisson_scorer,
            min_features_to_select=5,
            n_jobs=1,
        )
        rfecv.fit(X_full, y)

        selected_mask = rfecv.support_
        selected_features = [all_features[i] for i in range(n_total_features) if selected_mask[i]]

        if verbose:
            print(f"  RFECV selected {len(selected_features)}/{n_total_features} features")
            print(f"  Params: {best_params}")

        return {
            'params': {**best_params, 'selected_features': selected_features},
            'cv_score': study.best_value,
        }

    def _tune_minutes_in_process(self, n_iter: int, verbose: bool,
                                  df_train: pd.DataFrame, space: dict) -> dict:
        """Tune the two-stage MinutesModel: classifier + starter regressor + sub regressor.

        Each sub-model gets its own Optuna + RFECV pass. Returns nested params dict.
        """
        import optuna
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.feature_selection import RFECV
        from sklearn.metrics import make_scorer, log_loss
        import xgboost as xgb

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        tscv = TimeSeriesSplit(n_splits=5)

        df_played = df_train[df_train['minutes'] >= 1].copy()
        df_played = df_played.sort_values(['season', 'gameweek'])

        if verbose:
            print(f"\nTuning MINUTES (two-stage, {n_iter} trials each)...")
            print(f"  Total played: {len(df_played):,}")

        # ---- 1. StarterClassifier (log-loss) ----
        cls_features = [f for f in MINUTES_ALL_FEATURES if f in df_played.columns]
        X_cls = df_played[cls_features].fillna(0).values
        y_cls = (df_played['minutes'] >= 60).astype(int).values

        if verbose:
            print(f"\n  [1/3] StarterClassifier ({len(cls_features)} features, {y_cls.mean():.1%} starters)")

        def cls_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
                'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree'][0], space['colsample_bytree'][1]),
                'subsample': trial.suggest_float('subsample', space['subsample'][0], space['subsample'][1]),
                'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha'][0], space['reg_alpha'][1], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda'][0], space['reg_lambda'][1], log=True),
                'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
                'eval_metric': 'logloss', 'use_label_encoder': False,
            }
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_cls, y_cls, cv=tscv, scoring='neg_log_loss', n_jobs=1)
            return -scores.mean()

        study_cls = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study_cls.optimize(cls_objective, n_trials=n_iter, show_progress_bar=verbose)
        cls_best = study_cls.best_params.copy()

        if verbose:
            print(f"    Best CV LogLoss: {study_cls.best_value:.4f}")
            print(f"    RFECV feature selection...")

        rfecv_cls = RFECV(
            estimator=xgb.XGBClassifier(**cls_best, random_state=42, verbosity=0, n_jobs=-1,
                                         eval_metric='logloss', use_label_encoder=False),
            step=1, cv=tscv, scoring='neg_log_loss', min_features_to_select=5, n_jobs=1,
        )
        rfecv_cls.fit(X_cls, y_cls)
        cls_selected = [cls_features[i] for i in range(len(cls_features)) if rfecv_cls.support_[i]]
        if verbose:
            print(f"    Selected {len(cls_selected)}/{len(cls_features)} features")

        # ---- 2. StarterMinutesModel (MAE, trained on 60+ only) ----
        df_starters = df_played[df_played['minutes'] >= 60].copy()
        starter_features = [f for f in STARTER_FEATURES if f in df_starters.columns]
        X_start = df_starters[starter_features].fillna(0).values
        y_start = df_starters['minutes'].values

        if verbose:
            print(f"\n  [2/3] StarterMinutesModel ({len(starter_features)} features, {len(df_starters):,} samples)")

        def starter_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
                'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree'][0], space['colsample_bytree'][1]),
                'subsample': trial.suggest_float('subsample', space['subsample'][0], space['subsample'][1]),
                'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha'][0], space['reg_alpha'][1], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda'][0], space['reg_lambda'][1], log=True),
                'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
            }
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_start, y_start, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
            return -scores.mean()

        study_start = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=43))
        study_start.optimize(starter_objective, n_trials=n_iter, show_progress_bar=verbose)
        starter_best = study_start.best_params.copy()

        if verbose:
            print(f"    Best CV MAE: {study_start.best_value:.4f}")
            print(f"    RFECV feature selection...")

        rfecv_start = RFECV(
            estimator=xgb.XGBRegressor(**starter_best, random_state=42, verbosity=0, n_jobs=-1),
            step=1, cv=tscv, scoring='neg_mean_absolute_error', min_features_to_select=5, n_jobs=1,
        )
        rfecv_start.fit(X_start, y_start)
        starter_selected = [starter_features[i] for i in range(len(starter_features)) if rfecv_start.support_[i]]
        if verbose:
            print(f"    Selected {len(starter_selected)}/{len(starter_features)} features")

        # ---- 3. SubMinutesModel (MAE, trained on 1-59 only) ----
        df_subs = df_played[(df_played['minutes'] >= 1) & (df_played['minutes'] < 60)].copy()
        sub_features = [f for f in SUB_FEATURES if f in df_subs.columns]
        X_sub = df_subs[sub_features].fillna(0).values
        y_sub = df_subs['minutes'].values

        if verbose:
            print(f"\n  [3/3] SubMinutesModel ({len(sub_features)} features, {len(df_subs):,} samples)")

        def sub_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
                'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
                'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree'][0], space['colsample_bytree'][1]),
                'subsample': trial.suggest_float('subsample', space['subsample'][0], space['subsample'][1]),
                'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha'][0], space['reg_alpha'][1], log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda'][0], space['reg_lambda'][1], log=True),
                'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
            }
            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_sub, y_sub, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
            return -scores.mean()

        study_sub = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=44))
        study_sub.optimize(sub_objective, n_trials=n_iter, show_progress_bar=verbose)
        sub_best = study_sub.best_params.copy()

        if verbose:
            print(f"    Best CV MAE: {study_sub.best_value:.4f}")
            print(f"    RFECV feature selection...")

        rfecv_sub = RFECV(
            estimator=xgb.XGBRegressor(**sub_best, random_state=42, verbosity=0, n_jobs=-1),
            step=1, cv=tscv, scoring='neg_mean_absolute_error', min_features_to_select=3, n_jobs=1,
        )
        rfecv_sub.fit(X_sub, y_sub)
        sub_selected = [sub_features[i] for i in range(len(sub_features)) if rfecv_sub.support_[i]]
        if verbose:
            print(f"    Selected {len(sub_selected)}/{len(sub_features)} features")

        # Build nested params
        nested_params = {
            'classifier_params': {**cls_best, 'selected_features': cls_selected},
            'starter_params': {**starter_best, 'selected_features': starter_selected},
            'sub_params': {**sub_best, 'selected_features': sub_selected},
        }

        # Combined CV score: weighted average of classifier log-loss (as proxy)
        combined_cv = study_cls.best_value

        return {
            'params': nested_params,
            'cv_score': combined_cv,
        }

    def _tune_with_subprocess(self, models: list, n_iter: int, verbose: bool, df_train: pd.DataFrame) -> Dict:
        """Run each model's tuning in a separate subprocess to save RAM.

        Two-phase approach per model:
        1. Optuna tunes hyperparams (including regularization) with TimeSeriesSplit CV
        2. RFECV with best hyperparams finds the optimal feature subset

        Uses appropriate loss functions:
        - Goals, Assists: Poisson deviance (count data)
        - Saves: MAE
        - Defcon: RMSE
        - Minutes: Huber loss (robust to outliers)
        - Goals Against (clean sheet): Poisson deviance (count data)
        """
        import subprocess
        import sys
        import json
        import tempfile
        import os

        SCORING_NAMES = {
            'goals': 'Poisson Deviance',
            'assists': 'Poisson Deviance',
            'defcon': 'RMSE',
            'minutes': 'Huber Loss',
            'clean_sheet': 'Poisson Deviance',
            'saves': 'MAE',
        }

        SEARCH_SPACES_CS = {
            'n_estimators': (50, 300),
            'max_depth': (3, 8),
            'learning_rate': (0.01, 0.3, 'log'),
            'min_child_weight': (1, 10),
            'colsample_bytree': (0.4, 1.0),
            'subsample': (0.6, 1.0),
            'reg_alpha': (1e-3, 10.0, 'log'),
            'reg_lambda': (1e-3, 10.0, 'log'),
        }

        tuned_params = {}
        cv_scores = {}

        # Save training data to temp file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            temp_data_path = f.name
            df_train.to_csv(f, index=False)

        try:
            for model_name in models:
                # Handle clean_sheet specially (needs team-level aggregation)
                if model_name == 'clean_sheet':
                    cs_result = self._tune_clean_sheet_in_process(
                        n_iter, verbose, df_train, SEARCH_SPACES_CS
                    )
                    tuned_params['clean_sheet'] = cs_result['params']
                    cv_scores['clean_sheet'] = cs_result['cv_score']
                    continue

                # Handle minutes specially (three sub-models)
                if model_name == 'minutes':
                    mins_result = self._tune_minutes_in_process(
                        n_iter, verbose, df_train, {
                            'n_estimators': (100, 400),
                            'max_depth': (3, 7),
                            'learning_rate': (0.01, 0.3, 'log'),
                            'min_child_weight': (1, 15),
                            'colsample_bytree': (0.4, 1.0),
                            'subsample': (0.6, 1.0),
                            'reg_alpha': (1e-3, 10.0, 'log'),
                            'reg_lambda': (1e-3, 10.0, 'log'),
                        }
                    )
                    tuned_params['minutes'] = mins_result['params']
                    cv_scores['minutes'] = mins_result['cv_score']
                    continue

                score_name = SCORING_NAMES.get(model_name, 'RMSE')
                if verbose:
                    print(f"\nTuning {model_name.upper()} ({n_iter} trials, TimeSeriesSplit CV, {score_name}) in subprocess...")

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
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, mean_poisson_deviance
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, r"{self.data_dir.parent}")
from src.models import GoalsModel, AssistsModel, MinutesModel, DefconModel, SavesModel

def huber_loss(y_true, y_pred, delta=10.0):
    residual = np.abs(y_true - y_pred)
    quadratic = np.minimum(residual, delta)
    linear = residual - quadratic
    return np.mean(0.5 * quadratic**2 + delta * linear)

huber_scorer = make_scorer(huber_loss, greater_is_better=False)

def safe_poisson_deviance(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-8, None)
    y_true = np.clip(y_true, 0, None)
    return mean_poisson_deviance(y_true, y_pred)

poisson_scorer = make_scorer(safe_poisson_deviance, greater_is_better=False)

SCORING = {{
    'goals': poisson_scorer,
    'assists': poisson_scorer,
    'defcon': 'neg_root_mean_squared_error',
    'minutes': huber_scorer,
    'saves': 'neg_mean_absolute_error',
}}

MODEL_CLASSES = {{
    'goals': GoalsModel,
    'assists': AssistsModel,
    'minutes': MinutesModel,
    'defcon': DefconModel,
    'saves': SavesModel,
}}

SEARCH_SPACES = {{
    'goals': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
        'colsample_bytree': (0.4, 1.0),
        'subsample': (0.6, 1.0),
        'reg_alpha': (1e-3, 10.0),
        'reg_lambda': (1e-3, 10.0),
    }},
    'assists': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
        'colsample_bytree': (0.4, 1.0),
        'subsample': (0.6, 1.0),
        'reg_alpha': (1e-3, 10.0),
        'reg_lambda': (1e-3, 10.0),
    }},
    'minutes': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 15),
        'colsample_bytree': (0.4, 1.0),
        'subsample': (0.6, 1.0),
        'reg_alpha': (1e-3, 10.0),
        'reg_lambda': (1e-3, 10.0),
    }},
    'defcon': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
        'colsample_bytree': (0.4, 1.0),
        'subsample': (0.6, 1.0),
        'reg_alpha': (1e-3, 10.0),
        'reg_lambda': (1e-3, 10.0),
    }},
    'saves': {{
        'n_estimators': (100, 400),
        'max_depth': (3, 7),
        'learning_rate': (0.01, 0.3),
        'min_child_weight': (1, 10),
        'colsample_bytree': (0.4, 1.0),
        'subsample': (0.6, 1.0),
        'reg_alpha': (1e-3, 10.0),
        'reg_lambda': (1e-3, 10.0),
    }},
}}

model_name = "{model_name}"
n_iter = {n_iter}
score_name = "{score_name}"

df_train = pd.read_csv(r"{temp_data_path}", encoding='utf-8')
model_class = MODEL_CLASSES[model_name]
space = SEARCH_SPACES[model_name]
scorer = SCORING[model_name]

model_instance = model_class()
all_features = [f for f in model_instance.FEATURES if f in df_train.columns]
target = model_instance.TARGET
n_total_features = len(all_features)

# Filter to GKs only for saves model
if model_name == 'saves':
    df_train = df_train[df_train['is_gk'] == 1].copy()

X_full = df_train[all_features].fillna(0).values
y = df_train[target].fillna(0).values

tscv = TimeSeriesSplit(n_splits=5)

# Phase 1: Optuna tunes hyperparams
def objective(trial):
    params = {{
        'n_estimators': trial.suggest_int('n_estimators', space['n_estimators'][0], space['n_estimators'][1]),
        'max_depth': trial.suggest_int('max_depth', space['max_depth'][0], space['max_depth'][1]),
        'learning_rate': trial.suggest_float('learning_rate', space['learning_rate'][0], space['learning_rate'][1], log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', space['min_child_weight'][0], space['min_child_weight'][1]),
        'colsample_bytree': trial.suggest_float('colsample_bytree', space['colsample_bytree'][0], space['colsample_bytree'][1]),
        'subsample': trial.suggest_float('subsample', space['subsample'][0], space['subsample'][1]),
        'reg_alpha': trial.suggest_float('reg_alpha', space['reg_alpha'][0], space['reg_alpha'][1], log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', space['reg_lambda'][0], space['reg_lambda'][1], log=True),
        'random_state': 42,
        'verbosity': 0,
        'n_jobs': -1,
    }}

    # Use Poisson objective for goals model
    if model_name == 'goals':
        params['objective'] = 'count:poisson'

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_full, y, cv=tscv, scoring=scorer, n_jobs=1)
    return -scores.mean()

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

best_params = study.best_params.copy()
if model_name == 'goals':
    best_params['objective'] = 'count:poisson'

print(f"Phase 1 Best CV {{score_name}}: {{study.best_value:.4f}}")
print(f"Phase 2: RFECV feature selection...")

# Phase 2: RFECV with best hyperparams
min_feats = 15 if model_name == 'goals' else 5
rfecv_model = xgb.XGBRegressor(**best_params, random_state=42, verbosity=0, n_jobs=-1)
rfecv = RFECV(
    estimator=rfecv_model,
    step=1,
    cv=tscv,
    scoring=scorer,
    min_features_to_select=min_feats,
    n_jobs=1,
)
rfecv.fit(X_full, y)

selected_mask = rfecv.support_
selected_features = [all_features[i] for i in range(n_total_features) if selected_mask[i]]

print(f"RFECV selected {{len(selected_features)}}/{{n_total_features}} features")
print(f"Params: {{best_params}}")

result = {{
    'best_params': best_params,
    'best_score': study.best_value,
    'selected_features': selected_features,
}}

with open(r"{temp_result_path}", 'w') as f:
    json.dump(result, f)
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
                    tuned_params[model_name] = {
                        **tune_result['best_params'],
                        'selected_features': tune_result.get('selected_features', []),
                    }
                    cv_scores[model_name] = tune_result.get('best_score')
                except Exception as e:
                    if verbose:
                        print(f"  Failed to read results: {e}")
                finally:
                    try:
                        os.unlink(temp_result_path)
                    except:
                        pass
        finally:
            try:
                os.unlink(temp_data_path)
            except:
                pass

        return tuned_params, cv_scores

    def _evaluate_on_test_set(self, models: list, df_train: pd.DataFrame,
                               df_test: pd.DataFrame, verbose: bool):
        """Train models on train set with tuned params and evaluate on test set.
        
        Reports metrics matching the tuning loss functions:
        - Goals, Assists: Poisson deviance (count data)
        - Defcon: RMSE (continuous metric)
        - Minutes: Huber loss + MAE
        - Clean Sheet: Poisson deviance (actual count data)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_poisson_deviance
        import xgboost as xgb
        
        # Custom Huber loss
        def huber_loss(y_true, y_pred, delta=10.0):
            residual = np.abs(y_true - y_pred)
            quadratic = np.minimum(residual, delta)
            linear = residual - quadratic
            return np.mean(0.5 * quadratic**2 + delta * linear)
        
        # RMSE helper
        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Safe Poisson deviance (used for clean sheet goals conceded evaluation)
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
            'saves': SavesModel,
        }

        # Model-specific primary metrics (for regressors)
        PRIMARY_METRICS = {
            'goals': ('Poisson Dev', safe_poisson_deviance),
            'assists': ('Poisson Dev', safe_poisson_deviance),
            'defcon': ('RMSE', rmse),
            'minutes': ('Huber Loss', huber_loss),
            'saves': ('MAE', mean_absolute_error),
        }
        
        # Store test metrics
        test_metrics = {}
        # Store per-player predictions for FPL points calculation
        pred_store = {}

        # Handle regression models
        for model_name in models:
            if model_name in ('clean_sheet', 'minutes'):
                continue  # Handle separately below
            if model_name not in MODEL_CLASSES or model_name not in self.tuned_params:
                continue

            model_class = MODEL_CLASSES[model_name]
            params = self.tuned_params[model_name]
            metric_name, metric_fn = PRIMARY_METRICS[model_name]

            # Create model with tuned params (selected_features handled by constructor)
            model_instance = model_class(**params)
            features = [f for f in model_instance.features_to_use if f in df_train.columns]
            target = model_instance.TARGET

            # Filter to GKs only for saves model
            eval_train = df_train
            eval_test = df_test
            if model_name == 'saves':
                eval_train = df_train[df_train['is_gk'] == 1]
                eval_test = df_test[df_test['is_gk'] == 1]

            # Prepare train data
            X_train = eval_train[features].fillna(0).values
            y_train = eval_train[target].fillna(0).values

            # Prepare test data
            X_test = eval_test[features].fillna(0).values
            y_test = eval_test[target].fillna(0).values

            # Train on train set (filter out non-XGB params like selected_features)
            xgb_params = {k: v for k, v in params.items() if k != 'selected_features'}
            xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42, verbosity=0, n_jobs=-1)
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
            pred_store[model_name] = {
                'index': eval_test.index,
                'y_pred': y_pred,
                'y_test': y_test,
            }

        # Handle minutes (two-stage model) separately
        if 'minutes' in models and 'minutes' in self.tuned_params:
            from sklearn.metrics import log_loss, accuracy_score, roc_auc_score as _roc_auc
            params = self.tuned_params['minutes']
            mins_model = MinutesModel(**params)
            mins_model.fit(df_train, verbose=False)

            # Test on played players only
            test_played = df_test[df_test['minutes'] >= 1].copy()
            y_test = test_played['minutes'].values
            y_pred = mins_model.predict(test_played)

            # Combined metrics
            combined_huber = huber_loss(y_test, y_pred)
            combined_mae = mean_absolute_error(y_test, y_pred)

            test_metrics['minutes'] = {
                'metric_name': 'Huber Loss',
                'primary': combined_huber,
                'MAE': combined_mae,
            }
            pred_store['minutes'] = {
                'index': test_played.index,
                'y_pred': y_pred,
                'y_test': y_test,
            }

            # Additional classifier metrics
            p_start = mins_model.classifier.predict_proba(test_played)
            y_binary = (test_played['minutes'] >= 60).astype(int).values
            cls_auc = _roc_auc(y_binary, p_start)
            cls_acc = accuracy_score(y_binary, (p_start >= 0.5).astype(int))
            cls_logloss = log_loss(y_binary, p_start)

            # Starter/sub regressor MAEs on their subsets
            starter_mask = test_played['minutes'] >= 60
            sub_mask = (test_played['minutes'] >= 1) & (test_played['minutes'] < 60)
            starter_mae = mean_absolute_error(y_test[starter_mask], y_pred[starter_mask]) if starter_mask.any() else 0
            sub_mae = mean_absolute_error(y_test[sub_mask], y_pred[sub_mask]) if sub_mask.any() else 0

            test_metrics['minutes']['classifier'] = {
                'AUC': cls_auc, 'Accuracy': cls_acc, 'LogLoss': cls_logloss,
            }
            test_metrics['minutes']['starter_MAE'] = starter_mae
            test_metrics['minutes']['sub_MAE'] = sub_mae

        # Handle clean_sheet (goals against regression) separately (team-level data)
        if 'clean_sheet' in models and 'clean_sheet' in self.tuned_params:
            params = {k: v for k, v in self.tuned_params['clean_sheet'].items() if k != 'selected_features'}
            selected_features = self.tuned_params['clean_sheet'].get('selected_features', None)
            cs_model = CleanSheetModel()
            
            # Prepare team-level train data
            team_train = cs_model.prepare_team_features(df_train)
            team_train = team_train.dropna(subset=['team_conceded_roll5', 'goals_conceded'])
            
            # Prepare team-level test data
            team_test = cs_model.prepare_team_features(df_test)
            team_test = team_test.dropna(subset=['team_conceded_roll5', 'goals_conceded'])
            
            features = selected_features if selected_features else [f for f in cs_model.FEATURES if f in team_train.columns]
            
            X_train = team_train[features].fillna(0).values
            y_train = team_train['goals_conceded'].fillna(0).values
            X_test = team_test[features].fillna(0).values
            y_test = team_test['goals_conceded'].fillna(0).values
            
            # Train Poisson regressor
            xgb_model = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1, objective='count:poisson')
            xgb_model.fit(X_train, y_train)
            
            # Predict expected goals against
            y_pred = np.clip(xgb_model.predict(X_test), 1e-8, None)
            
            # Calculate metrics
            poisson_dev = safe_poisson_deviance(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            test_metrics['clean_sheet'] = {
                'metric_name': 'Poisson Dev',
                'primary': poisson_dev,
                'MAE': mae
            }
        
        # Compute aggregate FPL points: actual vs predicted
        # Uses actual stats to calculate "actual FPL points" and model predictions
        # to calculate "predicted FPL points", then reports MAE between them.
        fpl_mae = None
        try:
            fpl_mae = self._compute_fpl_points_mae(df_test, pred_store)
        except Exception as e:
            if verbose:
                print(f"\nWarning: Could not compute FPL points MAE: {e}")

        # Print test set performance
        if verbose:
            print(f"\n{'Model':<15} {'Metric':<15} {'Test Value':<12} {'MAE':<12}")
            print("-" * 54)
            for model_name, metrics in test_metrics.items():
                print(f"{model_name.upper():<15} {metrics['metric_name']:<15} {metrics['primary']:<12.4f} {metrics['MAE']:<12.4f}")
                # Print minutes sub-model details
                if model_name == 'minutes' and 'classifier' in metrics:
                    cls = metrics['classifier']
                    print(f"  {'Classifier':<13} {'AUC':<15} {cls['AUC']:<12.4f} {'LogLoss':<6} {cls['LogLoss']:.4f}")
                    print(f"  {'Starter Reg':<13} {'MAE':<15} {metrics['starter_MAE']:<12.4f}")
                    print(f"  {'Sub Reg':<13} {'MAE':<15} {metrics['sub_MAE']:<12.4f}")
            if fpl_mae is not None:
                print("-" * 54)
                print(f"{'FPL POINTS':<15} {'ex-bonus MAE':<15} {fpl_mae['mae_ex_bonus']:<12.4f}")
                print(f"{'':<15} {'inc-bonus MAE':<15} {fpl_mae['mae_inc_bonus']:<12.4f}")

            # Add context for interpretation
            print("\n(Lower is better for all metrics)")
            print("ex-bonus: apples-to-apples (no bonus either side)")
            print("inc-bonus: includes actual bonus in target (shows full gap)")

        test_metrics['_fpl_points_mae'] = fpl_mae
        return test_metrics

    def _compute_fpl_points_mae(self, df_test: pd.DataFrame, pred_store: dict) -> dict:
        """Compute MAE between actual and predicted FPL points on the test set.

        Calculates FPL points from actual stats and from model predictions,
        then returns MAE metrics. Bonus is included in actuals but NOT in
        predictions (since BonusModel isn't part of tuning), so we report
        both with-bonus and without-bonus MAE for fair comparison.

        Returns dict with:
            mae_ex_bonus: MAE excluding bonus from both sides (apples-to-apples)
            mae_inc_bonus: MAE with actual bonus included (shows full gap)
        """
        from sklearn.metrics import mean_absolute_error

        POS_MAP = {0: 'GK', 1: 'DEF', 2: 'MID', 3: 'FWD'}

        df = df_test.copy()
        df['pos_label'] = df['position'].map(POS_MAP).fillna('MID')

        # Helper to safely get numeric value (handles NaN and missing columns)
        def _safe(row, col, default=0):
            val = row.get(col, default)
            if pd.isna(val):
                return default
            return val

        # Use team_conceded_roll1 as proxy for per-match goals conceded when
        # the raw goals_conceded column is NaN (it's only populated for some rows)
        if 'goals_conceded' in df.columns:
            df['_gc'] = df['goals_conceded'].fillna(df.get('team_conceded_roll1', pd.Series(dtype=float))).fillna(1)
        else:
            df['_gc'] = df.get('team_conceded_roll1', pd.Series(1, index=df.index)).fillna(1)

        # --- Actual FPL points from real stats ---
        def actual_fpl_points(row):
            pos = row['pos_label']
            mins = _safe(row, 'minutes')

            # Appearance
            if mins >= 60:
                pts = FPL_POINTS['appearance_60']
            elif mins >= 1:
                pts = FPL_POINTS['appearance_1']
            else:
                return 0, 0  # (pts_ex_bonus, bonus)

            # Goals (actual count * position multiplier)
            pts += _safe(row, 'goals') * FPL_POINTS['goal'].get(pos, 5)

            # Assists
            pts += _safe(row, 'assists') * FPL_POINTS['assist']

            # Clean sheet: 1 if goals_conceded == 0 and mins >= 60
            gc = _safe(row, '_gc', 1)
            if mins >= 60 and gc == 0:
                pts += FPL_POINTS['clean_sheet'].get(pos, 0)

            # Goals conceded penalty (GK/DEF, 60+ mins)
            if pos in ('GK', 'DEF') and mins >= 60:
                pts += (int(gc) // 2) * FPL_POINTS['goals_conceded_2'].get(pos, 0)

            # Saves (GK) — 1 point per 3
            if pos == 'GK':
                pts += (int(_safe(row, 'saves')) // 3) * FPL_POINTS['saves_per_3']

            # Bonus (actual from data — may not exist)
            bonus = _safe(row, 'bonus')

            return pts, bonus

        results = df.apply(actual_fpl_points, axis=1, result_type='expand')
        df['actual_pts_ex_bonus'] = results[0]
        df['actual_bonus'] = results[1]
        df['actual_fpl_pts'] = df['actual_pts_ex_bonus'] + df['actual_bonus']

        # --- Predicted FPL points from model outputs ---
        # Map predictions back to the test DataFrame as Series
        for key in ('minutes', 'goals', 'assists', 'saves', 'defcon'):
            col = f'_pred_{key}'
            if key in pred_store:
                s = pd.Series(pred_store[key]['y_pred'], index=pred_store[key]['index'])
                df[col] = s.reindex(df.index).fillna(0)
            else:
                df[col] = 0

        # Use actual minutes as fallback if minutes model wasn't tuned
        if 'minutes' not in pred_store:
            df['_pred_minutes'] = df['minutes']

        def pred_fpl_points(row):
            pos = row['pos_label']
            mins = row['_pred_minutes']

            if mins >= 60:
                pts = FPL_POINTS['appearance_60']
            elif mins >= 1:
                pts = FPL_POINTS['appearance_1']
            else:
                return 0

            # Goals: pred rate * (pred_minutes / 90)
            exp_goals = row['_pred_goals'] * (mins / 90)
            pts += exp_goals * FPL_POINTS['goal'].get(pos, 5)

            # Assists
            exp_assists = row['_pred_assists'] * (mins / 90)
            pts += exp_assists * FPL_POINTS['assist']

            # Saves (GK)
            if pos == 'GK':
                exp_saves = row['_pred_saves'] * (mins / 90)
                pts += (exp_saves / 3) * FPL_POINTS['saves_per_3']

            return pts

        df['pred_fpl_pts'] = df.apply(pred_fpl_points, axis=1)

        # Only compare rows where player actually played and values are valid
        mask = (df['minutes'] >= 1) & df['actual_pts_ex_bonus'].notna() & df['pred_fpl_pts'].notna()
        played = df.loc[mask]

        return {
            'mae_ex_bonus': mean_absolute_error(played['actual_pts_ex_bonus'], played['pred_fpl_pts']),
            'mae_inc_bonus': mean_absolute_error(played['actual_fpl_pts'], played['pred_fpl_pts']),
        }

    def experiment_history(self, model: str = None) -> pd.DataFrame:
        """Return experiment history as a DataFrame.

        Args:
            model: If provided, filter to a specific model name.
        """
        return get_history(str(self.data_dir), model=model)

    def best_run(self, model_name: str):
        """Return the best test score row for a given model (lowest test_score)."""
        return get_best_run(str(self.data_dir), model_name)

    def prediction_history(self, gameweek: int = None, season: str = None,
                           player_name: str = None) -> pd.DataFrame:
        """Query stored predictions from the DB.

        Args:
            gameweek: Filter to a specific gameweek.
            season: Filter to a specific season.
            player_name: Substring match (case-insensitive).
        """
        return get_predictions(str(self.data_dir), gameweek=gameweek,
                               season=season, player_name=player_name)

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

        # Recompute interaction features (opponent stats were updated by _build_test_set)
        test_df = self._recompute_interaction_features(test_df)

        if len(test_df) == 0:
            print("WARNING: No players found for prediction")
            return pd.DataFrame()
        
        # Generate predictions
        test_df['pred_minutes'] = self.models['minutes'].predict(test_df)
        
        # Clean sheet / goals against FIRST (needed for pred_team_goals feature)
        cs_probs, two_plus_probs, pred_goals_against = self._predict_clean_sheet(test_df, gameweek, season, verbose)
        test_df['pred_cs_prob'] = cs_probs
        test_df['pred_2plus_conceded'] = two_plus_probs
        test_df['pred_goals_against'] = pred_goals_against
        
        # Inject pred_team_goals: for each player, predicted goals their team will score
        # = predicted goals conceded by the OPPONENT (from CleanSheetModel)
        test_df['pred_team_goals'] = self._get_pred_team_goals(test_df, gameweek, season)
        
        # Goals/Assists now use pred_team_goals as a feature
        test_df['pred_goals_per90'] = self.models['goals'].predict(test_df)
        test_df['pred_exp_goals'] = self.models['goals'].predict_expected(test_df, test_df['pred_minutes'])
        test_df['pred_assists_per90'] = self.models['assists'].predict(test_df)
        test_df['pred_exp_assists'] = self.models['assists'].predict_expected(test_df, test_df['pred_minutes'])
        test_df['pred_defcon_prob'] = self.models['defcon'].predict_threshold_prob(test_df, test_df['pred_minutes'])

        # Cards predictions
        test_df['pred_yellow_prob'] = self.models['cards'].predict_expected_yellows(test_df, test_df['pred_minutes'])
        test_df['pred_red_prob'] = self.models['cards'].predict_expected_reds(test_df, test_df['pred_minutes'])

        # Saves predictions (GK only, 0 for outfield)
        test_df['pred_exp_saves'] = 0.0
        gk_mask = test_df['is_gk'] == 1
        if gk_mask.any():
            gk_df = test_df[gk_mask]
            test_df.loc[gk_mask, 'pred_exp_saves'] = self.models['saves'].predict_expected_saves(
                gk_df, gk_df['pred_minutes'].values
            )

        # 4+ goals conceded probability (from Poisson on predicted goals against)
        from scipy.stats import poisson
        test_df['pred_4plus_conceded'] = 1.0 - poisson.cdf(3, test_df['pred_goals_against'].values)

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

        # Store per-simulation arrays (keyed by player_name) for distribution plots
        bonus_sims = self.models['bonus'].get_last_simulations()
        self.last_simulations = {
            'player_names': test_df['player_name'].values.tolist(),
            **bonus_sims,
        }

        # Calculate expected points (per fixture)
        test_df = self._calculate_expected_points(test_df)

        # Save per-fixture predictions
        output_path = self.data_dir / 'predictions' / f'gw{gameweek}_{season.replace("/", "-")}.csv'
        output_path.parent.mkdir(exist_ok=True)
        test_df.to_csv(output_path, index=False)

        # Aggregate DGW players: sum points across fixtures
        test_df = self._aggregate_dgw(test_df, verbose)

        # Log predictions to DB
        try:
            n_logged = log_predictions(str(self.data_dir), test_df, gameweek, season)
            if verbose:
                print(f"Logged {n_logged} predictions to DB")
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to log predictions to DB: {e}")

        if verbose:
            print(f"\nSaved predictions to: {output_path}")
            print(f"Total players: {len(test_df)}")

        return test_df
    
    def _aggregate_dgw(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Aggregate DGW players: sum expected points across their multiple fixtures.

        Players with a single fixture pass through unchanged.
        DGW players get their points summed and opponents joined (e.g. 'BRE, WOL').
        """
        if 'fixture_num' not in df.columns:
            return df

        # Check if any DGW players exist
        fixture_counts = df.groupby('player_id')['fixture_num'].count()
        dgw_players = fixture_counts[fixture_counts > 1].index
        if len(dgw_players) == 0:
            return df

        # Split into SGW and DGW
        sgw_df = df[~df['player_id'].isin(dgw_players)].copy()
        dgw_df = df[df['player_id'].isin(dgw_players)].copy()

        # Columns to sum across fixtures
        sum_cols = [c for c in df.columns if c.startswith('exp_') or c.startswith('pred_exp_')]

        # Columns to take from first fixture (player identity, rolling features, etc.)
        skip_cols = set(sum_cols + ['opponent', 'is_home', 'fixture_num',
                                      'pred_minutes', 'pred_goals_per90', 'pred_assists_per90',
                                      'pred_cs_prob', 'pred_2plus_conceded', 'pred_4plus_conceded',
                                      'pred_goals_against', 'pred_team_goals',
                                      'pred_defcon_prob', 'pred_yellow_prob', 'pred_red_prob',
                                      'pred_bonus'])

        # Aggregate DGW rows per player
        agg_rows = []
        for pid, group in dgw_df.groupby('player_id'):
            # Take identity/features from first fixture row
            base = group.iloc[0].to_dict()

            # Sum points columns
            for col in sum_cols:
                if col in group.columns:
                    base[col] = group[col].sum()

            # Sum key prediction columns too
            for col in ['pred_minutes', 'pred_exp_goals', 'pred_exp_assists',
                        'pred_bonus', 'pred_defcon_prob',
                        'pred_yellow_prob', 'pred_red_prob',
                        'pred_exp_saves']:
                if col in group.columns:
                    base[col] = group[col].sum()

            # Average probability-based columns (don't sum probabilities)
            for col in ['pred_cs_prob', 'pred_2plus_conceded', 'pred_4plus_conceded',
                        'pred_goals_against', 'pred_team_goals',
                        'pred_goals_per90', 'pred_assists_per90']:
                if col in group.columns:
                    base[col] = group[col].mean()

            # Join opponent names
            base['opponent'] = ', '.join(group['opponent'].astype(str).tolist())
            base['is_home'] = ', '.join(group['is_home'].astype(str).tolist())
            base['fixture_num'] = len(group)

            agg_rows.append(base)

        agg_df = pd.DataFrame(agg_rows)
        result = pd.concat([sgw_df, agg_df], ignore_index=True)

        if verbose:
            print(f"  Aggregated {len(dgw_players)} DGW players ({len(dgw_df)} fixtures -> {len(agg_df)} rows)")

        return result

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
        fixtures = self._get_gw_fixtures(gameweek, season, verbose)

        if len(fixtures) == 0:
            if verbose:
                print("WARNING: No fixtures found - opponents will be 'Unknown'")
            latest['opponent'] = 'Unknown'
            latest['is_home'] = 0
            latest['fixture_num'] = 1
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

            # Find ALL matching fixtures (handles DGW where a team plays twice)
            fixture_num = 0
            for fix in fixture_teams:
                # Check if player's team matches home or away team
                if player_team == fix['home_norm'] or player_team in fix['home_norm'] or fix['home_norm'] in player_team:
                    fixture_num += 1
                    row = player.to_dict()
                    row['opponent'] = fix['away_team']
                    row['is_home'] = 1
                    row['gameweek'] = gameweek
                    row['season'] = season
                    row['fixture_num'] = fixture_num
                    # Update team features to latest (fixes stale stats for players who missed games)
                    self._update_team_features(row, player.get('team', ''), team_stats_lookup)
                    # Update opponent features based on actual opponent
                    self._update_opponent_features(row, fix['away_team'], team_stats_lookup)
                    test_rows.append(row)
                elif player_team == fix['away_norm'] or player_team in fix['away_norm'] or fix['away_norm'] in player_team:
                    fixture_num += 1
                    row = player.to_dict()
                    row['opponent'] = fix['home_team']
                    row['is_home'] = 0
                    row['gameweek'] = gameweek
                    row['season'] = season
                    row['fixture_num'] = fixture_num
                    # Update team features to latest (fixes stale stats for players who missed games)
                    self._update_team_features(row, player.get('team', ''), team_stats_lookup)
                    # Update opponent features based on actual opponent
                    self._update_opponent_features(row, fix['home_team'], team_stats_lookup)
                    test_rows.append(row)

        if verbose:
            if test_rows:
                n_players = len(set(r.get('player_id') for r in test_rows))
                n_dgw = len(test_rows) - n_players
                print(f"Matched {len(test_rows)} player-fixtures to GW{gameweek} ({n_players} players)")
                if n_dgw > 0:
                    print(f"  DGW: {n_dgw} players have 2 fixtures")
            else:
                print(f"WARNING: No players matched to GW{gameweek} fixtures!")
                # Show what team names we have vs fixture teams
                player_teams = set(normalize_team_name(t) for t in latest['team'].unique())
                fixture_team_names = set()
                for fix in fixture_teams:
                    fixture_team_names.add(fix['home_norm'])
                    fixture_team_names.add(fix['away_norm'])
                unmatched = player_teams - fixture_team_names
                if unmatched:
                    print(f"  Unmatched player teams: {sorted(unmatched)[:10]}")
                    print(f"  Fixture teams: {sorted(fixture_team_names)}")

        if test_rows:
            return pd.DataFrame(test_rows)
        else:
            # Don't return stale opponents from historical data
            latest['opponent'] = 'Unknown'
            latest['is_home'] = 0
            latest['fixture_num'] = 1
            return latest
    
    def _build_team_stats_lookup(self) -> dict:
        """Build lookup of latest team stats (offensive + defensive) for feature updates."""
        # Get latest stats per team - include ALL team-level features
        team_cols = ['team', 'season', 'gameweek', 
                     # Offensive stats
                     'team_goals_roll5', 'team_goals_roll10', 'team_xg_roll5', 'team_xg_roll10',
                     # Defensive stats (multiple time horizons)
                     'team_conceded_roll1', 'team_conceded_roll3', 'team_conceded_roll5', 'team_conceded_roll10', 'team_conceded_roll30',
                     'team_xga_roll1', 'team_xga_roll3', 'team_xga_roll5', 'team_xga_roll10', 'team_xga_roll30',
                     # Clean sheet rates (multiple time horizons)
                     'team_cs_rate_roll1', 'team_cs_rate_roll3', 'team_cs_rate_roll5', 'team_cs_rate_roll10', 'team_cs_rate_roll30']
        
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
            # Opponent's defensive weakness = their team_conceded_roll (multiple time horizons)
            for window in [1, 3, 5, 10, 30]:
                if f'team_conceded_roll{window}' in opp_stats:
                    row[f'opp_conceded_roll{window}'] = opp_stats[f'team_conceded_roll{window}']
                if f'team_xga_roll{window}' in opp_stats:
                    row[f'opp_xga_roll{window}'] = opp_stats[f'team_xga_roll{window}']
            # Opponent's clean sheet rate
            for window in [5, 10]:
                if f'team_cs_rate_roll{window}' in opp_stats:
                    row[f'opp_cs_rate_roll{window}'] = opp_stats[f'team_cs_rate_roll{window}']
    
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
            # Update team defensive stats (multiple time horizons)
            for window in [1, 3, 5, 10, 30]:
                if f'team_conceded_roll{window}' in team_stats:
                    row[f'team_conceded_roll{window}'] = team_stats[f'team_conceded_roll{window}']
                if f'team_xga_roll{window}' in team_stats:
                    row[f'team_xga_roll{window}'] = team_stats[f'team_xga_roll{window}']
                if f'team_cs_rate_roll{window}' in team_stats:
                    row[f'team_cs_rate_roll{window}'] = team_stats[f'team_cs_rate_roll{window}']
            # Update team offensive stats too
            if 'team_goals_roll5' in team_stats:
                row['team_goals_roll5'] = team_stats['team_goals_roll5']
            if 'team_goals_roll10' in team_stats:
                row['team_goals_roll10'] = team_stats['team_goals_roll10']
            if 'team_xg_roll5' in team_stats:
                row['team_xg_roll5'] = team_stats['team_xg_roll5']
            if 'team_xg_roll10' in team_stats:
                row['team_xg_roll10'] = team_stats['team_xg_roll10']
    
    def _recompute_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute interaction features after opponent stats have been updated.

        _update_opponent_features overwrites opp_* columns with actual fixture
        opponent stats, but interaction columns (products of player x opponent)
        remain stale from historical data. This recalculates them.
        """
        df = df.copy()
        df['xg_x_opp_conceded'] = df['xg_per90_roll5'].fillna(0) * df['opp_conceded_roll5'].fillna(0)
        df['xa_x_opp_conceded'] = df['xa_per90_roll5'].fillna(0) * df['opp_conceded_roll5'].fillna(0)
        df['team_goals_x_opp_conceded'] = df['team_goals_roll5'].fillna(0) * df['opp_conceded_roll5'].fillna(0)
        df['defcon_x_opp_xg'] = df['defcon_per90_roll5'].fillna(0) * df['opp_xg_roll5'].fillna(0)
        return df

    def _get_gw_fixtures(self, gameweek: int, season: str, verbose: bool = True) -> pd.DataFrame:
        """Get fixtures for a gameweek from FPL API or local data."""
        import requests
        try:
            bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
            teams = {t['id']: t['name'] for t in bootstrap['teams']}

            fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/", timeout=10).json()
            gw_fixtures = [f for f in fixtures if f.get('event') == gameweek]

            if gw_fixtures:
                result = pd.DataFrame([{
                    'home_team': teams.get(f['team_h'], 'Unknown'),
                    'away_team': teams.get(f['team_a'], 'Unknown'),
                } for f in gw_fixtures])
                if verbose:
                    print(f"  Fetched {len(result)} GW{gameweek} fixtures from FPL API")
                    for _, fix in result.iterrows():
                        print(f"    {fix['home_team']} vs {fix['away_team']}")
                return result
            elif verbose:
                print(f"  WARNING: FPL API returned 0 fixtures for GW{gameweek}")
        except Exception as e:
            if verbose:
                print(f"  WARNING: FPL API request failed: {e}")

        # Fallback to local fixtures
        for fname in ['fixtures.csv', 'all_fixtures_8_seasons.csv']:
            fixtures_file = self.data_dir / fname
            if fixtures_file.exists():
                try:
                    local_fixtures = pd.read_csv(fixtures_file)
                    # Handle both 'round' (raw) and 'gameweek' (renamed) column names
                    gw_col = 'gameweek' if 'gameweek' in local_fixtures.columns else 'round'
                    result = local_fixtures[
                        (local_fixtures['season'] == season) & (local_fixtures[gw_col] == gameweek)
                    ][['home_team', 'away_team']]
                    if len(result) > 0:
                        if verbose:
                            print(f"  Using {len(result)} fixtures from local {fname}")
                        return result
                except Exception as e:
                    if verbose:
                        print(f"  WARNING: Failed to read local fixtures: {e}")

        if verbose:
            print(f"  WARNING: No fixtures found for GW{gameweek} from any source")
        return pd.DataFrame()
    
    def _build_cs_prediction_row(self, team_row, opp_row, is_home, league_avg_goals):
        """Build a proper CS prediction row by combining team's defensive features
        with the actual opponent's offensive features and recomputing interactions.

        The team's latest row has correct defensive features but WRONG opponent features
        (from their last match). This method swaps in the real opponent's stats.
        """
        pred = team_row.copy()
        pred['is_home'] = is_home

        # --- Fix venue-specific defensive stats ---
        # The latest row may be from a different venue than the prediction.
        # Swap in the correct venue's season stats.
        if is_home == 1:
            home_xga = pred.get('home_season_xga')
            home_cs = pred.get('home_season_cs_rate')
            if not pd.isna(home_xga):
                pred['ha_season_xga'] = home_xga
            if not pd.isna(home_cs):
                pred['ha_season_cs_rate'] = home_cs
        else:
            away_xga = pred.get('away_season_xga')
            away_cs = pred.get('away_season_cs_rate')
            if not pd.isna(away_xga):
                pred['ha_season_xga'] = away_xga
            if not pd.isna(away_cs):
                pred['ha_season_cs_rate'] = away_cs

        if opp_row is None:
            # Recompute prior_lambda with correct venue stats
            team_xga = pred.get('ha_season_xga', pred.get('season_xga_per_game', league_avg_goals))
            if pd.isna(team_xga):
                team_xga = league_avg_goals
            pred['prior_lambda'] = team_xga
            pred['naive_cs_prob'] = np.exp(-pred['prior_lambda'])
            return pd.DataFrame([pred])

        # --- Swap in actual opponent's offensive rolling stats ---
        # The opponent's team row has team_scored_roll5/10 and team_xg_scored_roll5/10
        # which are the opponent's OWN goals/xG rolling averages.
        opp_xg5 = opp_row.get('team_xg_scored_roll5', opp_row.get('season_xg_per_game', league_avg_goals))
        opp_xg10 = opp_row.get('team_xg_scored_roll10', opp_row.get('season_xg_per_game', league_avg_goals))
        opp_goals5 = opp_row.get('team_scored_roll5', opp_row.get('season_goals_per_game', league_avg_goals))
        opp_goals10 = opp_row.get('team_scored_roll10', opp_row.get('season_goals_per_game', league_avg_goals))

        # Handle NaN
        if pd.isna(opp_xg5):
            opp_xg5 = opp_row.get('season_xg_per_game', league_avg_goals)
        if pd.isna(opp_xg10):
            opp_xg10 = opp_row.get('season_xg_per_game', league_avg_goals)
        if pd.isna(opp_goals5):
            opp_goals5 = opp_row.get('season_goals_per_game', league_avg_goals)
        if pd.isna(opp_goals10):
            opp_goals10 = opp_row.get('season_goals_per_game', league_avg_goals)

        pred['opp_xg_roll5'] = opp_xg5
        pred['opp_xg_roll10'] = opp_xg10

        # Season-level opponent identity
        pred['opp_season_goals_per_game'] = opp_row.get('season_goals_per_game',
            opp_row.get('opp_season_goals_per_game', league_avg_goals))
        pred['opp_season_xg_per_game'] = opp_row.get('season_xg_per_game',
            opp_row.get('opp_season_xg_per_game', league_avg_goals))

        # Opponent venue-specific scoring — use overall season as best proxy
        pred['opp_ha_season_goals'] = pred['opp_season_goals_per_game']
        pred['opp_ha_season_xg'] = pred['opp_season_xg_per_game']

        # --- Recompute prior_lambda with correct opponent ---
        team_xga = pred.get('ha_season_xga', pred.get('season_xga_per_game', league_avg_goals))
        if pd.isna(team_xga):
            team_xga = league_avg_goals
        opp_goals_season = pred['opp_ha_season_goals']
        if pd.isna(opp_goals_season):
            opp_goals_season = league_avg_goals
        pred['prior_lambda'] = team_xga * (opp_goals_season / league_avg_goals)

        # Naive CS probability anchor
        pred['naive_cs_prob'] = np.exp(-pred['prior_lambda'])

        # --- Recompute interaction/ratio features ---
        team_xga5 = pred.get('team_xga_roll5', 1.0)
        team_xga10 = pred.get('team_xga_roll10', 1.0)
        if pd.isna(team_xga5):
            team_xga5 = 1.0
        if pd.isna(team_xga10):
            team_xga10 = 1.0

        pred['xga_x_opp_xg_roll5'] = team_xga5 * opp_xg5
        pred['xga_x_opp_xg_roll10'] = team_xga10 * opp_xg10
        pred['xga_div_opp_xg_roll5'] = team_xga5 / (opp_xg5 + 0.01)
        pred['xga_div_opp_xg_roll10'] = team_xga10 / (opp_xg10 + 0.01)

        # Opponent shots on target proxy from their xG and goals rolling stats
        opp_shots_ot5 = opp_row.get('opp_shots_ot_roll5', opp_xg5 * 3.5)
        opp_shots_ot10 = opp_row.get('opp_shots_ot_roll10', opp_xg10 * 3.5)
        # These are the opponent's OWN shots on target — but the opp_shots_ot in their row
        # is their OPPONENTS' shots on target. Use goals as better proxy.
        pred['opp_shots_ot_roll5'] = opp_goals5 * 3.0
        pred['opp_shots_ot_roll10'] = opp_goals10 * 3.0
        pred['opp_key_passes_roll10'] = opp_xg10 * 7

        pred['def_actions_x_opp_shots_roll5'] = pred.get('team_def_actions_roll5', 0) * pred['opp_shots_ot_roll5']

        # --- Recompute home advantage interactions ---
        pred['home_x_team_xga_roll10'] = is_home * team_xga10
        pred['home_x_season_cs_rate'] = is_home * pred.get('season_cs_rate', 0.25)

        return pd.DataFrame([pred])

    def _predict_clean_sheet(self, test_df: pd.DataFrame, gameweek: int,
                             season: str, verbose: bool) -> tuple:
        """Predict goals against, then derive CS prob and 2+ conceded prob for each player's team.

        Swaps in the actual GW opponent's offensive stats (not the last-match opponent's).

        Returns:
            Tuple of (cs_probs, two_plus_probs, pred_goals_against) as numpy arrays
        """
        # Get team-level features
        team_features = self.models['clean_sheet'].prepare_team_features(self.df)

        # Get latest team features
        latest_team = team_features.sort_values(['team', 'season', 'gameweek']).groupby('team').last().reset_index()

        # Build lookup by lowercase team name
        team_lookup = {}
        for _, t in latest_team.iterrows():
            team_lookup[str(t['team']).lower()] = t

        # League average for fallbacks
        league_avg_goals = team_features['goals_conceded'].mean()
        if league_avg_goals < 0.5:
            league_avg_goals = 1.3

        # Also build a lookup for the opponent's season-level scoring identity
        # We need each team's own scoring stats (goals/xG per game this season)
        # These are computed in prepare_team_features as 'goals' per match
        season_scoring = team_features[team_features['season'] == season].groupby('team').agg(
            season_goals_per_game=('goals', 'mean'),
            season_xg_per_game_off=('xg', 'mean'),
        ).reset_index()
        scoring_lookup = {}
        for _, r in season_scoring.iterrows():
            scoring_lookup[str(r['team']).lower()] = r

        # Map to players — one prediction per unique (team, opponent, is_home)
        match_cache = {}
        cs_probs = []
        two_plus_probs = []
        goals_against = []

        for _, row in test_df.iterrows():
            team = str(row.get('team', '')).lower()
            opponent = str(row.get('opponent', '')).lower()
            is_home = row.get('is_home', 0)
            cache_key = (team, opponent, is_home)

            if cache_key in match_cache:
                cs_prob, two_plus, ga = match_cache[cache_key]
            else:
                # Find matching team and opponent
                team_row = self._fuzzy_team_lookup(team, team_lookup)
                opp_row = self._fuzzy_team_lookup(opponent, team_lookup)

                # Also get opponent's scoring identity
                opp_scoring = self._fuzzy_team_lookup(opponent, scoring_lookup)

                if team_row is not None:
                    # Merge opponent scoring stats into opp_row for lookup
                    opp_info = opp_row.copy() if opp_row is not None else pd.Series()
                    if opp_scoring is not None:
                        opp_info['season_goals_per_game'] = opp_scoring['season_goals_per_game']
                        opp_info['season_xg_per_game'] = opp_scoring['season_xg_per_game_off']

                    team_pred = self._build_cs_prediction_row(
                        team_row, opp_info if len(opp_info) > 0 else None,
                        is_home, league_avg_goals
                    )
                    cs_prob = self.models['clean_sheet'].predict_cs_prob(team_pred)[0]
                    two_plus = self.models['clean_sheet'].predict_2plus_conceded_prob(team_pred)[0]
                    ga = self.models['clean_sheet'].predict_goals_against(team_pred)[0]
                else:
                    cs_prob = 0.25
                    two_plus = 0.40
                    ga = 1.2

                match_cache[cache_key] = (cs_prob, two_plus, ga)

            cs_probs.append(cs_prob)
            two_plus_probs.append(two_plus)
            goals_against.append(ga)

        return np.array(cs_probs), np.array(two_plus_probs), np.array(goals_against)

    @staticmethod
    def _fuzzy_team_lookup(name, lookup):
        """Find a team in a lookup dict by fuzzy substring matching."""
        name = name.lower()
        if name in lookup:
            return lookup[name]
        for key, val in lookup.items():
            if name in key or key in name:
                return val
        return None

    def _get_pred_team_goals(self, test_df: pd.DataFrame, gameweek: int, season: str) -> np.ndarray:
        """Predict how many goals each player's team will score.

        This equals the predicted goals conceded by the OPPONENT.
        Uses the CleanSheetModel with correct matchup features.
        """
        # Get team-level features
        team_features = self.models['clean_sheet'].prepare_team_features(self.df)
        latest_team = team_features.sort_values(['team', 'season', 'gameweek']).groupby('team').last().reset_index()

        team_lookup = {}
        for _, t in latest_team.iterrows():
            team_lookup[str(t['team']).lower()] = t

        league_avg_goals = team_features['goals_conceded'].mean()
        if league_avg_goals < 0.5:
            league_avg_goals = 1.3

        # Opponent's scoring identity for the team whose goals we want to predict
        season_scoring = team_features[team_features['season'] == season].groupby('team').agg(
            season_goals_per_game=('goals', 'mean'),
            season_xg_per_game_off=('xg', 'mean'),
        ).reset_index()
        scoring_lookup = {}
        for _, r in season_scoring.iterrows():
            scoring_lookup[str(r['team']).lower()] = r

        match_cache = {}
        pred_team_goals = []
        for _, row in test_df.iterrows():
            team = str(row.get('team', '')).lower()
            opponent = str(row.get('opponent', '')).lower()
            is_home = row.get('is_home', 0)
            opp_is_home = 1 - is_home
            cache_key = (opponent, team, opp_is_home)  # predict opponent's goals conceded

            if cache_key in match_cache:
                ga = match_cache[cache_key]
            else:
                # For pred_team_goals: predict how many the OPPONENT concedes
                # = how many the player's TEAM scores
                # So the "team" for CS model is the opponent, and the "opponent" is the player's team
                opp_row = self._fuzzy_team_lookup(opponent, team_lookup)
                team_scoring = self._fuzzy_team_lookup(team, scoring_lookup)

                if opp_row is not None:
                    # The player's team is the "attacker" (opponent of the defensive team)
                    attacker_info = pd.Series()
                    if team_scoring is not None:
                        attacker_info['season_goals_per_game'] = team_scoring['season_goals_per_game']
                        attacker_info['season_xg_per_game'] = team_scoring['season_xg_per_game_off']

                    opp_pred = self._build_cs_prediction_row(
                        opp_row, attacker_info if len(attacker_info) > 0 else None,
                        opp_is_home, league_avg_goals
                    )
                    ga = self.models['clean_sheet'].predict_goals_against(opp_pred)[0]
                else:
                    ga = 1.3

                match_cache[cache_key] = ga

            pred_team_goals.append(ga)

        return np.array(pred_team_goals)
    
    def _calculate_expected_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected FPL points."""
        from scipy.stats import poisson
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
            
            # Goals conceded penalty (GK/DEF only, 60+ mins)
            # FPL: lose 1 point per 2 goals conceded (i.e., floor(goals/2) * -1)
            # Expected penalty = sum over k>=2: floor(k/2) * P(goals_against=k) * (-1)
            conceded_penalty = 0
            if pos in ['GK', 'DEF'] and mins >= 60:
                lam = row.get('pred_goals_against', 1.2)
                # Compute expected floor(k/2) penalty using Poisson probabilities
                # Sum P(k) * floor(k/2) for k=2..10 (truncate at 10, negligible beyond)
                expected_neg = 0
                for k in range(2, 11):
                    expected_neg += poisson.pmf(k, lam) * (k // 2)
                # Also add the tail probability (11+) approximated
                tail_prob = 1 - poisson.cdf(10, lam)
                expected_neg += tail_prob * 5  # Conservative: 5 for 11+ goals
                conceded_penalty = -expected_neg  # Negative points
            
            # Defcon (DEF/MID only, 60+ mins)
            defcon_pts = 0
            if pos in ['DEF', 'MID'] and mins >= 60:
                defcon_pts = row.get('pred_defcon_prob', 0) * FPL_POINTS['defcon']
            
            # Saves (GK only) - 1 point per 3 saves
            saves_pts = 0
            if pos == 'GK':
                exp_saves = row.get('pred_exp_saves', 0)
                saves_pts = (exp_saves / 3) * FPL_POINTS['saves_per_3']

            # Bonus
            bonus_pts = row.get('pred_bonus', 0)

            # Yellow/Red cards (all positions)
            yellow_pts = row.get('pred_yellow_prob', 0) * FPL_POINTS['yellow_card']
            red_pts = row.get('pred_red_prob', 0) * FPL_POINTS['red_card']

            return pd.Series({
                'exp_goals_pts': goal_pts,
                'exp_assists_pts': assist_pts,
                'exp_cs_pts': cs_pts,
                'exp_conceded_penalty': conceded_penalty,
                'exp_saves_pts': saves_pts,
                'exp_defcon_pts': defcon_pts,
                'exp_bonus_pts': bonus_pts,
                'exp_yellow_pts': yellow_pts,
                'exp_red_pts': red_pts,
                'exp_appearance_pts': app_pts,
                'exp_total_pts': app_pts + goal_pts + assist_pts + cs_pts + conceded_penalty + saves_pts + defcon_pts + bonus_pts + yellow_pts + red_pts
            })
        
        points_df = df.apply(calc_points, axis=1)
        return pd.concat([df, points_df], axis=1)
    
    def get_top_players(self, predictions: pd.DataFrame, n: int = 30) -> pd.DataFrame:
        """Get top N players by expected points."""
        cols = ['player_name', 'team', 'fpl_position', 'fixture_num', 'opponent', 'is_home',
                'pred_minutes', 'pred_exp_goals', 'pred_exp_assists',
                'pred_team_goals', 'pred_cs_prob', 'pred_2plus_conceded', 'pred_4plus_conceded',
                'pred_goals_against', 'pred_defcon_prob', 'pred_yellow_prob', 'pred_red_prob',
                'pred_bonus', 'exp_total_pts']
        available_cols = [c for c in cols if c in predictions.columns]
        return predictions.nlargest(n, 'exp_total_pts')[available_cols]
