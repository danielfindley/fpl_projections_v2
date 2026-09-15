"""
FPL Prediction Pipeline

Main pipeline for loading data, computing features, training models, and generating predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from .data_loader import load_player_stats, load_fixtures, merge_fixtures, get_fpl_positions, map_fpl_position, get_fpl_availability, merge_fpl_card_data, get_fpl_current_squads, normalize_player_name
from .features import (compute_rolling_features, build_appearance_grid,
                       chronological_frame, deadline_splits,
                       resolve_defcon_positions)
from .models import GoalsModel, AssistsModel, MinutesModel, DefconModel, CleanSheetModel, BonusModel, CardsModel, SavesModel
from .models.minutes import StarterClassifier, StarterMinutesModel, SubMinutesModel, ALL_FEATURES as MINUTES_ALL_FEATURES, STARTER_FEATURES, SUB_FEATURES
from .experiment_log import log_experiment, get_history, get_best_run, log_predictions, get_predictions, clear_experiments


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
    'afc bournemouth': 'bournemouth',
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


def _tune_worker(data_dir, name, n_iter, verbose, frame, params):
    pipeline = FPLPipeline(data_dir)
    pipeline.tuned_params = params
    return pipeline._tune_in_process([name], n_iter, verbose, frame)


class FPLPipeline:
    """Complete FPL prediction pipeline."""
    
    # Only use seasons from 2020/21 onwards for training
    MIN_SEASON = '2020/2021'
    
    def __init__(self, data_dir: str = 'data', n_sims: int = 5000):
        self.data_dir = Path(data_dir)
        self.df = None
        self.models: Dict = {}
        self.tuned_params: Dict = {}  # Store tuned hyperparameters
        self.fpl_positions = {}
        self.current_season_players = set()  # Track players in current season
        self.n_sims = n_sims
    
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

        # Merge FPL yellow/red card data (required for CardsModel)
        self.df = merge_fpl_card_data(self.df, str(self.data_dir), verbose)

        # Exact season/GW positions from the FPL cache take precedence; the
        # current FPL API fills older rows for players still in the game.
        # Feature engineering uses FotMob only for the remaining historical gaps.
        self.fpl_positions = get_fpl_positions()
        api_position = self.df.apply(
            lambda row: map_fpl_position(
                row.get('position'), row.get('player_name'), self.fpl_positions,
                fallback_to_fotmob=False),
            axis=1,
        ).astype('string').str.upper()
        cached_position = self.df.get(
            'fpl_position', pd.Series(pd.NA, index=self.df.index, dtype='string')
        ).astype('string').str.upper()
        valid_cached = cached_position.isin(['GK', 'DEF', 'MID', 'FWD'])
        self.df['fpl_position'] = cached_position.where(valid_cached, api_position)
        if verbose:
            known_position = self.df['fpl_position'].isin(['GK', 'DEF', 'MID', 'FWD'])
            print(f"  FPL positions: {int(known_position.sum()):,}/{len(self.df):,} rows resolved; "
                  "historical gaps use FotMob for DefCon")

        # Carry actual match timing into every feature and validation window.
        date_path = self.data_dir / 'matches' / 'match_details.csv'
        if date_path.exists():
            dates = pd.read_csv(date_path, usecols=['match_id', 'match_date']).drop_duplicates('match_id')
            self.df = self.df.drop(columns=['match_date'], errors='ignore').merge(
                dates, on='match_id', how='left', validate='many_to_one')
            missing_dates = pd.to_datetime(self.df['match_date'], errors='coerce', utc=True).isna()
            if missing_dates.any():
                if verbose:
                    print(f"  Excluding {self.df.loc[missing_dates, 'match_id'].nunique()} matches without kickoff timestamps")
                self.df = self.df.loc[~missing_dates].copy()
        else:
            raise ValueError("match_details.csv with real kickoff timestamps is required for temporal training")
        self.df = chronological_frame(self.df)
        self.df.attrs['data_dir'] = str(self.data_dir)

        # Snapshot raw data before feature engineering. Used at predict time to
        # build synthetic target-gameweek rows so rolling features include the
        # player's most recent actual match (shift(1) excludes the synthetic row).
        self.raw_df = self.df.copy()

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
        Training order matters:
        1. MinutesModel first (generates pred_minutes feature for Goals/Assists)
        2. CleanSheetModel (generates pred_team_goals feature via OOF)
        3. Goals/Assists models consume both pred_minutes and pred_team_goals
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
        
        # Minutes model (independent). The appearance grid adds the weeks players did
        # not feature — rows that exist nowhere in self.df, which only holds appearances.
        mins_params = self.tuned_params.get('minutes', {})
        self.models['minutes'] = MinutesModel(**mins_params)
        self.appearance_grid = build_appearance_grid(self.df, verbose)
        self.models['minutes'].fit(self.df, verbose, appearance_grid=self.appearance_grid)
        
        # Clean sheet model FIRST (needed for pred_team_goals feature)
        cs_params = self.tuned_params.get('clean_sheet', {})
        self.models['clean_sheet'] = CleanSheetModel(**cs_params)
        self.models['clean_sheet'].fit(self.df, verbose)
        
        # Generate leak-free OOF predicted team goals for training data
        self._generate_oof_team_goals(verbose)

        # Downstream training sees cross-fitted minutes, never the current match's outcome.
        self.df['pred_minutes'] = self._generate_oof_minutes(self.df, mins_params, verbose)

        # Goals model (uses pred_team_goals and pred_minutes)
        goals_params = self.tuned_params.get('goals', {})
        # Ensure pred_team_goals and pred_minutes are included in selected features
        if goals_params.get('selected_features'):
            for feat in ['pred_team_goals', 'pred_minutes']:
                if feat not in goals_params['selected_features']:
                    goals_params['selected_features'] = goals_params['selected_features'] + [feat]
        self.models['goals'] = GoalsModel(**goals_params)
        self.models['goals'].fit(self.df, verbose)
        
        # Assists model (uses pred_team_goals and pred_minutes)
        assists_params = self.tuned_params.get('assists', {})
        if assists_params.get('selected_features'):
            for feat in ['pred_team_goals', 'pred_minutes']:
                if feat not in assists_params['selected_features']:
                    assists_params['selected_features'] = assists_params['selected_features'] + [feat]
        self.models['assists'] = AssistsModel(**assists_params)
        self.models['assists'].fit(self.df, verbose)
        
        # Defcon model (independent)
        defcon_params = self.tuned_params.get('defcon', {})
        self.models['defcon'] = DefconModel(**defcon_params)
        self.models['defcon'].fit(self.df, verbose)
        
        # Bonus model (not tuned - uses Monte Carlo simulation)
        self.models['bonus'] = BonusModel(n_simulations=self.n_sims)
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
    
    @staticmethod
    def _model_class(name):
        return {'minutes': MinutesModel, 'clean_sheet': CleanSheetModel,
                'goals': GoalsModel, 'assists': AssistsModel,
                'defcon': DefconModel, 'saves': SavesModel}[name]

    @staticmethod
    def _fit_model(name, frame, params=None):
        """One fitting contract, used by CV, holdout, OOF and final training."""
        model = FPLPipeline._model_class(name)(**(params or {}))
        if name == 'clean_sheet':
            model.fit_prepared(frame, verbose=False)
        elif name == 'minutes':
            model.fit(frame, verbose=False,
                      appearance_grid=build_appearance_grid(frame, verbose=False))
        else:
            model.fit(frame, verbose=False)
        return model

    @staticmethod
    def _model_predict(name, model, frame):
        if name == 'clean_sheet':
            return model.predict_goals_against(frame)
        if name == 'saves':
            return model.predict_per90(frame)
        return model.predict(frame)

    @staticmethod
    def _loss(name, actual, predicted):
        from sklearn.metrics import mean_poisson_deviance, mean_absolute_error
        if name == 'minutes':
            residual = np.abs(np.asarray(actual) - predicted)
            quadratic = np.minimum(residual, 10.0)
            return float(np.mean(0.5 * quadratic**2 + 10 * (residual - quadratic)))
        if name == 'saves':
            return float(mean_absolute_error(actual, predicted))
        return float(mean_poisson_deviance(np.maximum(actual, 0),
                                          np.maximum(predicted, 1e-8)))

    @staticmethod
    def _map_team_prediction(players, teams, values, opponent=False):
        """Fixture identity prevents collisions in double gameweeks."""
        lookup = teams[['match_id', 'team']].copy()
        lookup['_team'] = lookup['team'].map(normalize_team_name)
        lookup['_prediction'] = np.asarray(values)
        lookup = lookup.drop_duplicates(['match_id', '_team']).set_index(['match_id', '_team'])
        keys = pd.MultiIndex.from_arrays([
            players['match_id'],
            players['opponent' if opponent else 'team'].map(normalize_team_name)])
        return lookup['_prediction'].reindex(keys).fillna(1.3).to_numpy()

    def _generate_oof_minutes(self, df_train, mins_params, verbose=False):
        frame = chronological_frame(df_train).reset_index(drop=True)
        # Cold-start rows have no fitted predecessor. Use only prior information,
        # never their actual minutes. These rows are not scored as CV validation.
        prediction = frame.get('minutes_roll5', pd.Series(60., index=frame.index))
        prediction = prediction.fillna(60).clip(1, 90).to_numpy(copy=True, dtype=float)
        for ti, vi in deadline_splits(frame):
            model = self._fit_model('minutes', frame.iloc[ti], mins_params)
            prediction[vi] = model.predict(frame.iloc[vi])
        return prediction

    def _generate_oof_team_goals_for_tuning(self, df_train, cs_params, verbose=False):
        frame = chronological_frame(df_train).copy()
        teams = CleanSheetModel().prepare_team_features(frame).reset_index(drop=True)
        prediction = teams['prior_lambda'].fillna(1.3).to_numpy(copy=True)
        for ti, vi in deadline_splits(teams):
            model = self._fit_model('clean_sheet', teams.iloc[ti], cs_params)
            prediction[vi] = model.predict_goals_against(teams.iloc[vi])
        frame['pred_team_goals'] = self._map_team_prediction(frame, teams, prediction, opponent=True)
        return frame

    def _generate_oof_team_goals(self, verbose=True):
        self.df = self._generate_oof_team_goals_for_tuning(
            self.df, self.tuned_params.get('clean_sheet', {}), verbose)

    def _dependency_frames(self, train, valid=None, need_team=True):
        """Cross-fit upstream features inside this training window, not outside CV."""
        train = chronological_frame(train).copy()
        train['pred_minutes'] = self._generate_oof_minutes(
            train, self.tuned_params.get('minutes', {}))
        if need_team:
            train = self._generate_oof_team_goals_for_tuning(
                train, self.tuned_params.get('clean_sheet', {}))
        if valid is None:
            return train
        valid = chronological_frame(valid).copy()
        minutes = self._fit_model('minutes', train, self.tuned_params.get('minutes', {}))
        valid['pred_minutes'] = minutes.predict(valid)
        if need_team:
            # Preparing history together is safe: each feature's information
            # cutoff is its own deadline. Fitting uses training outcomes only.
            teams = CleanSheetModel().prepare_team_features(pd.concat([train, valid], ignore_index=True))
            train_teams = teams[teams['match_id'].isin(train['match_id'])]
            cs = self._fit_model('clean_sheet', train_teams,
                                 self.tuned_params.get('clean_sheet', {}))
            valid['pred_team_goals'] = self._map_team_prediction(
                valid, teams, cs.predict_goals_against(teams), opponent=True)
        return train, valid

    def tune(self, models: list = None, n_iter: int = 100, test_size: float = 0.2,
             test_season: str = '2025/2026', test_start_gw: int = None,
             verbose: bool = True, use_subprocess: bool = False,
             description: str = '') -> 'FPLPipeline':
        """Tune model hyperparameters using Optuna with holdout test set evaluation.

        This method:
        1. Splits data into train/test (2025/2026 season held out by default)
        2. Tunes hyperparameters using 5-fold CV on training set
        3. Evaluates best models on held-out test set
        4. Stores tuned params (does NOT train final models - call train() for that)

        Args:
            models: List of model names to tune. Defaults to ['goals', 'assists', 'minutes', 'defcon', 'clean_sheet']
            n_iter: Number of Optuna trials per model (default 100)
            test_size: Ignored when test_season is set (kept for backward compat).
            test_season: Season to use as holdout test set (default '2025/2026').
                         Seasons after the holdout are always excluded from tuning
                         to preserve chronological evaluation.
                         Set to None to fall back to percentage-based temporal split.
            test_start_gw: If set, only gameweeks >= this value in test_season are held out.
                           Earlier gameweeks from test_season are included in training data.
            verbose: Print progress
            use_subprocess: If True, run each model's tuning in a separate subprocess to save RAM
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() and compute_features() first.")

        # Default model order: minutes first (OOF pred_minutes for downstream),
        # clean_sheet next (OOF pred_team_goals for goals/assists), then the rest.
        # Order is enforced in _tune_in_process/_tune_with_subprocess regardless,
        # but the list determines which models get tuned.
        if models is None:
            models = ['minutes', 'clean_sheet', 'goals', 'assists', 'defcon', 'saves']

        if verbose:
            print("\n" + "=" * 60)
            print("TUNING HYPERPARAMETERS WITH HOLDOUT TEST SET")
            print("=" * 60)

        # Create train/test split
        df_sorted = self.df[self.df['minutes'] >= 1].copy()
        df_sorted = chronological_frame(df_sorted).sort_values('match_date')

        if test_season is not None:
            # Split by season (optionally filtered by gameweek)
            season_mask = df_sorted['season'] == test_season
            if test_start_gw is not None:
                # Only GWs >= test_start_gw in test_season are held out
                df_test = df_sorted[season_mask & (df_sorted['gameweek'] >= test_start_gw)].copy()
                df_train = df_sorted[
                    (df_sorted['season'] < test_season)
                    | (season_mask & (df_sorted['gameweek'] < test_start_gw))
                ].copy()
            else:
                df_test = df_sorted[season_mask].copy()
                df_train = df_sorted[df_sorted['season'] < test_season].copy()
            if len(df_test) == 0:
                raise ValueError(f"No data found for test season '{test_season}'" +
                                 (f" GW>={test_start_gw}" if test_start_gw else ""))
        else:
            # Fall back to percentage-based temporal split
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1")
            deadlines = np.sort(df_sorted['forecast_time'].unique())
            cutoff = deadlines[max(1, min(len(deadlines) - 1,
                                        int(len(deadlines) * (1 - test_size))))]
            df_train = df_sorted[df_sorted['forecast_time'] < cutoff].copy()
            df_test = df_sorted[df_sorted['forecast_time'] >= cutoff].copy()

        # A rescheduled older-GW result may not yet exist at the holdout origin.
        df_train = df_train[df_train['result_time'] < df_test['forecast_time'].min()].copy()
        if df_train.empty or df_test.empty:
            raise ValueError("Temporal split must contain training and holdout rows")

        if verbose:
            split_label = f"season={test_season}" if test_season else f"temporal {test_size:.0%}"
            print(f"\nData split ({split_label}):")
            print(f"  Train: {len(df_train):,} samples ({sorted(df_train['season'].unique())})")
            print(f"  Test:  {len(df_test):,} samples ({sorted(df_test['season'].unique())})")
            test_gws = df_test.groupby('season')['gameweek'].agg(['min', 'max'])
            for s, row in test_gws.iterrows():
                print(f"    {s}: GW{row['min']}-GW{row['max']}")
        
        if verbose:
            print("\n" + "-" * 60)
            print("PHASE 1: Hyperparameter + Feature Selection Tuning (5-fold CV)")
            print("-" * 60)
        
        if use_subprocess:
            new_params, cv_scores = self._tune_with_subprocess(models, n_iter, verbose, df_train)
        else:
            new_params, cv_scores = self._tune_in_process(models, n_iter, verbose, df_train)
        self.tuned_params.update(new_params)

        # Evaluate on held-out test set
        if verbose:
            print("\n" + "-" * 60)
            print("PHASE 2: Evaluation on Held-Out Test Set")
            print("-" * 60)

        test_metrics = self._evaluate_on_test_set(models, df_train, df_test, verbose)

        # Store test metrics for later use (e.g., viz)
        self.last_test_metrics = test_metrics.copy()

        # Log experiment to SQLite
        try:
            fpl_mae_dict = test_metrics.pop('_fpl_points_mae', None)
            fpl_ex = fpl_mae_dict['mae_ex_bonus'] if isinstance(fpl_mae_dict, dict) else fpl_mae_dict
            fpl_inc = fpl_mae_dict['mae_inc_bonus'] if isinstance(fpl_mae_dict, dict) else None
            fpl_top25 = None  # deprecated; column kept for schema compat
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
                fpl_points_mae_top25_inc_bonus=fpl_top25,
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

        Feature selection is integrated into Optuna: ranking method and number of
        features are hyperparameters. Rankings are pre-computed once per model before
        trials begin, so each trial just slices the top-N from the chosen ranking.

        Uses TimeSeriesSplit for time-aware cross-validation.
        """
        import optuna
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.metrics import make_scorer, mean_poisson_deviance
        import xgboost as xgb
        from .feature_selection import compute_feature_rankings, select_features

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
        SCORING = {
            'goals': ('Poisson Deviance', poisson_scorer),
            'assists': ('Poisson Deviance', poisson_scorer),
            'defcon': ('Poisson Deviance', poisson_scorer),
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

        # Min features per model (excluding protected features)
        MIN_FEATURES = {
            'goals': 15, 'assists': 5, 'defcon': 5, 'saves': 5,
        }

        # Protected features that must always be included
        PROTECTED = {
            'goals': ['pred_team_goals', 'pred_minutes'],
            'assists': ['pred_team_goals', 'pred_minutes'],
            'defcon': ['pred_minutes'],
            'saves': [],
        }

        RANKING_METHODS = ['xgb_gain', 'xgb_cover', 'lgbm', 'permutation', 'mutual_info']

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

        # --- Dependency-ordered tuning with OOF feature generation ---
        # 1. Minutes first → OOF pred_minutes for downstream models
        # 2. Clean sheet → OOF pred_team_goals for goals/assists
        # 3. Everything else uses the OOF predictions as features

        # Stage 1: Tune minutes (no upstream dependencies)
        if 'minutes' in models:
            mins_result = self._tune_minutes_in_process(
                n_iter, verbose, df_train, SEARCH_SPACES['minutes']
            )
            tuned_params['minutes'] = mins_result['params']
            cv_scores['minutes'] = mins_result['cv_score']

            # Generate OOF pred_minutes for downstream models
            if verbose:
                print(f"\n  Generating OOF pred_minutes for downstream models...")
            df_train['pred_minutes'] = self._generate_oof_minutes(df_train, tuned_params['minutes'], verbose)
        else:
            # Fallback: use actual minutes
            df_train['pred_minutes'] = df_train['minutes']

        # Stage 2: Tune clean_sheet (no upstream dependencies)
        if 'clean_sheet' in models:
            cs_result = self._tune_clean_sheet_in_process(
                n_iter, verbose, df_train, SEARCH_SPACES['clean_sheet']
            )
            tuned_params['clean_sheet'] = cs_result['params']
            cv_scores['clean_sheet'] = cs_result['cv_score']

            # Generate OOF pred_team_goals for downstream models
            if verbose:
                print(f"\n  Generating OOF pred_team_goals for downstream models...")
            df_train = self._generate_oof_team_goals_for_tuning(df_train, tuned_params['clean_sheet'], verbose)

        # Stage 3: Tune remaining models (goals, assists, defcon, saves)
        # These now have OOF pred_minutes and pred_team_goals available as features
        remaining = [m for m in models if m not in ('minutes', 'clean_sheet')]
        for model_name in remaining:
            if model_name not in MODEL_CLASSES:
                continue

            model_class = MODEL_CLASSES[model_name]
            space = SEARCH_SPACES.get(model_name, {})
            score_name, scorer = SCORING.get(model_name, ('RMSE', 'neg_root_mean_squared_error'))

            model_instance = model_class()
            all_features = [f for f in model_instance.FEATURES if f in df_train.columns]
            target = model_instance.TARGET
            n_total_features = len(all_features)
            protected = [f for f in PROTECTED.get(model_name, []) if f in all_features]
            min_feats = MIN_FEATURES.get(model_name, 5)

            # Filter to GKs only for saves model
            tune_df = df_train
            if model_name == 'saves':
                tune_df = df_train[df_train['is_gk'] == 1].copy()
            elif model_name == 'defcon':
                tune_df = df_train[
                    df_train['defcon_position'].isin(['DEF', 'MID'])
                    & df_train['defcon'].notna()
                ].copy()

            X_full = tune_df[all_features].fillna(0).values
            y = tune_df[target].fillna(0).values

            if verbose:
                print(f"\nTuning {model_name.upper()} ({n_iter} trials, TimeSeriesSplit CV, {score_name})...")
                print(f"  Pre-computing feature rankings ({n_total_features} features, {len(RANKING_METHODS)} methods)...")

            # Pre-compute rankings once
            xgb_hint = {'objective': 'count:poisson'} if model_name in ('goals', 'assists', 'defcon') else {}
            rankings = compute_feature_rankings(X_full, y, all_features, task='regression', xgb_params=xgb_hint)

            if verbose:
                print(f"  Rankings computed. Starting Optuna search...")

            # Optuna tunes hyperparams + feature selection jointly
            def objective(trial, _rankings=rankings, _all_features=all_features,
                          _protected=protected, _min_feats=min_feats, _n_total=n_total_features):
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

                if model_name in ('goals', 'assists', 'defcon'):
                    params['objective'] = 'count:poisson'

                # Feature selection as hyperparameters
                feat_method = trial.suggest_categorical('feat_method', RANKING_METHODS)
                n_selectable = _n_total - len(_protected)
                n_features = trial.suggest_int('n_features', _min_feats, n_selectable)

                selected = select_features(_rankings, feat_method, n_features, _protected)
                feat_idx = [_all_features.index(f) for f in selected]
                X_sel = X_full[:, feat_idx]

                model = xgb.XGBRegressor(**params)
                scores = cross_val_score(model, X_sel, y, cv=tscv, scoring=scorer, n_jobs=1)
                return -scores.mean()

            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=n_iter, show_progress_bar=verbose)

            best_params = study.best_params.copy()

            # Extract feature selection params
            best_feat_method = best_params.pop('feat_method')
            best_n_features = best_params.pop('n_features')
            selected_features = select_features(rankings, best_feat_method, best_n_features, protected)

            if model_name in ('goals', 'assists', 'defcon'):
                best_params['objective'] = 'count:poisson'

            tuned_params[model_name] = {
                **best_params,
                'selected_features': selected_features,
            }
            cv_scores[model_name] = study.best_value

            if verbose:
                print(f"  Best CV {score_name}: {study.best_value:.4f}")
                print(f"  Feature method: {best_feat_method}, selected {len(selected_features)}/{n_total_features} features")

        return tuned_params, cv_scores

    def _tune_clean_sheet_in_process(self, n_iter: int, verbose: bool,
                                      df_train: pd.DataFrame, space: dict) -> Dict:
        """Tune CleanSheetModel (Poisson regression for goals against).

        Feature selection is integrated into Optuna: ranking method and number of
        features are hyperparameters alongside XGBoost params.
        """
        import optuna
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.metrics import make_scorer, mean_poisson_deviance
        import xgboost as xgb
        from .feature_selection import compute_feature_rankings, select_features

        RANKING_METHODS = ['xgb_gain', 'xgb_cover', 'lgbm', 'permutation', 'mutual_info']

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
            print(f"  Pre-computing feature rankings ({n_total_features} features, {len(RANKING_METHODS)} methods)...")
            print(f"  Team-matches: {len(X_full)}, Avg conceded: {y.mean():.3f}")

        # Pre-compute rankings
        rankings = compute_feature_rankings(X_full, y, all_features, task='regression',
                                            xgb_params={'objective': 'count:poisson'})

        if verbose:
            print(f"  Rankings computed. Starting Optuna search...")

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

            # Feature selection as hyperparameters
            feat_method = trial.suggest_categorical('feat_method', RANKING_METHODS)
            n_features = trial.suggest_int('n_features', 5, n_total_features)

            selected = select_features(rankings, feat_method, n_features)
            feat_idx = [all_features.index(f) for f in selected]
            X_sel = X_full[:, feat_idx]

            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_sel, y, cv=tscv, scoring=poisson_scorer, n_jobs=1)
            return -scores.mean()

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_iter, show_progress_bar=verbose)

        best_params = study.best_params.copy()

        # Extract feature selection params
        best_feat_method = best_params.pop('feat_method')
        best_n_features = best_params.pop('n_features')
        selected_features = select_features(rankings, best_feat_method, best_n_features)

        if verbose:
            print(f"  Best CV Poisson Deviance: {study.best_value:.4f}")
            print(f"  Feature method: {best_feat_method}, selected {len(selected_features)}/{n_total_features} features")

        return {
            'params': {**best_params, 'selected_features': selected_features},
            'cv_score': study.best_value,
        }

    def _tune_minutes_in_process(self, n_iter: int, verbose: bool,
                                  df_train: pd.DataFrame, space: dict) -> dict:
        """Tune the two-stage MinutesModel: classifier + starter regressor + sub regressor.

        Each sub-model gets its own Optuna pass with integrated feature selection.
        Returns nested params dict.
        """
        import optuna
        from sklearn.model_selection import cross_val_score, TimeSeriesSplit
        from sklearn.metrics import make_scorer, log_loss
        import xgboost as xgb
        from .feature_selection import compute_feature_rankings, select_features

        RANKING_METHODS = ['xgb_gain', 'xgb_cover', 'lgbm', 'permutation', 'mutual_info']

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
        n_cls_features = len(cls_features)

        if verbose:
            print(f"\n  [1/3] StarterClassifier ({n_cls_features} features, {y_cls.mean():.1%} starters)")
            print(f"    Pre-computing feature rankings...")

        cls_rankings = compute_feature_rankings(X_cls, y_cls, cls_features, task='classification')

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
            feat_method = trial.suggest_categorical('feat_method', RANKING_METHODS)
            n_features = trial.suggest_int('n_features', 5, n_cls_features)
            selected = select_features(cls_rankings, feat_method, n_features)
            feat_idx = [cls_features.index(f) for f in selected]
            X_sel = X_cls[:, feat_idx]

            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_sel, y_cls, cv=tscv, scoring='neg_log_loss', n_jobs=1)
            return -scores.mean()

        study_cls = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study_cls.optimize(cls_objective, n_trials=n_iter, show_progress_bar=verbose)
        cls_best = study_cls.best_params.copy()
        cls_feat_method = cls_best.pop('feat_method')
        cls_n_features = cls_best.pop('n_features')
        cls_selected = select_features(cls_rankings, cls_feat_method, cls_n_features)

        if verbose:
            print(f"    Best CV LogLoss: {study_cls.best_value:.4f}")
            print(f"    Feature method: {cls_feat_method}, selected {len(cls_selected)}/{n_cls_features} features")

        # ---- 2. StarterMinutesModel (MAE, trained on 60+ only) ----
        df_starters = df_played[df_played['minutes'] >= 60].copy()
        starter_features = [f for f in STARTER_FEATURES if f in df_starters.columns]
        X_start = df_starters[starter_features].fillna(0).values
        y_start = df_starters['minutes'].values
        n_starter_features = len(starter_features)

        if verbose:
            print(f"\n  [2/3] StarterMinutesModel ({n_starter_features} features, {len(df_starters):,} samples)")
            print(f"    Pre-computing feature rankings...")

        starter_rankings = compute_feature_rankings(X_start, y_start, starter_features, task='regression')

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
            feat_method = trial.suggest_categorical('feat_method', RANKING_METHODS)
            n_features = trial.suggest_int('n_features', 5, n_starter_features)
            selected = select_features(starter_rankings, feat_method, n_features)
            feat_idx = [starter_features.index(f) for f in selected]
            X_sel = X_start[:, feat_idx]

            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_sel, y_start, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
            return -scores.mean()

        study_start = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=43))
        study_start.optimize(starter_objective, n_trials=n_iter, show_progress_bar=verbose)
        starter_best = study_start.best_params.copy()
        starter_feat_method = starter_best.pop('feat_method')
        starter_n_features = starter_best.pop('n_features')
        starter_selected = select_features(starter_rankings, starter_feat_method, starter_n_features)

        if verbose:
            print(f"    Best CV MAE: {study_start.best_value:.4f}")
            print(f"    Feature method: {starter_feat_method}, selected {len(starter_selected)}/{n_starter_features} features")

        # ---- 3. SubMinutesModel (MAE, trained on 1-59 only) ----
        df_subs = df_played[(df_played['minutes'] >= 1) & (df_played['minutes'] < 60)].copy()
        sub_features = [f for f in SUB_FEATURES if f in df_subs.columns]
        X_sub = df_subs[sub_features].fillna(0).values
        y_sub = df_subs['minutes'].values
        n_sub_features = len(sub_features)

        if verbose:
            print(f"\n  [3/3] SubMinutesModel ({n_sub_features} features, {len(df_subs):,} samples)")
            print(f"    Pre-computing feature rankings...")

        sub_rankings = compute_feature_rankings(X_sub, y_sub, sub_features, task='regression')

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
            feat_method = trial.suggest_categorical('feat_method', RANKING_METHODS)
            n_features = trial.suggest_int('n_features', 3, n_sub_features)
            selected = select_features(sub_rankings, feat_method, n_features)
            feat_idx = [sub_features.index(f) for f in selected]
            X_sel = X_sub[:, feat_idx]

            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(model, X_sel, y_sub, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1)
            return -scores.mean()

        study_sub = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=44))
        study_sub.optimize(sub_objective, n_trials=n_iter, show_progress_bar=verbose)
        sub_best = study_sub.best_params.copy()
        sub_feat_method = sub_best.pop('feat_method')
        sub_n_features = sub_best.pop('n_features')
        sub_selected = select_features(sub_rankings, sub_feat_method, sub_n_features)

        if verbose:
            print(f"    Best CV MAE: {study_sub.best_value:.4f}")
            print(f"    Feature method: {sub_feat_method}, selected {len(sub_selected)}/{n_sub_features} features")

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

    def _legacy_generate_oof_minutes_unused(self, df_train: pd.DataFrame, mins_params: dict,
                                             verbose: bool = True) -> np.ndarray:
        """Generate out-of-fold predicted minutes on training data.

        Uses TimeSeriesSplit to produce leak-free pred_minutes that downstream
        models (goals, assists, defcon) see during tuning — matching what they'll
        see at inference time when actual minutes are unknown.

        Returns an array of OOF predictions aligned to df_train's index.
        """
        from sklearn.model_selection import TimeSeriesSplit

        df_played = df_train[df_train['minutes'] >= 1].copy()
        df_played = df_played.sort_values(['season', 'gameweek']).reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=5)
        oof_preds = np.full(len(df_played), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df_played)):
            fold_train = df_played.iloc[train_idx]
            fold_val = df_played.iloc[val_idx]

            fold_model = MinutesModel(**mins_params)
            fold_model.fit(fold_train, verbose=False)
            oof_preds[val_idx] = fold_model.predict(fold_val)

        # For rows not covered by any val fold (early data), use actual minutes
        missing = np.isnan(oof_preds)
        oof_preds[missing] = df_played.loc[missing, 'minutes'].values

        # Map back to original df_train index
        result = pd.Series(df_train['minutes'].values, index=df_train.index)
        # df_played was sorted and reset, so map via player_id + season + gameweek
        oof_series = pd.Series(oof_preds, index=df_played.index)
        # Since df_played is a filtered/sorted copy, merge back by matching original indices
        played_original_idx = df_train[df_train['minutes'] >= 1].sort_values(['season', 'gameweek']).index
        for i, orig_idx in enumerate(played_original_idx):
            result.loc[orig_idx] = oof_preds[i]

        if verbose:
            valid = (~np.isnan(oof_preds)).sum()
            mae = np.nanmean(np.abs(oof_preds - df_played['minutes'].values))
            print(f"  OOF pred_minutes: {valid:,} predictions, MAE={mae:.2f}")

        return result.values

    def _legacy_generate_oof_team_goals_for_tuning_unused(
            self, df_train: pd.DataFrame, cs_params: dict,
            verbose: bool = True) -> pd.DataFrame:
        """Generate out-of-fold predicted team goals on training data.

        Uses TimeSeriesSplit on team-level data to produce leak-free pred_team_goals,
        then maps back to player-level rows so goals/assists models see realistic
        predictions during tuning.

        Returns df_train with pred_team_goals column added/updated.
        """
        from sklearn.model_selection import TimeSeriesSplit
        import xgboost as xgb

        cs_model = CleanSheetModel()
        team_df = cs_model.prepare_team_features(df_train)
        team_df = team_df.dropna(subset=['team_conceded_roll5', 'goals_conceded'])
        team_df = team_df.sort_values(['season', 'gameweek']).reset_index(drop=True)

        selected_features = cs_params.get('selected_features', None)
        features = selected_features if selected_features else [f for f in cs_model.FEATURES if f in team_df.columns]
        X = team_df[features].fillna(0).values
        y = team_df['goals_conceded'].fillna(0).values

        xgb_params = {k: v for k, v in cs_params.items() if k not in ('selected_features',)}
        xgb_params.setdefault('objective', 'count:poisson')
        xgb_params.setdefault('random_state', 42)
        xgb_params.setdefault('verbosity', 0)

        tscv = TimeSeriesSplit(n_splits=5)
        oof_preds = np.full(len(y), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = np.clip(model.predict(X[val_idx]), 1e-6, 10.0)

        # Fill early folds with actual values
        missing = np.isnan(oof_preds)
        oof_preds[missing] = y[missing]

        team_df['oof_goals_conceded'] = oof_preds

        # Map to player-level: pred_team_goals = opponent's predicted goals conceded
        def normalize_name(name):
            if pd.isna(name):
                return ''
            return str(name).lower().replace(' ', '_').replace("'", "").strip()

        opp_lookup = team_df[['team', 'season', 'gameweek', 'oof_goals_conceded']].copy()
        opp_lookup = opp_lookup.rename(columns={
            'team': 'opponent',
            'oof_goals_conceded': 'pred_team_goals'
        })
        opp_lookup['opponent_norm'] = opp_lookup['opponent'].apply(normalize_name)
        opp_lookup = opp_lookup.drop_duplicates(subset=['opponent_norm', 'season', 'gameweek'], keep='first')

        df_train['opponent_norm'] = df_train['opponent'].apply(normalize_name)

        if 'pred_team_goals' in df_train.columns:
            df_train = df_train.drop(columns=['pred_team_goals'])

        n_before = len(df_train)
        df_train = df_train.merge(
            opp_lookup[['opponent_norm', 'season', 'gameweek', 'pred_team_goals']],
            on=['opponent_norm', 'season', 'gameweek'],
            how='left'
        )
        if len(df_train) != n_before:
            df_train = df_train.drop_duplicates(subset=['player_id', 'season', 'gameweek'], keep='first')

        df_train['pred_team_goals'] = df_train['pred_team_goals'].fillna(1.3)
        df_train = df_train.drop(columns=['opponent_norm'], errors='ignore')

        if verbose:
            valid = df_train['pred_team_goals'].notna().sum()
            print(f"  OOF pred_team_goals: mapped to {valid:,}/{len(df_train):,} player rows")
            print(f"  Mean pred_team_goals: {df_train['pred_team_goals'].mean():.3f}")

        return df_train

    def _tune_with_subprocess(self, models: list, n_iter: int, verbose: bool, df_train: pd.DataFrame) -> Dict:
        """Run each model's tuning in a separate subprocess to save RAM.

        Dependency-ordered: minutes and clean_sheet tune first (in-process) to
        generate OOF predictions. Remaining models run in subprocesses with
        OOF pred_minutes and pred_team_goals available as features.
        """
        import subprocess
        import sys
        import json
        import tempfile
        import os

        SCORING_NAMES = {
            'goals': 'Poisson Deviance',
            'assists': 'Poisson Deviance',
            'defcon': 'Poisson Deviance',
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

        # --- Stage 1: Tune minutes (in-process) + generate OOF pred_minutes ---
        if 'minutes' in models:
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

            if verbose:
                print(f"\n  Generating OOF pred_minutes for downstream models...")
            df_train['pred_minutes'] = self._generate_oof_minutes(df_train, tuned_params['minutes'], verbose)
        else:
            df_train['pred_minutes'] = df_train['minutes']

        # --- Stage 2: Tune clean_sheet (in-process) + generate OOF pred_team_goals ---
        if 'clean_sheet' in models:
            cs_result = self._tune_clean_sheet_in_process(
                n_iter, verbose, df_train, SEARCH_SPACES_CS
            )
            tuned_params['clean_sheet'] = cs_result['params']
            cv_scores['clean_sheet'] = cs_result['cv_score']

            if verbose:
                print(f"\n  Generating OOF pred_team_goals for downstream models...")
            df_train = self._generate_oof_team_goals_for_tuning(df_train, tuned_params['clean_sheet'], verbose)

        # --- Stage 3: Remaining models in subprocess (with OOF features in CSV) ---
        remaining = [m for m in models if m not in ('minutes', 'clean_sheet')]

        # Save training data (now including OOF pred_minutes and pred_team_goals)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            temp_data_path = f.name
            df_train.to_csv(f, index=False)

        try:
            for model_name in remaining:
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
from sklearn.metrics import make_scorer, mean_poisson_deviance
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, r"{self.data_dir.parent}")
from src.models import GoalsModel, AssistsModel, MinutesModel, DefconModel, SavesModel
from src.feature_selection import compute_feature_rankings, select_features

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
    'defcon': poisson_scorer,
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

PROTECTED = {{
    'goals': ['pred_team_goals', 'pred_minutes'],
    'assists': ['pred_team_goals', 'pred_minutes'],
    'defcon': ['pred_minutes'],
    'saves': [],
}}
MIN_FEATURES = {{
    'goals': 15, 'assists': 5, 'defcon': 5, 'saves': 5,
}}
RANKING_METHODS = ['xgb_gain', 'xgb_cover', 'lgbm', 'permutation', 'mutual_info']

model_name = "{model_name}"
n_iter = {n_iter}
score_name = "{score_name}"

df_train = pd.read_csv(r"{temp_data_path}", encoding='utf-8')
# pred_minutes and pred_team_goals are already in the CSV (OOF values from upstream models)
model_class = MODEL_CLASSES[model_name]
space = SEARCH_SPACES[model_name]
scorer = SCORING[model_name]

model_instance = model_class()
all_features = [f for f in model_instance.FEATURES if f in df_train.columns]
target = model_instance.TARGET
n_total_features = len(all_features)
protected = [f for f in PROTECTED.get(model_name, []) if f in all_features]
min_feats = MIN_FEATURES.get(model_name, 5)

# Filter to GKs only for saves model
if model_name == 'saves':
    df_train = df_train[df_train['is_gk'] == 1].copy()
elif model_name == 'defcon':
    df_train = df_train[
        df_train['defcon_position'].isin(['DEF', 'MID'])
        & df_train['defcon'].notna()
    ].copy()

X_full = df_train[all_features].fillna(0).values
y = df_train[target].fillna(0).values

tscv = TimeSeriesSplit(n_splits=5)

# Pre-compute feature rankings
xgb_hint = {{'objective': 'count:poisson'}} if model_name in ('goals', 'assists', 'defcon') else {{}}
print(f"Pre-computing feature rankings ({{n_total_features}} features, {{len(RANKING_METHODS)}} methods)...")
rankings = compute_feature_rankings(X_full, y, all_features, task='regression', xgb_params=xgb_hint)
print(f"Rankings computed. Starting Optuna search...")

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

    if model_name in ('goals', 'assists', 'defcon'):
        params['objective'] = 'count:poisson'

    # Feature selection as hyperparameters
    feat_method = trial.suggest_categorical('feat_method', RANKING_METHODS)
    n_selectable = n_total_features - len(protected)
    n_features = trial.suggest_int('n_features', min_feats, n_selectable)

    selected = select_features(rankings, feat_method, n_features, protected)
    feat_idx = [all_features.index(f) for f in selected]
    X_sel = X_full[:, feat_idx]

    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X_sel, y, cv=tscv, scoring=scorer, n_jobs=1)
    return -scores.mean()

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=n_iter, show_progress_bar=False)

best_params = study.best_params.copy()
best_feat_method = best_params.pop('feat_method')
best_n_features = best_params.pop('n_features')
selected_features = select_features(rankings, best_feat_method, best_n_features, protected)

if model_name in ('goals', 'assists', 'defcon'):
    best_params['objective'] = 'count:poisson'

print(f"Best CV {{score_name}}: {{study.best_value:.4f}}")
print(f"Feature method: {{best_feat_method}}, selected {{len(selected_features)}}/{{n_total_features}} features")

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

    def _evaluate_on_test_set(self, models, df_train, df_test, verbose=True):
        """Holdout uses production wrappers and causal upstream features.

        Component/legacy reconstructed-points metrics remain conditional on an
        appearance. This is not an official-points or all-roster evaluation.
        """
        from sklearn.metrics import mean_absolute_error
        train, test = self._dependency_frames(df_train, df_test)
        fitted = {'minutes': self._fit_model('minutes', train, self.tuned_params.get('minutes', {}))}
        teams = CleanSheetModel().prepare_team_features(pd.concat([train, test], ignore_index=True))
        fitted['clean_sheet'] = self._fit_model('clean_sheet',
            teams[teams['match_id'].isin(train['match_id'])], self.tuned_params.get('clean_sheet', {}))
        against = fitted['clean_sheet'].predict_goals_against(teams)
        test['pred_goals_against'] = self._map_team_prediction(test, teams, against)
        test['pred_team_goals'] = self._map_team_prediction(test, teams, against, opponent=True)
        test['pred_cs_prob'] = np.exp(-test['pred_goals_against'])
        metrics, pred_store = {}, {}
        for name in ('minutes', 'goals', 'assists', 'defcon', 'saves'):
            if name != 'minutes':
                fitted[name] = self._fit_model(name, train, self.tuned_params.get(name, {}))
            if name == 'saves':
                target_rows = test[test['is_gk'] == 1]
            elif name == 'defcon':
                target_rows = test[
                    test['defcon_position'].isin(['DEF', 'MID']) & test['defcon'].notna()
                ]
            else:
                target_rows = test
            if target_rows.empty:
                continue
            actual = target_rows[fitted[name].TARGET].fillna(0).to_numpy()
            predicted = self._model_predict(name, fitted[name], target_rows)
            pred_store[name] = {'index': target_rows.index, 'y_pred': predicted, 'y_test': actual}
            if name == 'defcon':
                pred_store[name]['dispersion_r'] = fitted[name].dispersion_r
            if name in models:
                metrics[name] = {
                    'metric_name': 'Huber Loss' if name == 'minutes' else 'MAE' if name == 'saves' else 'Poisson Dev',
                    'primary': self._loss(name, actual, predicted),
                    'MAE': float(mean_absolute_error(actual, predicted))}
        if 'clean_sheet' in models:
            mask = teams['match_id'].isin(test['match_id'])
            actual, predicted = teams.loc[mask, 'goals_conceded'].to_numpy(), against[mask]
            metrics['clean_sheet'] = {
                'metric_name': 'Poisson Dev', 'primary': self._loss('clean_sheet', actual, predicted),
                'MAE': float(mean_absolute_error(actual, predicted))}
        fitted['cards'] = CardsModel(**self.tuned_params.get('cards', {})).fit(train, verbose=False)
        fitted['bonus'] = BonusModel(n_simulations=self.n_sims).fit(train, verbose=False)
        test = self._predict_player_components(test, fitted)
        # Bound memory by processing one deadline at a time; no files/live API calls.
        for _, batch in test.groupby('forecast_time', sort=True):
            probability, state_minutes = fitted['minutes'].predict_distribution(batch, appear_prob=np.ones(len(batch)))
            scored, _ = self._simulate_points(batch, probability, state_minutes, fitted)
            test.loc[batch.index, '_pred_points_shared'] = scored['exp_total_pts'].to_numpy()
            test.loc[batch.index, '_pred_bonus'] = scored['pred_bonus'].to_numpy()
        test['_pred_cs_prob'] = test['pred_cs_prob']
        test['_pred_goals_against'] = test['pred_goals_against']
        test['_pred_yellow'] = test['pred_yellow_prob']
        test['_pred_red'] = test['pred_red_prob']
        fpl = self._compute_fpl_points_mae(test, pred_store)
        if verbose:
            for name, result in metrics.items():
                print(f"  {name}: {result['metric_name']}={result['primary']:.4f}, MAE={result['MAE']:.4f}")
            print(f"  Reconstructed played-only FPL MAE (ex bonus): {fpl['mae_ex_bonus']:.4f}")
        metrics['_fpl_points_mae'] = fpl
        return metrics

    def _compute_fpl_points_mae(self, df_test: pd.DataFrame, pred_store: dict) -> dict:
        """Compute MAE between actual and predicted FPL points on the test set.

        Calculates FPL points from actual stats and from model predictions,
        including all components: appearance, goals, assists, clean sheet,
        goals conceded penalty, saves, defcon, and cards. Bonus is excluded
        from both sides (BonusModel isn't part of tuning).

        Expects df_test to have columns _pred_cs_prob, _pred_goals_against,
        _pred_yellow, _pred_red (added by _evaluate_on_test_set).

        Returns dict with:
            mae_ex_bonus: MAE excluding bonus from both sides (apples-to-apples)
            mae_inc_bonus: MAE with actual bonus included (shows full gap)
        """
        from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
        from scipy.stats import poisson as _poisson, nbinom as _nbinom, spearmanr

        _defcon_r = pred_store.get('defcon', {}).get('dispersion_r', None)

        df = df_test.copy()
        df['pos_label'] = df.get(
            'fpl_position', pd.Series(pd.NA, index=df.index, dtype='string')
        ).astype('string').str.upper().fillna('')
        df['defcon_pos_label'] = df.get(
            'defcon_position', pd.Series(pd.NA, index=df.index, dtype='string')
        ).astype('string').str.upper().fillna('')

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
            defcon_pos = row['defcon_pos_label']
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

            # Defcon (DEF/MID, 60+ mins) — actual threshold hit
            if defcon_pos in ('DEF', 'MID') and mins >= 60:
                pts += _safe(row, 'hit_threshold') * FPL_POINTS['defcon']

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

        # Ensure CS/cards columns exist (set by _evaluate_on_test_set)
        for col, default in [('_pred_cs_prob', 0), ('_pred_goals_against', 1.3),
                             ('_pred_yellow', 0), ('_pred_red', 0)]:
            if col not in df.columns:
                df[col] = default

        def pred_fpl_points(row):
            pos = row['pos_label']
            defcon_pos = row['defcon_pos_label']
            mins = row['_pred_minutes']

            if mins >= 60:
                pts = FPL_POINTS['appearance_60']
            elif mins >= 1:
                pts = FPL_POINTS['appearance_1']
            else:
                return 0

            # Goals: model predicts raw match counts directly
            exp_goals = row['_pred_goals']
            pts += exp_goals * FPL_POINTS['goal'].get(pos, 5)

            # Assists: model predicts raw match counts directly
            exp_assists = row['_pred_assists']
            pts += exp_assists * FPL_POINTS['assist']

            # Clean sheet (60+ mins only)
            if mins >= 60:
                pts += row['_pred_cs_prob'] * FPL_POINTS['clean_sheet'].get(pos, 0)

            # Goals conceded penalty (GK/DEF, 60+ mins)
            if pos in ('GK', 'DEF') and mins >= 60:
                lam = row['_pred_goals_against']
                expected_neg = 0
                for k in range(2, 11):
                    expected_neg += _poisson.pmf(k, lam) * (k // 2)
                tail_prob = 1 - _poisson.cdf(10, lam)
                expected_neg += tail_prob * 5
                pts -= expected_neg

            # Saves (GK)
            if pos == 'GK':
                exp_saves = row['_pred_saves'] * (mins / 90)
                pts += (exp_saves / 3) * FPL_POINTS['saves_per_3']

            # Defcon (DEF/MID, 60+ mins)
            if defcon_pos in ('DEF', 'MID') and mins >= 60:
                exp_defcon = max(row['_pred_defcon'], 0.01)
                threshold = 10 if defcon_pos == 'DEF' else 12
                if _defcon_r is not None:
                    p = _defcon_r / (_defcon_r + exp_defcon)
                    defcon_prob = 1 - _nbinom.cdf(threshold - 1, _defcon_r, p)
                else:
                    defcon_prob = 1 - _poisson.cdf(threshold - 1, exp_defcon)
                pts += defcon_prob * FPL_POINTS['defcon']

            # Cards
            pts += row['_pred_yellow'] * FPL_POINTS['yellow_card']
            pts += row['_pred_red'] * FPL_POINTS['red_card']

            return pts

        if '_pred_points_shared' in df:
            df['pred_fpl_pts'] = df['_pred_points_shared'] - df['_pred_bonus'].fillna(0)
        else:
            df['pred_fpl_pts'] = df.apply(pred_fpl_points, axis=1)

        # Add predicted bonus to prediction side for inc-bonus comparison
        if '_pred_bonus' in df.columns:
            df['pred_fpl_pts_inc_bonus'] = df['pred_fpl_pts'] + df['_pred_bonus'].fillna(0)
        else:
            df['pred_fpl_pts_inc_bonus'] = df['pred_fpl_pts']

        # Only compare rows where player actually played and values are valid
        mask = (df['minutes'] >= 1) & df['actual_pts_ex_bonus'].notna() & df['pred_fpl_pts'].notna()
        played = df.loc[mask]

        # --- Overall-points metrics on the inc-bonus side ---
        y_true = played['actual_fpl_pts'].values.astype(float)
        y_pred = played['pred_fpl_pts_inc_bonus'].values.astype(float)

        # Poisson deviance: clip true >= 0 (FPL pts can be negative due to cards/conceded)
        # and clip pred to a tiny positive (deviance is undefined at 0).
        y_true_nn = np.maximum(y_true, 0.0)
        y_pred_pos = np.maximum(y_pred, 1e-3)
        poisson_dev_inc = mean_poisson_deviance(y_true_nn, y_pred_pos) if len(y_true) else None

        # Naive baseline: predict the median of actual points for everyone.
        median_val = float(np.median(y_true_nn)) if len(y_true_nn) else 0.0
        y_pred_naive = np.full(len(y_true_nn), max(median_val, 1e-3))
        poisson_dev_naive = (
            mean_poisson_deviance(y_true_nn, y_pred_naive) if len(y_true_nn) else None
        )

        # Spearman rank correlation
        if len(y_true) > 1:
            sp_corr, _ = spearmanr(y_true, y_pred)
            spearman_inc = float(sp_corr) if not np.isnan(sp_corr) else None
        else:
            spearman_inc = None

        # Calibration buckets: group rows by predicted-points bucket and compare
        # mean predicted vs mean actual. Reveals systematic over/under-prediction.
        bucket_edges = [(1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3), (3, 3.5),
                        (3.5, 4), (4, 4.5), (4.5, 5), (5, 5.5), (5.5, 6),
                        (6, 6.5), (6.5, 50)]
        calibration = []
        for lo, hi in bucket_edges:
            mask_b = (played['pred_fpl_pts_inc_bonus'] >= lo) & (played['pred_fpl_pts_inc_bonus'] < hi)
            n = int(mask_b.sum())
            if n == 0:
                continue
            calibration.append({
                'bucket': f'{lo}-{hi}' if hi < 50 else f'{lo}+',
                'n': n,
                'pred': float(played.loc[mask_b, 'pred_fpl_pts_inc_bonus'].mean()),
                'actual': float(played.loc[mask_b, 'actual_fpl_pts'].mean()),
            })

        result = {
            'mae_ex_bonus': mean_absolute_error(played['actual_pts_ex_bonus'], played['pred_fpl_pts']),
            'mae_inc_bonus': mean_absolute_error(played['actual_fpl_pts'], played['pred_fpl_pts_inc_bonus']),
            'poisson_dev_inc_bonus': poisson_dev_inc,
            'poisson_dev_naive_median': poisson_dev_naive,
            'spearman_inc_bonus': spearman_inc,
            'naive_median_value': median_val,
            'calibration_inc_bonus': calibration,
        }

        # Standalone bonus MAE (predicted bonus vs actual bonus)
        if '_pred_bonus' in df.columns:
            bonus_mask = mask & df['_pred_bonus'].notna()
            bonus_played = df.loc[bonus_mask]
            if len(bonus_played) > 0:
                result['bonus_mae'] = mean_absolute_error(
                    bonus_played['actual_bonus'], bonus_played['_pred_bonus'])
                # Report the sparse zero baseline, but do not treat it as the
                # sole expected-value benchmark: MAE naturally favors the
                # conditional median (usually zero) for a rank-awarded target.
                result['bonus_zero_mae'] = mean_absolute_error(
                    bonus_played['actual_bonus'], np.zeros(len(bonus_played)))
                result['bonus_mae_skill_vs_zero'] = (
                    result['bonus_zero_mae'] - result['bonus_mae'])
                if 'match_id' in bonus_played:
                    group = bonus_played.groupby('match_id')
                    uniform_bonus = (
                        group['actual_bonus'].transform('sum') /
                        group['actual_bonus'].transform('size'))
                    actual_fixture = group['actual_bonus'].sum()
                    predicted_fixture = group['_pred_bonus'].sum()
                    result['bonus_fixture_total_mae'] = mean_absolute_error(
                        actual_fixture, predicted_fixture)
                else:
                    uniform_bonus = np.full(
                        len(bonus_played), bonus_played['actual_bonus'].mean())
                result['bonus_uniform_mae'] = mean_absolute_error(
                    bonus_played['actual_bonus'], uniform_bonus)
                result['bonus_mae_skill_vs_uniform'] = (
                    result['bonus_uniform_mae'] - result['bonus_mae'])
                actual_bonus = bonus_played['actual_bonus'].to_numpy(dtype=float)
                predicted_bonus = bonus_played['_pred_bonus'].to_numpy(dtype=float)
                result['bonus_rmse'] = float(np.sqrt(np.mean(
                    (actual_bonus - predicted_bonus) ** 2)))
                result['bonus_zero_rmse'] = float(np.sqrt(np.mean(actual_bonus ** 2)))
                if (len(bonus_played) > 1 and np.ptp(actual_bonus) > 0 and
                        np.ptp(predicted_bonus) > 0):
                    bonus_corr, _ = spearmanr(actual_bonus, predicted_bonus)
                    result['bonus_spearman'] = (
                        float(bonus_corr) if not np.isnan(bonus_corr) else None)
                else:
                    result['bonus_spearman'] = None
                result['actual_bonus_mean'] = float(actual_bonus.mean())
                result['predicted_bonus_mean'] = float(predicted_bonus.mean())

        # Store detailed DataFrame for breakdown analysis
        self._last_fpl_detail = played.copy()
        self._last_defcon_dispersion_r = _defcon_r

        return result

    def points_breakdown(self) -> pd.DataFrame:
        """Return per-category breakdown of predicted vs actual FPL points.

        Must be called after tune() which populates self._last_fpl_detail.
        Returns a DataFrame with columns: category, pred_value_avg, pred_pts_avg,
        actual_value_avg, actual_pts_avg, pts_diff, abs_pts_diff.
        """
        import pandas as pd
        from scipy.stats import poisson as _poisson, nbinom as _nbinom

        if not hasattr(self, '_last_fpl_detail') or self._last_fpl_detail is None:
            raise ValueError("No test evaluation data. Run tune() first.")

        _defcon_r = getattr(self, '_last_defcon_dispersion_r', None)

        df = self._last_fpl_detail.copy()
        df['pos_label'] = df.get(
            'fpl_position', pd.Series(pd.NA, index=df.index, dtype='string')
        ).astype('string').str.upper().fillna('')
        df['defcon_pos_label'] = df.get(
            'defcon_position', pd.Series(pd.NA, index=df.index, dtype='string')
        ).astype('string').str.upper().fillna('')

        def _safe(col, default=0):
            if col in df.columns:
                return df[col].fillna(default)
            return pd.Series(default, index=df.index)

        rows = []

        # --- Appearance ---
        actual_app = df['minutes'].apply(lambda m: 2 if m >= 60 else (1 if m >= 1 else 0))
        pred_app = df['_pred_minutes'].apply(lambda m: 2 if m >= 60 else (1 if m >= 1 else 0))
        rows.append({
            'category': 'Appearance',
            'pred_value_avg': df['_pred_minutes'].mean(),
            'pred_pts_avg': pred_app.mean(),
            'actual_value_avg': df['minutes'].mean(),
            'actual_pts_avg': actual_app.mean(),
        })

        # --- Goals ---
        goal_mult = df['pos_label'].map(FPL_POINTS['goal']).fillna(5)
        pred_exp_goals = df['_pred_goals']
        pred_goal_pts = pred_exp_goals * goal_mult
        actual_goals = _safe('goals')
        actual_goal_pts = actual_goals * goal_mult
        rows.append({
            'category': 'Goals',
            'pred_value_avg': pred_exp_goals.mean(),
            'pred_pts_avg': pred_goal_pts.mean(),
            'actual_value_avg': actual_goals.mean(),
            'actual_pts_avg': actual_goal_pts.mean(),
        })

        # --- Assists ---
        pred_exp_assists = df['_pred_assists']
        pred_assist_pts = pred_exp_assists * FPL_POINTS['assist']
        actual_assists = _safe('assists')
        actual_assist_pts = actual_assists * FPL_POINTS['assist']
        rows.append({
            'category': 'Assists',
            'pred_value_avg': pred_exp_assists.mean(),
            'pred_pts_avg': pred_assist_pts.mean(),
            'actual_value_avg': actual_assists.mean(),
            'actual_pts_avg': actual_assist_pts.mean(),
        })

        # --- Clean Sheet ---
        pred_cs_pts = pd.Series(0.0, index=df.index)
        actual_cs = pd.Series(0.0, index=df.index)
        actual_cs_pts = pd.Series(0.0, index=df.index)
        cs_mult = df['pos_label'].map(FPL_POINTS['clean_sheet']).fillna(0)

        # Predicted: prob * multiplier (60+ mins only)
        mask60 = df['_pred_minutes'] >= 60
        pred_cs_pts[mask60] = df.loc[mask60, '_pred_cs_prob'] * cs_mult[mask60]

        # Actual: binary CS (goals_conceded == 0 and mins >= 60)
        gc = _safe('_gc', 1)
        actual_mask = (df['minutes'] >= 60) & (gc == 0)
        actual_cs[actual_mask] = 1
        actual_cs_pts[actual_mask] = cs_mult[actual_mask]

        rows.append({
            'category': 'Clean Sheet',
            'pred_value_avg': df['_pred_cs_prob'].mean(),
            'pred_pts_avg': pred_cs_pts.mean(),
            'actual_value_avg': actual_cs.mean(),
            'actual_pts_avg': actual_cs_pts.mean(),
        })

        # --- Goals Conceded Penalty (GK/DEF only) ---
        gk_def = df['pos_label'].isin(['GK', 'DEF']) & (df['minutes'] >= 60)
        pred_gc_pen = pd.Series(0.0, index=df.index)
        actual_gc_pen = pd.Series(0.0, index=df.index)
        if gk_def.any():
            for idx in df.index[gk_def]:
                lam = df.loc[idx, '_pred_goals_against']
                expected_neg = 0
                for k in range(2, 11):
                    expected_neg += _poisson.pmf(k, lam) * (k // 2)
                tail_prob = 1 - _poisson.cdf(10, lam)
                expected_neg += tail_prob * 5
                pred_gc_pen[idx] = -expected_neg
            actual_gc_pen[gk_def] = -(gc[gk_def].astype(int) // 2)

        rows.append({
            'category': 'Goals Conceded',
            'pred_value_avg': df.loc[gk_def, '_pred_goals_against'].mean() if gk_def.any() else 0,
            'pred_pts_avg': pred_gc_pen.mean(),
            'actual_value_avg': gc[gk_def].mean() if gk_def.any() else 0,
            'actual_pts_avg': actual_gc_pen.mean(),
        })

        # --- Saves (GK only) ---
        is_gk = df['pos_label'] == 'GK'
        pred_saves = df['_pred_saves'] * (df['_pred_minutes'] / 90)
        pred_saves_pts = pd.Series(0.0, index=df.index)
        actual_saves_pts = pd.Series(0.0, index=df.index)
        if is_gk.any():
            pred_saves_pts[is_gk] = (pred_saves[is_gk] / 3) * FPL_POINTS['saves_per_3']
            actual_saves_pts[is_gk] = (_safe('saves')[is_gk].astype(int) // 3) * FPL_POINTS['saves_per_3']
        rows.append({
            'category': 'Saves',
            'pred_value_avg': pred_saves[is_gk].mean() if is_gk.any() else 0,
            'pred_pts_avg': pred_saves_pts.mean(),
            'actual_value_avg': _safe('saves')[is_gk].mean() if is_gk.any() else 0,
            'actual_pts_avg': actual_saves_pts.mean(),
        })

        # --- Defcon (DEF/MID, 60+ mins) ---
        def_mid_60 = df['defcon_pos_label'].isin(['DEF', 'MID']) & (df['_pred_minutes'] >= 60)
        pred_defcon_pts = pd.Series(0.0, index=df.index)
        actual_defcon_pts = pd.Series(0.0, index=df.index)
        if def_mid_60.any():
            exp_defcon = df['_pred_defcon']
            threshold = df['defcon_pos_label'].map({'DEF': 10, 'MID': 12}).fillna(12)
            for idx in df.index[def_mid_60]:
                mu = max(exp_defcon[idx], 0.01)
                thr = threshold[idx]
                if _defcon_r is not None:
                    p = _defcon_r / (_defcon_r + mu)
                    defcon_prob = 1 - _nbinom.cdf(thr - 1, _defcon_r, p)
                else:
                    defcon_prob = 1 - _poisson.cdf(thr - 1, mu)
                pred_defcon_pts[idx] = defcon_prob * FPL_POINTS['defcon']
            actual_defcon_pts[def_mid_60] = _safe('hit_threshold')[def_mid_60] * FPL_POINTS['defcon']
        rows.append({
            'category': 'Defcon',
            'pred_value_avg': pred_defcon_pts[def_mid_60].mean() / FPL_POINTS['defcon'] if def_mid_60.any() else 0,
            'pred_pts_avg': pred_defcon_pts.mean(),
            'actual_value_avg': _safe('hit_threshold')[def_mid_60].mean() if def_mid_60.any() else 0,
            'actual_pts_avg': actual_defcon_pts.mean(),
        })

        # --- Yellow Cards ---
        pred_yellow_pts = df['_pred_yellow'] * FPL_POINTS['yellow_card']
        actual_yellows = _safe('yellow_cards')
        actual_yellow_pts = actual_yellows * FPL_POINTS['yellow_card']
        rows.append({
            'category': 'Yellow Cards',
            'pred_value_avg': df['_pred_yellow'].mean(),
            'pred_pts_avg': pred_yellow_pts.mean(),
            'actual_value_avg': actual_yellows.mean(),
            'actual_pts_avg': actual_yellow_pts.mean(),
        })

        # --- Red Cards ---
        pred_red_pts = df['_pred_red'] * FPL_POINTS['red_card']
        actual_reds = _safe('red_card') if 'red_card' in df.columns else _safe('red_cards')
        actual_red_pts = actual_reds * FPL_POINTS['red_card']
        rows.append({
            'category': 'Red Cards',
            'pred_value_avg': df['_pred_red'].mean(),
            'pred_pts_avg': pred_red_pts.mean(),
            'actual_value_avg': actual_reds.mean(),
            'actual_pts_avg': actual_red_pts.mean(),
        })

        # --- Bonus ---
        pred_bonus = _safe('_pred_bonus')
        actual_bonus = _safe('actual_bonus')
        rows.append({
            'category': 'Bonus',
            'pred_value_avg': pred_bonus.mean(),
            'pred_pts_avg': pred_bonus.mean(),  # bonus value = bonus pts (1:1)
            'actual_value_avg': actual_bonus.mean(),
            'actual_pts_avg': actual_bonus.mean(),
        })

        result = pd.DataFrame(rows)
        result['pts_diff'] = result['pred_pts_avg'] - result['actual_pts_avg']
        result['abs_pts_diff'] = result['pts_diff'].abs()

        # Add totals row
        totals = pd.DataFrame([{
            'category': 'TOTAL',
            'pred_value_avg': None,
            'pred_pts_avg': result['pred_pts_avg'].sum(),
            'actual_value_avg': None,
            'actual_pts_avg': result['actual_pts_avg'].sum(),
            'pts_diff': result['pts_diff'].sum(),
            'abs_pts_diff': result['abs_pts_diff'].sum(),
        }])
        result = pd.concat([result, totals], ignore_index=True)

        return result

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
        
        # Live squad/availability APIs are not historical snapshots.
        known = chronological_frame(self.df)
        if season < known['season'].max():
            raise ValueError("Historical seasons require chronological holdout evaluation, not live squad/availability data.")
        target_known = known[known['season'].eq(season) & known['gameweek'].eq(gameweek)]
        if not target_known.empty and known['result_time'].max() >= target_known['forecast_time'].min():
            raise ValueError("predict() is for unplayed gameweeks. Use chronological holdout evaluation for historical forecasts.")
        # Get test data - latest features for players
        test_df = self._build_test_set(gameweek, season, verbose)

        # Recompute interaction features (opponent stats were updated by _build_test_set)
        test_df = self._recompute_interaction_features(test_df)

        if len(test_df) == 0:
            print("WARNING: No players found for prediction")
            return pd.DataFrame()
        
        # Predicted lineups, when supplied, answer the one question the model is
        # worst at: who actually starts. Everything downstream consumes pred_minutes
        # as a feature, so correcting it here propagates through goals, assists,
        # defcon, saves and bonus without any of them changing.
        lineup_p_start, lineup_available = None, None
        if getattr(self, 'last_lineups', None) is not None:
            from .lineups import match_lineups, build_overrides
            matched = match_lineups(self.last_lineups, test_df, verbose=verbose)
            ov = build_overrides(matched, test_df, verbose=verbose)
            lineup_p_start = ov['lineup_p_start'].values
            lineup_available = ov['lineup_available'].values

        # Combine history and lineup priors as joint role probabilities.
        # Live availability is applied once, after lineup evidence.
        appear = (self.models['minutes'].predict_appear_proba(test_df)
                  if self.models['minutes'].appear_is_fitted else np.ones(len(test_df)))
        available = pd.to_numeric(test_df.get('fpl_chance_of_playing',
            pd.Series(100, index=test_df.index)), errors='coerce').fillna(100).to_numpy() / 100.
        if lineup_available is not None:
            available *= np.where(np.isfinite(lineup_available), lineup_available, 1.)
        probability, state_minutes = self.models['minutes'].predict_distribution(
            test_df, lineup_p_start=lineup_p_start, appear_prob=appear, availability=available)
        test_df['pred_appear_prob'] = 1 - probability[:, 0]
        test_df['pred_minutes'] = np.divide(
            (probability * state_minutes).sum(axis=1), test_df['pred_appear_prob'],
            out=np.zeros(len(test_df)), where=test_df['pred_appear_prob'].to_numpy() > 0)

        # Clean sheet / goals against FIRST (needed for pred_team_goals feature)
        cs_probs, two_plus_probs, pred_goals_against = self._predict_clean_sheet(test_df, gameweek, season, verbose)
        test_df['pred_cs_prob'] = cs_probs
        test_df['pred_2plus_conceded'] = two_plus_probs
        test_df['pred_goals_against'] = pred_goals_against
        
        # Inject pred_team_goals: for each player, predicted goals their team will score
        # = predicted goals conceded by the OPPONENT (from CleanSheetModel)
        test_df['pred_team_goals'] = self._get_pred_team_goals(test_df, gameweek, season)

        # Refresh the target-week classification before DefCon prediction. A
        # failed API lookup may retain an already cached FPL position, but it may
        # never infer one from FotMob's on-pitch role.
        self.fpl_positions = get_fpl_positions()
        live_position = test_df.apply(
            lambda row: map_fpl_position(
                row.get('position'), row.get('player_name'), self.fpl_positions,
                fallback_to_fotmob=False),
            axis=1,
        ).astype('string').str.upper()
        cached_position = test_df.get(
            'fpl_position', pd.Series(pd.NA, index=test_df.index, dtype='string')
        ).astype('string').str.upper()
        test_df['fpl_position'] = live_position.where(
            live_position.isin(['GK', 'DEF', 'MID', 'FWD']), cached_position)
        test_df = resolve_defcon_positions(test_df)
        if verbose:
            unresolved = ~test_df['fpl_position'].isin(['GK', 'DEF', 'MID', 'FWD'])
            if unresolved.any():
                print(f"  WARNING: {int(unresolved.sum())} players have no FPL API position; "
                      "DefCon uses their FotMob position fallback")

        test_df = self._predict_player_components(test_df, self.models)

        test_df, bonus_sims = self._simulate_points(test_df, probability, state_minutes)
        self.last_simulations = {
            'player_names': test_df['player_name'].tolist(),
            'player_ids': test_df['player_id'].tolist(),
            **bonus_sims,
        }

        # Save per-fixture predictions
        output_path = self.data_dir / 'predictions' / f'gw{gameweek}_{season.replace("/", "-")}.csv'
        output_path.parent.mkdir(exist_ok=True)
        test_df.to_csv(output_path, index=False)

        # Snapshot per-fixture predictions for viz (sim arrays are indexed by row position here)
        _pf = test_df.reset_index(drop=True).copy()
        _pf['_sim_idx'] = range(len(_pf))
        self.last_predictions_per_fixture = _pf

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
                                      'pred_minutes',
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
                        'pred_goals_against', 'pred_team_goals']:
                if col in group.columns:
                    base[col] = group[col].mean()

            if 'exp_total_pts_uncond' in group:
                appear = 1 - np.prod(1 - group['pred_appear_prob'].fillna(1).to_numpy())
                base['pred_appear_prob'] = appear
                base['exp_total_pts_cond'] = base['exp_total_pts_uncond'] / appear if appear > 0 else 0
                base['pred_minutes_uncond'] = group['pred_minutes_uncond'].sum()
                base['pred_60_prob'] = 1 - np.prod(1 - group['pred_60_prob'].to_numpy())
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

    def get_last_gw_review(self, gameweek: int, season: str, top_n: int = 10):
        """Top-N predicted players from the previous gameweek, with what they actually scored.

        Reads the most recent saved run for ``gameweek - 1`` and joins the FPL
        actuals cache. Returns None when there is no prior run or no actuals yet
        (e.g. a season opener, or before the gameweek has been scored).
        """
        import unicodedata

        prev_gw = int(gameweek) - 1
        if prev_gw < 1:
            return None

        runs = sorted((self.data_dir / 'runs').glob(f'gw{prev_gw}_*')) if (self.data_dir / 'runs').exists() else []
        if not runs:
            return None
        preds_file = runs[-1] / 'predictions.csv'
        if not preds_file.exists():
            return None

        preds = pd.read_csv(preds_file)
        if 'exp_total_pts' not in preds.columns or not len(preds):
            return None
        top = preds.nlargest(top_n, 'exp_total_pts')

        actuals_file = self.data_dir / 'fpl_actual_points.csv'
        if not actuals_file.exists():
            return None
        act = pd.read_csv(actuals_file)
        act = act[(act['season'] == season) & (pd.to_numeric(act['gameweek'], errors='coerce') == prev_gw)]
        if not len(act):
            return None

        def _norm(name):
            if pd.isna(name):
                return ''
            return ''.join(
                c for c in unicodedata.normalize('NFD', str(name))
                if unicodedata.category(c) != 'Mn'
            ).lower().strip()

        # name -> row, with web_name and surname as progressively looser keys
        lookup = {}
        for _, r in act.iterrows():
            for key in (_norm(r['player_name']), _norm(r.get('web_name'))):
                if key:
                    lookup.setdefault(key, r)
            parts = _norm(r['player_name']).split()
            if len(parts) > 1:
                lookup.setdefault(parts[-1], r)

        pos_names = {0: 'GK', 1: 'DEF', 2: 'MID', 3: 'FWD'}
        rows = []
        for _, p in top.iterrows():
            n = _norm(p['player_name'])
            hit = lookup.get(n)
            if hit is None:
                parts = n.split()
                if len(parts) > 1:
                    hit = lookup.get(parts[-1])
            try:
                pos = pos_names.get(int(float(p.get('position'))), '')
            except (TypeError, ValueError):
                pos = ''
            rows.append({
                'player_name': p['player_name'],
                'team': p.get('team', ''),
                'position': pos,
                'predicted': float(p['exp_total_pts']),
                'actual': None if hit is None else float(hit['actual_total_points']),
                'minutes': None if hit is None else float(hit['minutes']),
            })

        matched = [r for r in rows if r['actual'] is not None]
        if not matched:
            return None
        return {
            'gameweek': prev_gw,
            'rows': rows,
            'mean_predicted': sum(r['predicted'] for r in matched) / len(matched),
            'mean_actual': sum(r['actual'] for r in matched) / len(matched),
        }

    def get_viz_metrics(self):
        """Format last_test_metrics for generate_distribution_html(metrics=...).

        Returns a dict ``{'sections': [...], 'calibration': [...]}`` with two
        sections — sub-models and overall FPL points — plus calibration buckets.
        """
        if not hasattr(self, 'last_test_metrics') or not self.last_test_metrics:
            return None

        sub_rows = []
        for model_name, m in self.last_test_metrics.items():
            if model_name.startswith('_'):
                continue
            if isinstance(m, dict) and 'metric_name' in m and 'primary' in m:
                sub_rows.append({
                    'model': model_name.capitalize(),
                    'metric': m['metric_name'],
                    'score': f"{m['primary']:.4f}",
                })
        fpl_bonus = self.last_test_metrics.get('_fpl_points_mae')
        if isinstance(fpl_bonus, dict) and fpl_bonus.get('bonus_mae') is not None:
            sub_rows.append({
                'model': 'Bonus',
                'metric': 'MAE',
                'score': f"{fpl_bonus['bonus_mae']:.4f}",
            })

        overall_rows = []
        calibration = None
        fpl = self.last_test_metrics.get('_fpl_points_mae')
        if isinstance(fpl, dict):
            if fpl.get('mae_inc_bonus') is not None:
                overall_rows.append({'metric': 'MAE (inc-bonus)', 'score': f"{fpl['mae_inc_bonus']:.4f}"})
            if fpl.get('poisson_dev_inc_bonus') is not None:
                overall_rows.append({'metric': 'Poisson Deviance', 'score': f"{fpl['poisson_dev_inc_bonus']:.4f}"})
            if fpl.get('spearman_inc_bonus') is not None:
                overall_rows.append({'metric': 'Spearman ρ (rank corr)', 'score': f"{fpl['spearman_inc_bonus']:.4f}"})
            calibration = fpl.get('calibration_inc_bonus') or None

        sections = []
        if sub_rows:
            sections.append({'title': 'Sub-Models (Holdout Test Set)', 'rows': sub_rows})
        if overall_rows:
            sections.append({'title': 'Overall FPL Points (Holdout Test Set)', 'rows': overall_rows})

        if not sections and not calibration:
            return None
        return {'sections': sections, 'calibration': calibration}

    def load_lineups(self, gameweek: int = None, season: str = '2026/2027',
                     snapshot: bool = True, html: str = None,
                     verbose: bool = True) -> pd.DataFrame:
        """Fetch predicted lineups and arm them for the next predict() call.

        Must be called before predict(). Pass ``html`` to parse a cached page instead
        of hitting the network. Snapshots to data/lineups/ because RotoWire serves
        only the current slate and the history cannot be recovered afterwards.
        """
        from .lineups import (fetch_lineups, parse_lineups, save_lineups,
                              evaluate_snapshots)

        if verbose:
            print("\n" + "=" * 60)
            print("PREDICTED LINEUPS")
            print("=" * 60)

        # Score every past snapshot against real minutes before adding a new one, so
        # the feed's accuracy is reported on every deploy rather than taken on trust.
        if self.df is not None:
            try:
                self.last_lineup_accuracy = evaluate_snapshots(
                    self.df, str(self.data_dir), verbose=verbose)
            except Exception as exc:
                print(f"  (lineup accuracy check skipped: {type(exc).__name__}: {exc})")

        raw = parse_lineups(html if html is not None else fetch_lineups())
        if raw.empty:
            print("  WARNING: no lineups parsed — page layout may have changed. "
                  "Predictions will fall back to the model.")
            self.last_lineups = None
            return raw

        xi = raw[raw['is_predicted_starter']]
        if verbose:
            print(f"  {raw['fixture'].nunique()} fixtures, {xi['team'].nunique()} teams, "
                  f"{len(xi)} predicted starters"
                  f"{' (CONFIRMED)' if raw['confirmed'].any() else ' (predicted, not confirmed)'}")
        if snapshot and gameweek is not None:
            save_lineups(raw, gameweek, season, str(self.data_dir), verbose)

        self.last_lineups = raw
        return raw

    def optimize_squad(self, predictions: pd.DataFrame, gameweek: int = None,
                       season: str = '2026/2027', budget: float = 100.0,
                       snapshot: bool = True, verbose: bool = True) -> dict:
        """Pick the best 15 under a budget, weighting bench spend by P(appears).

        Prices are fetched live — nothing in the repo stores them — and snapshotted to
        data/fpl_prices.csv, since the FPL API drops per-gameweek price history at
        season rollover and it cannot be reconstructed afterwards.

        Requires a pred_appear_prob column, which predict() produces once the
        MinutesModel has fitted its AppearClassifier.
        """
        from .optimizer import fetch_fpl_prices, snapshot_prices, attach_prices, optimize_squad

        if verbose:
            print("\n" + "=" * 60)
            print(f"OPTIMIZING SQUAD (budget £{budget:.1f}m)")
            print("=" * 60)

        prices = fetch_fpl_prices(verbose=verbose)
        if snapshot and gameweek is not None:
            snapshot_prices(prices, gameweek, season, str(self.data_dir), verbose)
        priced = attach_prices(predictions, prices, verbose=verbose)
        result = optimize_squad(priced, budget=budget, verbose=verbose)
        self.last_squad = result
        return result

    def save_run(self, predictions: pd.DataFrame, gameweek: int, season: str = '2025/2026',
                 description: str = '', verbose: bool = True) -> Path:
        """Save a complete run: predictions, simulations, tuned params, metrics.

        Creates a timestamped directory under data/runs/ with everything needed
        to regenerate distributions without retraining.

        Returns:
            Path to the saved run directory.
        """
        import json, pickle
        from datetime import datetime

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.data_dir / 'runs' / f'gw{gameweek}_{ts}'
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Predictions CSV
        predictions.to_csv(run_dir / 'predictions.csv', index=False)

        # 2. Simulations (numpy arrays)
        if hasattr(self, 'last_simulations') and self.last_simulations:
            sim_dir = run_dir / 'simulations'
            sim_dir.mkdir(exist_ok=True)
            for key, val in self.last_simulations.items():
                if isinstance(val, np.ndarray):
                    np.save(sim_dir / f'{key}.npy', val)
                elif isinstance(val, list):
                    with open(sim_dir / f'{key}.json', 'w') as f:
                        json.dump(val, f)

        if getattr(self, 'last_predictions_per_fixture', None) is not None:
            self.last_predictions_per_fixture.to_csv(run_dir / 'predictions_per_fixture.csv', index=False)

        # 3. Tuned params
        if self.tuned_params:
            # Convert numpy types for JSON serialization
            def _convert(obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            serializable = {}
            for model_name, params in self.tuned_params.items():
                serializable[model_name] = {k: _convert(v) for k, v in params.items()}
            with open(run_dir / 'tuned_params.json', 'w') as f:
                json.dump(serializable, f, indent=2, default=_convert)

        # 3b. Optimized squad
        if getattr(self, 'last_squad', None):
            sq = self.last_squad
            sq['squad'].to_csv(run_dir / 'squad.csv', index=False)
            with open(run_dir / 'squad_meta.json', 'w') as f:
                json.dump({k: v for k, v in sq.items()
                           if k not in ('squad', 'xi', 'bench')}, f, indent=2)

        # 4. Test metrics
        if hasattr(self, 'last_test_metrics') and self.last_test_metrics:
            with open(run_dir / 'test_metrics.json', 'w') as f:
                json.dump(self.last_test_metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

        # 5. Viz metrics (formatted for HTML)
        viz_metrics = self.get_viz_metrics()
        if viz_metrics:
            with open(run_dir / 'viz_metrics.json', 'w') as f:
                json.dump(viz_metrics, f, indent=2)

        # 6. Run metadata
        meta = {
            'gameweek': gameweek,
            'season': season,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'n_sims': self.n_sims,
            'n_players': len(predictions),
            'points_semantics': 'unconditional' if 'exp_total_pts_uncond' in predictions else 'legacy_conditional',
            'simulation_schema': 2 if 'exp_total_pts_uncond' in predictions else 1,
        }
        with open(run_dir / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        # 7. Resolved FPL name-match table (audit trail for the squad override)
        if getattr(self, 'last_fpl_name_matches', None) is not None:
            self.last_fpl_name_matches.to_csv(run_dir / 'fpl_name_matches.csv', index=False)

        if verbose:
            print(f"\nRun saved to: {run_dir}")
            print(f"  predictions.csv: {len(predictions)} players")
            if hasattr(self, 'last_simulations') and self.last_simulations:
                n_sims = next((v.shape[0] for v in self.last_simulations.values() if isinstance(v, np.ndarray)), 0)
                print(f"  simulations: {n_sims} sims")
            print(f"  tuned_params.json: {len(self.tuned_params)} models")
            if hasattr(self, 'last_test_metrics') and self.last_test_metrics:
                fpl = self.last_test_metrics.get('_fpl_points_mae', {})
                if isinstance(fpl, dict) and fpl.get('mae_inc_bonus') is not None:
                    parts = [f"inc-bonus MAE={fpl['mae_inc_bonus']:.4f}"]
                    if fpl.get('poisson_dev_inc_bonus') is not None:
                        parts.append(f"PoissonDev={fpl['poisson_dev_inc_bonus']:.4f}")
                    if fpl.get('spearman_inc_bonus') is not None:
                        parts.append(f"Spearman={fpl['spearman_inc_bonus']:.4f}")
                    print(f"  FPL points: {', '.join(parts)}")

        return run_dir

    @classmethod
    def load_run(cls, run_dir: str, verbose: bool = True):
        """Load a saved run for viz generation without retraining.

        Returns:
            (predictions_df, simulations_dict, viz_metrics, meta)
        """
        import json
        run_dir = Path(run_dir)

        # Predictions
        predictions = pd.read_csv(run_dir / 'predictions.csv')

        # Simulations
        simulations = {}
        sim_dir = run_dir / 'simulations'
        if sim_dir.exists():
            for f in sim_dir.iterdir():
                if f.suffix == '.npy':
                    simulations[f.stem] = np.load(f)
                elif f.suffix == '.json':
                    with open(f) as fh:
                        simulations[f.stem] = json.load(fh)

        # Viz metrics
        viz_metrics = None
        vm_path = run_dir / 'viz_metrics.json'
        if vm_path.exists():
            with open(vm_path) as f:
                viz_metrics = json.load(f)

        # Metadata
        meta = {}
        meta_path = run_dir / 'meta.json'
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        if verbose:
            print(f"Loaded run from: {run_dir}")
            print(f"  {len(predictions)} players, GW{meta.get('gameweek', '?')}")
            if viz_metrics:
                for m in viz_metrics:
                    print(f"  {m['model']}: {m['metric']} = {m['score']}")

        return predictions, simulations, viz_metrics, meta

    def _build_test_set(self, gameweek: int, season: str, verbose: bool) -> pd.DataFrame:
        """Build test set for target gameweek by appending synthetic prediction rows
        to the raw historical data and recomputing rolling features.

        Rolling features are defined as shift(1).rolling(N), which at a synthetic
        GW-N prediction row yields the mean of the player's N most recent actual
        matches. This fixes the stale-feature bug where predictions were effectively
        based on the player's state as of the match BEFORE their most recent one.
        """
        if not hasattr(self, 'raw_df') or self.raw_df is None:
            # Fallback: reconstruct a pseudo raw frame from self.df by keeping only
            # the columns that existed before feature engineering.
            raise ValueError("raw_df not available. Call load_data() before predict().")

        fixtures = self._get_gw_fixtures(gameweek, season, verbose)
        if len(fixtures) == 0:
            if verbose:
                print(f"WARNING: No fixtures found for GW{gameweek}")
            return pd.DataFrame()

        fixture_dates = pd.to_datetime(fixtures.get('match_date', pd.Series(dtype=object)),
                                       errors='coerce', utc=True)
        if fixture_dates.notna().any():
            cutoff = fixture_dates.min() - pd.Timedelta(minutes=90)
        else:
            known_target = chronological_frame(self.raw_df)
            known_target = known_target[known_target['season'].eq(season) & known_target['gameweek'].eq(gameweek)]
            cutoff = (known_target['forecast_time'].min() if not known_target.empty else
                      pd.Timestamp.now(tz='UTC'))
        # Current live forecasts cannot include results obtained after now, even
        # when the target fixture is several weeks away.
        cutoff = min(cutoff, pd.Timestamp.now(tz='UTC'))
        history = chronological_frame(self.raw_df)
        prior_raw = history[history['result_time'] < cutoff].copy()
        if prior_raw.empty:
            raise ValueError("No historical results are available before this forecast origin")

        latest_identity = (
            prior_raw.sort_values(['player_id', 'match_date'])
            .groupby('player_id').last().reset_index()
        )

        # Map from normalized team name -> FotMob team name (latest naming wins).
        # FPL API names ('Spurs', 'Man City') must be translated to FotMob names
        # so synthetic rows join team/opponent rolling-stat history.
        fotmob_by_norm = {}
        for _, r in prior_raw[['team', 'season']].dropna().drop_duplicates().sort_values('season').iterrows():
            fotmob_by_norm[normalize_team_name(r['team'])] = r['team']

        def _to_fotmob(name):
            return fotmob_by_norm.get(normalize_team_name(name), name)

        # FPL-squad team override: place each player in their CURRENT club's
        # fixture (summer transfers), and reinstate players on a current FPL
        # squad whose last FotMob appearance was before this season (e.g.
        # promoted-team players with prior PL history). Form features are
        # unaffected — they follow the player's own match history by player_id.
        in_current = latest_identity['player_id'].isin(self.current_season_players)
        try:
            fpl_full_names, fpl_variants = get_fpl_current_squads()
        except Exception as e:
            if verbose:
                print(f"  WARNING: could not fetch FPL squads ({e}); using last observed teams")
            fpl_full_names, fpl_variants = {}, {}

        # Manual corrections, checked before any automatic matching.
        # data/fpl_name_overrides.json: {FotMob name: FPL team name | null}
        # null = force-exclude (player is not in FPL despite what matching says)
        import json as _json
        name_overrides = {}
        overrides_path = self.data_dir / 'fpl_name_overrides.json'
        if overrides_path.exists():
            with open(overrides_path) as f:
                name_overrides = {normalize_player_name(k): v
                                  for k, v in _json.load(f).items()}

        if fpl_variants:
            def _token_subset_team(key):
                # FotMob 'David Raya' vs FPL 'David Raya Martin': match when the
                # FotMob tokens are a subset of exactly one FPL full name's tokens
                toks = set(key.split())
                if len(toks) < 2:
                    return None
                hits = {t for full, t in fpl_full_names.items() if toks <= set(full.split())}
                return next(iter(hits)) if len(hits) == 1 else None

            def _fpl_team(name, strict=False):
                key = normalize_player_name(name)
                if not key:
                    return None
                if key in name_overrides:
                    return name_overrides[key]
                # exact full name, then token subset (handles FPL middle names:
                # 'Bruno Fernandes' -> 'Bruno Borges Fernandes')
                team = fpl_full_names.get(key) or _token_subset_team(key)
                if team or strict:
                    return team
                # exact web_name / second_name variant
                team = fpl_variants.get(key)
                if team:
                    return team
                # last resort: surname — only if it appears in exactly ONE FPL
                # full name overall (variant-level uniqueness is not enough:
                # 'Fernandes' was unique as a web_name yet belongs to two
                # players) AND the first names are compatible (Yerson Mosquera
                # is not FPL's Cristhian Mosquera; Danny/Daniel Ings is fine)
                if ' ' in key:
                    last = key.split()[-1]
                    hits = [(full, t) for full, t in fpl_full_names.items() if last in full.split()]
                    if len(hits) == 1:
                        fpl_first, first = hits[0][0].split()[0], key.split()[0]
                        if fpl_first.startswith(first[:3]) or first.startswith(fpl_first[:3]):
                            return hits[0][1]
                return None

            mapped = latest_identity['player_name'].apply(_fpl_team)
            fotmob_team_orig = latest_identity['team'].copy()
            mapped_fotmob = mapped.dropna().apply(_to_fotmob)
            changed = mapped_fotmob.index[
                mapped_fotmob.apply(normalize_team_name)
                != latest_identity.loc[mapped_fotmob.index, 'team'].apply(normalize_team_name)
            ]
            latest_identity.loc[changed, 'team'] = mapped_fotmob[changed]

            # Reinstatement requires an exact full-name match (conservative)
            strict_match = latest_identity['player_name'].apply(
                lambda n: _fpl_team(n, strict=True) is not None
            )
            # Current players must appear in the FPL bootstrap — a player with no
            # FPL element (departed/released) can't be picked in FPL at all
            keep = (in_current & mapped.notna()) | strict_match
            if verbose:
                n_transfers = int(in_current.loc[changed].sum())
                n_reinstated = int((~in_current & strict_match).sum())
                n_departed = int((in_current & mapped.isna()).sum())
                print(f"  FPL squad override: {n_transfers} transfers re-teamed, "
                      f"{n_reinstated} prior-season players reinstated, "
                      f"{n_departed} not in FPL (departed) dropped")

            # Persist the resolved name-match table for audit (saved by save_run)
            self.last_fpl_name_matches = pd.DataFrame({
                'player_name': latest_identity['player_name'],
                'fotmob_team': fotmob_team_orig,
                'fpl_team': mapped,
                'overridden': latest_identity['player_name'].apply(
                    lambda n: normalize_player_name(n) in name_overrides),
                'kept': keep,
            }).sort_values(['kept', 'player_name'], ascending=[False, True])

            latest_identity = latest_identity[keep].copy()
        else:
            latest_identity = latest_identity[in_current].copy()

        if verbose:
            print(f"Found {len(latest_identity)} active players with historical data")

        # FPL availability filter
        fpl_availability = get_fpl_availability()
        if fpl_availability:
            def get_availability(name):
                if pd.isna(name):
                    return 100
                name_lower = str(name).lower()
                if name_lower in fpl_availability:
                    chance = fpl_availability[name_lower].get('chance_of_playing')
                    status = fpl_availability[name_lower].get('status', 'a')
                    if status in ['i', 's', 'u'] and chance == 0:
                        return 0
                    return chance if chance is not None else 100
                parts = str(name).split()
                if len(parts) > 1:
                    last = parts[-1].lower()
                    if last in fpl_availability:
                        chance = fpl_availability[last].get('chance_of_playing')
                        status = fpl_availability[last].get('status', 'a')
                        if status in ['i', 's', 'u'] and chance == 0:
                            return 0
                        return chance if chance is not None else 100
                return 100

            latest_identity['fpl_chance_of_playing'] = latest_identity['player_name'].apply(get_availability)
            n_before = len(latest_identity)
            latest_identity = latest_identity[latest_identity['fpl_chance_of_playing'] > 0].copy()
            if verbose and (n_before - len(latest_identity)) > 0:
                print(f"  Filtered out {n_before - len(latest_identity)} unavailable players (injured/suspended)")

        # Normalize target-GW fixture teams once. Team names are translated to
        # FotMob naming so synthetic rows' team/opponent values join the
        # rolling-stat history (FPL API names like 'Spurs' would silently miss).
        fixture_teams = []
        for fixture_index, (_, fix) in enumerate(fixtures.iterrows()):
            fixture_teams.append({
                'match_id': fix.get('match_id', -900000 - fixture_index),
                'match_date': pd.to_datetime(fix.get('match_date', cutoff), utc=True),
                'home_norm': normalize_team_name(fix['home_team']),
                'away_norm': normalize_team_name(fix['away_team']),
                'home_team': _to_fotmob(fix['home_team']),
                'away_team': _to_fotmob(fix['away_team']),
            })

        # Raw stat columns (everything except identity / fixture-linkage) are
        # zeroed out on synthetic rows so they don't contribute to rolling stats.
        raw_cols = list(self.raw_df.columns)
        preserve = {'player_id', 'player_name', 'team', 'position', 'fpl_position', 'shirt_number',
                    'season', 'match_id', 'gameweek', 'opponent', 'is_home',
                    'match_date', 'forecast_time', 'result_time'}
        stat_cols = [c for c in raw_cols if c not in preserve]

        test_rows = []
        unmatched_teams = set()
        for _, player in latest_identity.iterrows():
            player_team_norm = normalize_team_name(player.get('team', ''))
            fixture_num = 0
            any_match = False
            for fix in fixture_teams:
                is_home_match = (
                    player_team_norm == fix['home_norm']
                    or (player_team_norm and fix['home_norm'] and
                        (player_team_norm in fix['home_norm'] or fix['home_norm'] in player_team_norm))
                )
                is_away_match = (not is_home_match) and (
                    player_team_norm == fix['away_norm']
                    or (player_team_norm and fix['away_norm'] and
                        (player_team_norm in fix['away_norm'] or fix['away_norm'] in player_team_norm))
                )
                if not (is_home_match or is_away_match):
                    continue
                any_match = True
                fixture_num += 1
                row = {c: player.get(c) for c in raw_cols}
                for c in stat_cols:
                    row[c] = 0  # zero stat cols on synthetic row
                row['minutes'] = 0
                row['gameweek'] = gameweek
                row['season'] = season
                row['opponent'] = fix['away_team'] if is_home_match else fix['home_team']
                row['is_home'] = 1 if is_home_match else 0
                row['match_id'] = fix['match_id']
                row['match_date'] = fix['match_date'] if pd.notna(fix['match_date']) else cutoff
                row['forecast_time'] = cutoff
                row['result_time'] = row['match_date'] + pd.Timedelta(hours=3)
                row['_synthetic'] = True
                row['_fixture_num'] = fixture_num
                row['_fpl_chance_of_playing'] = player.get('fpl_chance_of_playing', 100)
                test_rows.append(row)
            if not any_match and player_team_norm:
                unmatched_teams.add(player_team_norm)

        if verbose:
            if test_rows:
                n_players = len(set(r['player_id'] for r in test_rows))
                n_dgw = len(test_rows) - n_players
                print(f"Matched {len(test_rows)} player-fixtures to GW{gameweek} ({n_players} players)")
                if n_dgw > 0:
                    print(f"  DGW: {n_dgw} extra fixture rows for players with 2 fixtures")
            else:
                print(f"WARNING: No players matched to GW{gameweek} fixtures!")
                if unmatched_teams:
                    print(f"  Unmatched player teams (sample): {sorted(unmatched_teams)[:10]}")
                return pd.DataFrame()

        synthetic_df = pd.DataFrame(test_rows)

        # Append synthetic rows only to information available at this forecast origin.
        raw_full = prior_raw.copy()
        raw_full['_synthetic'] = False
        raw_full['_fixture_num'] = 1
        raw_full['_fpl_chance_of_playing'] = 100

        augmented = pd.concat([raw_full, synthetic_df], ignore_index=True, sort=False)

        # Recompute all rolling features on the augmented dataset. At each
        # synthetic row, shift(1).rolling(N) over player/team/opponent groups
        # yields the mean of the N most recent actual matches.
        augmented.attrs['data_dir'] = str(self.data_dir)
        augmented = compute_rolling_features(augmented, verbose=False)

        test_df = augmented[augmented['_synthetic'] == True].copy()
        test_df = test_df.rename(columns={
            '_fixture_num': 'fixture_num',
            '_fpl_chance_of_playing': 'fpl_chance_of_playing',
        })
        test_df = test_df.drop(columns=['_synthetic'], errors='ignore')

        return test_df
    
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
                    'match_id': -1000000 - int(f['id']),
                    'match_date': f.get('kickoff_time'),
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
                    ].copy()
                    dates_path = self.data_dir / 'matches' / 'match_details.csv'
                    if 'match_date' not in result and dates_path.exists():
                        dates = pd.read_csv(dates_path, usecols=['match_id', 'match_date']).drop_duplicates('match_id')
                        result = result.merge(dates, on='match_id', how='left')
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

    @staticmethod
    def _cs_match_key(team, opponent, is_home):
        """Canonical key shared by team-level predictions and player rows."""
        return (
            CleanSheetModel._normalize_name(team),
            CleanSheetModel._normalize_name(opponent),
            int(is_home),
        )

    def _build_clean_sheet_match_predictions(self, test_df: pd.DataFrame,
                                             gameweek: int, season: str,
                                             verbose: bool = True) -> dict:
        """Build one shift-safe CleanSheetModel prediction per target fixture."""
        fixture_cols = ['team', 'opponent', 'season', 'gameweek', 'is_home']
        fixture_cols += [c for c in ('match_id', 'match_date', 'forecast_time', 'result_time') if c in test_df]
        fixture_cols += [c for c in test_df if c.startswith('manager_raw_') or c == 'manager_prior_games']
        fixture_cols += [
            f'manager_emb_{i}' for i in range(8)
            if f'manager_emb_{i}' in test_df.columns
        ]
        fixtures = test_df[fixture_cols].drop_duplicates(
            subset=['team', 'opponent', 'season', 'gameweek', 'is_home'],
            keep='first'
        )
        team_features = self.models['clean_sheet'].prepare_team_features(
            chronological_frame(self.df).loc[lambda history: history['result_time'] <
                chronological_frame(test_df)['forecast_time'].min()], prediction_fixtures=fixtures
        )
        prediction_mask = (
            team_features['_is_prediction'].astype(bool)
            if '_is_prediction' in team_features.columns
            else pd.Series(False, index=team_features.index)
        )
        target = team_features[
            prediction_mask
            & team_features['season'].eq(season)
            & pd.to_numeric(team_features['gameweek'], errors='coerce').eq(gameweek)
        ].copy()

        if target.empty:
            if verbose:
                print("  WARNING: No target clean-sheet feature rows were built")
            return {}

        goals_against = self.models['clean_sheet'].predict_goals_against(target)
        cs_probs = np.exp(-goals_against)
        two_plus_probs = 1.0 - np.exp(-goals_against) * (1.0 + goals_against)

        match_predictions = {}
        for (_, row), cs_prob, two_plus, ga in zip(
                target.iterrows(), cs_probs, two_plus_probs, goals_against):
            key = self._cs_match_key(row['team'], row['opponent'], row['is_home'])
            match_predictions[key] = (float(cs_prob), float(two_plus), float(ga))
        return match_predictions

    def _predict_clean_sheet(self, test_df: pd.DataFrame, gameweek: int,
                             season: str, verbose: bool) -> tuple:
        """Predict goals against, then derive CS prob and 2+ conceded prob for each player's team.

        Swaps in the actual GW opponent's offensive stats (not the last-match opponent's).

        Returns:
            Tuple of (cs_probs, two_plus_probs, pred_goals_against) as numpy arrays
        """
        match_predictions = self._build_clean_sheet_match_predictions(
            test_df, gameweek, season, verbose
        )
        self._last_clean_sheet_match_predictions = match_predictions

        # Map to players — one prediction per unique (team, opponent, is_home)
        cs_probs = []
        two_plus_probs = []
        goals_against = []
        missing = set()

        for _, row in test_df.iterrows():
            key = self._cs_match_key(
                row.get('team', ''), row.get('opponent', ''), row.get('is_home', 0)
            )
            if key in match_predictions:
                cs_prob, two_plus, ga = match_predictions[key]
            else:
                missing.add(key)
                cs_prob, two_plus, ga = 0.25, 0.40, 1.2

            cs_probs.append(cs_prob)
            two_plus_probs.append(two_plus)
            goals_against.append(ga)

        if missing and verbose:
            print(f"  WARNING: Used CS fallback for {len(missing)} unmatched team fixtures")
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
        match_predictions = getattr(
            self, '_last_clean_sheet_match_predictions', None
        )
        if match_predictions is None:
            match_predictions = self._build_clean_sheet_match_predictions(
                test_df, gameweek, season, verbose=False
            )

        pred_team_goals = []
        missing = set()
        for _, row in test_df.iterrows():
            is_home = row.get('is_home', 0)
            reverse_key = self._cs_match_key(
                row.get('opponent', ''), row.get('team', ''), 1 - is_home
            )
            if reverse_key in match_predictions:
                # Goals scored by this player's team equal goals conceded by the
                # opponent in the same target fixture.
                ga = match_predictions[reverse_key][2]
            else:
                missing.add(reverse_key)
                ga = 1.3
            pred_team_goals.append(ga)

        if missing:
            print(f"  WARNING: Used team-goals fallback for {len(missing)} unmatched fixtures")
        return np.array(pred_team_goals)
    
    @staticmethod
    def _predict_player_components(frame, models):
        """Raw model expectations conditional on appearing; minutes remains a feature."""
        frame = frame.copy()
        frame['pred_exp_goals'] = models['goals'].predict(frame)
        frame['pred_exp_assists'] = models['assists'].predict(frame)
        frame['pred_exp_defcon'] = models['defcon'].predict(frame)
        frame['pred_defcon_prob'] = models['defcon'].predict_threshold_prob(frame)
        frame['pred_yellow_prob'] = models['cards'].predict_yellow_prob(frame, frame['pred_minutes'].to_numpy())
        frame['pred_red_prob'] = models['cards'].predict_red_prob(frame, frame['pred_minutes'].to_numpy())
        frame['pred_exp_saves'] = 0.
        gk = frame['is_gk'] == 1
        if gk.any():
            frame.loc[gk, 'pred_exp_saves'] = models['saves'].predict_expected_saves(
                frame.loc[gk], frame.loc[gk, 'pred_minutes'].to_numpy())
        from scipy.stats import poisson
        frame['pred_4plus_conceded'] = poisson.sf(3, frame['pred_goals_against'])
        return frame

    def _simulate_points(self, frame, probability, state_minutes, models=None,
                         n_simulations=None, seed=42):
        models = self.models if models is None else models
        frame = frame.copy()
        state_minutes = np.asarray(state_minutes)
        if state_minutes.ndim == 1:
            state_minutes = np.broadcast_to(state_minutes, np.asarray(probability).shape)
        probability = np.asarray(probability)
        frame['pred_minutes_uncond'] = (probability * state_minutes).sum(axis=1)
        frame['pred_appear_prob'] = (probability * (state_minutes > 0)).sum(axis=1)
        p60 = (probability * (state_minutes >= 60)).sum(axis=1)
        frame['pred_60_prob'] = p60
        frame['pred_60_prob_cond'] = np.divide(p60, frame['pred_appear_prob'],
            out=np.zeros(len(frame)), where=frame['pred_appear_prob'].to_numpy() > 0)
        simulations = models['bonus'].simulate(
            frame, probability, state_minutes, n_simulations=n_simulations,
            seed=seed, defcon_r=models['defcon'].dispersion_r)
        return self._calculate_expected_points(frame, simulations), simulations

    def _calculate_expected_points(self, df, simulations=None):
        """Unconditional point expectations from the same draws shown in charts.

        exp_total_pts_uncond is an explicit alias; exp_total_pts_cond retains the
        old conditional interpretation for consumers that need it. Historical
        saved runs without the alias keep their legacy meaning in the optimizer.
        """
        from .models.bonus import score_simulations
        if simulations is None:
            probability, state_minutes = self.models['minutes'].predict_distribution(
                df, appear_prob=df.get('pred_appear_prob', pd.Series(1., index=df.index)).fillna(1).to_numpy())
            return self._simulate_points(df, probability, state_minutes)[0]
        frame = df.copy()
        components = score_simulations(frame, simulations, FPL_POINTS)
        for column, draws in components.items():
            frame[column] = draws.mean(axis=0)
        frame['exp_total_pts'] = simulations['total_points'].mean(axis=0)
        frame['exp_total_pts_uncond'] = frame['exp_total_pts']
        frame['pred_bonus'] = simulations['bonus'].mean(axis=0)
        appear = frame.get('pred_appear_prob', pd.Series(1., index=frame.index)).to_numpy()
        frame['exp_total_pts_cond'] = np.divide(frame['exp_total_pts'], appear,
                                              out=np.zeros(len(frame)), where=appear > 0)
        return frame

    def get_top_players(self, predictions: pd.DataFrame, n: int = 30) -> pd.DataFrame:
        """Get top N players by expected points."""
        cols = ['player_name', 'team', 'fpl_position', 'fixture_num', 'opponent', 'is_home',
                'pred_minutes', 'pred_exp_goals', 'pred_exp_assists',
                'pred_team_goals', 'pred_cs_prob', 'pred_2plus_conceded', 'pred_4plus_conceded',
                'pred_goals_against', 'pred_defcon_prob', 'pred_yellow_prob', 'pred_red_prob',
                'pred_bonus', 'exp_total_pts']
        available_cols = [c for c in cols if c in predictions.columns]
        return predictions.nlargest(n, 'exp_total_pts')[available_cols]

    # ------------------------------------------------------------------
    # Season evaluation: predicted vs actual FPL API points (2025/2026)
    # ------------------------------------------------------------------

    # FPL API team name -> our canonical team name
    _FPL_TEAM_MAP = {
        'arsenal': 'arsenal',
        'aston villa': 'aston villa',
        'bournemouth': 'bournemouth',
        'brentford': 'brentford',
        'brighton': 'brighton and hove albion',
        'burnley': 'burnley',
        'chelsea': 'chelsea',
        'crystal palace': 'crystal palace',
        'everton': 'everton',
        'fulham': 'fulham',
        'leeds': 'leeds united',
        'liverpool': 'liverpool',
        'man city': 'manchester city',
        'man utd': 'manchester united',
        'newcastle': 'newcastle united',
        "nott'm forest": 'nottingham forest',
        'spurs': 'tottenham hotspur',
        'sunderland': 'sunderland',
        'west ham': 'west ham united',
        'wolves': 'wolverhampton wanderers',
    }

    def evaluate_season(self, season: str = '2025/2026',
                        gameweeks: list = None,
                        use_saved: bool = True,
                        verbose: bool = True) -> pd.DataFrame:
        """Evaluate predictions against actual FPL API points for a season.

        For each gameweek, matches predicted exp_total_pts (inc bonus) against
        the official FPL total_points from the API. This is the definitive
        accuracy metric.

        Args:
            season: Season to evaluate (default '2025/2026').
            gameweeks: Specific GWs to evaluate. If None, uses all GWs with
                       saved predictions (or all completed GWs if use_saved=False).
            use_saved: If True, load predictions from data/predictions/ CSVs.
                       If False, run predict() fresh for each GW (slow but accurate).
            verbose: Print detailed results.

        Returns:
            DataFrame with per-player per-GW comparison: player_name, team,
            gameweek, pred_pts, actual_pts, error, plus summary metrics.
        """
        from sklearn.metrics import mean_absolute_error
        from .data_loader import fetch_fpl_actual_points

        season_tag = season.replace('/', '-')

        # --- Load predictions ---
        if use_saved:
            pred_files = sorted(self.data_dir.glob(f'predictions/gw*_{season_tag}.csv'))
            if not pred_files:
                raise ValueError(f"No saved predictions found for {season} in data/predictions/")

            all_preds = []
            for f in pred_files:
                gw = int(f.stem.split('_')[0].replace('gw', ''))
                if gameweeks and gw not in gameweeks:
                    continue
                df = pd.read_csv(f)
                df['gameweek'] = gw
                all_preds.append(df)

            preds_df = pd.concat(all_preds, ignore_index=True)
            eval_gws = sorted(preds_df['gameweek'].unique())
        else:
            # Run fresh predictions for each GW
            if not self.models:
                raise ValueError("Models not trained. Call train() first.")
            if gameweeks is None:
                raise ValueError("Must specify gameweeks when use_saved=False")
            all_preds = []
            for gw in gameweeks:
                try:
                    pred = self.predict(gameweek=gw, season=season, verbose=False)
                    pred['gameweek'] = gw
                    all_preds.append(pred)
                except Exception as e:
                    if verbose:
                        print(f"  GW{gw}: Could not predict ({e})")
            preds_df = pd.concat(all_preds, ignore_index=True)
            eval_gws = sorted(preds_df['gameweek'].unique())

        if verbose:
            print(f"\n{'='*60}")
            print(f"SEASON EVALUATION: {season}")
            print(f"{'='*60}")
            print(f"Evaluating GWs: {eval_gws}")
            print(f"Predictions: {len(preds_df):,} player-GW rows")

        # --- Fetch actual FPL points ---
        actuals_df = fetch_fpl_actual_points(
            gameweeks=eval_gws,
            cache_dir=str(self.data_dir),
            verbose=verbose
        )

        # Normalize team names for matching
        actuals_df['team_norm'] = actuals_df['team'].str.lower().map(self._FPL_TEAM_MAP)
        actuals_df['team_norm'] = actuals_df['team_norm'].fillna(actuals_df['team'].str.lower())

        preds_df['team_norm'] = preds_df['team'].str.lower().str.strip()

        # --- Match predictions to actuals ---
        # Strategy: match by (player_name, team, gameweek)
        # FPL API has full_name and web_name; our data has player_name
        # Try multiple matching strategies

        def _normalize_name(name):
            """Normalize player name for matching."""
            import unicodedata
            if pd.isna(name):
                return ''
            # Decompose unicode, strip accents
            nfkd = unicodedata.normalize('NFKD', str(name))
            ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
            return ascii_name.lower().strip()

        actuals_df['name_norm'] = actuals_df['player_name'].apply(_normalize_name)
        actuals_df['web_norm'] = actuals_df['web_name'].apply(_normalize_name)
        preds_df['name_norm'] = preds_df['player_name'].apply(_normalize_name)

        # Build lookup: (name_norm, team_norm, gw) -> actual row
        actual_lookup = {}
        for _, row in actuals_df.iterrows():
            gw = row['gameweek']
            team = row['team_norm']
            # Index by full name and web name
            for name_key in [row['name_norm'], row['web_norm']]:
                actual_lookup[(name_key, team, gw)] = row
            # Also index by last name only (fallback)
            parts = row['name_norm'].split()
            if len(parts) > 1:
                actual_lookup[(parts[-1], team, gw)] = row

        matched_rows = []
        unmatched = []
        for _, pred_row in preds_df.iterrows():
            gw = pred_row['gameweek']
            team = pred_row['team_norm']
            name = pred_row['name_norm']

            # Try full name match
            actual = actual_lookup.get((name, team, gw))
            if actual is None:
                # Try last name
                parts = name.split()
                if len(parts) > 1:
                    actual = actual_lookup.get((parts[-1], team, gw))

            if actual is not None:
                matched_rows.append({
                    'player_name': pred_row['player_name'],
                    'team': pred_row.get('team', ''),
                    'fpl_position': pred_row.get('fpl_position', actual.get('fpl_position', '')),
                    'gameweek': gw,
                    'pred_pts': pred_row.get('exp_total_pts', 0),
                    'pred_minutes': pred_row.get('pred_minutes', 0),
                    'pred_goals': pred_row.get('pred_exp_goals', 0),
                    'pred_assists': pred_row.get('pred_exp_assists', 0),
                    'pred_bonus': pred_row.get('pred_bonus', 0),
                    'actual_pts': actual['actual_total_points'],
                    'actual_minutes': actual['minutes'],
                    'actual_goals': actual['goals_scored'],
                    'actual_assists': actual['assists'],
                    'actual_bonus': actual['bonus'],
                })
            else:
                unmatched.append((pred_row['player_name'], team, gw))

        results_df = pd.DataFrame(matched_rows)

        if len(results_df) == 0:
            if verbose:
                print("No matches found between predictions and actuals!")
                print(f"Unmatched: {len(unmatched)}")
            return pd.DataFrame()

        results_df['error'] = results_df['pred_pts'] - results_df['actual_pts']
        results_df['abs_error'] = results_df['error'].abs()

        # --- Compute metrics ---
        # Only evaluate players who actually played (minutes > 0)
        played = results_df[results_df['actual_minutes'] > 0]

        overall_mae = mean_absolute_error(played['actual_pts'], played['pred_pts'])
        overall_mae_all = mean_absolute_error(results_df['actual_pts'], results_df['pred_pts'])

        if verbose:
            print(f"\nMatched: {len(results_df):,} / {len(preds_df):,} predictions "
                  f"({len(unmatched)} unmatched)")
            print(f"Players who played: {len(played):,}")
            print(f"\n{'-'*60}")
            print(f"{'FPL POINTS MAE (inc bonus)':<35} {overall_mae:.4f}")
            print(f"{'FPL POINTS MAE (all predicted)':<35} {overall_mae_all:.4f}")

            # Per-GW breakdown
            print(f"\n{'GW':<6} {'Matched':<10} {'Played':<10} {'MAE':<10} {'Mean Pred':<12} {'Mean Actual':<12}")
            print("-" * 60)
            for gw in eval_gws:
                gw_data = played[played['gameweek'] == gw]
                gw_all = results_df[results_df['gameweek'] == gw]
                if len(gw_data) > 0:
                    gw_mae = mean_absolute_error(gw_data['actual_pts'], gw_data['pred_pts'])
                    print(f"{gw:<6} {len(gw_all):<10} {len(gw_data):<10} {gw_mae:<10.4f} "
                          f"{gw_data['pred_pts'].mean():<12.2f} {gw_data['actual_pts'].mean():<12.2f}")

            # Per-position breakdown
            print(f"\n{'Position':<10} {'N':<8} {'MAE':<10} {'Mean Pred':<12} {'Mean Actual':<12}")
            print("-" * 52)
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                pos_data = played[played['fpl_position'] == pos]
                if len(pos_data) > 0:
                    pos_mae = mean_absolute_error(pos_data['actual_pts'], pos_data['pred_pts'])
                    print(f"{pos:<10} {len(pos_data):<8} {pos_mae:<10.4f} "
                          f"{pos_data['pred_pts'].mean():<12.2f} {pos_data['actual_pts'].mean():<12.2f}")

            # Bias check
            mean_error = played['error'].mean()
            print(f"\nMean error (bias): {mean_error:+.4f} "
                  f"({'over-predicting' if mean_error > 0 else 'under-predicting'})")

        # --- Store results ---
        self._store_season_eval(season, eval_gws, results_df, played, verbose)

        return results_df

    def _store_season_eval(self, season: str, gameweeks: list,
                           results_df: pd.DataFrame, played: pd.DataFrame,
                           verbose: bool):
        """Store season evaluation results in experiments.db."""
        import sqlite3
        from sklearn.metrics import mean_absolute_error
        from datetime import datetime

        db_path = self.data_dir / 'experiments.db'
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create season_eval table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS season_eval (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                season TEXT,
                gameweeks TEXT,
                n_matched INTEGER,
                n_played INTEGER,
                mae_inc_bonus REAL,
                mae_played REAL,
                mean_bias REAL,
                mae_gk REAL,
                mae_def REAL,
                mae_mid REAL,
                mae_fwd REAL
            )
        """)

        # Create per-GW detail table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS season_eval_gw (
                eval_id INTEGER,
                gameweek INTEGER,
                n_played INTEGER,
                mae REAL,
                mean_pred REAL,
                mean_actual REAL,
                FOREIGN KEY (eval_id) REFERENCES season_eval(id)
            )
        """)

        # Compute metrics
        overall_mae = mean_absolute_error(played['actual_pts'], played['pred_pts'])
        overall_mae_all = mean_absolute_error(results_df['actual_pts'], results_df['pred_pts'])
        mean_bias = played['error'].mean()

        pos_maes = {}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            pos_data = played[played['fpl_position'] == pos]
            pos_maes[pos] = mean_absolute_error(pos_data['actual_pts'], pos_data['pred_pts']) if len(pos_data) > 0 else None

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO season_eval
            (timestamp, season, gameweeks, n_matched, n_played,
             mae_inc_bonus, mae_played, mean_bias,
             mae_gk, mae_def, mae_mid, mae_fwd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, season, str(gameweeks), len(results_df), len(played),
              overall_mae_all, overall_mae, mean_bias,
              pos_maes.get('GK'), pos_maes.get('DEF'),
              pos_maes.get('MID'), pos_maes.get('FWD')))

        eval_id = cursor.lastrowid

        # Store per-GW details
        for gw in gameweeks:
            gw_data = played[played['gameweek'] == gw]
            if len(gw_data) > 0:
                gw_mae = mean_absolute_error(gw_data['actual_pts'], gw_data['pred_pts'])
                cursor.execute("""
                    INSERT INTO season_eval_gw (eval_id, gameweek, n_played, mae, mean_pred, mean_actual)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (eval_id, gw, len(gw_data), gw_mae,
                      gw_data['pred_pts'].mean(), gw_data['actual_pts'].mean()))

        conn.commit()
        conn.close()

        if verbose:
            print(f"\nStored evaluation (id={eval_id}) in experiments.db")
