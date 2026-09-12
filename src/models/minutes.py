"""Minutes prediction model — two-stage architecture.

AppearClassifier: predicts P(minutes >= 1) — the only model trained on non-appearances
StarterClassifier: predicts P(60+ minutes | appears) via XGBClassifier
StarterMinutesModel: regressor for 60+ minute players
SubMinutesModel: regressor for 1-59 minute players
MinutesModel: backward-compatible wrapper that blends all three
"""
from ..features import ManagerFeatureMixin, chronological_frame, deadline_splits
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score

from ..features import APPEARANCE_FEATURES


# Shared feature list (classifier uses all, regressors use subsets)
ALL_FEATURES = [
    # Minutes history
    'last_minutes', 'minutes_roll1', 'minutes_roll2', 'minutes_roll3', 'minutes_roll5', 'minutes_roll7', 'minutes_roll10',

    # Starting likelihood
    'starter_score',
    'starter_rate_roll1', 'starter_rate_roll2', 'starter_rate_roll3', 'starter_rate_roll5', 'starter_rate_roll7', 'starter_rate_roll10',
    'full90_rate_roll1', 'full90_rate_roll2', 'full90_rate_roll3', 'full90_rate_roll5', 'full90_rate_roll7', 'full90_rate_roll10',
    'last_was_starter', 'last_was_full_90',

    # Lifetime profile
    'lifetime_minutes', 'lifetime_mins_per_app',

    # Current season
    'current_season_minutes', 'current_season_apps', 'current_season_mins_per_app',
    'gw_gap_since_last_appearance', 'last_app_prev_season',

    # Goal involvement (key players play more)
    'goals_roll2', 'goals_roll3', 'goals_roll5', 'goals_roll7', 'goals_roll10',
    'assists_roll2', 'assists_roll3', 'assists_roll5', 'assists_roll7', 'assists_roll10',
    'goal_involvements_roll5',

    # Position
    'is_gk', 'is_def', 'is_mid', 'is_fwd',

    # Form trend
    'minutes_trend',

    # Match context
    'is_home',
    'team_goals_roll1', 'team_goals_roll2', 'team_goals_roll3', 'team_goals_roll5', 'team_goals_roll7', 'team_goals_roll10',

    # Opponent strength (rotation signal)
    'opp_goals_roll1', 'opp_goals_roll2', 'opp_goals_roll3', 'opp_goals_roll5', 'opp_goals_roll7', 'opp_goals_roll10',
    'opp_xg_roll1', 'opp_xg_roll2', 'opp_xg_roll3', 'opp_xg_roll5', 'opp_xg_roll7', 'opp_xg_roll10',

    # Fixture context
    'gameweek',

    # Player importance to team
    'goal_share_roll1', 'goal_share_roll2', 'goal_share_roll3', 'goal_share_roll5', 'goal_share_roll7', 'goal_share_roll10',
    'xg_per90_roll1', 'xg_per90_roll2', 'xg_per90_roll3', 'xg_per90_roll5', 'xg_per90_roll7', 'xg_per90_roll10',

    # Manager embeddings (rotation/playstyle signal — particularly relevant to minutes)
    'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
    'manager_emb_4', 'manager_emb_5', 'manager_emb_6', 'manager_emb_7',
]

# Starter regressor: features that matter for how long starters play
STARTER_FEATURES = [
    'last_minutes', 'minutes_roll1', 'minutes_roll2', 'minutes_roll3', 'minutes_roll5', 'minutes_roll7', 'minutes_roll10',
    'starter_score',
    'full90_rate_roll1', 'full90_rate_roll2', 'full90_rate_roll3', 'full90_rate_roll5', 'full90_rate_roll7', 'full90_rate_roll10',
    'last_was_full_90',
    'lifetime_mins_per_app',
    'current_season_mins_per_app', 'current_season_apps',
    'gw_gap_since_last_appearance', 'last_app_prev_season',
    'goals_roll2', 'goals_roll3', 'goals_roll5', 'goals_roll7', 'goals_roll10',
    'goal_involvements_roll5',
    'is_gk', 'is_def', 'is_mid', 'is_fwd',
    'minutes_trend',
    'is_home',
    'gameweek',
    'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
    'manager_emb_4', 'manager_emb_5', 'manager_emb_6', 'manager_emb_7',
]

# Sub regressor: features focused on sub patterns
SUB_FEATURES = [
    'last_minutes', 'minutes_roll1', 'minutes_roll2', 'minutes_roll3', 'minutes_roll5', 'minutes_roll7', 'minutes_roll10',
    'starter_score',
    'starter_rate_roll1', 'starter_rate_roll2', 'starter_rate_roll3', 'starter_rate_roll5', 'starter_rate_roll7', 'starter_rate_roll10',
    'last_was_starter',
    'lifetime_mins_per_app',
    'current_season_mins_per_app', 'current_season_apps',
    'gw_gap_since_last_appearance', 'last_app_prev_season',
    'is_gk', 'is_def', 'is_mid', 'is_fwd',
    'minutes_trend',
    'is_home',
    'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
    'manager_emb_4', 'manager_emb_5', 'manager_emb_6', 'manager_emb_7',
]


class AppearClassifier(ManagerFeatureMixin):
    """XGBClassifier predicting P(minutes >= 1) — whether a player features at all.

    Unlike every other model in the pipeline, this one trains on the calendar grid
    from ``features.build_appearance_grid`` rather than the per-match frame, so it
    sees the weeks a player was available and not picked. Those rows are the entire
    signal: a frame of appearances only has no counterexamples.

    Kept separate from ``MinutesModel.predict``, which stays conditional on playing
    (E[minutes | appears]). Folding P(appears) into pred_minutes would change the
    meaning of a feature that goals/assists/defcon/saves/bonus were all tuned
    against, forcing a full retune.
    """

    FEATURES = APPEARANCE_FEATURES
    TARGET = 'appeared'

    def __init__(self, **xgb_params):
        self.selected_features = xgb_params.pop('selected_features', None)
        default_params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'random_state': 42,
            'min_child_weight': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self.scaler = StandardScaler()
        self.calibrator = None
        self.is_fitted = False

    @property
    def features_to_use(self):
        return self.selected_features if self.selected_features else self.FEATURES

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.manager_features(df)
        for feat in self.features_to_use:
            if feat not in df.columns:
                df[feat] = 0
        return df[self.features_to_use].fillna(0).astype(float)

    def fit(self, grid: pd.DataFrame, verbose: bool = True):
        """Train on the appearance grid. Expects an ``appeared`` column.

        Raw XGB output is badly calibrated in the region that matters here: on a
        2025/26 holdout it put P(blank) at .090 for likely starters whose actual
        blank rate was .050, which would roughly double every bench weight. So the
        most recent season is held out to fit an isotonic map, then the booster is
        refit on everything and the map applied on top.
        """
        y_all = grid[self.TARGET].astype(int).values
        if not len(y_all):
            raise ValueError("Appearance training grid is empty")
        self.calibrator = None
        self.constant_probability = float(y_all[0]) if len(np.unique(y_all)) == 1 else None
        if self.constant_probability is not None:
            self.is_fitted = True
            return self
        if verbose:
            print(f"  AppearClassifier: {len(grid):,} player-gameweeks, {y_all.mean():.1%} appeared")

        seasons = sorted(grid['season'].dropna().unique()) if 'season' in grid.columns else []
        if len(seasons) >= 2 and grid[grid['season'] != seasons[-1]][self.TARGET].nunique() > 1:
            holdout = seasons[-1]
            calib = chronological_frame(grid[grid['season'] == holdout])
            core = chronological_frame(grid[grid['season'] != holdout])
            core = core[core['result_time'] < calib['forecast_time'].min()]
            scaler = StandardScaler()
            pre = xgb.XGBClassifier(**self.model.get_params())
            pre.fit(scaler.fit_transform(self._prepare_X(core)),
                    core[self.TARGET].astype(int).values)
            p_cal = pre.predict_proba(scaler.transform(self._prepare_X(calib)))[:, 1]
            self.calibrator = IsotonicRegression(out_of_bounds='clip').fit(
                p_cal, calib[self.TARGET].astype(int).values
            )
            if verbose:
                print(f"    isotonic calibration fit on {holdout} ({len(calib):,} rows)")

        self.fit_manager_features(grid)
        X_scaled = self.scaler.fit_transform(self._prepare_X(grid))
        self.model.fit(X_scaled, y_all)
        self.is_fitted = True

        if verbose:
            try:
                auc = roc_auc_score(y_all, self.model.predict_proba(X_scaled)[:, 1])
                print(f"    train AUC: {auc:.3f}")
            except ValueError:
                pass
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(plays >= 1 minute) for each row."""
        if not self.is_fitted:
            raise ValueError("AppearClassifier not fitted")
        if getattr(self, 'constant_probability', None) is not None:
            return np.full(len(df), self.constant_probability)
        X = self._prepare_X(df)
        p = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)


class StarterClassifier(ManagerFeatureMixin):
    """XGBClassifier predicting P(minutes >= 60)."""

    FEATURES = ALL_FEATURES
    TARGET = 'minutes'

    def __init__(self, **xgb_params):
        self.selected_features = xgb_params.pop('selected_features', None)
        default_params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42,
            'min_child_weight': 5,
            'eval_metric': 'logloss',
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    @property
    def features_to_use(self):
        return self.selected_features if self.selected_features else self.FEATURES

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = self.manager_features(df)
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on all players with minutes >= 1. Target: minutes >= 60."""
        df = df[df['minutes'] >= 1].copy()
        self.fit_manager_features(df)
        X = self._prepare_X(df)
        y = (df['minutes'] >= 60).astype(int).values
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"  StarterClassifier: {len(X):,} samples, {y.mean():.1%} starters")

        self.constant_probability = float(y[0]) if len(np.unique(y)) == 1 else None
        if self.constant_probability is None:
            self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(minutes >= 60 | appears), not lineup-start probability."""
        if not self.is_fitted:
            raise ValueError("StarterClassifier not fitted")
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        if getattr(self, 'constant_probability', None) is not None:
            return np.full(len(df), self.constant_probability)
        return self.model.predict_proba(X_scaled)[:, 1]


class StarterMinutesModel(ManagerFeatureMixin):
    """Regressor for players who start (minutes >= 60). Output clipped to [60, 90]."""

    FEATURES = STARTER_FEATURES
    TARGET = 'minutes'

    def __init__(self, **xgb_params):
        self.selected_features = xgb_params.pop('selected_features', None)
        default_params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42,
            'min_child_weight': 5,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    @property
    def features_to_use(self):
        return self.selected_features if self.selected_features else self.FEATURES

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = self.manager_features(df)
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on starters only (minutes >= 60)."""
        df = df[df['minutes'] >= 60].copy()
        self.constant_minutes = 80.0 if df.empty else None
        if df.empty:
            self.is_fitted = True
            return self
        self.fit_manager_features(df)
        X = self._prepare_X(df)
        y = df['minutes'].values
        X_scaled = self.scaler.fit_transform(X)

        # Fit the state's conditional mean without a full-90 preference.
        weights = np.ones(len(y))
        # Uniform weights retain E[minutes | state] for mixture expectations.

        if verbose:
            print(f"  StarterMinutesModel: {len(X):,} samples, mean={y.mean():.1f}")

        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("StarterMinutesModel not fitted")
        if getattr(self, 'constant_minutes', None) is not None:
            return np.full(len(df), self.constant_minutes)
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 60, 90)


class SubMinutesModel(ManagerFeatureMixin):
    """Regressor for substitutes (1 <= minutes < 60). Output clipped to [1, 59]."""

    FEATURES = SUB_FEATURES
    TARGET = 'minutes'

    def __init__(self, **xgb_params):
        self.selected_features = xgb_params.pop('selected_features', None)
        default_params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.1,
            'random_state': 42,
            'min_child_weight': 5,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    @property
    def features_to_use(self):
        return self.selected_features if self.selected_features else self.FEATURES

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = self.manager_features(df)
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on subs only (1 <= minutes < 60)."""
        df = df[(df['minutes'] >= 1) & (df['minutes'] < 60)].copy()
        self.constant_minutes = 20.0 if df.empty else None
        if df.empty:
            self.is_fitted = True
            return self
        self.fit_manager_features(df)
        X = self._prepare_X(df)
        y = df['minutes'].values
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"  SubMinutesModel: {len(X):,} samples, mean={y.mean():.1f}")

        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("SubMinutesModel not fitted")
        if getattr(self, 'constant_minutes', None) is not None:
            return np.full(len(df), self.constant_minutes)
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 1, 59)


class MinutesModel:
    """Backward-compatible wrapper: blends StarterClassifier + two regressors.

    Accepts either:
    - Nested params: {'classifier_params': {...}, 'starter_params': {...}, 'sub_params': {...}}
    - Flat params (legacy): passed to all sub-models as defaults
    """

    # Exposed for pipeline tuning introspection
    FEATURES = ALL_FEATURES
    TARGET = 'minutes'

    def __init__(self, **params):
        # Detect nested vs flat params
        if 'classifier_params' in params:
            cls_params = params.get('classifier_params', {})
            starter_params = params.get('starter_params', {})
            sub_params = params.get('sub_params', {})
        else:
            # Legacy flat params — use for all sub-models
            cls_params = {k: v for k, v in params.items() if k not in ('appear_params',)}
            starter_params = dict(cls_params)
            sub_params = dict(cls_params)

        self.classifier = StarterClassifier(**cls_params)
        self.starter_model = StarterMinutesModel(**starter_params)
        self.sub_model = SubMinutesModel(**sub_params)
        self.appear_model = AppearClassifier(**params.get('appear_params', {}))
        self.is_fitted = False
        self.appear_is_fitted = False

        # Store selected_features for compatibility (not used directly)
        self.selected_features = None

    @property
    def features_to_use(self):
        return self.FEATURES

    def fit(self, df: pd.DataFrame, verbose: bool = True, appearance_grid: pd.DataFrame = None):
        """Train all three sub-models, plus the appearance model when a grid is given.

        ``appearance_grid`` comes from ``features.build_appearance_grid`` and is the
        only input here containing non-appearances. Without it the minutes models
        still train exactly as before and ``predict_appear_proba`` is unavailable.
        """
        if verbose:
            print(f"Training MinutesModel (two-stage) on {len(df[df['minutes'] >= 1]):,} samples...")

        if 'started' in df:
            started = df.loc[pd.to_numeric(df['started'], errors='coerce').eq(1)]
            self.p60_given_start = float((started['minutes'] >= 60).mean()) if len(started) >= 100 else 0.90
        else:
            self.p60_given_start = 0.90
        self.classifier.fit(df, verbose)
        self.starter_model.fit(df, verbose)
        self.sub_model.fit(df, verbose)
        self.is_fitted = True

        if appearance_grid is not None and len(appearance_grid):
            self.appear_model.fit(appearance_grid, verbose)
            self.appear_is_fitted = True

        if verbose:
            # Report combined training MAE
            played = df[df['minutes'] >= 1].copy()
            y_pred = self._blend(played)
            y_true = played['minutes'].values
            print(f"  Combined MAE: {mean_absolute_error(y_true, y_pred):.1f}")

        return self

    def predict_distribution(self, df: pd.DataFrame, lineup_p_start=None,
                             lineup_weight: float = 0.7, appear_prob=None,
                             availability=None):
        """Joint probabilities for absent, 1-59 minutes and 60-90 minutes.

        The classifier estimates P(60+ | appears); a lineup feed estimates
        P(start). Convert the latter to a joint 60+ prior with a completion rate,
        then blend whole distributions. A probable starter must also be probable
        to appear. Injury/suspension availability discounts the mixture once.
        """
        if not self.is_fitted:
            raise ValueError("MinutesModel not fitted")
        if not 0 <= lineup_weight <= 1:
            raise ValueError("lineup_weight must lie in [0, 1]")
        if appear_prob is None:
            appear_prob = self.predict_appear_proba(df) if self.appear_is_fitted else np.ones(len(df))
        appear = np.clip(np.asarray(appear_prob, dtype=float), 0, 1)
        p60_joint = appear * np.clip(self.classifier.predict_proba(df), 0, 1)
        if lineup_p_start is not None:
            lineup = np.asarray(lineup_p_start, dtype=float)
            have = np.isfinite(lineup)
            starts = np.clip(lineup, 0, 1)
            feed_appear = np.maximum(appear, starts)
            feed_sixty = starts * getattr(self, 'p60_given_start', 0.90)
            p60_joint = np.where(have,
                (1 - lineup_weight) * p60_joint + lineup_weight * feed_sixty, p60_joint)
            appear = np.where(have,
                (1 - lineup_weight) * appear + lineup_weight * feed_appear, appear)
        if availability is not None:
            available = np.clip(np.asarray(availability, dtype=float), 0, 1)
            appear, p60_joint = appear * available, p60_joint * available
        probability = np.column_stack([1 - appear, appear - p60_joint, p60_joint])
        state_minutes = np.column_stack([np.zeros(len(df)), self.sub_model.predict(df),
                                        self.starter_model.predict(df)])
        return probability, state_minutes

    def predict(self, df: pd.DataFrame, lineup_p_start=None, lineup_weight: float = 0.7):
        """E[minutes | appears], using the same mixture as points and simulation."""
        probability, minutes = self.predict_distribution(df, lineup_p_start, lineup_weight)
        appear = 1 - probability[:, 0]
        return np.divide(np.sum(probability * minutes, axis=1), appear,
                         out=np.zeros(len(df)), where=appear > 0)

    def predict_appear_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(plays >= 1 minute). Complements predict(), which is E[minutes | appears]."""
        if not self.appear_is_fitted:
            raise ValueError(
                "AppearClassifier not fitted — pass appearance_grid to MinutesModel.fit()"
            )
        return self.appear_model.predict_proba(df)

    def _blend(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict(df)

    def feature_importance(self) -> pd.DataFrame:
        """Return classifier feature importances."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.classifier.features_to_use,
            'importance': self.classifier.model.feature_importances_
        }).sort_values('importance', ascending=False)
