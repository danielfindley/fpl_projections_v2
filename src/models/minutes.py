"""Minutes prediction model — two-stage architecture.

AppearClassifier: predicts P(minutes >= 1) — the only model trained on non-appearances
StarterClassifier: predicts P(starter) via XGBClassifier
StarterMinutesModel: regressor for 60+ minute players
SubMinutesModel: regressor for 1-59 minute players
MinutesModel: backward-compatible wrapper that blends all three
"""
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


class AppearClassifier:
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
        df = df.copy()
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
        if verbose:
            print(f"  AppearClassifier: {len(grid):,} player-gameweeks, {y_all.mean():.1%} appeared")

        seasons = sorted(grid['season'].dropna().unique()) if 'season' in grid.columns else []
        if len(seasons) >= 2:
            holdout = seasons[-1]
            core = grid[grid['season'] != holdout]
            calib = grid[grid['season'] == holdout]
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
        X = self._prepare_X(df)
        p = self.model.predict_proba(self.scaler.transform(X))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)


class StarterClassifier:
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
            'use_label_encoder': False,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self.scaler = StandardScaler()
        self.is_fitted = False

    @property
    def features_to_use(self):
        return self.selected_features if self.selected_features else self.FEATURES

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on all players with minutes >= 1. Target: minutes >= 60."""
        df = df[df['minutes'] >= 1].copy()
        X = self._prepare_X(df)
        y = (df['minutes'] >= 60).astype(int).values
        X_scaled = self.scaler.fit_transform(X)

        if verbose:
            print(f"  StarterClassifier: {len(X):,} samples, {y.mean():.1%} starters")

        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(starter) for each row."""
        if not self.is_fitted:
            raise ValueError("StarterClassifier not fitted")
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


class StarterMinutesModel:
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
        df = df.copy()
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on starters only (minutes >= 60)."""
        df = df[df['minutes'] >= 60].copy()
        X = self._prepare_X(df)
        y = df['minutes'].values
        X_scaled = self.scaler.fit_transform(X)

        # Weight toward full-90 games
        weights = np.ones(len(y))
        weights[y >= 89] = 3.0
        weights[(y >= 75) & (y < 89)] = 1.5

        if verbose:
            print(f"  StarterMinutesModel: {len(X):,} samples, mean={y.mean():.1f}")

        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("StarterMinutesModel not fitted")
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 60, 90)


class SubMinutesModel:
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
        df = df.copy()
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on subs only (1 <= minutes < 60)."""
        df = df[(df['minutes'] >= 1) & (df['minutes'] < 60)].copy()
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
            cls_params = {k: v for k, v in params.items() if k != 'selected_features'}
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

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Blend predictions: P(start)*starter_pred + (1-P(start))*sub_pred, then cap."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        preds = self._blend(df)
        preds = self._apply_caps(df, preds)
        return np.clip(preds, 1, 90)

    def predict_appear_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(plays >= 1 minute). Complements predict(), which is E[minutes | appears]."""
        if not self.appear_is_fitted:
            raise ValueError(
                "AppearClassifier not fitted — pass appearance_grid to MinutesModel.fit()"
            )
        return self.appear_model.predict_proba(df)

    def _blend(self, df: pd.DataFrame) -> np.ndarray:
        """Sigmoid-sharpened blend: push P(start) toward 0/1 before weighting."""
        p_start = self.classifier.predict_proba(df)
        p_sharp = self._sharpen(p_start, temp=0.3)
        starter_preds = self.starter_model.predict(df)
        sub_preds = self.sub_model.predict(df)
        return p_sharp * starter_preds + (1 - p_sharp) * sub_preds

    @staticmethod
    def _sharpen(p: np.ndarray, temp: float = 0.3) -> np.ndarray:
        """Sharpen probabilities via sigmoid with temperature scaling."""
        p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
        logit = np.log(p_clipped / (1 - p_clipped))
        return 1 / (1 + np.exp(-logit / temp))

    def _apply_caps(self, df: pd.DataFrame, preds: np.ndarray) -> np.ndarray:
        """Apply season-based capping logic (preserved from original)."""
        preds = preds.copy()

        current_season_mins = df['current_season_minutes'].fillna(0).values
        current_season_apps = df['current_season_apps'].fillna(0).values
        current_season_mins_per_app = df['current_season_mins_per_app'].fillna(0).values
        roll5 = df['minutes_roll5'].fillna(60).values
        # gw gap is 0-filled upstream, so 0 = no prior appearance; real gaps are >= 1
        gw_gap = df['gw_gap_since_last_appearance'].fillna(0).values if 'gw_gap_since_last_appearance' in df.columns else np.zeros(len(preds))

        for i in range(len(preds)):
            if current_season_apps[i] == 0:
                # Season opener for a recently active player (e.g. played the end
                # of last season): trust rolling form instead of the cold-start cap
                if 1 <= gw_gap[i] <= 3 and roll5[i] > 0:
                    preds[i] = min(preds[i], max(roll5[i] * 1.2, 30))
                else:
                    preds[i] = min(preds[i], 30)
            elif current_season_apps[i] >= 1 and current_season_mins[i] < 90:
                max_reasonable = max(current_season_mins_per_app[i] * 1.2, 15)
                preds[i] = min(preds[i], max_reasonable)
            elif current_season_apps[i] >= 3:
                max_reasonable = min(90, current_season_mins_per_app[i] * 1.2)
                preds[i] = min(preds[i], max(max_reasonable, 30))

            if current_season_apps[i] >= 5 and current_season_mins_per_app[i] >= 80:
                if roll5[i] >= 80:
                    preds[i] = max(preds[i], 88)
                elif roll5[i] >= 70:
                    preds[i] = max(preds[i], 80)

        return preds

    def feature_importance(self) -> pd.DataFrame:
        """Return classifier feature importances."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.classifier.features_to_use,
            'importance': self.classifier.model.feature_importances_
        }).sort_values('importance', ascending=False)
