"""Goalkeeper saves prediction model.

Predicts saves_per90 for goalkeepers, then converts to expected saves
and FPL save points (1 point per 3 saves).
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


class SavesModel:
    """Predicts saves per 90 for goalkeepers."""

    FEATURES = [
        # Saves history (rolling per90 rates)
        'saves_per90_roll1', 'saves_per90_roll2', 'saves_per90_roll3', 'saves_per90_roll5', 'saves_per90_roll7', 'saves_per90_roll10',

        # Recent saves form (raw counts)
        'saves_last1', 'saves_roll2', 'saves_roll3', 'saves_roll5', 'saves_roll7', 'saves_roll10',

        # Shot quality faced (xGoT faced per 90)
        'xgot_faced_per90_roll1', 'xgot_faced_per90_roll2', 'xgot_faced_per90_roll3', 'xgot_faced_per90_roll5', 'xgot_faced_per90_roll7', 'xgot_faced_per90_roll10',

        # Lifetime goalkeeper profile
        'lifetime_saves_per90',
        'lifetime_minutes',

        # Team defensive context (leaky defense = more saves)
        'team_conceded_roll1', 'team_conceded_roll2', 'team_conceded_roll3', 'team_conceded_roll5', 'team_conceded_roll7', 'team_conceded_roll10',
        'team_xga_roll1', 'team_xga_roll2', 'team_xga_roll3', 'team_xga_roll5', 'team_xga_roll7', 'team_xga_roll10',

        # Opponent attacking strength (stronger attack = more shots = more saves)
        'opp_xg_roll1', 'opp_xg_roll2', 'opp_xg_roll3', 'opp_xg_roll5', 'opp_xg_roll7', 'opp_xg_roll10',
        'opp_goals_roll1', 'opp_goals_roll2', 'opp_goals_roll3', 'opp_goals_roll5', 'opp_goals_roll7', 'opp_goals_roll10',

        # Match context
        'is_home',
    ]

    TARGET = 'saves_per90'

    def __init__(self, **xgb_params):
        self.selected_features = xgb_params.pop('selected_features', None)

        default_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.08,
            'random_state': 42,
            'min_child_weight': 3,
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
        """Train on goalkeepers who played 1+ minutes."""
        df = df[(df['minutes'] >= 1) & (df['is_gk'] == 1)].copy()

        X = self._prepare_X(df)
        y = np.clip(df[self.TARGET].fillna(0).values, 0, 12.0)

        X_scaled = self.scaler.fit_transform(X)
        weights = df['minutes'].values / df['minutes'].mean()

        if verbose:
            n_features = len(self.features_to_use)
            feature_info = f"({n_features} features"
            if self.selected_features:
                feature_info += ", tuned selection)"
            else:
                feature_info += ")"
            print(f"Training SavesModel on {len(X):,} GK samples {feature_info}...")
            print(f"  Mean saves/90: {y.mean():.2f}")

        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True

        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.3f}")

        return self

    def predict_per90(self, df: pd.DataFrame) -> np.ndarray:
        """Predict saves per 90 minutes."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, 12.0)

    def predict_expected_saves(self, df: pd.DataFrame, pred_minutes: np.ndarray) -> np.ndarray:
        """Predict expected saves for the match."""
        per90 = self.predict_per90(df)
        return per90 * (np.array(pred_minutes) / 90)

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.features_to_use,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
