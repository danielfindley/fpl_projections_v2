"""Yellow and red card prediction model.

Predicts fouls_committed_per90 using XGBoost, then converts to expected
card probabilities using calibrated Premier League conversion rates.

PL averages (per player per 90):
- ~0.93 fouls committed
- ~0.13 yellow cards  → ~0.14 yellows per foul
- ~0.004 red cards    → ~0.004 reds per foul
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# Calibrated from PL historical data
YELLOW_PER_FOUL = 0.14
RED_PER_FOUL = 0.004


class CardsModel:
    """Predicts fouls per 90, then derives yellow/red card expectations."""

    FEATURES = [
        # Fouls history (rolling)
        'fouls_committed_per90_roll3', 'fouls_committed_per90_roll5', 'fouls_committed_per90_roll10',

        # Lifetime fouling profile
        'lifetime_fouls_committed_per90',
        'lifetime_minutes',

        # Defensive activity (correlated with fouling)
        'tackles_per90_roll5', 'interceptions_per90_roll5',
        'defcon_per90_roll5',

        # Position (defenders/mids foul more)
        'is_def', 'is_mid', 'is_fwd',

        # Opponent context (stronger opponents = more fouls)
        'opp_xg_roll5', 'opp_goals_roll5',

        # Match context
        'is_home',
    ]

    TARGET = 'fouls_committed_per90'

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
        df = df[df['minutes'] >= 1].copy()

        X = self._prepare_X(df)
        y = np.clip(df[self.TARGET].fillna(0).values, 0, 6.0)

        X_scaled = self.scaler.fit_transform(X)
        weights = df['minutes'].values / df['minutes'].mean()

        if verbose:
            n_features = len(self.features_to_use)
            feature_info = f"({n_features} features"
            if self.selected_features:
                feature_info += ", tuned selection)"
            else:
                feature_info += ")"
            print(f"Training CardsModel on {len(X):,} samples {feature_info}...")
            print(f"  Mean fouls/90: {y.mean():.3f}")

        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True

        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.3f}")

        return self

    def predict_fouls_per90(self, df: pd.DataFrame) -> np.ndarray:
        """Predict fouls committed per 90 minutes."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, 6.0)

    def predict_expected_yellows(self, df: pd.DataFrame, pred_minutes: np.ndarray) -> np.ndarray:
        """Predict expected yellow cards for the match."""
        fouls_per90 = self.predict_fouls_per90(df)
        expected_fouls = fouls_per90 * (np.array(pred_minutes) / 90)
        return expected_fouls * YELLOW_PER_FOUL

    def predict_expected_reds(self, df: pd.DataFrame, pred_minutes: np.ndarray) -> np.ndarray:
        """Predict expected red cards for the match."""
        fouls_per90 = self.predict_fouls_per90(df)
        expected_fouls = fouls_per90 * (np.array(pred_minutes) / 90)
        return expected_fouls * RED_PER_FOUL

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.features_to_use,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
