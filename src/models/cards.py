"""Yellow and red card prediction model.

Trains an XGBoost binary classifier on actual yellow card data from the FPL API
to predict P(yellow card) per match directly. Requires yellow_cards column in
training data (merged via merge_fpl_card_data() in load_data()).

Red cards are predicted via fouls regression (too rare for direct classification).
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss


# Calibrated from PL historical data (used for red card prediction only)
RED_PER_FOUL = 0.004


class CardsModel:
    """Predicts yellow card probability via binary classifier, red via fouls."""

    FEATURES = [
        # Fouls history (primary predictor)
        'fouls_committed_per90_roll3', 'fouls_committed_per90_roll5', 'fouls_committed_per90_roll10',

        # Yellow card history (from FPL merge)
        'yellow_cards_roll3', 'yellow_cards_roll5', 'yellow_cards_roll10',
        'yellow_per_foul_roll10',
        'lifetime_yellow_cards_per90',

        # Lifetime fouling profile
        'lifetime_fouls_committed_per90',
        'lifetime_minutes',

        # Defensive activity (correlated with card-worthy fouls)
        'tackles_per90_roll5', 'interceptions_per90_roll5',
        'defcon_per90_roll5',

        # Position (defenders/mids get carded more)
        'is_def', 'is_mid', 'is_fwd',

        # Opponent context (stronger opponents = more tactical fouls)
        'opp_xg_roll5', 'opp_goals_roll5',

        # Match context
        'is_home',
    ]

    # Features for fouls regression (red card prediction)
    _FOULS_FEATURES = [
        'fouls_committed_per90_roll3', 'fouls_committed_per90_roll5', 'fouls_committed_per90_roll10',
        'lifetime_fouls_committed_per90',
        'lifetime_minutes',
        'tackles_per90_roll5', 'interceptions_per90_roll5',
        'defcon_per90_roll5',
        'is_def', 'is_mid', 'is_fwd',
        'opp_xg_roll5', 'opp_goals_roll5',
        'is_home',
    ]

    TARGET = 'yellow_cards'

    def __init__(self, **xgb_params):
        self.selected_features = xgb_params.pop('selected_features', None)
        self._xgb_params = xgb_params
        self.is_fitted = False
        self._model = None
        self._fouls_model = None

    @property
    def features_to_use(self):
        return self.selected_features if self.selected_features else self.FEATURES

    def _prepare_X(self, df: pd.DataFrame, features: list = None) -> np.ndarray:
        df = df.copy()
        features = features or self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float).values

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        df = df[df['minutes'] >= 1].copy()

        # Require yellow_cards column
        if 'yellow_cards' not in df.columns:
            raise ValueError(
                "CardsModel requires 'yellow_cards' column. "
                "Ensure load_data() successfully merged FPL card data "
                "(merge_fpl_card_data). Check your internet connection "
                "and that the FPL API is reachable."
            )

        n_with_data = df['yellow_cards'].notna().sum()
        if n_with_data == 0:
            raise ValueError(
                "yellow_cards column exists but has no non-null values. "
                "FPL card data merge likely failed to match any rows. "
                "Check player name / team / gameweek matching."
            )

        if verbose and n_with_data < len(df):
            pct = n_with_data / len(df) * 100
            print(f"  Warning: yellow_cards available for {n_with_data:,}/{len(df):,} "
                  f"rows ({pct:.0f}%) — training on matched rows only")

        # Filter to rows with yellow card data
        df_yc = df[df['yellow_cards'].notna()].copy()

        features = self.features_to_use
        X = self._prepare_X(df_yc, features)
        # Binary 0/1 per match (DGW rows already excluded as NaN upstream)
        y = np.clip(df_yc['yellow_cards'].astype(int).values, 0, 1)

        weights = df_yc['minutes'].values / df_yc['minutes'].mean()

        params = {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.08,
            'random_state': 42,
            'min_child_weight': 5,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
        }
        params.update({k: v for k, v in self._xgb_params.items()
                       if k not in ('objective', 'eval_metric')})

        self._model = xgb.XGBClassifier(**params)
        self._model.fit(X, y, sample_weight=weights)
        self._features_used = features

        if verbose:
            n_yellow = y.sum()
            y_pred = self._model.predict_proba(X)[:, 1]
            ll = log_loss(y, y_pred)
            print(f"Training CardsModel on {len(X):,} samples "
                  f"({len(features)} features)...")
            print(f"  Yellow cards: {n_yellow:,}/{len(y):,} "
                  f"({n_yellow/len(y)*100:.1f}%)")
            print(f"  LogLoss: {ll:.4f}")

        # Fit fouls model for red card prediction
        self._fit_fouls_model(df, verbose=False)

        self.is_fitted = True
        return self

    def _fit_fouls_model(self, df: pd.DataFrame, verbose: bool):
        """Train fouls regression model (used for red card prediction only)."""
        features = self._FOULS_FEATURES
        X = self._prepare_X(df, features)
        y = np.clip(df['fouls_committed_per90'].fillna(0).values, 0, 6.0)
        weights = df['minutes'].values / df['minutes'].mean()

        self._fouls_model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            random_state=42, min_child_weight=3,
        )
        self._fouls_model.fit(X, y, sample_weight=weights)
        self._fouls_features = features

    def predict_yellow_prob(self, df: pd.DataFrame, pred_minutes: np.ndarray = None) -> np.ndarray:
        """Predict probability of getting a yellow card in the match."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = self._prepare_X(df, self._features_used)
        return self._model.predict_proba(X)[:, 1]

    def predict_red_prob(self, df: pd.DataFrame, pred_minutes: np.ndarray = None) -> np.ndarray:
        """Predict probability of getting a red card in the match.

        Uses fouls model (reds are too rare for direct classification).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X = self._prepare_X(df, self._fouls_features)
        fouls_per90 = np.clip(self._fouls_model.predict(X), 0, 6.0)
        if pred_minutes is not None:
            expected_fouls = fouls_per90 * (np.array(pred_minutes) / 90)
        else:
            expected_fouls = fouls_per90
        return expected_fouls * RED_PER_FOUL

    # Backward-compatible aliases
    def predict_expected_yellows(self, df: pd.DataFrame, pred_minutes: np.ndarray) -> np.ndarray:
        return self.predict_yellow_prob(df, pred_minutes)

    def predict_expected_reds(self, df: pd.DataFrame, pred_minutes: np.ndarray) -> np.ndarray:
        return self.predict_red_prob(df, pred_minutes)

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self._features_used,
            'importance': self._model.feature_importances_
        }).sort_values('importance', ascending=False)
