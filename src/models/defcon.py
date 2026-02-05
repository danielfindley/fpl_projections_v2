"""Defensive contribution (defcon) prediction model."""
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


class DefconModel:
    """Predicts defensive contribution per 90 and threshold probability."""
    
    FEATURES = [
        # Defcon rolling
        'defcon_per90_roll5', 'defcon_per90_roll10',
        'hit_threshold_roll5', 'hit_threshold_roll10',
        
        # Component stats (rolling)
        'tackles_per90_roll5', 'interceptions_per90_roll5',
        'clearances_per90_roll5', 'blocks_per90_roll5', 'recoveries_per90_roll5',
        
        # Lifetime defensive profile
        'lifetime_minutes',
        'lifetime_defcon_per90',
        'lifetime_tackles_per90', 'lifetime_interceptions_per90', 'lifetime_clearances_per90',
        
        # Position
        'is_def', 'is_mid',
        
        # Opponent context (more attacks = more defensive actions)
        'opp_xg_roll5', 'opp_goals_roll5', 'opp_xg_roll10',
        
        # Match context
        'is_home',
    ]
    
    TARGET = 'defcon_per90'
    
    def __init__(self, **xgb_params):
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
    
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for feat in self.FEATURES:
            if feat not in df.columns:
                df[feat] = 0
        return df[self.FEATURES].fillna(0).astype(float)
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        df = df[df['minutes'] >= 1].copy()
        
        X = self._prepare_X(df)
        y = np.clip(df['defcon_per90'].fillna(0).values, 0, 30)
        
        X_scaled = self.scaler.fit_transform(X)
        weights = df['minutes'].values / df['minutes'].mean()
        
        if verbose:
            print(f"Training DefconModel on {len(X):,} samples...")
            print(f"  Mean defcon/90: {y.mean():.2f}")
        
        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.2f}")
        
        return self
    
    def predict_per90(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, 30)
    
    def predict_expected(self, df, pred_minutes) -> np.ndarray:
        per90 = self.predict_per90(df)
        return per90 * (np.array(pred_minutes) / 90)
    
    def predict_threshold_prob(self, df: pd.DataFrame, pred_minutes) -> np.ndarray:
        """Predict P(defcon >= threshold) using Poisson distribution."""
        expected = self.predict_expected(df, pred_minutes)
        expected = np.maximum(expected, 0.01)
        
        # Threshold: DEF=10, MID/FWD=12
        thresholds = np.where(df['is_def'] == 1, 10, 12)
        
        # P(X >= threshold) = 1 - P(X <= threshold-1)
        return np.clip(1 - poisson.cdf(thresholds - 1, expected), 0, 1)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.FEATURES,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
