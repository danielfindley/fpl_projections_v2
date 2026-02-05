"""Minutes prediction model."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


class MinutesModel:
    """Predicts expected minutes for players who will play."""
    
    FEATURES = [
        # Minutes history
        'last_minutes', 'minutes_roll3', 'minutes_roll5', 'minutes_roll10',
        
        # Starting likelihood
        'starter_score', 'starter_rate_roll5', 'starter_rate_roll10',
        'full90_rate_roll5', 'full90_rate_roll10',
        'last_was_starter', 'last_was_full_90',
        
        # Lifetime profile
        'lifetime_minutes', 'lifetime_mins_per_app',
        
        # Current season (to detect limited playing time this season)
        'current_season_minutes', 'current_season_apps', 'current_season_mins_per_app',
        
        # Goal involvement (key players play more)
        'goals_roll5', 'assists_roll5', 'goal_involvements_roll5',
        
        # Position
        'is_gk', 'is_def', 'is_mid', 'is_fwd',
        
        # Match context
        'is_home', 'team_goals_roll5',
    ]
    
    TARGET = 'minutes'
    
    def __init__(self, **xgb_params):
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
    
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for feat in self.FEATURES:
            if feat not in df.columns:
                df[feat] = 0
        return df[self.FEATURES].fillna(0).astype(float)
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on players who played 1+ minutes."""
        df = df[df['minutes'] >= 1].copy()
        
        X = self._prepare_X(df)
        y = df['minutes'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Heavy weighting toward full games
        weights = np.ones(len(y))
        weights[y >= 89] = 3.0
        weights[(y >= 60) & (y < 89)] = 2.0
        weights[y < 30] = 0.5
        
        if verbose:
            print(f"Training MinutesModel on {len(X):,} samples...")
            print(f"  90 min: {(y >= 89).mean():.1%}, 60+ min: {(y >= 60).mean():.1%}")
        
        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True
        
        if verbose:
            y_pred = np.clip(self.model.predict(X_scaled), 1, 90)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.1f}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected minutes."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        preds = np.clip(self.model.predict(X_scaled), 1, 90)
        
        # Get current season stats (to detect limited playing time this season)
        current_season_mins = df['current_season_minutes'].fillna(0).values
        current_season_apps = df['current_season_apps'].fillna(0).values
        current_season_mins_per_app = df['current_season_mins_per_app'].fillna(0).values
        roll5 = df['minutes_roll5'].fillna(60).values
        
        for i in range(len(preds)):
            # CAP predictions for players with limited current-season playing time
            # If player has very few minutes THIS SEASON, don't trust cross-season averages
            if current_season_apps[i] == 0:
                # NO current-season appearances before this game
                # This player hasn't established themselves this season - be very conservative
                # Cap at 30 mins (sub appearance) unless model predicts less
                preds[i] = min(preds[i], 30)
            elif current_season_apps[i] >= 1 and current_season_mins[i] < 90:
                # Very limited current-season time (few minutes total this season)
                # Cap based on their actual current-season average
                max_reasonable = max(current_season_mins_per_app[i] * 1.2, 15)
                preds[i] = min(preds[i], max_reasonable)
            elif current_season_apps[i] >= 3:
                # Enough this-season data - use current season avg as reasonable cap
                max_reasonable = min(90, current_season_mins_per_app[i] * 1.2)
                preds[i] = min(preds[i], max(max_reasonable, 30))
            
            # Boost only if CURRENT SEASON supports it (not just historical)
            if current_season_apps[i] >= 5 and current_season_mins_per_app[i] >= 80:
                if roll5[i] >= 80:
                    preds[i] = max(preds[i], 88)
                elif roll5[i] >= 70:
                    preds[i] = max(preds[i], 80)
        
        return np.clip(preds, 1, 90)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.FEATURES,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
