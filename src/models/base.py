"""Base model class for FPL prediction models."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for FPL prediction models."""
    
    FEATURES = []
    TARGET = ''
    
    def __init__(self, **xgb_params):
        # Extract selected_features if provided (from tuning)
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
        """Return selected features if available, otherwise all FEATURES."""
        return self.selected_features if self.selected_features else self.FEATURES
    
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix using selected features if available."""
        df = df.copy()
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        X = df[features].fillna(0).astype(float)
        return X
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the model."""
        df = df[df['minutes'] >= 1].copy()
        
        X = self._prepare_X(df)
        y = df[self.TARGET].fillna(0).values
        y = np.clip(y, 0, self._get_y_max())
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Weight by minutes
        weights = df['minutes'].values / df['minutes'].mean()
        
        if verbose:
            n_features = len(self.features_to_use)
            feature_info = f"({n_features} features"
            if self.selected_features:
                feature_info += ", tuned selection)"
            else:
                feature_info += ")"
            print(f"Training {self.__class__.__name__} on {len(X):,} samples {feature_info}...")
            print(f"  Target mean: {y.mean():.4f}")
        
        self.model.fit(X_scaled, y, sample_weight=weights)
        self.is_fitted = True
        
        if verbose:
            y_pred = self.model.predict(X_scaled)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.4f}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = self._prepare_X(df)
        X_scaled = self.scaler.transform(X)
        return np.clip(self.model.predict(X_scaled), 0, self._get_y_max())
    
    def feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.features_to_use,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
    @abstractmethod
    def _get_y_max(self) -> float:
        """Maximum value for target clipping."""
        pass
