"""Base model class for FPL prediction models."""
from ..features import ManagerFeatureMixin
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from abc import ABC, abstractmethod


def log_exposure(minutes):
    """Poisson base margin for a per-90 rate: log(minutes / 90), minutes in [1, 90]."""
    minutes = pd.to_numeric(pd.Series(np.asarray(minutes, dtype=float)), errors='coerce')
    return np.log(minutes.fillna(90).clip(1, 90).to_numpy() / 90)


class ExposureXGBRegressor(xgb.XGBRegressor):
    """Tuning twin of ``BaseModel.EXPOSURE_OFFSET`` for sklearn CV.

    X's last two columns are the training base margin (actual minutes) and the
    inference base margin (``pred_minutes``), so each CV fold fits a rate on what
    was played and scores it on what was forecast.
    """

    def fit(self, X, y, **kwargs):
        X = np.asarray(X)
        return super().fit(X[:, :-2], y, base_margin=X[:, -2], **kwargs)

    def predict(self, X, **kwargs):
        X = np.asarray(X)
        return super().predict(X[:, :-2], base_margin=X[:, -1], **kwargs)


class BaseModel(ManagerFeatureMixin, ABC):
    """Abstract base class for FPL prediction models."""

    FEATURES = []
    TARGET = ''
    # Count models fitted as a per-90 rate: log(minutes / 90) is the base margin,
    # actual minutes in training and ``pred_minutes`` at inference. Without it the
    # minute-weighted fit learned starters' output and gave cameo players more
    # expected goals than starters (2.4x the actual rate at <=30 predicted minutes).
    EXPOSURE_OFFSET = False

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
        self.is_fitted = False
    
    @property
    def features_to_use(self):
        """Return selected features if available, otherwise all FEATURES."""
        return self.selected_features if self.selected_features else self.FEATURES
    
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix using selected features if available."""
        df = self.manager_features(df)
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        X = df[features].fillna(0).astype(float)
        return X
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train the model."""
        df = df[df['minutes'] >= 1].copy()
        
        self.fit_manager_features(df)
        X = self._prepare_X(df)
        y = df[self.TARGET].fillna(0).values
        y = np.clip(y, 0, self._get_y_max())
        
        # Weight by minutes (rate models need no weights: the offset carries exposure)
        weights = None if self.EXPOSURE_OFFSET else df['minutes'].values / df['minutes'].mean()

        if verbose:
            n_features = len(self.features_to_use)
            feature_info = f"({n_features} features"
            if self.selected_features:
                feature_info += ", tuned selection)"
            else:
                feature_info += ")"
            print(f"Training {self.__class__.__name__} on {len(X):,} samples {feature_info}...")
            print(f"  Target mean: {y.mean():.4f}")

        if self.EXPOSURE_OFFSET:
            self.model.fit(X, y, base_margin=log_exposure(df['minutes']))
        else:
            self.model.fit(X, y, sample_weight=weights)
        self.is_fitted = True

        if verbose:
            y_pred = self.predict(df)
            print(f"  MAE: {mean_absolute_error(y, y_pred):.4f}")
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = self._prepare_X(df)
        if self.EXPOSURE_OFFSET:
            exposure = df['pred_minutes'] if 'pred_minutes' in df.columns else df['minutes']
            return np.clip(self.model.predict(X, base_margin=log_exposure(exposure)),
                           0, self._get_y_max())
        return np.clip(self.model.predict(X), 0, self._get_y_max())
    
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
