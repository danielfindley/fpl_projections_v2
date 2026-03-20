"""Defensive contribution (defcon) prediction model — predicts raw match counts.

Uses Poisson objective for mean estimation (consistent even under overdispersion),
but Negative Binomial CDF for threshold probabilities to account for the heavy
overdispersion in defcon counts (var/mean ~ 2.5–3.3x).
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from .base import BaseModel


class DefconModel(BaseModel):
    """Predicts expected defensive contributions per match using Poisson objective on raw counts."""

    FEATURES = [
        # Raw defcon rolling counts
        'defcon_roll3', 'defcon_roll5', 'defcon_roll10',
        'defcon_last1',

        # Defcon per-90 rolling (rate features)
        'defcon_per90_roll5', 'defcon_per90_roll10',
        'hit_threshold_roll5', 'hit_threshold_roll10',

        # Component stats (rolling per-90 rates)
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

        # Interaction (defensive work x opponent attacking strength)
        'defcon_x_opp_xg',

        # Form trend
        'defcon_trend',

        # Predicted minutes (from MinutesModel — trained first)
        'pred_minutes',

        # Match context
        'is_home',
    ]

    TARGET = 'defcon'

    def __init__(self, **xgb_params):
        xgb_params.setdefault('objective', 'count:poisson')
        super().__init__(**xgb_params)
        self.dispersion_r = None

    def _get_y_max(self) -> float:
        return 30.0

    def _estimate_dispersion(self, df: pd.DataFrame):
        """Estimate Negative Binomial dispersion parameter r from training residuals.

        Uses Pearson chi-squared method: φ = Σ(y-μ)²/μ / (n-1).
        For NB: φ = 1 + mean(μ)/r, so r = mean(μ) / (φ - 1).
        """
        y = df[self.TARGET].fillna(0).values
        mu = np.maximum(self.predict(df), 0.01)

        pearson_chi2 = np.sum((y - mu) ** 2 / mu) / (len(y) - 1)

        if pearson_chi2 <= 1.0:
            self.dispersion_r = None
            return

        self.dispersion_r = float(np.mean(mu) / (pearson_chi2 - 1))
        self.dispersion_r = max(self.dispersion_r, 0.5)

    def fit(self, df: pd.DataFrame, verbose: bool = True):
        super().fit(df, verbose=verbose)

        train_df = df[df['minutes'] >= 1].copy()
        self._estimate_dispersion(train_df)

        if verbose:
            if self.dispersion_r is not None:
                print(f"  NB dispersion r={self.dispersion_r:.2f} "
                      f"(var/mean ≈ {1 + np.mean(self.predict(train_df)) / self.dispersion_r:.2f}x)")
            else:
                print("  No overdispersion detected, using Poisson CDF")

        return self

    def predict_threshold_prob(self, df, pred_minutes=None) -> np.ndarray:
        """Predict P(defcon >= threshold) using Negative Binomial distribution.

        Uses NB to account for overdispersion (var >> mean in defcon counts).
        Falls back to Poisson if no overdispersion was detected during fit.
        """
        expected = self.predict(df)
        expected = np.maximum(expected, 0.01)

        thresholds = np.where(df['is_def'] == 1, 10, 12)

        if self.dispersion_r is not None:
            r = self.dispersion_r
            p = r / (r + expected)
            probs = 1 - nbinom.cdf(thresholds - 1, r, p)
        else:
            from scipy.stats import poisson
            probs = 1 - poisson.cdf(thresholds - 1, expected)

        return np.clip(probs, 0, 1)
