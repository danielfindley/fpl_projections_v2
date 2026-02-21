"""Goals Against model (team-level) - predicts expected goals conceded per match.
Uses Poisson distribution to derive clean sheet probability and 2+ conceded probability.
Includes isotonic calibration for CS probabilities.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from sklearn.metrics import mean_absolute_error


class CleanSheetModel:
    """Predicts expected goals against per match, then derives:
    - P(clean sheet) = P(goals_against = 0) via Poisson
    - P(2+ conceded) = 1 - P(goals_against <= 1) via Poisson

    Uses isotonic calibration to map raw Poisson CS probabilities to
    calibrated probabilities that match observed CS rates.
    """

    FEATURES = [
        # Team defensive history (multiple time horizons)
        'team_conceded_roll1', 'team_conceded_roll3', 'team_conceded_roll5', 'team_conceded_roll10', 'team_conceded_roll30',
        'team_xga_roll1', 'team_xga_roll3', 'team_xga_roll5', 'team_xga_roll10', 'team_xga_roll30',
        'team_cs_roll1', 'team_cs_roll3', 'team_cs_roll5', 'team_cs_roll10', 'team_cs_roll30',

        # Opponent attacking history
        'opp_scored_roll5', 'opp_scored_roll10',
        'opp_xg_roll5', 'opp_xg_roll10',

        # Match context
        'is_home',
    ]

    TARGET = 'goals_conceded'

    def __init__(self, **xgb_params):
        # Extract selected_features if provided (from tuning)
        self.selected_features = xgb_params.pop('selected_features', None)

        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'objective': 'count:poisson',  # Poisson regression for count data
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBRegressor(**default_params)
        self.is_fitted = False
        self.cs_calibrator = None
    
    @property
    def features_to_use(self):
        """Return selected features if available, otherwise all FEATURES."""
        return self.selected_features if self.selected_features else self.FEATURES
    
    @staticmethod
    def _normalize_name(name):
        """Normalize team name for consistent matching."""
        if pd.isna(name):
            return ''
        return str(name).lower().replace(' ', '_').replace("'", "").strip()
    
    def prepare_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player data to team-match level and compute team features."""
        # Add normalized names for reliable matching
        df = df.copy()
        df['team_norm'] = df['team'].apply(self._normalize_name)
        df['opponent_norm'] = df['opponent'].apply(self._normalize_name)
        
        # Aggregate to team-match level
        team_match = df.groupby(['team_norm', 'opponent_norm', 'season', 'gameweek', 'is_home']).agg({
            'goals': 'sum',
            'xg': 'sum',
            'team': 'first',
            'opponent': 'first',
        }).reset_index()
        
        # Get goals conceded from opponent's goals (swap team_norm <-> opponent_norm)
        opp_goals = team_match[['team_norm', 'season', 'gameweek', 'goals', 'xg']].copy()
        opp_goals = opp_goals.rename(columns={
            'team_norm': 'opponent_norm',
            'goals': 'goals_conceded',
            'xg': 'xga'
        })
        team_match = team_match.merge(opp_goals, on=['opponent_norm', 'season', 'gameweek'], how='left')
        
        team_match['clean_sheet'] = (team_match['goals_conceded'] == 0).astype(int)
        team_match = team_match.sort_values(['team_norm', 'season', 'gameweek'])
        
        # Team defensive rolling (multiple time horizons)
        for col, source in [('team_conceded', 'goals_conceded'), ('team_xga', 'xga')]:
            for window in [1, 3, 5, 10, 30]:
                team_match[f'{col}_roll{window}'] = team_match.groupby('team_norm')[source].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
        
        # Clean sheet rate
        for window in [1, 3, 5, 10, 30]:
            team_match[f'team_cs_roll{window}'] = team_match.groupby('team_norm')['clean_sheet'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        # Opponent attacking rolling (what the opponent scores on average)
        opp_attack = team_match.groupby('opponent_norm').apply(
            lambda g: g.assign(
                opp_scored_roll5=g['goals_conceded'].shift(1).rolling(5, min_periods=1).mean(),
                opp_scored_roll10=g['goals_conceded'].shift(1).rolling(10, min_periods=1).mean(),
                opp_xg_team_roll5=g['xga'].shift(1).rolling(5, min_periods=1).mean(),
                opp_xg_team_roll10=g['xga'].shift(1).rolling(10, min_periods=1).mean(),
            )
        ).reset_index(drop=True)
        
        # Actually we need opponent's OWN attacking stats, so look them up via team_norm
        opp_offense = team_match.groupby('team_norm').apply(
            lambda g: g.assign(
                team_scored_roll5=g['goals'].shift(1).rolling(5, min_periods=1).mean(),
                team_scored_roll10=g['goals'].shift(1).rolling(10, min_periods=1).mean(),
                team_xg_off_roll5=g['xg'].shift(1).rolling(5, min_periods=1).mean(),
                team_xg_off_roll10=g['xg'].shift(1).rolling(10, min_periods=1).mean(),
            )
        ).reset_index(drop=True)
        
        opp_lookup = opp_offense[['team_norm', 'season', 'gameweek', 
                                   'team_scored_roll5', 'team_scored_roll10',
                                   'team_xg_off_roll5', 'team_xg_off_roll10']].copy()
        opp_lookup = opp_lookup.rename(columns={
            'team_norm': 'opponent_norm',
            'team_scored_roll5': 'opp_scored_roll5',
            'team_scored_roll10': 'opp_scored_roll10',
            'team_xg_off_roll5': 'opp_xg_roll5',
            'team_xg_off_roll10': 'opp_xg_roll10',
        })
        
        team_match = team_match.merge(opp_lookup, on=['opponent_norm', 'season', 'gameweek'], how='left')
        
        # Drop temp columns
        team_match = team_match.drop(columns=['team_norm', 'opponent_norm'], errors='ignore')
        
        return team_match
    
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        features = self.features_to_use
        for feat in features:
            if feat not in df.columns:
                df[feat] = 0
        return df[features].fillna(0).astype(float)
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on team-match data to predict goals conceded.

        After fitting the XGBoost model, runs isotonic calibration on
        OOF CS probabilities to correct systematic bias in P(CS) = e^(-lambda).
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.isotonic import IsotonicRegression

        team_df = self.prepare_team_features(df)
        team_df = team_df.dropna(subset=['team_conceded_roll5', 'goals_conceded'])

        # Ensure temporal ordering for OOF calibration
        team_df = team_df.sort_values(['season', 'gameweek']).reset_index(drop=True)

        X = self._prepare_X(team_df)
        y = team_df['goals_conceded'].fillna(0).values

        if verbose:
            n_features = len(self.features_to_use)
            feature_info = f"({n_features} features"
            if self.selected_features:
                feature_info += ", tuned selection)"
            else:
                feature_info += ")"
            print(f"Training CleanSheetModel (Goals Against) on {len(X):,} team-matches {feature_info}...")
            print(f"  Avg goals conceded: {y.mean():.3f}")
            print(f"  Actual CS rate: {(y == 0).mean():.1%}")

        # Step 1: Generate OOF predictions for calibration (before fitting on all data)
        tscv = TimeSeriesSplit(n_splits=5)
        oof_lambda = np.full(len(y), np.nan)

        for train_idx, val_idx in tscv.split(X):
            fold_model = xgb.XGBRegressor(**self.model.get_params())
            fold_model.fit(X.values[train_idx] if hasattr(X, 'values') else X[train_idx],
                          y[train_idx])
            X_val = X.values[val_idx] if hasattr(X, 'values') else X[val_idx]
            oof_lambda[val_idx] = np.clip(fold_model.predict(X_val), 1e-6, 10.0)

        # Only calibrate on rows that have OOF predictions (TimeSeriesSplit skips first fold)
        valid_mask = ~np.isnan(oof_lambda)
        if valid_mask.sum() > 50:
            raw_cs_prob = poisson.pmf(0, oof_lambda[valid_mask])
            actual_cs = (y[valid_mask] == 0).astype(float)

            self.cs_calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            self.cs_calibrator.fit(raw_cs_prob, actual_cs)

            if verbose:
                calibrated_cs = self.cs_calibrator.predict(raw_cs_prob)
                print(f"  Calibration: raw CS prob mean={raw_cs_prob.mean():.3f}, "
                      f"calibrated={calibrated_cs.mean():.3f}, actual={actual_cs.mean():.3f}")
        else:
            self.cs_calibrator = None
            if verbose:
                print("  Skipping calibration (insufficient OOF data)")

        # Step 2: Fit final model on ALL data
        self.model.fit(X.values if hasattr(X, 'values') else X, y)
        self.is_fitted = True

        if verbose:
            y_pred = self.predict_goals_against(team_df)
            cs_probs = self.predict_cs_prob(team_df)
            two_plus_probs = self.predict_2plus_conceded_prob(team_df)
            print(f"  MAE (goals against): {mean_absolute_error(y, y_pred):.3f}")
            print(f"  Predicted CS prob (mean): {cs_probs.mean():.1%}")
            print(f"  Predicted 2+ conceded prob (mean): {two_plus_probs.mean():.1%}")

        return self
    
    def predict_goals_against(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected goals against (lambda for Poisson)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = self._prepare_X(df)
        # Poisson regression output is already the rate (lambda), clip to avoid edge cases
        return np.clip(self.model.predict(X), 1e-6, 10.0)
    
    def predict_cs_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Predict clean sheet probability: P(goals_against = 0) = e^(-lambda).

        If isotonic calibrator is available, applies calibration to raw probabilities.
        """
        lambda_pred = self.predict_goals_against(df)
        raw_cs_prob = poisson.pmf(0, lambda_pred)

        if self.cs_calibrator is not None:
            return self.cs_calibrator.predict(raw_cs_prob)
        return raw_cs_prob
    
    def predict_2plus_conceded_prob(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of conceding 2+ goals: 1 - P(0) - P(1)."""
        lambda_pred = self.predict_goals_against(df)
        return 1.0 - poisson.cdf(1, lambda_pred)
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.features_to_use,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
