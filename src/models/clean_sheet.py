"""Clean sheet probability model (team-level)."""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score


class CleanSheetModel:
    """Predicts team clean sheet probability."""
    
    FEATURES = [
        # Team defensive history
        'team_conceded_roll5', 'team_conceded_roll10',
        'team_xga_roll5', 'team_xga_roll10',
        'team_cs_roll5', 'team_cs_roll10',
        
        # Opponent attacking history
        'opp_scored_roll5', 'opp_scored_roll10',
        'opp_xg_roll5', 'opp_xg_roll10',
        
        # Match context
        'is_home',
    ]
    
    TARGET = 'clean_sheet'
    
    def __init__(self, **xgb_params):
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss',
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self.is_fitted = False
    
    def prepare_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player data to team-match level and compute team features."""
        # Aggregate to team-match level
        team_match = df.groupby(['team', 'opponent', 'season', 'gameweek', 'is_home']).agg({
            'goals': 'sum',
            'xg': 'sum',
        }).reset_index()
        
        # Get goals conceded from opponent's goals
        opp_goals = team_match[['team', 'season', 'gameweek', 'goals', 'xg']].copy()
        opp_goals = opp_goals.rename(columns={
            'team': 'opponent',
            'goals': 'goals_conceded',
            'xg': 'xga'
        })
        team_match = team_match.merge(opp_goals, on=['opponent', 'season', 'gameweek'], how='left')
        
        team_match['clean_sheet'] = (team_match['goals_conceded'] == 0).astype(int)
        team_match = team_match.sort_values(['team', 'season', 'gameweek'])
        
        # Team defensive rolling
        for col, source in [('team_conceded', 'goals_conceded'), ('team_xga', 'xga')]:
            for window in [5, 10]:
                team_match[f'{col}_roll{window}'] = team_match.groupby('team')[source].transform(
                    lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                )
        
        # Clean sheet rate
        for window in [5, 10]:
            team_match[f'team_cs_roll{window}'] = team_match.groupby('team')['clean_sheet'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        # Opponent attacking rolling
        opp_attack = team_match.groupby('team').apply(
            lambda g: g.assign(
                opp_scored_roll5=g['goals'].shift(1).rolling(5, min_periods=1).mean(),
                opp_scored_roll10=g['goals'].shift(1).rolling(10, min_periods=1).mean(),
                opp_xg_team_roll5=g['xg'].shift(1).rolling(5, min_periods=1).mean(),
                opp_xg_team_roll10=g['xg'].shift(1).rolling(10, min_periods=1).mean(),
            )
        ).reset_index(drop=True)
        
        opp_lookup = opp_attack[['team', 'season', 'gameweek', 
                                  'opp_scored_roll5', 'opp_scored_roll10',
                                  'opp_xg_team_roll5', 'opp_xg_team_roll10']].copy()
        opp_lookup = opp_lookup.rename(columns={
            'team': 'opponent',
            'opp_xg_team_roll5': 'opp_xg_roll5',
            'opp_xg_team_roll10': 'opp_xg_roll10'
        })
        
        team_match = team_match.merge(opp_lookup, on=['opponent', 'season', 'gameweek'], how='left')
        
        return team_match
    
    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        for feat in self.FEATURES:
            if feat not in df.columns:
                df[feat] = 0
        return df[self.FEATURES].fillna(0).astype(float)
    
    def fit(self, df: pd.DataFrame, verbose: bool = True):
        """Train on team-match data."""
        team_df = self.prepare_team_features(df)
        team_df = team_df.dropna(subset=['team_conceded_roll5'])
        
        X = self._prepare_X(team_df)
        y = team_df['clean_sheet'].values
        
        if verbose:
            print(f"Training CleanSheetModel on {len(X):,} team-matches...")
            print(f"  CS rate: {y.mean():.1%}")
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        if verbose:
            y_pred = self.model.predict_proba(X)[:, 1]
            print(f"  AUC: {roc_auc_score(y, y_pred):.3f}")
        
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict clean sheet probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = self._prepare_X(df)
        return self.model.predict_proba(X)[:, 1]
    
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return pd.DataFrame({
            'feature': self.FEATURES,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
