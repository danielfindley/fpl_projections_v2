"""Goals per 90 prediction model."""
import numpy as np
from .base import BaseModel


class GoalsModel(BaseModel):
    """Predicts goals per 90 minutes rate."""
    
    FEATURES = [
        # Player xG/shooting
        'xg_per90_roll3', 'xg_per90_roll5', 'xg_per90_roll10',
        'shots_per90_roll3', 'shots_per90_roll5', 'shots_per90_roll10',
        'goals_per90_roll3', 'goals_per90_roll5', 'goals_per90_roll10',
        
        # Recent form
        'goals_last1', 'goals_roll3', 'goals_roll5',
        
        # Lifetime profile
        'lifetime_goals_per90', 'lifetime_xg_per90', 'lifetime_shots_per90',
        
        # Team context
        'team_goals_roll5', 'team_xg_roll5', 'team_goals_roll10',
        
        # Opponent attacking (defensive matchup context)
        'opp_goals_roll5', 'opp_xg_roll5', 'opp_goals_roll10',
        
        # Opponent defensive weakness (how many they concede = opportunity)
        'opp_conceded_roll5', 'opp_xga_roll5', 'opp_conceded_roll10', 'opp_xga_roll10',
        
        # Match-specific predicted team goals (from CleanSheetModel, leak-free OOF)
        'pred_team_goals',
        
        # Match context
        'is_home',
    ]
    
    TARGET = 'goals_per90'
    
    def _get_y_max(self) -> float:
        return 3.0
    
    def predict_expected(self, df, pred_minutes) -> np.ndarray:
        """Predict expected goals = per90 rate * (minutes/90)."""
        per90 = self.predict(df)
        return per90 * (np.array(pred_minutes) / 90)
