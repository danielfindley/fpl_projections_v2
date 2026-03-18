"""Goals prediction model — predicts raw match goal counts."""
import numpy as np
from .base import BaseModel


class GoalsModel(BaseModel):
    """Predicts expected goals per match using Poisson objective on raw counts."""

    FEATURES = [
        # Player xG/shooting
        'xg_per90_roll3', 'xg_per90_roll5', 'xg_per90_roll10',
        'shots_per90_roll3', 'shots_per90_roll5', 'shots_per90_roll10',
        'goals_per90_roll3', 'goals_per90_roll5', 'goals_per90_roll10',

        # Recent form
        'goals_last1', 'goals_roll3', 'goals_roll5',

        # Lifetime profile
        'lifetime_goals_per90', 'lifetime_xg_per90', 'lifetime_shots_per90',

        # Player centrality (share of team output)
        'xg_share_roll5', 'shot_share_roll5', 'goal_share_roll5',

        # Team context
        'team_goals_roll5', 'team_xg_roll5', 'team_goals_roll10',

        # Opponent attacking (defensive matchup context)
        'opp_goals_roll5', 'opp_xg_roll5', 'opp_goals_roll10',

        # Opponent defensive weakness (how many they concede = opportunity)
        'opp_conceded_roll5', 'opp_xga_roll5', 'opp_conceded_roll10', 'opp_xga_roll10',

        # Opponent clean sheet rate (higher = harder to score against)
        'opp_cs_rate_roll5', 'opp_cs_rate_roll10',

        # Interaction features (player ability x opponent weakness)
        'xg_x_opp_conceded', 'team_goals_x_opp_conceded',

        # Form trends (short-term vs medium-term momentum)
        'xg_trend', 'goals_trend',

        # xG over/underperformance (finishing quality / regression signal)
        'xg_overperformance_roll10', 'lifetime_xg_overperformance',

        # Match-specific predicted team goals (from CleanSheetModel, leak-free OOF)
        'pred_team_goals',

        # Predicted minutes (from MinutesModel — trained first)
        'pred_minutes',

        # Match context
        'is_home',
    ]

    TARGET = 'goals'

    def __init__(self, **xgb_params):
        xgb_params.setdefault('objective', 'count:poisson')
        super().__init__(**xgb_params)

    def _get_y_max(self) -> float:
        return 4.0
