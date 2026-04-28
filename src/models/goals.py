"""Goals prediction model — predicts raw match goal counts."""
import numpy as np
from .base import BaseModel


class GoalsModel(BaseModel):
    """Predicts expected goals per match using Poisson objective on raw counts."""

    FEATURES = [
        # Player xG/shooting
        'xg_per90_roll1', 'xg_per90_roll2', 'xg_per90_roll3', 'xg_per90_roll5', 'xg_per90_roll7', 'xg_per90_roll10',
        'shots_per90_roll1', 'shots_per90_roll2', 'shots_per90_roll3', 'shots_per90_roll5', 'shots_per90_roll7', 'shots_per90_roll10',
        'goals_per90_roll1', 'goals_per90_roll2', 'goals_per90_roll3', 'goals_per90_roll5', 'goals_per90_roll7', 'goals_per90_roll10',

        # Recent form (raw count sums over window)
        'goals_last1', 'goals_roll1', 'goals_roll2', 'goals_roll3', 'goals_roll5', 'goals_roll7', 'goals_roll10',

        # Lifetime profile
        'lifetime_goals_per90', 'lifetime_xg_per90', 'lifetime_shots_per90',

        # Player centrality (share of team output)
        'xg_share_roll1', 'xg_share_roll2', 'xg_share_roll3', 'xg_share_roll5', 'xg_share_roll7', 'xg_share_roll10',
        'shot_share_roll1', 'shot_share_roll2', 'shot_share_roll3', 'shot_share_roll5', 'shot_share_roll7', 'shot_share_roll10',
        'goal_share_roll1', 'goal_share_roll2', 'goal_share_roll3', 'goal_share_roll5', 'goal_share_roll7', 'goal_share_roll10',

        # Team context
        'team_goals_roll1', 'team_goals_roll2', 'team_goals_roll3', 'team_goals_roll5', 'team_goals_roll7', 'team_goals_roll10',
        'team_xg_roll1', 'team_xg_roll2', 'team_xg_roll3', 'team_xg_roll5', 'team_xg_roll7', 'team_xg_roll10',

        # Opponent attacking (defensive matchup context)
        'opp_goals_roll1', 'opp_goals_roll2', 'opp_goals_roll3', 'opp_goals_roll5', 'opp_goals_roll7', 'opp_goals_roll10',
        'opp_xg_roll1', 'opp_xg_roll2', 'opp_xg_roll3', 'opp_xg_roll5', 'opp_xg_roll7', 'opp_xg_roll10',

        # Opponent defensive weakness (how many they concede = opportunity)
        'opp_conceded_roll1', 'opp_conceded_roll2', 'opp_conceded_roll3', 'opp_conceded_roll5', 'opp_conceded_roll7', 'opp_conceded_roll10',
        'opp_xga_roll1', 'opp_xga_roll2', 'opp_xga_roll3', 'opp_xga_roll5', 'opp_xga_roll7', 'opp_xga_roll10',

        # Opponent clean sheet rate (higher = harder to score against)
        'opp_cs_rate_roll1', 'opp_cs_rate_roll2', 'opp_cs_rate_roll3', 'opp_cs_rate_roll5', 'opp_cs_rate_roll7', 'opp_cs_rate_roll10',

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

        # Manager embeddings (8-dim PCA over rolling-20-prior manager stats: minutes/GF/GA/formation)
        'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
        'manager_emb_4', 'manager_emb_5', 'manager_emb_6', 'manager_emb_7',
    ]

    TARGET = 'goals'

    def __init__(self, **xgb_params):
        xgb_params.setdefault('objective', 'count:poisson')
        super().__init__(**xgb_params)

    def _get_y_max(self) -> float:
        return 4.0
