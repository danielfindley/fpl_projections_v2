"""Assists prediction model — predicts raw match assist counts."""
import numpy as np
from .base import BaseModel


class AssistsModel(BaseModel):
    """Predicts expected assists per match using Poisson objective on raw counts."""

    FEATURES = [
        # Player xA/creativity
        'xa_per90_roll3', 'xa_per90_roll5', 'xa_per90_roll10',
        'key_passes_per90_roll3', 'key_passes_per90_roll5', 'key_passes_per90_roll10',
        'assists_per90_roll3', 'assists_per90_roll5', 'assists_per90_roll10',

        # Recent form
        'assists_last1', 'assists_roll3', 'assists_roll5',

        # Lifetime profile
        'lifetime_assists_per90', 'lifetime_xa_per90',

        # Player centrality (share of team output)
        'xg_share_roll5', 'shot_share_roll5',

        # Team context
        'team_goals_roll5', 'team_xg_roll5', 'team_goals_roll10',

        # Opponent attacking
        'opp_goals_roll5', 'opp_xg_roll5', 'opp_goals_roll10',

        # Opponent defensive weakness (more goals conceded = more assist opportunities)
        'opp_conceded_roll5', 'opp_xga_roll5', 'opp_conceded_roll10', 'opp_xga_roll10',

        # Opponent clean sheet rate (higher = harder to assist against)
        'opp_cs_rate_roll5', 'opp_cs_rate_roll10',

        # Interaction features (creativity x opponent weakness)
        'xa_x_opp_conceded', 'team_goals_x_opp_conceded',

        # Form trends (short-term vs medium-term momentum)
        'xa_trend', 'assists_trend',

        # xA over/underperformance (regression signal)
        'xa_overperformance_roll10',

        # Match-specific predicted team goals (from CleanSheetModel, leak-free OOF)
        'pred_team_goals',

        # Predicted minutes (from MinutesModel — trained first)
        'pred_minutes',

        # Match context
        'is_home',
    ]

    TARGET = 'assists'

    def __init__(self, **xgb_params):
        xgb_params.setdefault('objective', 'count:poisson')
        super().__init__(**xgb_params)

    def _get_y_max(self) -> float:
        return 4.0
