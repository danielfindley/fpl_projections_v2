"""Assists prediction model — predicts raw match assist counts."""
import numpy as np
from .base import BaseModel


class AssistsModel(BaseModel):
    """Predicts expected assists per match using Poisson objective on raw counts."""

    FEATURES = [
        # Player xA/creativity
        'xa_per90_roll1', 'xa_per90_roll2', 'xa_per90_roll3', 'xa_per90_roll5', 'xa_per90_roll7', 'xa_per90_roll10',
        'key_passes_per90_roll1', 'key_passes_per90_roll2', 'key_passes_per90_roll3', 'key_passes_per90_roll5', 'key_passes_per90_roll7', 'key_passes_per90_roll10',
        'assists_per90_roll1', 'assists_per90_roll2', 'assists_per90_roll3', 'assists_per90_roll5', 'assists_per90_roll7', 'assists_per90_roll10',

        # Recent form (raw count sums over window)
        'assists_last1', 'assists_roll1', 'assists_roll2', 'assists_roll3', 'assists_roll5', 'assists_roll7', 'assists_roll10',

        # Lifetime profile
        'lifetime_assists_per90', 'lifetime_xa_per90',

        # Player centrality (share of team output)
        'xg_share_roll1', 'xg_share_roll2', 'xg_share_roll3', 'xg_share_roll5', 'xg_share_roll7', 'xg_share_roll10',
        'shot_share_roll1', 'shot_share_roll2', 'shot_share_roll3', 'shot_share_roll5', 'shot_share_roll7', 'shot_share_roll10',

        # Team context
        'team_goals_roll1', 'team_goals_roll2', 'team_goals_roll3', 'team_goals_roll5', 'team_goals_roll7', 'team_goals_roll10',
        'team_xg_roll1', 'team_xg_roll2', 'team_xg_roll3', 'team_xg_roll5', 'team_xg_roll7', 'team_xg_roll10',

        # Opponent attacking
        'opp_goals_roll1', 'opp_goals_roll2', 'opp_goals_roll3', 'opp_goals_roll5', 'opp_goals_roll7', 'opp_goals_roll10',
        'opp_xg_roll1', 'opp_xg_roll2', 'opp_xg_roll3', 'opp_xg_roll5', 'opp_xg_roll7', 'opp_xg_roll10',

        # Opponent defensive weakness (more goals conceded = more assist opportunities)
        'opp_conceded_roll1', 'opp_conceded_roll2', 'opp_conceded_roll3', 'opp_conceded_roll5', 'opp_conceded_roll7', 'opp_conceded_roll10',
        'opp_xga_roll1', 'opp_xga_roll2', 'opp_xga_roll3', 'opp_xga_roll5', 'opp_xga_roll7', 'opp_xga_roll10',

        # Opponent clean sheet rate (higher = harder to assist against)
        'opp_cs_rate_roll1', 'opp_cs_rate_roll2', 'opp_cs_rate_roll3', 'opp_cs_rate_roll5', 'opp_cs_rate_roll7', 'opp_cs_rate_roll10',

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

        # Manager embeddings (8-dim PCA over rolling-20-prior manager stats)
        'manager_emb_0', 'manager_emb_1', 'manager_emb_2', 'manager_emb_3',
        'manager_emb_4', 'manager_emb_5', 'manager_emb_6', 'manager_emb_7',
    ]

    TARGET = 'assists'

    def __init__(self, **xgb_params):
        xgb_params.setdefault('objective', 'count:poisson')
        super().__init__(**xgb_params)

    def _get_y_max(self) -> float:
        return 4.0
