"""
Feature engineering for FPL prediction.
All rolling features use shift(1) to prevent data leakage.
"""
import pandas as pd
import numpy as np


# Standard rolling windows for player-level features
ROLLING_WINDOWS = [1, 2, 3, 5, 7, 10]
# Extended windows for team-level stats (includes season-long trend)
ROLLING_WINDOWS_LONG = [1, 2, 3, 5, 7, 10, 30]

# Caps for per90 stats to prevent inflation from low-minutes appearances
# These represent maximum values a player could achieve per 90 minutes
PER90_CAPS = {
    'goals_per90': 3.0,
    'assists_per90': 3.0,
    'xg_per90': 2.0,
    'xa_per90': 1.5,
    'shots_per90': 8.0,
    'key_passes_per90': 6.0,
}

# Minimum minutes to compute meaningful per90 stats
MIN_MINUTES_FOR_PER90 = 20


def compute_rolling_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute all rolling features for prediction models."""
    if verbose:
        print("Computing rolling features...")

    df = df.copy()
    df = df.sort_values(['player_id', 'season', 'gameweek']).reset_index(drop=True)

    # Ensure minutes is numeric
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0)
    mins_90 = np.maximum(df['minutes'] / 90, 0.01)

    # Mask for games with enough minutes to compute meaningful per90
    sufficient_minutes = df['minutes'] >= MIN_MINUTES_FOR_PER90

    # =========================================================================
    # PLAYER ROLLING STATS
    # =========================================================================

    # Per-90 rates for current game (targets)
    # Only compute for games with sufficient minutes, otherwise NaN (will be skipped in rolling)
    for col in ['goals', 'assists', 'xg', 'xa', 'shots', 'key_passes']:
        if col in df.columns:
            per90_col = f'{col}_per90'
            # Compute raw per90
            df[per90_col] = np.where(sufficient_minutes, df[col] / mins_90, np.nan)
            # Apply cap to prevent inflation
            if col in PER90_CAPS:
                cap = PER90_CAPS[f'{col}_per90']
                df[per90_col] = df[per90_col].clip(upper=cap)

    # Rolling per-90 rates (shifted to avoid leakage)
    for col in ['goals', 'assists', 'xg', 'xa', 'shots', 'key_passes']:
        if col not in df.columns:
            df[col] = 0
        per90_col = f'{col}_per90'
        if per90_col not in df.columns:
            # Compute with cap and min minutes filter
            df[per90_col] = np.where(sufficient_minutes, df[col] / mins_90, np.nan)
            if col in PER90_CAPS:
                cap = PER90_CAPS[f'{col}_per90']
                df[per90_col] = df[per90_col].clip(upper=cap)

        for window in ROLLING_WINDOWS:
            # Rolling mean will skip NaN values (low-minute games)
            df[f'{col}_per90_roll{window}'] = df.groupby('player_id')[per90_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    # Minutes features
    df['last_minutes'] = df.groupby('player_id')['minutes'].shift(1)
    for window in ROLLING_WINDOWS:
        df[f'minutes_roll{window}'] = df.groupby('player_id')['minutes'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Starter indicators
    df['was_starter'] = (df['minutes'] >= 60).astype(int)
    df['was_full_90'] = (df['minutes'] >= 89).astype(int)
    df['last_was_starter'] = df.groupby('player_id')['was_starter'].shift(1).fillna(0)
    df['last_was_full_90'] = df.groupby('player_id')['was_full_90'].shift(1).fillna(0)

    for window in ROLLING_WINDOWS:
        df[f'starter_rate_roll{window}'] = df.groupby('player_id')['was_starter'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'full90_rate_roll{window}'] = df.groupby('player_id')['was_full_90'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Starter score (composite)
    df['starter_score'] = (
        (df['minutes_roll5'].fillna(60) / 90) * 0.4 +
        (df['starter_rate_roll5'].fillna(0.5)) * 0.4 +
        (df['last_was_full_90'].fillna(0.5)) * 0.2
    )

    # Current season features (to detect players with limited this-season playing time)
    # These are cumulative stats WITHIN the current season only
    df['current_season_minutes'] = df.groupby(['player_id', 'season'])['minutes'].transform('cumsum') - df['minutes']
    df['current_season_apps'] = df.groupby(['player_id', 'season']).cumcount()
    df['current_season_mins_per_app'] = df['current_season_minutes'] / df['current_season_apps'].replace(0, 1)

    # Recent form (raw counts)
    for col in ['goals', 'assists']:
        df[f'{col}_last1'] = df.groupby('player_id')[col].shift(1).fillna(0)
        for window in ROLLING_WINDOWS:
            df[f'{col}_roll{window}'] = df.groupby('player_id')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).sum()
            ).fillna(0)

    for window in ROLLING_WINDOWS:
        df[f'goal_involvements_roll{window}'] = df[f'goals_roll{window}'] + df[f'assists_roll{window}']

    # =========================================================================
    # FOULS COMMITTED (for card prediction)
    # =========================================================================

    if 'fouls_committed' not in df.columns:
        df['fouls_committed'] = 0
    else:
        df['fouls_committed'] = pd.to_numeric(df['fouls_committed'], errors='coerce').fillna(0)

    df['fouls_committed_per90'] = np.where(sufficient_minutes, df['fouls_committed'] / mins_90, np.nan)
    df['fouls_committed_per90'] = df['fouls_committed_per90'].clip(upper=6.0)

    for window in ROLLING_WINDOWS:
        df[f'fouls_committed_per90_roll{window}'] = df.groupby('player_id')['fouls_committed_per90'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # =========================================================================
    # YELLOW / RED CARDS (from FPL API merge, if available)
    # =========================================================================

    if 'yellow_cards' in df.columns:
        df['yellow_cards'] = pd.to_numeric(df['yellow_cards'], errors='coerce').fillna(0)
        for window in ROLLING_WINDOWS:
            df[f'yellow_cards_roll{window}'] = df.groupby('player_id')['yellow_cards'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        # Fouls per yellow (personalized booking rate) — only where fouls > 0
        df['_fouls_with_yellow'] = np.where(
            df['fouls_committed'] > 0,
            df['yellow_cards'] / df['fouls_committed'],
            np.nan
        )
        df['yellow_per_foul_roll10'] = df.groupby('player_id')['_fouls_with_yellow'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        )
        df = df.drop(columns=['_fouls_with_yellow'], errors='ignore')

    if 'red_cards' in df.columns:
        df['red_cards'] = pd.to_numeric(df['red_cards'], errors='coerce').fillna(0)

    # =========================================================================
    # GOALKEEPER STATS (Saves, xGoT faced)
    # =========================================================================

    for col in ['saves', 'xgot_faced']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Saves per 90 (only meaningful for GKs, but compute for all - model filters)
    df['saves_per90'] = np.where(sufficient_minutes, df['saves'] / mins_90, np.nan)
    df['saves_per90'] = df['saves_per90'].clip(upper=12.0)

    for window in ROLLING_WINDOWS:
        df[f'saves_per90_roll{window}'] = df.groupby('player_id')['saves_per90'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Raw saves rolling (recent form counts)
    df['saves_last1'] = df.groupby('player_id')['saves'].shift(1).fillna(0)
    for window in ROLLING_WINDOWS:
        df[f'saves_roll{window}'] = df.groupby('player_id')['saves'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # xGoT faced per 90 (shot quality faced by GK)
    df['xgot_faced_per90'] = np.where(sufficient_minutes, df['xgot_faced'] / mins_90, np.nan)
    df['xgot_faced_per90'] = df['xgot_faced_per90'].clip(upper=5.0)

    for window in ROLLING_WINDOWS:
        df[f'xgot_faced_per90_roll{window}'] = df.groupby('player_id')['xgot_faced_per90'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # =========================================================================
    # DEFENSIVE STATS (DEFCON)
    # =========================================================================

    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['CBIT'] = df['clearances'] + df['blocks'] + df['interceptions'] + df['tackles']
    df['CBIRT'] = df['CBIT'] + df['recoveries']

    # Position indicators (position may be int, float, or string)
    pos = df['position'].fillna(2).astype(float).astype(int)
    df['is_gk'] = (pos == 0).astype(int)
    df['is_def'] = (pos == 1).astype(int)
    df['is_mid'] = (pos == 2).astype(int)
    df['is_fwd'] = (pos == 3).astype(int)

    # Defcon based on position
    df['defcon'] = np.where(df['is_def'] == 1, df['CBIT'], df['CBIRT'])
    df['defcon_threshold'] = np.where(df['is_def'] == 1, 10, 12)
    df['hit_threshold'] = (df['defcon'] >= df['defcon_threshold']).astype(int)

    # Defcon per 90
    df['defcon_per90'] = df['defcon'] / mins_90
    for window in ROLLING_WINDOWS:
        df[f'defcon_per90_roll{window}'] = df.groupby('player_id')['defcon_per90'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'hit_threshold_roll{window}'] = df.groupby('player_id')['hit_threshold'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Raw defcon rolling counts (for Poisson count model)
    for window in ROLLING_WINDOWS:
        df[f'defcon_roll{window}'] = df.groupby('player_id')['defcon'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    df['defcon_last1'] = df.groupby('player_id')['defcon'].transform(lambda x: x.shift(1))

    # Component stats per 90
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
        df[f'{col}_per90'] = df[col] / mins_90
        for window in ROLLING_WINDOWS:
            df[f'{col}_per90_roll{window}'] = df.groupby('player_id')[f'{col}_per90'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    # =========================================================================
    # LIFETIME PLAYER PROFILE
    # =========================================================================

    df = df.sort_values(['player_name', 'season', 'gameweek']).reset_index(drop=True)

    # Offensive stats
    for stat in ['goals', 'assists', 'xg', 'xa', 'minutes', 'shots']:
        if stat in df.columns:
            df[f'lifetime_{stat}'] = df.groupby('player_name')[stat].transform(
                lambda x: x.shift(1).expanding().sum()
            )

    # Goalkeeper stats (lifetime)
    for stat in ['saves', 'xgot_faced']:
        if stat in df.columns:
            df[f'lifetime_{stat}'] = df.groupby('player_name')[stat].transform(
                lambda x: x.shift(1).expanding().sum()
            )

    # Defensive stats (lifetime)
    df['lifetime_defcon'] = df.groupby('player_name')['defcon'].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries', 'fouls_committed']:
        df[f'lifetime_{col}'] = df.groupby('player_name')[col].transform(
            lambda x: x.shift(1).expanding().sum()
        )

    df['lifetime_minutes'] = df['lifetime_minutes'].fillna(0)
    lifetime_mins_90 = np.maximum(df['lifetime_minutes'] / 90, 0.01)

    # Per-90 rates for lifetime stats (with caps)
    for stat in ['goals', 'assists', 'xg', 'xa', 'shots']:
        if f'lifetime_{stat}' in df.columns:
            per90_col = f'lifetime_{stat}_per90'
            df[per90_col] = np.where(
                df['lifetime_minutes'] >= 90,
                df[f'lifetime_{stat}'] / lifetime_mins_90,
                0
            )
            # Apply cap to lifetime per90 as well
            cap_key = f'{stat}_per90'
            if cap_key in PER90_CAPS:
                df[per90_col] = df[per90_col].clip(upper=PER90_CAPS[cap_key])

    # Lifetime defensive per 90
    df['lifetime_defcon_per90'] = np.where(
        df['lifetime_minutes'] >= 90,
        df['lifetime_defcon'].fillna(0) / lifetime_mins_90,
        0
    )
    for col in ['tackles', 'interceptions', 'clearances', 'fouls_committed']:
        df[f'lifetime_{col}_per90'] = np.where(
            df['lifetime_minutes'] >= 90,
            df[f'lifetime_{col}'].fillna(0) / lifetime_mins_90,
            0
        )

    # Lifetime yellow card per 90 (if yellow_cards column exists)
    if 'yellow_cards' in df.columns:
        df['lifetime_yellow_cards'] = df.groupby('player_name')['yellow_cards'].transform(
            lambda x: x.shift(1).expanding().sum()
        ).fillna(0)
        df['lifetime_yellow_cards_per90'] = np.where(
            df['lifetime_minutes'] >= 90,
            df['lifetime_yellow_cards'] / lifetime_mins_90,
            0
        )

    # Lifetime goalkeeper per 90
    df['lifetime_saves_per90'] = np.where(
        df['lifetime_minutes'] >= 90,
        df['lifetime_saves'].fillna(0) / lifetime_mins_90,
        0
    )

    df['lifetime_appearances'] = df.groupby('player_name').cumcount()
    df['lifetime_mins_per_app'] = np.where(
        df['lifetime_appearances'] > 0,
        df['lifetime_minutes'] / df['lifetime_appearances'],
        0
    )

    # =========================================================================
    # TEAM ROLLING STATS (OFFENSIVE)
    # =========================================================================

    # Ensure own_goal column exists
    if 'own_goal' not in df.columns:
        df['own_goal'] = 0

    team_stats = df.groupby(['team', 'season', 'gameweek']).agg({
        'goals': 'sum', 'xg': 'sum', 'shots': 'sum', 'own_goal': 'sum'
    }).reset_index()

    # Add opponent own goals to team goals (own goals by opponent count as goals for this team)
    if 'opponent' in df.columns:
        opp_og = df.groupby(['opponent', 'season', 'gameweek'])['own_goal'].sum().reset_index()
        opp_og = opp_og.rename(columns={'opponent': 'team', 'own_goal': 'opp_own_goals'})
        team_stats = team_stats.merge(opp_og, on=['team', 'season', 'gameweek'], how='left')
        team_stats['opp_own_goals'] = team_stats['opp_own_goals'].fillna(0)
        team_stats['team_goals'] = team_stats['goals'] + team_stats['opp_own_goals']
    else:
        team_stats['team_goals'] = team_stats['goals']

    team_stats = team_stats.rename(columns={'xg': 'team_xg', 'shots': 'team_shots'})
    team_stats = team_stats.drop(columns=['goals', 'own_goal'], errors='ignore')
    team_stats = team_stats.sort_values(['team', 'season', 'gameweek'])

    for col in ['team_goals', 'team_xg', 'team_shots']:
        for window in ROLLING_WINDOWS:
            team_stats[f'{col}_roll{window}'] = team_stats.groupby('team')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    # Merge rolling team offensive stats (dynamic column selection)
    team_roll_cols = [c for c in team_stats.columns if '_roll' in c]
    df = df.merge(
        team_stats[['team', 'season', 'gameweek'] + team_roll_cols],
        on=['team', 'season', 'gameweek'], how='left'
    )

    # =========================================================================
    # PLAYER SHARE / CENTRALITY FEATURES
    # =========================================================================

    # Merge raw per-match team totals for share computation
    df = df.merge(
        team_stats[['team', 'season', 'gameweek', 'team_goals', 'team_xg', 'team_shots']],
        on=['team', 'season', 'gameweek'], how='left'
    )

    # Per-match share ratios (only for games with sufficient minutes and non-zero team totals)
    for player_col, team_col, share_name in [
        ('xg', 'team_xg', 'xg_share'),
        ('shots', 'team_shots', 'shot_share'),
        ('goals', 'team_goals', 'goal_share'),
    ]:
        df[share_name] = np.where(
            sufficient_minutes & (df[team_col] > 0),
            df[player_col] / df[team_col].clip(lower=0.1),
            np.nan
        )

    # Rolling share features (shifted to prevent leakage)
    for share_col in ['xg_share', 'shot_share', 'goal_share']:
        for window in ROLLING_WINDOWS:
            df[f'{share_col}_roll{window}'] = df.groupby('player_id')[share_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

    # Drop intermediate columns
    df = df.drop(columns=['team_goals', 'team_xg', 'team_shots',
                          'xg_share', 'shot_share', 'goal_share'], errors='ignore')

    # =========================================================================
    # TEAM DEFENSIVE STATS (Goals Conceded, xGA)
    # =========================================================================

    # Normalize team names for consistent matching across all team-related features
    def normalize_name(name):
        if pd.isna(name):
            return ''
        return str(name).lower().replace(' ', '_').replace("'", "").strip()

    if 'opponent' in df.columns:
        # Create normalized versions for matching
        df['team_norm'] = df['team'].apply(normalize_name)
        df['opponent_norm'] = df['opponent'].apply(normalize_name)

        # Get goals conceded by each team (= opponent's goals + this team's own goals)
        match_results = df.groupby(['team_norm', 'opponent_norm', 'season', 'gameweek']).agg({
            'goals': 'sum', 'xg': 'sum', 'own_goal': 'sum', 'team': 'first', 'opponent': 'first'
        }).reset_index()

        # Swap to get what the opponent scored against this team
        # After swap: team_norm becomes what was opponent_norm, so we get each team's conceded stats
        team_conceded = match_results.rename(columns={
            'team_norm': 'opponent_temp_norm',
            'opponent_norm': 'team_norm',
            'goals': 'opp_goals',
            'xg': 'xga',
            'own_goal': 'opp_own_goals',
        })

        # Goals conceded = opponent's player goals + this team's own goals
        # own goals by this team are in the original (pre-swap) match_results
        team_og = match_results[['team_norm', 'season', 'gameweek', 'own_goal']].rename(
            columns={'own_goal': 'self_own_goals'}
        )
        team_conceded = team_conceded.merge(team_og, on=['team_norm', 'season', 'gameweek'], how='left')
        team_conceded['self_own_goals'] = team_conceded['self_own_goals'].fillna(0)
        team_conceded['goals_conceded'] = team_conceded['opp_goals'] + team_conceded['self_own_goals']
        team_conceded = team_conceded.sort_values(['team_norm', 'season', 'gameweek'])

        # Rolling goals conceded and xGA (multiple windows for different time horizons)
        for window in ROLLING_WINDOWS_LONG:
            team_conceded[f'team_conceded_roll{window}'] = team_conceded.groupby('team_norm')['goals_conceded'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            team_conceded[f'team_xga_roll{window}'] = team_conceded.groupby('team_norm')['xga'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Clean sheet tracking
        team_conceded['clean_sheet'] = (team_conceded['goals_conceded'] == 0).astype(int)
        for window in ROLLING_WINDOWS_LONG:
            team_conceded[f'team_cs_rate_roll{window}'] = team_conceded.groupby('team_norm')['clean_sheet'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Merge team defensive stats (dynamic column selection)
        team_def_cols = [c for c in team_conceded.columns
                        if any(c.startswith(p) for p in ['team_conceded_roll', 'team_xga_roll', 'team_cs_rate_roll'])]
        df = df.merge(
            team_conceded[['team_norm', 'season', 'gameweek'] + team_def_cols],
            on=['team_norm', 'season', 'gameweek'], how='left'
        )

    # =========================================================================
    # OPPONENT ROLLING STATS (Offensive + Defensive)
    # =========================================================================

    if 'opponent' in df.columns:
        # Add normalized team name to team_stats for opponent matching
        team_stats['team_norm'] = team_stats['team'].apply(normalize_name)

        # Opponent's goals scored (their attacking strength)
        opp_offense = team_stats[['team_norm', 'season', 'gameweek', 'team_goals', 'team_xg']].copy()
        opp_offense = opp_offense.rename(columns={
            'team_norm': 'opponent_norm',
            'team_goals': 'opp_goals',
            'team_xg': 'opp_xg'
        })
        opp_offense = opp_offense.sort_values(['opponent_norm', 'season', 'gameweek'])

        for window in ROLLING_WINDOWS:
            opp_offense[f'opp_goals_roll{window}'] = opp_offense.groupby('opponent_norm')['opp_goals'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            opp_offense[f'opp_xg_roll{window}'] = opp_offense.groupby('opponent_norm')['opp_xg'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        opp_off_roll_cols = [c for c in opp_offense.columns if c.startswith('opp_') and '_roll' in c]
        df = df.merge(
            opp_offense[['opponent_norm', 'season', 'gameweek'] + opp_off_roll_cols],
            on=['opponent_norm', 'season', 'gameweek'], how='left'
        )

        # Opponent's defensive weakness (goals they concede = xGA) + CS rate
        opp_def_roll_cols = [c for c in team_conceded.columns
                           if any(c.startswith(p) for p in ['team_conceded_roll', 'team_xga_roll', 'team_cs_rate_roll'])]
        opp_defense = team_conceded[['team_norm', 'season', 'gameweek'] + opp_def_roll_cols].copy()
        rename_map = {'team_norm': 'opponent_norm'}
        for col in opp_def_roll_cols:
            new_name = (col.replace('team_conceded_', 'opp_conceded_')
                           .replace('team_xga_', 'opp_xga_')
                           .replace('team_cs_rate_', 'opp_cs_rate_'))
            rename_map[col] = new_name
        opp_defense = opp_defense.rename(columns=rename_map)

        opp_def_merged_cols = [c for c in opp_defense.columns if c.startswith('opp_')]
        df = df.merge(
            opp_defense[['opponent_norm', 'season', 'gameweek'] + opp_def_merged_cols],
            on=['opponent_norm', 'season', 'gameweek'], how='left'
        )

        # Clean up temporary normalized columns
        df = df.drop(columns=['team_norm', 'opponent_norm'], errors='ignore')

    # =========================================================================
    # FORM TREND FEATURES (short-term vs medium-term momentum)
    # =========================================================================

    df['xg_trend'] = df['xg_per90_roll3'] - df['xg_per90_roll10']
    df['goals_trend'] = df['goals_per90_roll3'] - df['goals_per90_roll10']
    df['xa_trend'] = df['xa_per90_roll3'] - df['xa_per90_roll10']
    df['assists_trend'] = df['assists_per90_roll3'] - df['assists_per90_roll10']
    df['minutes_trend'] = df['minutes_roll3'] - df['minutes_roll10']
    df['defcon_trend'] = df['defcon_per90_roll5'] - df['defcon_per90_roll10']

    # =========================================================================
    # xG OVER/UNDERPERFORMANCE (regression-to-mean signal)
    # =========================================================================

    df['xg_overperformance_roll10'] = df['goals_per90_roll10'] - df['xg_per90_roll10']
    df['xa_overperformance_roll10'] = df['assists_per90_roll10'] - df['xa_per90_roll10']
    df['lifetime_xg_overperformance'] = df['lifetime_goals_per90'] - df['lifetime_xg_per90']

    # =========================================================================
    # INTERACTION FEATURES (player ability x opponent weakness)
    # =========================================================================

    df['xg_x_opp_conceded'] = df['xg_per90_roll5'] * df['opp_conceded_roll5']
    df['xa_x_opp_conceded'] = df['xa_per90_roll5'] * df['opp_conceded_roll5']
    df['team_goals_x_opp_conceded'] = df['team_goals_roll5'] * df['opp_conceded_roll5']
    df['defcon_x_opp_xg'] = df['defcon_per90_roll5'] * df['opp_xg_roll5']

    # =========================================================================
    # CLEAN UP
    # =========================================================================

    df = df.drop(columns=['was_starter', 'was_full_90'], errors='ignore')

    # Fill NaN in rolling/derived columns
    rolling_cols = [c for c in df.columns if 'roll' in c or 'lifetime' in c or 'last_' in c
                    or 'trend' in c or 'overperformance' in c or '_x_' in c]
    for col in rolling_cols:
        df[col] = df[col].fillna(0)

    # Fill other defaults
    df['is_home'] = df['is_home'].fillna(0).astype(int)
    df['starter_score'] = df['starter_score'].fillna(0.5)

    if verbose:
        print(f"  Computed {len(rolling_cols)} rolling/lifetime features")

    return df
