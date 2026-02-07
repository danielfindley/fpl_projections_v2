"""
Feature engineering for FPL prediction.
All rolling features use shift(1) to prevent data leakage.
"""
import pandas as pd
import numpy as np


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
        
        for window in [3, 5, 10]:
            # Rolling mean will skip NaN values (low-minute games)
            df[f'{col}_per90_roll{window}'] = df.groupby('player_id')[per90_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
    
    # Minutes features
    df['last_minutes'] = df.groupby('player_id')['minutes'].shift(1)
    for window in [3, 5, 10]:
        df[f'minutes_roll{window}'] = df.groupby('player_id')['minutes'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    # Starter indicators
    df['was_starter'] = (df['minutes'] >= 60).astype(int)
    df['was_full_90'] = (df['minutes'] >= 89).astype(int)
    df['last_was_starter'] = df.groupby('player_id')['was_starter'].shift(1).fillna(0)
    df['last_was_full_90'] = df.groupby('player_id')['was_full_90'].shift(1).fillna(0)
    
    for window in [5, 10]:
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
        df[f'{col}_roll3'] = df.groupby('player_id')[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum()
        ).fillna(0)
        df[f'{col}_roll5'] = df.groupby('player_id')[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum()
        ).fillna(0)
    
    df['goal_involvements_roll5'] = df['goals_roll5'] + df['assists_roll5']
    
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
    
    # Position indicators
    pos = df['position'].fillna(2).astype(str)
    df['is_gk'] = (pos == '0').astype(int)
    df['is_def'] = (pos == '1').astype(int)
    df['is_mid'] = (pos == '2').astype(int)
    df['is_fwd'] = (pos == '3').astype(int)
    
    # Defcon based on position
    df['defcon'] = np.where(df['is_def'] == 1, df['CBIT'], df['CBIRT'])
    df['defcon_threshold'] = np.where(df['is_def'] == 1, 10, 12)
    df['hit_threshold'] = (df['defcon'] >= df['defcon_threshold']).astype(int)
    
    # Defcon per 90
    df['defcon_per90'] = df['defcon'] / mins_90
    for window in [5, 10]:
        df[f'defcon_per90_roll{window}'] = df.groupby('player_id')['defcon_per90'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'hit_threshold_roll{window}'] = df.groupby('player_id')['hit_threshold'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
    
    # Component stats per 90
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
        df[f'{col}_per90'] = df[col] / mins_90
        df[f'{col}_per90_roll5'] = df.groupby('player_id')[f'{col}_per90'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
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
    
    # Defensive stats (lifetime)
    df['lifetime_defcon'] = df.groupby('player_name')['defcon'].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
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
    for col in ['tackles', 'interceptions', 'clearances']:
        df[f'lifetime_{col}_per90'] = np.where(
            df['lifetime_minutes'] >= 90,
            df[f'lifetime_{col}'].fillna(0) / lifetime_mins_90,
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
    
    team_stats = df.groupby(['team', 'season', 'gameweek']).agg({
        'goals': 'sum', 'xg': 'sum', 'shots': 'sum'
    }).reset_index()
    team_stats = team_stats.rename(columns={'goals': 'team_goals', 'xg': 'team_xg', 'shots': 'team_shots'})
    team_stats = team_stats.sort_values(['team', 'season', 'gameweek'])
    
    for col in ['team_goals', 'team_xg', 'team_shots']:
        for window in [5, 10]:
            team_stats[f'{col}_roll{window}'] = team_stats.groupby('team')[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
    
    df = df.merge(
        team_stats[['team', 'season', 'gameweek', 'team_goals_roll5', 'team_xg_roll5', 
                    'team_goals_roll10', 'team_xg_roll10']],
        on=['team', 'season', 'gameweek'], how='left'
    )
    
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
        
        # Get goals conceded by each team (= opponent's goals scored that match)
        # First build a match-level lookup: team -> opponent's goals that game
        match_results = df.groupby(['team_norm', 'opponent_norm', 'season', 'gameweek']).agg({
            'goals': 'sum', 'xg': 'sum', 'team': 'first', 'opponent': 'first'
        }).reset_index()
        
        # Swap to get what the opponent scored against this team
        # After swap: team_norm becomes what was opponent_norm, so we get each team's conceded stats
        team_conceded = match_results.rename(columns={
            'team_norm': 'opponent_temp_norm', 
            'opponent_norm': 'team_norm',  # The opponent becomes the team for defensive stats
            'goals': 'goals_conceded',
            'xg': 'xga'
        })
        team_conceded = team_conceded.sort_values(['team_norm', 'season', 'gameweek'])
        
        # Rolling goals conceded and xGA (multiple windows for different time horizons)
        # Last game (roll1), recent form (roll3), short term (roll5), medium term (roll10), season trend (roll30)
        for window in [1, 3, 5, 10, 30]:
            team_conceded[f'team_conceded_roll{window}'] = team_conceded.groupby('team_norm')['goals_conceded'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            team_conceded[f'team_xga_roll{window}'] = team_conceded.groupby('team_norm')['xga'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        # Clean sheet tracking
        team_conceded['clean_sheet'] = (team_conceded['goals_conceded'] == 0).astype(int)
        for window in [1, 3, 5, 10, 30]:
            team_conceded[f'team_cs_rate_roll{window}'] = team_conceded.groupby('team_norm')['clean_sheet'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        # Merge using normalized team name
        df = df.merge(
            team_conceded[['team_norm', 'season', 'gameweek', 
                          'team_conceded_roll1', 'team_conceded_roll3', 'team_conceded_roll5', 'team_conceded_roll10', 'team_conceded_roll30',
                          'team_xga_roll1', 'team_xga_roll3', 'team_xga_roll5', 'team_xga_roll10', 'team_xga_roll30',
                          'team_cs_rate_roll1', 'team_cs_rate_roll3', 'team_cs_rate_roll5', 'team_cs_rate_roll10', 'team_cs_rate_roll30']],
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
        
        for window in [5, 10]:
            opp_offense[f'opp_goals_roll{window}'] = opp_offense.groupby('opponent_norm')['opp_goals'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            opp_offense[f'opp_xg_roll{window}'] = opp_offense.groupby('opponent_norm')['opp_xg'].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
        
        df = df.merge(
            opp_offense[['opponent_norm', 'season', 'gameweek', 'opp_goals_roll5', 'opp_xg_roll5',
                        'opp_goals_roll10', 'opp_xg_roll10']],
            on=['opponent_norm', 'season', 'gameweek'], how='left'
        )
        
        # Opponent's defensive weakness (goals they concede = xGA)
        opp_defense = team_conceded[['team_norm', 'season', 'gameweek', 
                                     'goals_conceded', 'xga',
                                     'team_conceded_roll5', 'team_conceded_roll10',
                                     'team_xga_roll5', 'team_xga_roll10']].copy()
        opp_defense = opp_defense.rename(columns={
            'team_norm': 'opponent_norm',
            'team_conceded_roll5': 'opp_conceded_roll5',
            'team_conceded_roll10': 'opp_conceded_roll10', 
            'team_xga_roll5': 'opp_xga_roll5',
            'team_xga_roll10': 'opp_xga_roll10'
        })
        
        df = df.merge(
            opp_defense[['opponent_norm', 'season', 'gameweek', 
                        'opp_conceded_roll5', 'opp_conceded_roll10',
                        'opp_xga_roll5', 'opp_xga_roll10']],
            on=['opponent_norm', 'season', 'gameweek'], how='left'
        )
        
        # Clean up temporary normalized columns
        df = df.drop(columns=['team_norm', 'opponent_norm'], errors='ignore')
    
    # =========================================================================
    # CLEAN UP
    # =========================================================================
    
    df = df.drop(columns=['was_starter', 'was_full_90'], errors='ignore')
    
    # Fill NaN in rolling columns
    rolling_cols = [c for c in df.columns if 'roll' in c or 'lifetime' in c or 'last_' in c]
    for col in rolling_cols:
        df[col] = df[col].fillna(0)
    
    # Fill other defaults
    df['is_home'] = df['is_home'].fillna(0).astype(int)
    df['starter_score'] = df['starter_score'].fillna(0.5)
    
    if verbose:
        print(f"  Computed {len(rolling_cols)} rolling/lifetime features")
    
    return df
