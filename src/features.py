"""
Feature engineering for FPL prediction.
All rolling features use shift(1) to prevent data leakage.
"""
import pandas as pd
import numpy as np
from pathlib import Path


# Standard rolling windows for player-level features
ROLLING_WINDOWS = [1, 2, 3, 5, 7, 10]

# Manager embedding config
MANAGER_EMB_DIM = 8
MANAGER_EMB_WINDOW = 20
MANAGER_EMB_MIN_GAMES = 3
MANAGER_EMB_COLS = [f'manager_emb_{i}' for i in range(MANAGER_EMB_DIM)]
TEAM_NAME_MAP = {
    'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton and Hove Albion',
}
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


def _compute_calendar_minutes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a complete (player_id, season, gameweek) timeline filling missed gameweeks
    with 0 minutes, then compute calendar-aware minutes / starter / full-90 rolling features
    plus weeks-since-last-appearance.

    Returns a DataFrame keyed by (player_id, season, gameweek) with the new feature columns.
    Use to replace the appearances-only versions, which over-credit players with sparse
    appearances (e.g., a player who started one match 7 GWs ago looks identical to a
    regular starter in the appearance-only rolling).
    """
    base = df[['player_id', 'season', 'gameweek', 'minutes']].copy()
    base['minutes'] = pd.to_numeric(base['minutes'], errors='coerce').fillna(0)
    base['was_starter'] = (base['minutes'] >= 60).astype(int)
    base['was_full_90'] = (base['minutes'] >= 89).astype(int)
    base['appeared'] = (base['minutes'] > 0).astype(int)

    # Aggregate DGW (multiple matches in one gameweek) to a single calendar entry.
    agg = base.groupby(['player_id', 'season', 'gameweek'], as_index=False).agg(
        minutes=('minutes', 'sum'),
        was_starter=('was_starter', 'max'),
        was_full_90=('was_full_90', 'max'),
        appeared=('appeared', 'max'),
    )

    bounds = agg.groupby(['player_id', 'season'], as_index=False).agg(
        gw_min=('gameweek', 'min'),
        gw_max=('gameweek', 'max'),
    )

    grids = []
    for _, b in bounds.iterrows():
        gws = np.arange(int(b['gw_min']), int(b['gw_max']) + 1)
        grids.append(pd.DataFrame({
            'player_id': b['player_id'],
            'season': b['season'],
            'gameweek': gws,
        }))
    if not grids:
        return pd.DataFrame(columns=['player_id', 'season', 'gameweek'])
    grid = pd.concat(grids, ignore_index=True)

    grid = grid.merge(agg, on=['player_id', 'season', 'gameweek'], how='left')
    grid['minutes'] = grid['minutes'].fillna(0)
    grid['was_starter'] = grid['was_starter'].fillna(0).astype(int)
    grid['was_full_90'] = grid['was_full_90'].fillna(0).astype(int)
    grid['appeared'] = grid['appeared'].fillna(0).astype(int)

    grid = grid.sort_values(['player_id', 'season', 'gameweek']).reset_index(drop=True)
    g = grid.groupby(['player_id', 'season'])

    grid['last_minutes'] = g['minutes'].shift(1)
    grid['last_was_starter'] = g['was_starter'].shift(1).fillna(0)
    grid['last_was_full_90'] = g['was_full_90'].shift(1).fillna(0)

    for window in ROLLING_WINDOWS:
        grid[f'minutes_roll{window}'] = g['minutes'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        grid[f'starter_rate_roll{window}'] = g['was_starter'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        grid[f'full90_rate_roll{window}'] = g['was_full_90'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Weeks since last appearance: gameweek minus the most-recent prior gameweek where appeared=1.
    last_app_gw = grid['gameweek'].where(grid['appeared'] == 1)
    grid['_last_app_gw'] = last_app_gw.groupby([grid['player_id'], grid['season']]).transform(
        lambda x: x.shift(1).ffill()
    )
    grid['gw_gap_since_last_appearance'] = grid['gameweek'] - grid['_last_app_gw']

    feature_cols = (
        ['last_minutes', 'last_was_starter', 'last_was_full_90', 'gw_gap_since_last_appearance']
        + [f'minutes_roll{w}' for w in ROLLING_WINDOWS]
        + [f'starter_rate_roll{w}' for w in ROLLING_WINDOWS]
        + [f'full90_rate_roll{w}' for w in ROLLING_WINDOWS]
    )
    return grid[['player_id', 'season', 'gameweek'] + feature_cols]


def _parse_formation(formation_str):
    """Parse FotMob formation strings like '4-2-3-1' into (def, mid, fwd) counts.
    Convention: first number = defenders, last = forwards, middle nums summed = midfielders.
    Returns (def, mid, fwd) as ints; (np.nan, np.nan, np.nan) if parse fails.
    """
    if not isinstance(formation_str, str) or not formation_str.strip():
        return (np.nan, np.nan, np.nan)
    parts = formation_str.strip().split('-')
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return (np.nan, np.nan, np.nan)
    if len(nums) < 2:
        return (np.nan, np.nan, np.nan)
    return (float(nums[0]), float(sum(nums[1:-1])), float(nums[-1]))


def _team_match_minute_features(df_team_match: pd.DataFrame) -> dict:
    """Per (match, team) minute-distribution features from player_stats rows.

    Captures minutes management / rotation / squad-usage signal:
      mins_mean / median / std / max — central tendency + spread
      num_players_used — squad depth used in this match
      num_full_90 — count of players going the full 90 (low rotation indicator)
      num_subs_made — appearances beyond the starting 11
      mins_concentration_top11 — share of total minutes from top 11 players
      mins_entropy — Shannon entropy of normalized minutes distribution
    """
    mins = pd.to_numeric(df_team_match['minutes'], errors='coerce').fillna(0)
    mins = mins[mins > 0]
    if len(mins) == 0:
        return {k: np.nan for k in [
            'mins_mean', 'mins_median', 'mins_std', 'mins_max',
            'num_players_used', 'num_full_90', 'num_subs_made',
            'mins_concentration_top11', 'mins_entropy',
        ]}
    arr = mins.values
    total = arr.sum()
    p = arr / total if total > 0 else np.zeros_like(arr)
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    top11 = np.sort(arr)[::-1][:11].sum()
    return {
        'mins_mean': float(arr.mean()),
        'mins_median': float(np.median(arr)),
        'mins_std': float(arr.std(ddof=0)),
        'mins_max': float(arr.max()),
        'num_players_used': int(len(arr)),
        'num_full_90': int((arr >= 89).sum()),
        'num_subs_made': max(0, int(len(arr) - 11)),
        'mins_concentration_top11': float(top11 / total) if total > 0 else 0.0,
        'mins_entropy': entropy,
    }


def add_manager_embeddings(
    df: pd.DataFrame,
    data_dir: str = 'data',
    n_components: int = MANAGER_EMB_DIM,
    window: int = MANAGER_EMB_WINDOW,
    min_games: int = MANAGER_EMB_MIN_GAMES,
    verbose: bool = True,
) -> pd.DataFrame:
    """Append leak-free manager embedding features to df (one set per (match_id, team) row).

    Pipeline:
      1. Load manager-per-match cache (data/match_managers.csv) and match_details.
      2. Build per (manager, match) feature vector: minute-distribution stats, GF/GA,
         formation breakdown.
      3. Sort each manager's matches chronologically; rolling-mean over the prior
         `window` games (shift(1) — strictly prior — so the current game's outcome
         never leaks into its own embedding).
      4. Standardize + fit PCA(n_components) on rows with >= min_games prior history.
      5. Transform every row; managers below the threshold get a zero vector.
      6. For synthetic future-match rows in df, assume the team's most recent
         manager and compute their embedding from the latest `window` real games.
      7. Merge `manager_emb_0..n-1` onto df by (match_id, team).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data_path = Path(data_dir)
    mm_path = data_path / 'match_managers.csv'
    md_path = data_path / 'matches' / 'match_details.csv'
    if not mm_path.exists() or not md_path.exists():
        if verbose:
            print(f"  [manager_emb] Missing cache ({mm_path} or {md_path}), skipping")
        for c in MANAGER_EMB_COLS:
            df[c] = 0.0
        return df

    if verbose:
        print(f"  Computing manager embeddings (dim={n_components}, window={window}, min_games={min_games})...")

    mm = pd.read_csv(mm_path)
    md = pd.read_csv(md_path)[['match_id', 'match_date']]
    for col in ('home_team', 'away_team'):
        mm[col] = mm[col].replace(TEAM_NAME_MAP)

    # Long form: one row per (match_id, team)
    home = mm[['match_id', 'home_team', 'home_manager_id', 'home_manager_name', 'home_formation']].rename(
        columns={'home_team': 'team', 'home_manager_id': 'manager_id',
                 'home_manager_name': 'manager_name', 'home_formation': 'formation'})
    home['is_home'] = 1
    away = mm[['match_id', 'away_team', 'away_manager_id', 'away_manager_name', 'away_formation']].rename(
        columns={'away_team': 'team', 'away_manager_id': 'manager_id',
                 'away_manager_name': 'manager_name', 'away_formation': 'formation'})
    away['is_home'] = 0
    mt = pd.concat([home, away], ignore_index=True)

    mt = mt.merge(md, on='match_id', how='left')
    mt['match_date'] = pd.to_datetime(mt['match_date'], errors='coerce')

    # GF/GA from player_stats sums (match_details.csv has no scores in this repo).
    df_goals = df.copy()
    df_goals['goals'] = pd.to_numeric(df_goals.get('goals', 0), errors='coerce').fillna(0)
    df_goals['own_goal'] = pd.to_numeric(df_goals.get('own_goal', 0), errors='coerce').fillna(0)
    team_goals = (df_goals.groupby(['match_id', 'team'], dropna=False)
                  .agg(team_goals=('goals', 'sum'), team_og=('own_goal', 'sum'))
                  .reset_index())
    mt = mt.merge(team_goals.rename(columns={'team': 'team', 'team_goals': 'gf_self', 'team_og': 'og_self'}),
                  on=['match_id', 'team'], how='left')
    # Opponent's goals for ga: each match has 2 teams; pull other team's goals.
    pair = team_goals.rename(columns={'team': 'opp_team', 'team_goals': 'opp_goals', 'team_og': 'opp_og'})
    mt_pair = mt[['match_id', 'team']].merge(pair, on='match_id')
    mt_pair = mt_pair[mt_pair['team'] != mt_pair['opp_team']]
    mt = mt.merge(mt_pair[['match_id', 'team', 'opp_goals', 'opp_og']],
                  on=['match_id', 'team'], how='left')
    # GF = own player goals + opponent own-goals; GA = opp player goals + own own-goals
    mt['gf'] = mt['gf_self'].fillna(0) + mt['opp_og'].fillna(0)
    mt['ga'] = mt['opp_goals'].fillna(0) + mt['og_self'].fillna(0)
    # If no goals data was found at all for this match-team, treat as missing rather than zero
    no_data = mt['gf_self'].isna() & mt['opp_goals'].isna()
    mt.loc[no_data, ['gf', 'ga']] = np.nan
    mt = mt.drop(columns=['gf_self', 'og_self', 'opp_goals', 'opp_og'])

    # Per (match_id, team) minute distribution from current df
    if verbose:
        print(f"  [manager_emb] Aggregating minute distributions...")
    min_feats = (
        df.groupby(['match_id', 'team'], dropna=False)
          .apply(_team_match_minute_features)
          .apply(pd.Series)
          .reset_index()
    )
    mt = mt.merge(min_feats, on=['match_id', 'team'], how='left')

    # Formation parse
    form = mt['formation'].apply(_parse_formation)
    mt['form_def'] = form.apply(lambda t: t[0])
    mt['form_mid'] = form.apply(lambda t: t[1])
    mt['form_fwd'] = form.apply(lambda t: t[2])

    feat_cols = [
        'gf', 'ga',
        'mins_mean', 'mins_median', 'mins_std', 'mins_max',
        'num_players_used', 'num_full_90', 'num_subs_made',
        'mins_concentration_top11', 'mins_entropy',
        'form_def', 'form_mid', 'form_fwd',
    ]

    # Drop rows with no manager identity, no score (unplayed), or no min-distribution
    # (the match never made it into player_stats — typically pre-season / abandoned).
    mt = mt[mt['manager_id'].notna()
            & mt['gf'].notna()
            & mt['ga'].notna()
            & mt['mins_mean'].notna()].copy()
    mt['manager_id'] = mt['manager_id'].astype('int64')

    # Sort each manager chronologically; ties broken by match_id
    mt = mt.sort_values(['manager_id', 'match_date', 'match_id']).reset_index(drop=True)

    # Rolling mean of prior `window` games (shift(1) for strict leak-freedom)
    g = mt.groupby('manager_id', group_keys=False)
    n_prior = g.cumcount()
    mt['n_prior_games'] = n_prior.values
    rolled = pd.DataFrame(index=mt.index)
    for col in feat_cols:
        rolled[col] = g[col].apply(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        ).values

    valid_mask = (mt['n_prior_games'] >= min_games).values
    X_valid = rolled.loc[valid_mask, feat_cols].fillna(0.0).values

    if X_valid.shape[0] < n_components:
        if verbose:
            print(f"  [manager_emb] Only {X_valid.shape[0]} valid rows, skipping PCA")
        for c in MANAGER_EMB_COLS[:n_components]:
            df[c] = 0.0
        return df

    scaler = StandardScaler()
    X_valid_scaled = scaler.fit_transform(X_valid)
    pca = PCA(n_components=n_components)
    pca.fit(X_valid_scaled)

    X_all = rolled[feat_cols].fillna(0.0).values
    X_all_scaled = scaler.transform(X_all)
    emb_all = pca.transform(X_all_scaled)
    emb_all[~valid_mask] = 0.0

    for i in range(n_components):
        mt[f'manager_emb_{i}'] = emb_all[:, i]

    # Embeddings for synthetic future rows (match_ids absent from cache).
    # For each team, look up the most recent manager and their up-to-date rolled
    # feature vector (no shift since it's "current state").
    df_matches = set(df['match_id'].dropna().unique())
    cache_matches = set(mt['match_id'].unique())
    synth_matches = df_matches - cache_matches
    synth_rows = []
    if synth_matches:
        # Latest known manager per team (by match_date)
        team_latest_mgr = (
            mt.sort_values(['team', 'match_date'])
              .groupby('team').tail(1)[['team', 'manager_id']]
              .set_index('team')['manager_id']
              .to_dict()
        )
        # Per-manager "current" rolled feature vector: mean of their latest `window`
        # real games (no shift — represents now).
        cur_per_mgr = (
            mt.groupby('manager_id', group_keys=False)
              .apply(lambda g: g.tail(window)[feat_cols].mean())
        )
        cur_n = mt.groupby('manager_id').size()

        synth_team_match = df[df['match_id'].isin(synth_matches)][['match_id', 'team']].drop_duplicates()
        for _, r in synth_team_match.iterrows():
            mgr_id = team_latest_mgr.get(r['team'])
            if mgr_id is None or mgr_id not in cur_per_mgr.index or cur_n.get(mgr_id, 0) < min_games:
                emb = np.zeros(n_components)
            else:
                feats = cur_per_mgr.loc[mgr_id].fillna(0.0).values.reshape(1, -1)
                emb = pca.transform(scaler.transform(feats))[0]
            row = {'match_id': r['match_id'], 'team': r['team'], 'manager_id': mgr_id}
            for i in range(n_components):
                row[f'manager_emb_{i}'] = float(emb[i])
            synth_rows.append(row)

    emb_df = mt[['match_id', 'team'] + MANAGER_EMB_COLS[:n_components]].copy()
    if synth_rows:
        emb_df = pd.concat([emb_df, pd.DataFrame(synth_rows)[emb_df.columns]], ignore_index=True)

    df = df.merge(emb_df, on=['match_id', 'team'], how='left')
    for c in MANAGER_EMB_COLS[:n_components]:
        df[c] = df[c].fillna(0.0)

    if verbose:
        n_zero = (df[MANAGER_EMB_COLS[:n_components]].abs().sum(axis=1) == 0).sum()
        print(f"  [manager_emb] Done. {n_zero} of {len(df)} rows have zero embedding (interim/missing).")
    return df


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

    # Calendar-aware minutes features (treats missed gameweeks as 0 minutes / not started).
    # Without this, a player who started one match 7 GWs ago has the same `last_minutes` and
    # rolling-minutes profile as a player who started last week.
    df['was_starter'] = (df['minutes'] >= 60).astype(int)
    df['was_full_90'] = (df['minutes'] >= 89).astype(int)
    cal_features = _compute_calendar_minutes_features(df)
    cal_cols = [c for c in cal_features.columns if c not in ('player_id', 'season', 'gameweek')]
    df = df.drop(columns=[c for c in cal_cols if c in df.columns], errors='ignore')
    df = df.merge(cal_features, on=['player_id', 'season', 'gameweek'], how='left')

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
    # MANAGER EMBEDDINGS (leak-free PCA over rolling-prior manager stats)
    # =========================================================================

    df = add_manager_embeddings(df, verbose=verbose)

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
