"""
Feature engineering for FPL prediction.
Rolling statistics read only results completed before each forecast deadline.
This is the timestamp-aware equivalent of shift(1), including double gameweeks.
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

# Team-level rolling stats need a real sample before they mean anything. With
# min_periods=1 a promoted club's roll5 is just its one played match, so a 3-0
# opener reads as "scores 3, concedes 0, always clean sheet". Windows shorter
# than this keep their own length (roll1 is honestly one match).
TEAM_ROLL_MIN_PERIODS = 3

# Team "identity" stats (scoring rate, xGA, clean-sheet rate) roll over this many
# matches ACROSS seasons rather than resetting each August. Season-to-date is one
# match in GW2, and prior_lambda is multiplicative in the opponent's scoring term,
# so a club that happened to blank in GW1 would read as unable to score at all.
# Newly promoted clubs are excluded by the cold-start guard and stay on the
# promoted-cohort prior until they have TEAM_ROLL_MIN_PERIODS games.
SEASON_STAT_WINDOW = 10

# ...but only while the current season is too thin to speak for itself. Once a club
# has this many games in the bag, season-to-date is the better estimator: by March
# it averages 25+ matches, where a 10-game window is just noisier. So the rolling
# window is an early-season bridge, not a replacement.
SEASON_STAT_MIN_GAMES = 5


def chronological_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Attach match times and the information deadline for each gameweek.

    Historical rows use actual kickoff times. The boundary defaults to 90 minutes
    before a gameweek's earliest kickoff (an explicit forecast_time wins); postponed results cannot enter earlier
    training windows. Small synthetic/unit-test frames may use season/GW order.
    """
    out = df.copy()
    if 'match_date' not in out:
        out['match_date'] = pd.NaT
    dates = pd.to_datetime(out['match_date'], errors='coerce', utc=True)
    years = out['season'].astype(str).str[:4].astype(int)
    fallback = pd.to_datetime(years.astype(str) + '-07-01', utc=True)
    fallback += pd.to_timedelta(pd.to_numeric(out['gameweek']).fillna(0) * 7, unit='D')
    out['match_date'] = dates.fillna(fallback)
    if 'forecast_time' not in out:
        out['forecast_time'] = (out.groupby(['season', 'gameweek'])['match_date'].transform('min') - pd.Timedelta(minutes=90))
    out['forecast_time'] = pd.to_datetime(out['forecast_time'], utc=True).fillna(
        out.groupby(['season', 'gameweek'])['match_date'].transform('min'))
    if 'result_time' not in out:
        out['result_time'] = out['match_date'] + pd.Timedelta(hours=3)
    out['result_time'] = pd.to_datetime(out['result_time'], utc=True).fillna(
        out['match_date'] + pd.Timedelta(hours=3))
    return out


def deadline_splits(df: pd.DataFrame, n_splits: int = 5):
    """Expanding folds of whole forecast deadlines, purged of unplayed results."""
    from sklearn.model_selection import TimeSeriesSplit
    frame = chronological_frame(df)
    deadlines = np.sort(frame['forecast_time'].unique())
    if len(deadlines) < 3:
        return
    splitter = TimeSeriesSplit(n_splits=min(n_splits, len(deadlines) - 1))
    for train_blocks, valid_blocks in splitter.split(deadlines):
        cutoff = deadlines[valid_blocks[0]]
        train = frame['forecast_time'].isin(deadlines[train_blocks])
        train &= frame['result_time'] < cutoff
        valid = frame['forecast_time'].isin(deadlines[valid_blocks])
        ti, vi = np.flatnonzero(train), np.flatnonzero(valid)
        if len(ti) and len(vi):
            yield ti, vi


def freeze_prior_features(frame: pd.DataFrame, group: str, columns: list) -> pd.DataFrame:
    """Read shifted historical state at the deadline, not at a later kickoff.

    Each selected column already excludes its own match. The first match at or
    after the deadline therefore carries exactly the available historical state.
    This also prevents one DGW fixture from becoming history for the other.
    """
    if not columns or 'match_date' not in frame:
        return frame
    out = frame.copy()
    for _, part in out.groupby(group, sort=False):
        ordered = part.sort_values('match_date', kind='stable')
        times = ordered['match_date'].astype('int64').to_numpy()
        cutoffs = pd.to_datetime(part['forecast_time'], utc=True).astype('int64').to_numpy()
        source = np.searchsorted(times, cutoffs, side='left').clip(0, len(times) - 1)
        out.loc[part.index, columns] = ordered.iloc[source][columns].to_numpy()
    return out


def prior_stat(frame, group, source, window=None, aggregation='mean', min_periods=1, span=None):
    """Aggregate completed observations strictly before each row's deadline."""
    if not {'match_date', 'forecast_time', 'result_time'} <= set(frame.columns):
        frame = chronological_frame(frame)
    groups = [group] if isinstance(group, str) else list(group)
    frame = frame[list(dict.fromkeys(groups + [source, 'result_time', 'forecast_time']))]
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, part in frame.groupby(group, sort=False):
        ordered = part.sort_values('result_time', kind='stable')
        values = pd.to_numeric(ordered[source], errors='coerce')
        if aggregation == 'last':
            state = values
        else:
            rolling = (values.ewm(span=span, min_periods=min_periods) if span else
                       values.rolling(window, min_periods=min_periods) if window else
                       values.expanding(min_periods=min_periods))
            state = getattr(rolling, aggregation)()
        times = ordered['result_time'].astype('int64').to_numpy()
        cutoffs = part['forecast_time'].astype('int64').to_numpy()
        previous = np.searchsorted(times, cutoffs, side='left') - 1
        available = previous >= 0
        result.loc[part.index[available]] = state.iloc[previous[available]].to_numpy()
    return result


def prior_exposure_rate(frame, group, source, window, cap=None, min_periods=1):
    """Per-90 rate from pooled prior exposure: 90 * sum(stat) / sum(minutes).

    Averaging match-level per-90 values lets a short cameo carry the same weight as
    a full match. This keeps the same deadline-aware observation window as
    ``prior_stat`` while weighting every action by the minutes that produced it.
    """
    if not {'match_date', 'forecast_time', 'result_time'} <= set(frame.columns):
        frame = chronological_frame(frame)
    groups = [group] if isinstance(group, str) else list(group)
    columns = list(dict.fromkeys(groups + [source, 'minutes', 'result_time', 'forecast_time']))
    working = frame[columns]
    result = pd.Series(np.nan, index=working.index, dtype=float)
    for _, part in working.groupby(group, sort=False):
        ordered = part.sort_values('result_time', kind='stable')
        raw_values = pd.to_numeric(ordered[source], errors='coerce')
        valid = raw_values.notna()
        values = raw_values.where(valid)
        minutes = pd.to_numeric(ordered['minutes'], errors='coerce').clip(lower=0).where(valid)
        numerator = values.rolling(window, min_periods=min_periods).sum()
        denominator = minutes.rolling(window, min_periods=min_periods).sum()
        rate = 90 * numerator / denominator.replace(0, np.nan)
        times = ordered['result_time'].astype('int64').to_numpy()
        cutoffs = part['forecast_time'].astype('int64').to_numpy()
        previous = np.searchsorted(times, cutoffs, side='left') - 1
        available = previous >= 0
        result.loc[part.index[available]] = rate.iloc[previous[available]].to_numpy()
    if cap is not None:
        result = result.clip(upper=cap)
    return result


class ManagerFeatureMixin:
    """Fit the manager PCA only on a model's training rows; reuse at inference."""

    def fit_manager_features(self, df):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        columns = sorted(c for c in df if c.startswith('manager_raw_'))
        self.manager_basis = None
        if columns:
            valid = df.get('manager_prior_games', pd.Series(0, index=df.index)) >= MANAGER_EMB_MIN_GAMES
            # Managers shared by many player rows must not gain extra PCA weight.
            x = df.loc[valid, columns].fillna(0).drop_duplicates()
            if len(x) >= MANAGER_EMB_DIM and len(columns) >= MANAGER_EMB_DIM:
                scaler = StandardScaler().fit(x)
                pca = PCA(n_components=MANAGER_EMB_DIM, svd_solver='full').fit(scaler.transform(x))
                self.manager_basis = (columns, scaler, pca)
        return self

    def manager_features(self, df):
        out = df.copy()
        if not any(c.startswith('manager_raw_') for c in out):
            return out
        values = np.zeros((len(out), MANAGER_EMB_DIM))
        basis = getattr(self, 'manager_basis', None)
        if basis is not None:
            columns, scaler, pca = basis
            x = out.reindex(columns=columns).fillna(0)
            values = pca.transform(scaler.transform(x))
            valid = out.get('manager_prior_games', pd.Series(0, index=out.index)) >= MANAGER_EMB_MIN_GAMES
            values[~valid.to_numpy()] = 0
        out[MANAGER_EMB_COLS] = values
        return out

def promoted_team_seasons(df, team_col='team_norm', season_col='season'):
    """(team, season) pairs where the club was absent from the PL the season before.

    Catches clubs returning after a gap (Ipswich, Leeds) as well as clubs that
    have never been up -- a club promoted after two years away still has stale
    top-flight form on file, and carrying it into the new season is exactly the
    error the cold-start guard exists to prevent. The first season in the data
    has no predecessor, so nothing counts as promoted there.
    """
    seasons_sorted = sorted(df[season_col].dropna().unique())
    pairs = set()
    for prev_s, cur_s in zip(seasons_sorted, seasons_sorted[1:]):
        prev_teams = set(df.loc[df[season_col] == prev_s, team_col].unique())
        cur_teams = set(df.loc[df[season_col] == cur_s, team_col].unique())
        pairs |= {(t, cur_s) for t in cur_teams - prev_teams}
    return pairs

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


def resolve_defcon_positions(frame: pd.DataFrame) -> pd.DataFrame:
    """Resolve the position used by DefCon, preferring authoritative FPL data.

    Historical FPL position data is incomplete. When it is unavailable, retain
    the full defensive-contribution history by falling back to the player's
    FotMob position for that match. Rows missing both sources remain unknown.
    """
    out = frame.copy()
    fpl_position = out.get(
        'fpl_position', pd.Series(pd.NA, index=out.index, dtype='string')
    ).astype('string').str.upper()
    fpl_known = fpl_position.isin(['GK', 'DEF', 'MID', 'FWD'])
    fotmob_position = pd.to_numeric(
        out.get('position', pd.Series(np.nan, index=out.index)), errors='coerce'
    ).map({0: 'GK', 1: 'DEF', 2: 'MID', 3: 'FWD'}).astype('string')

    out['defcon_position'] = fpl_position.where(fpl_known, fotmob_position)
    fotmob_known = fotmob_position.isin(['GK', 'DEF', 'MID', 'FWD'])
    out['defcon_position_source'] = pd.Series(
        np.where(fpl_known, 'FPL', np.where(fotmob_known, 'FotMob', pd.NA)),
        index=out.index, dtype='string')
    out['fpl_is_def'] = fpl_position.eq('DEF').fillna(False).astype(int)
    out['fpl_is_mid'] = fpl_position.eq('MID').fillna(False).astype(int)
    out['fpl_is_fwd'] = fpl_position.eq('FWD').fillna(False).astype(int)
    out['defcon_is_def'] = out['defcon_position'].eq('DEF').fillna(False).astype(int)
    out['defcon_is_mid'] = out['defcon_position'].eq('MID').fillna(False).astype(int)
    out['defcon_is_fwd'] = out['defcon_position'].eq('FWD').fillna(False).astype(int)
    return out


def _compute_calendar_minutes_features(df: pd.DataFrame,
                                       include_appeared: bool = False,
                                       per_fixture: bool = False,
                                       roster_spells: pd.DataFrame = None) -> pd.DataFrame:
    """Playing-time history on scheduled fixtures, including non-appearances.

    Explicit registration intervals can be supplied through roster_spells (or
    df.attrs['roster_spells']): player_id, team, season, start_date, end_date.
    Without them, membership starts with the first observed squad appearance and
    continues until a known transfer or the club's last observed fixture. This
    keeps terminal injury/rotation absences; unobserved transfers remain unknown.
    Blank gameweeks have no fixture and are not labeled as selection failures.
    """
    frame = chronological_frame(df)
    spells = roster_spells if roster_spells is not None else df.attrs.get('roster_spells')
    frame.attrs = {k: v for k, v in frame.attrs.items() if k != 'roster_spells'}
    keys = ['player_id', 'season', 'gameweek']
    if {'team', 'match_id'} <= set(frame.columns):
        timing = ['match_id', 'season', 'gameweek', 'match_date', 'forecast_time', 'result_time']
        schedule = frame[['team'] + timing].drop_duplicates(['team', 'match_id'])
        schedule = schedule.sort_values('match_date')
        if spells is None:
            records = []
            for (player, season), history in frame.groupby(['player_id', 'season'], sort=False):
                history = history.sort_values('match_date').drop_duplicates('match_id')
                changes = history['team'].ne(history['team'].shift())
                starts = history.loc[changes, ['team', 'match_date']]
                for i, (_, row) in enumerate(starts.iterrows()):
                    end_date = (starts.iloc[i + 1]['match_date'] if i + 1 < len(starts)
                                else schedule.loc[schedule['season'].eq(season), 'result_time'].max()
                                + pd.Timedelta(days=1))
                    records.append(dict(player_id=player, season=season, team=row['team'],
                                        start_date=row['match_date'], end_date=end_date))
            spells = pd.DataFrame(records)
        spells = pd.DataFrame(spells)
        grids = []
        for _, spell in spells.iterrows():
            start_date = pd.to_datetime(spell['start_date'], utc=True)
            end_date = pd.to_datetime(spell.get('end_date'), utc=True)
            eligible = schedule['team'].eq(spell['team']) & schedule['season'].eq(spell['season'])
            eligible &= schedule['match_date'].ge(start_date)
            if pd.notna(end_date):
                eligible &= schedule['match_date'].lt(end_date)
            part = schedule.loc[eligible].copy()
            part['player_id'] = spell['player_id']
            grids.append(part)
        if not grids:
            return pd.DataFrame(columns=keys + ['match_id', 'appeared'])
        grid = pd.concat(grids, ignore_index=True).drop_duplicates(['player_id', 'match_id'])
        actual = frame.groupby(['player_id', 'match_id'], as_index=False)['minutes'].max()
        grid = grid.merge(actual, on=['player_id', 'match_id'], how='left', validate='one_to_one')
    else:
        # Compatibility for weekly input without a fixture table.
        actual = frame.groupby(keys, as_index=False)['minutes'].max()
        grids = []
        for (player, season), part in actual.groupby(['player_id', 'season']):
            grids.append(pd.DataFrame({'player_id': player, 'season': season,
                                       'gameweek': range(int(part.gameweek.min()), int(part.gameweek.max()) + 1)}))
        grid = chronological_frame(pd.concat(grids, ignore_index=True).merge(actual, on=keys, how='left'))
    grid['minutes'] = pd.to_numeric(grid['minutes'], errors='coerce').fillna(0).clip(0, 90)
    grid = grid.sort_values(['player_id', 'match_date'], kind='stable').reset_index(drop=True)
    grid['was_starter'] = (grid['minutes'] >= 60).astype(int)
    grid['was_full_90'] = (grid['minutes'] >= 89).astype(int)
    grid['appeared'] = (grid['minutes'] > 0).astype(int)
    for source, dest in [('minutes', 'last_minutes'), ('was_starter', 'last_was_starter'),
                         ('was_full_90', 'last_was_full_90')]:
        grid[dest] = prior_stat(grid, 'player_id', source, aggregation='last').fillna(0)
    for window in ROLLING_WINDOWS:
        for source, prefix in [('minutes', 'minutes'), ('was_starter', 'starter_rate'),
                               ('was_full_90', 'full90_rate')]:
            grid[f'{prefix}_roll{window}'] = prior_stat(grid, 'player_id', source, window).fillna(0)
    year = grid['season'].str[:4].astype(int)
    grid['_ordinal'] = (year * 38 + grid['gameweek']).where(grid['appeared'].eq(1))
    grid['_app_year'] = year.where(grid['appeared'].eq(1))
    # Forward fill is historical within a player, then read at the deadline.
    grid['_ordinal'] = grid.groupby('player_id')['_ordinal'].ffill()
    grid['_app_year'] = grid.groupby('player_id')['_app_year'].ffill()
    last = prior_stat(grid, 'player_id', '_ordinal', aggregation='last')
    last_year = prior_stat(grid, 'player_id', '_app_year', aggregation='last')
    grid['gw_gap_since_last_appearance'] = (year * 38 + grid['gameweek'] - last).fillna(0)
    grid['last_app_prev_season'] = (last_year < year).fillna(False).astype(int)
    feature_cols = (
        ['last_minutes', 'last_was_starter', 'last_was_full_90',
         'gw_gap_since_last_appearance', 'last_app_prev_season']
        + [f'minutes_roll{w}' for w in ROLLING_WINDOWS]
        + [f'starter_rate_roll{w}' for w in ROLLING_WINDOWS]
        + [f'full90_rate_roll{w}' for w in ROLLING_WINDOWS])
    if include_appeared:
        feature_cols += ['appeared']
    extra = [c for c in ['match_id', 'team', 'match_date', 'forecast_time', 'result_time'] if c in grid]
    if per_fixture:
        return grid[keys + extra + feature_cols]
    return grid[keys + feature_cols].drop_duplicates(keys, keep='first')


# Features the appearance model trains on. Restricted to the calendar grid, which is
# the only feature set defined on a week the player did not play — anything derived
# from match events (xG, touches, opponent stats) simply does not exist on those rows.
APPEARANCE_FEATURES = (
    ['last_minutes', 'last_was_starter', 'last_was_full_90', 'gw_gap_since_last_appearance']
    + [f'minutes_roll{w}' for w in ROLLING_WINDOWS]
    + [f'starter_rate_roll{w}' for w in ROLLING_WINDOWS]
    + [f'full90_rate_roll{w}' for w in ROLLING_WINDOWS]
    + ['is_gk', 'is_def', 'is_mid', 'is_fwd']
)


def build_appearance_grid(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """One eligible player-fixture row, including terminal non-appearances.

    Uses registered intervals if supplied, otherwise observed club spells. A
    blank gameweek is not a fixture. Unknown departures/registration dates cannot
    be recovered from appearances alone; see _compute_calendar_minutes_features.
    """
    grid = _compute_calendar_minutes_features(df, include_appeared=True, per_fixture=True)

    # Drop each player-season's opening row. The span starts at their first appearance,
    # so that row always has appeared == 1 — and it carries the cross-season gap, the
    # largest gw_gap value in the data. Left in, it teaches the exact inverse of the
    # truth: that a player absent for 40+ gameweeks is certain to feature. Removing it
    # keeps long gaps associated only with genuine mid-season absences.
    first = grid.groupby(['player_id', 'season'])['gameweek'].transform('min')
    n_before = len(grid)
    if df.attrs.get('roster_spells') is None:
        grid = grid[grid['gameweek'] > first].copy()

    # Position is a player attribute, not a per-match one; carry it over from any
    # match row so the grid's 0-minute weeks still get position dummies.
    pos_cols = [c for c in ('is_gk', 'is_def', 'is_mid', 'is_fwd') if c in df.columns]
    if pos_cols:
        pos = df.sort_values(['season', 'gameweek']).groupby('player_id')[pos_cols].first().reset_index()
        grid = grid.merge(pos, on='player_id', how='left')
    for c in ('is_gk', 'is_def', 'is_mid', 'is_fwd'):
        if c not in grid.columns:
            grid[c] = 0
        grid[c] = grid[c].fillna(0).astype(int)

    if verbose:
        n_zero = int((grid['appeared'] == 0).sum())
        print(f"  Appearance grid: {len(grid):,} player-gameweeks "
              f"({n_zero:,} non-appearances, base rate {grid['appeared'].mean():.3f}; "
              f"dropped {n_before - len(grid):,} season-opening rows)")
    return grid


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
    """Build manager raw priors using results strictly before each deadline.

    Manager identity comes from the team's latest completed match. Raw playing
    style, formation and minute-distribution averages are retained; each model
    fits and saves its own scaler/PCA using only its training window. The zero
    embedding columns are placeholders, not a globally fitted projection.
    """
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

    mt['result_time'] = pd.to_datetime(mt['match_date'], utc=True) + pd.Timedelta(hours=3)
    mt = mt.sort_values('result_time').reset_index(drop=True)
    raw_cols = ['manager_raw_' + c for c in feat_cols]
    # Resolve the manager from completed team history, including cache-missing
    # historical rows. Never fill an old missing row from the latest future manager.
    targets = chronological_frame(df)[['team', 'forecast_time']].drop_duplicates()
    records = []
    by_team = {team: part for team, part in mt.groupby('team')}
    by_manager = {manager: part for manager, part in mt.groupby('manager_id')}
    for _, target in targets.iterrows():
        record = {'team': target['team'], 'forecast_time': target['forecast_time'],
                  'manager_prior_games': 0, **dict.fromkeys(raw_cols, 0.)}
        team_history = by_team.get(target['team'])
        if team_history is not None:
            prior = team_history[team_history['result_time'] < target['forecast_time']]
            if not prior.empty:
                history = by_manager[prior.iloc[-1]['manager_id']]
                history = history[history['result_time'] < target['forecast_time']]
                record['manager_prior_games'] = len(history)
                record.update(zip(raw_cols, history.tail(window)[feat_cols].mean().fillna(0)))
        records.append(record)
    stale = [c for c in df if c.startswith('manager_raw_') or c == 'manager_prior_games']
    df = chronological_frame(df).drop(columns=stale, errors='ignore').merge(
        pd.DataFrame(records), on=['team', 'forecast_time'], how='left', validate='many_to_one')
    df[MANAGER_EMB_COLS] = 0.0
    return df


def compute_rolling_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute all rolling features for prediction models."""
    if verbose:
        print("Computing rolling features...")

    data_dir = df.attrs.get('data_dir', 'data')
    input_attrs = df.attrs.copy()
    if isinstance(input_attrs.get('roster_spells'), pd.DataFrame):
        input_attrs['roster_spells'] = input_attrs['roster_spells'].to_dict('records')
    df = df.copy()
    df.attrs = input_attrs
    df = chronological_frame(df).sort_values(['player_id', 'match_date'], kind='stable').reset_index(drop=True)

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
            if f'{col}_per90' in PER90_CAPS:
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
            if f'{col}_per90' in PER90_CAPS:
                cap = PER90_CAPS[f'{col}_per90']
                df[per90_col] = df[per90_col].clip(upper=cap)

        for window in ROLLING_WINDOWS:
            df[f'{col}_per90_roll{window}'] = prior_exposure_rate(
                df, 'player_id', col, window, PER90_CAPS.get(per90_col))

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
    df['current_season_minutes'] = prior_stat(df, ['player_id', 'season'], 'minutes', aggregation='sum').fillna(0)
    df['current_season_apps'] = prior_stat(df.assign(_one=1), ['player_id', 'season'], '_one', aggregation='sum').fillna(0)
    df['current_season_mins_per_app'] = df['current_season_minutes'] / df['current_season_apps'].replace(0, 1)

    # Recent form (raw counts)
    for col in ['goals', 'assists']:
        df[f'{col}_last1'] = prior_stat(df, 'player_id', col, aggregation='last').fillna(0)
        for window in ROLLING_WINDOWS:
            df[f'{col}_roll{window}'] = prior_stat(df, 'player_id', col, window, 'sum', 1).fillna(0)

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
        df[f'fouls_committed_per90_roll{window}'] = prior_exposure_rate(
            df, 'player_id', 'fouls_committed', window, 6.0)

    # =========================================================================
    # YELLOW / RED CARDS (from FPL API merge, if available)
    # =========================================================================

    if 'yellow_cards' in df.columns:
        df['yellow_cards'] = pd.to_numeric(df['yellow_cards'], errors='coerce').fillna(0)
        for window in ROLLING_WINDOWS:
            df[f'yellow_cards_roll{window}'] = prior_stat(df, 'player_id', 'yellow_cards', window, 'mean', 1)
        # Fouls per yellow (personalized booking rate) — only where fouls > 0
        df['_fouls_with_yellow'] = np.where(
            df['fouls_committed'] > 0,
            df['yellow_cards'] / df['fouls_committed'],
            np.nan
        )
        df['yellow_per_foul_roll10'] = prior_stat(df, 'player_id', '_fouls_with_yellow', 10, min_periods=3)
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
        df[f'saves_per90_roll{window}'] = prior_exposure_rate(
            df, 'player_id', 'saves', window, 12.0)

    # Raw saves rolling (recent form counts)
    df['saves_last1'] = prior_stat(df, 'player_id', 'saves', aggregation='last').fillna(0)
    for window in ROLLING_WINDOWS:
        df[f'saves_roll{window}'] = prior_stat(df, 'player_id', 'saves', window, 'mean', 1)

    # xGoT faced per 90 (shot quality faced by GK)
    df['xgot_faced_per90'] = np.where(sufficient_minutes, df['xgot_faced'] / mins_90, np.nan)
    df['xgot_faced_per90'] = df['xgot_faced_per90'].clip(upper=5.0)

    for window in ROLLING_WINDOWS:
        df[f'xgot_faced_per90_roll{window}'] = prior_exposure_rate(
            df, 'player_id', 'xgot_faced', window, 5.0)

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

    # FPL position is authoritative whenever known (so an FPL DEF playing as a
    # winger still uses CBIT/10). Historical gaps fall back to that match's
    # FotMob role rather than discarding otherwise useful DefCon history.
    df = resolve_defcon_positions(df)
    defcon_def = df['defcon_position'].eq('DEF').fillna(False)
    defcon_mid = df['defcon_position'].eq('MID').fillna(False)
    eligible_defcon = defcon_def | defcon_mid
    df['defcon'] = np.where(defcon_def, df['CBIT'],
                            np.where(defcon_mid, df['CBIRT'], np.nan))
    df['defcon_threshold'] = np.where(defcon_def, 10,
                                      np.where(defcon_mid, 12, np.nan))
    df['hit_threshold'] = np.where(
        eligible_defcon, (df['defcon'] >= df['defcon_threshold']).astype(int), np.nan)

    # Defcon per 90
    df['defcon_per90'] = df['defcon'] / mins_90
    for window in ROLLING_WINDOWS:
        df[f'defcon_per90_roll{window}'] = prior_exposure_rate(
            df, 'player_id', 'defcon', window)
        df[f'hit_threshold_roll{window}'] = prior_stat(df, 'player_id', 'hit_threshold', window, 'mean', 1)

    # Raw defcon rolling counts (for Poisson count model)
    for window in ROLLING_WINDOWS:
        df[f'defcon_roll{window}'] = prior_stat(df, 'player_id', 'defcon', window, 'mean', 1)
    df['defcon_last1'] = prior_stat(df, 'player_id', 'defcon', aggregation='last')

    # Component stats per 90
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries']:
        df[f'{col}_per90'] = df[col] / mins_90
        for window in ROLLING_WINDOWS:
            df[f'{col}_per90_roll{window}'] = prior_exposure_rate(
                df, 'player_id', col, window)

    # =========================================================================
    # LIFETIME PLAYER PROFILE
    # =========================================================================

    # Keyed on player_id, not player_name: FotMob re-spells players mid-history
    # (Pascal Gross -> Pascal Groß, Bruno Guimaraes -> Bruno Guimarães, 60 players
    # / 5.6% of rows), which splits one career into two lifetime profiles, and it
    # reuses names across different players (Aaron Ramsey, Danilo, Joshua King),
    # which merges two careers into one. Every other rolling feature already keys
    # on player_id. The sort has to match the grouping — expanding() depends on
    # row order, and sorting by name would interleave a re-spelled player's blocks.
    df = df.sort_values(['player_id', 'match_date'], kind='stable').reset_index(drop=True)

    # Offensive stats
    for stat in ['goals', 'assists', 'xg', 'xa', 'minutes', 'shots']:
        if stat in df.columns:
            df[f'lifetime_{stat}'] = prior_stat(df, 'player_id', stat, aggregation='sum')

    # Goalkeeper stats (lifetime)
    for stat in ['saves', 'xgot_faced']:
        if stat in df.columns:
            df[f'lifetime_{stat}'] = prior_stat(df, 'player_id', stat, aggregation='sum')

    # Defensive stats (lifetime)
    df['lifetime_defcon'] = prior_stat(df, 'player_id', 'defcon', aggregation='sum')
    df['_defcon_minutes'] = df['minutes'].where(df['defcon'].notna())
    df['lifetime_defcon_minutes'] = prior_stat(
        df, 'player_id', '_defcon_minutes', aggregation='sum').fillna(0)
    for col in ['tackles', 'interceptions', 'clearances', 'blocks', 'recoveries', 'fouls_committed']:
        df[f'lifetime_{col}'] = prior_stat(df, 'player_id', col, aggregation='sum')

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
    lifetime_defcon_mins_90 = np.maximum(df['lifetime_defcon_minutes'] / 90, 0.01)
    df['lifetime_defcon_per90'] = np.where(
        df['lifetime_defcon_minutes'] >= 90,
        df['lifetime_defcon'].fillna(0) / lifetime_defcon_mins_90,
        0
    )
    df = df.drop(columns=['_defcon_minutes'], errors='ignore')
    for col in ['tackles', 'interceptions', 'clearances', 'fouls_committed']:
        df[f'lifetime_{col}_per90'] = np.where(
            df['lifetime_minutes'] >= 90,
            df[f'lifetime_{col}'].fillna(0) / lifetime_mins_90,
            0
        )

    # Lifetime yellow card per 90 (if yellow_cards column exists)
    if 'yellow_cards' in df.columns:
        df['lifetime_yellow_cards'] = prior_stat(df, 'player_id', 'yellow_cards', aggregation='sum').fillna(0)
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

    df['lifetime_appearances'] = prior_stat(df.assign(_one=1), 'player_id', '_one', aggregation='sum').fillna(0)
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

    team_stats = df.groupby(['team', 'season', 'gameweek', 'match_id']).agg({
        'goals': 'sum', 'xg': 'sum', 'shots': 'sum', 'own_goal': 'sum'
    }).reset_index()

    # Add opponent own goals to team goals (own goals by opponent count as goals for this team)
    if 'opponent' in df.columns:
        opp_og = df.groupby(['opponent', 'season', 'gameweek', 'match_id'])['own_goal'].sum().reset_index()
        opp_og = opp_og.rename(columns={'opponent': 'team', 'own_goal': 'opp_own_goals'})
        team_stats = team_stats.merge(opp_og, on=['team', 'season', 'gameweek', 'match_id'], how='left')
        team_stats['opp_own_goals'] = team_stats['opp_own_goals'].fillna(0)
        team_stats['team_goals'] = team_stats['goals'] + team_stats['opp_own_goals']
    else:
        team_stats['team_goals'] = team_stats['goals']

    team_stats = team_stats.rename(columns={'xg': 'team_xg', 'shots': 'team_shots'})
    team_stats = team_stats.drop(columns=['goals', 'own_goal'], errors='ignore')
    timing = df[['match_id', 'match_date', 'forecast_time', 'result_time']].drop_duplicates('match_id')
    team_stats = team_stats.merge(timing, on='match_id', validate='many_to_one')
    team_stats = team_stats.sort_values(['team', 'match_date'])

    for col in ['team_goals', 'team_xg', 'team_shots']:
        for window in ROLLING_WINDOWS:
            team_stats[f'{col}_roll{window}'] = prior_stat(team_stats, 'team', col, window, 'mean', min(TEAM_ROLL_MIN_PERIODS, window))

    # Merge rolling team offensive stats (dynamic column selection)
    team_roll_cols = [c for c in team_stats.columns if '_roll' in c]
    df = df.merge(
        team_stats[['team', 'season', 'gameweek', 'match_id'] + team_roll_cols],
        on=['team', 'season', 'gameweek', 'match_id'], how='left'
    )

    # =========================================================================
    # PLAYER SHARE / CENTRALITY FEATURES
    # =========================================================================

    # Merge raw per-match team totals for share computation
    df = df.merge(
        team_stats[['team', 'season', 'gameweek', 'match_id', 'team_goals', 'team_xg', 'team_shots']],
        on=['team', 'season', 'gameweek', 'match_id'], how='left'
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
            df[f'{share_col}_roll{window}'] = prior_stat(df, 'player_id', share_col, window, 'mean', 1)

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
        # 'team' comes from player_stats, 'opponent' from fixtures, and the two
        # spell Bournemouth and Brighton differently. Without this the defensive
        # stats (keyed by opponent_norm after the swap) never join back.
        name = TEAM_NAME_MAP.get(str(name).strip(), name)
        return str(name).lower().replace(' ', '_').replace("'", "").strip()

    if 'opponent' in df.columns:
        # Create normalized versions for matching
        df['team_norm'] = df['team'].apply(normalize_name)
        df['opponent_norm'] = df['opponent'].apply(normalize_name)

        # Get goals conceded by each team (= opponent's goals + this team's own goals)
        match_results = df.groupby(['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id']).agg({
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
        # Include opponent_norm in key to prevent cartesian product for DGW teams
        team_og = match_results[['team_norm', 'opponent_norm', 'season', 'gameweek', 'match_id', 'own_goal']].rename(
            columns={'own_goal': 'self_own_goals', 'opponent_norm': 'opponent_temp_norm'}
        )
        team_conceded = team_conceded.merge(team_og, on=['team_norm', 'opponent_temp_norm', 'season', 'gameweek', 'match_id'], how='left')
        team_conceded['self_own_goals'] = team_conceded['self_own_goals'].fillna(0)
        team_conceded['goals_conceded'] = team_conceded['opp_goals'] + team_conceded['self_own_goals']
        team_conceded = team_conceded.merge(timing, on='match_id', validate='many_to_one')
        team_conceded = team_conceded.sort_values(['team_norm', 'match_date'])

        # Rolling goals conceded and xGA (multiple windows for different time horizons)
        for window in ROLLING_WINDOWS_LONG:
            team_conceded[f'team_conceded_roll{window}'] = prior_stat(team_conceded, 'team_norm', 'goals_conceded', window, 'mean', min(TEAM_ROLL_MIN_PERIODS, window))
            team_conceded[f'team_xga_roll{window}'] = prior_stat(team_conceded, 'team_norm', 'xga', window, 'mean', min(TEAM_ROLL_MIN_PERIODS, window))

        # Clean sheet tracking
        team_conceded['clean_sheet'] = (team_conceded['goals_conceded'] == 0).astype(int)
        for window in ROLLING_WINDOWS_LONG:
            team_conceded[f'team_cs_rate_roll{window}'] = prior_stat(team_conceded, 'team_norm', 'clean_sheet', window, 'mean', min(TEAM_ROLL_MIN_PERIODS, window))

        # Merge team defensive stats (dynamic column selection)
        # Deduplicate per (team, season, gameweek) to avoid cartesian for DGW teams
        team_def_cols = [c for c in team_conceded.columns
                        if any(c.startswith(p) for p in ['team_conceded_roll', 'team_xga_roll', 'team_cs_rate_roll'])]
        team_conceded_dedup = team_conceded[['team_norm', 'season', 'gameweek', 'match_id'] + team_def_cols].drop_duplicates(
            subset=['team_norm', 'season', 'gameweek', 'match_id'], keep='first'
        )
        df = df.merge(
            team_conceded_dedup,
            on=['team_norm', 'season', 'gameweek', 'match_id'], how='left'
        )

    # =========================================================================
    # OPPONENT ROLLING STATS (Offensive + Defensive)
    # =========================================================================

    if 'opponent' in df.columns:
        # Add normalized team name to team_stats for opponent matching
        team_stats['team_norm'] = team_stats['team'].apply(normalize_name)

        # Opponent's goals scored (their attacking strength)
        opp_offense = team_stats[['team_norm', 'season', 'gameweek', 'match_id', 'team_goals', 'team_xg']].copy()
        opp_offense = opp_offense.rename(columns={
            'team_norm': 'opponent_norm',
            'team_goals': 'opp_goals',
            'team_xg': 'opp_xg'
        })
        opp_offense = opp_offense.merge(timing, on='match_id', validate='many_to_one')
        opp_offense = opp_offense.sort_values(['opponent_norm', 'match_date'])

        for window in ROLLING_WINDOWS:
            opp_offense[f'opp_goals_roll{window}'] = prior_stat(opp_offense, 'opponent_norm', 'opp_goals', window, 'mean', min(TEAM_ROLL_MIN_PERIODS, window))
            opp_offense[f'opp_xg_roll{window}'] = prior_stat(opp_offense, 'opponent_norm', 'opp_xg', window, 'mean', min(TEAM_ROLL_MIN_PERIODS, window))

        opp_off_roll_cols = [c for c in opp_offense.columns if c.startswith('opp_') and '_roll' in c]
        df = df.merge(
            opp_offense[['opponent_norm', 'season', 'gameweek', 'match_id'] + opp_off_roll_cols],
            on=['opponent_norm', 'season', 'gameweek', 'match_id'], how='left'
        )

        # Opponent's defensive weakness (goals they concede = xGA) + CS rate
        opp_def_roll_cols = [c for c in team_conceded.columns
                           if any(c.startswith(p) for p in ['team_conceded_roll', 'team_xga_roll', 'team_cs_rate_roll'])]
        opp_defense = team_conceded[['team_norm', 'season', 'gameweek', 'match_id'] + opp_def_roll_cols].copy()
        rename_map = {'team_norm': 'opponent_norm'}
        for col in opp_def_roll_cols:
            new_name = (col.replace('team_conceded_', 'opp_conceded_')
                           .replace('team_xga_', 'opp_xga_')
                           .replace('team_cs_rate_', 'opp_cs_rate_'))
            rename_map[col] = new_name
        opp_defense = opp_defense.rename(columns=rename_map)

        opp_def_merged_cols = [c for c in opp_defense.columns if c.startswith('opp_')]
        opp_defense_dedup = opp_defense[['opponent_norm', 'season', 'gameweek', 'match_id'] + opp_def_merged_cols].drop_duplicates(
            subset=['opponent_norm', 'season', 'gameweek', 'match_id'], keep='first'
        )
        df = df.merge(
            opp_defense_dedup,
            on=['opponent_norm', 'season', 'gameweek', 'match_id'], how='left'
        )

        # ================================================================
        # COLD-START GUARD
        # min_periods=min(3, w) still lets roll1 through, so a club with a
        # single PL match has that one result as its only real signal while
        # every wider window is a prior -- Hull's clean sheet reads as
        # "never concedes", Coventry's 0-3 as "always concedes 3". Blank every
        # team aggregate until the club has TEAM_ROLL_MIN_PERIODS games, and
        # every opponent aggregate until the OPPONENT does, so the promoted
        # prior below covers them uniformly. Only newly promoted clubs qualify --
        # established clubs keep their cross-season history untouched.
        # ================================================================
        _promoted = promoted_team_seasons(df)
        _tm = df[['team_norm', 'season', 'gameweek', 'match_id']].drop_duplicates().sort_values(
            ['team_norm', 'season', 'gameweek', 'match_id'])
        # Games so far THIS season: a club returning after a relegation spell has
        # plenty of career games but no current-season form, and its old top-flight
        # numbers are the stale ones we are trying not to carry over.
        _tm = _tm.merge(timing, on='match_id', validate='many_to_one')
        _tm['_one'] = 1
        _tm['_prior_games'] = prior_stat(_tm, ['team_norm', 'season'], '_one', aggregation='sum').fillna(0)
        _tm['_cold'] = [
            (t, sn) in _promoted and g < TEAM_ROLL_MIN_PERIODS
            for t, sn, g in zip(_tm['team_norm'], _tm['season'], _tm['_prior_games'])
        ]
        _tm = _tm.drop(columns=['_prior_games', '_one', 'match_date', 'forecast_time', 'result_time'])
        df = df.merge(_tm, on=['team_norm', 'season', 'gameweek', 'match_id'], how='left')
        df = df.merge(
            _tm.rename(columns={'team_norm': 'opponent_norm', '_cold': '_opp_cold'}),
            on=['opponent_norm', 'season', 'gameweek', 'match_id'], how='left')

        _cold = df['_cold'].fillna(False).astype(bool)
        _opp_cold = df['_opp_cold'].fillna(False).astype(bool)
        for _c in [c for c in df.columns if c.startswith('team_') and '_roll' in c]:
            df.loc[_cold, _c] = np.nan
        for _c in [c for c in df.columns if c.startswith('opp_') and '_roll' in c]:
            df.loc[_opp_cold, _c] = np.nan
        df = df.drop(columns=['_cold', '_opp_cold'], errors='ignore')

        # ================================================================
        # PROMOTED-TEAM PRIOR
        # Teams with no PL history in the dataset (newly promoted sides)
        # produce NaN team/opponent rolling stats, which the blanket fill
        # below would turn into 0 ("never scores, never concedes"). Fill
        # them instead with the promoted-cohort average: the early-season
        # (GW<=10) rolling values of previously promoted teams.
        # ================================================================
        seasons_sorted = sorted(df['season'].dropna().unique())
        promoted_pairs = set()
        for prev_s, cur_s in zip(seasons_sorted, seasons_sorted[1:]):
            prev_teams = set(df.loc[df['season'] == prev_s, 'team_norm'].unique())
            cur_teams = set(df.loc[df['season'] == cur_s, 'team_norm'].unique())
            promoted_pairs |= {(t, cur_s) for t in cur_teams - prev_teams}

        team_roll_prior_cols = [c for c in df.columns if c.startswith('team_') and '_roll' in c]
        opp_roll_cols = [c for c in df.columns if c.startswith('opp_') and '_roll' in c]

        if team_roll_prior_cols:
            is_promoted_row = pd.Series(list(zip(df['team_norm'], df['season'])), index=df.index).isin(promoted_pairs)
            for season in seasons_sorted:
                # A new season's prior is frozen from earlier seasons only.
                cohort = df[is_promoted_row & (df['gameweek'] <= 10) & (df['season'] < season)]
                season_mask = df['season'].eq(season)
                prior = {c: cohort[c].mean() for c in team_roll_prior_cols}
                for c in team_roll_prior_cols + opp_roll_cols:
                    team_c = (c.replace('opp_conceded_', 'team_conceded_')
                               .replace('opp_xga_', 'team_xga_').replace('opp_cs_rate_', 'team_cs_rate_')
                               .replace('opp_goals_', 'team_goals_').replace('opp_xg_', 'team_xg_'))
                    value = prior.get(team_c, np.nan)
                    if not np.isfinite(value):
                        value = 0.25 if 'cs_rate' in c else 12.0 if 'shots' in c else 1.3
                    df.loc[season_mask, c] = df.loc[season_mask, c].fillna(value)

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

    df = add_manager_embeddings(df, data_dir=data_dir, verbose=verbose)

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
    df.attrs.update(input_attrs)

    if verbose:
        print(f"  Computed {len(rolling_cols)} rolling/lifetime features")

    return df
