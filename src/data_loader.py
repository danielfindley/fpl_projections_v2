"""
Data loading utilities for FotMob data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import time


# Column name mapping from FotMob to internal names
COLUMN_MAP = {
    'name': 'player_name',
    'minutes_played': 'minutes',
    'expected_goals_(xg)': 'xg',
    'expected_assists_(xa)': 'xa',
    'xg_non-penalty': 'npxg',
    'total_shots': 'shots',
    'chances_created': 'key_passes',
    'round': 'gameweek',
}


def load_player_stats(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Load player stats from FotMob CSV files."""
    data_dir = Path(data_dir)

    # Try canonical path first, fall back to timestamped glob
    player_file = data_dir / 'players' / 'player_stats.csv'
    if not player_file.exists():
        player_files = list((data_dir / 'players').glob('player_stats_*.csv'))
        if not player_files:
            raise FileNotFoundError(f"No player stats files found in {data_dir / 'players'}")
        player_file = sorted(player_files)[-1]

    if verbose:
        print(f"Loading player stats from: {player_file.name}")
    
    df = pd.read_csv(player_file, on_bad_lines='skip')
    
    # Rename columns
    df = df.rename(columns=COLUMN_MAP)
    
    # Ensure numeric columns
    numeric_cols = ['minutes', 'goals', 'assists', 'xg', 'xa', 'shots', 'key_passes',
                    'tackles', 'interceptions', 'clearances', 'blocks', 'recoveries',
                    'fotmob_rating', 'touches', 'touches_in_opposition_box']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Create player_id if not exists
    if 'player_id' not in df.columns:
        df['player_id'] = df['player_name'].astype(str) + '_' + df['team'].astype(str)
    
    # CRITICAL: Remove duplicate rows (same player, same match)
    # This can happen from scraping issues or data appending
    n_before = len(df)
    
    # Normalize strings to handle whitespace/encoding issues
    df['_player_norm'] = df['player_name'].astype(str).str.strip().str.lower()
    df['_team_norm'] = df['team'].astype(str).str.strip().str.lower()
    
    # Deduplicate using normalized columns
    df = df.drop_duplicates(subset=['match_id', '_player_norm', '_team_norm'], keep='first')
    
    # Drop temp columns
    df = df.drop(columns=['_player_norm', '_team_norm'])
    
    n_dupes = n_before - len(df)
    
    if verbose:
        print(f"  Loaded {len(df):,} player-match records")
        if n_dupes > 0:
            print(f"  Removed {n_dupes:,} duplicate rows")
        print(f"  Seasons: {sorted(df['season'].unique())}")
    
    return df


def load_fixtures(data_dir: Path, verbose: bool = True) -> pd.DataFrame:
    """Load fixtures from CSV."""
    data_dir = Path(data_dir)

    # Try canonical path first, fall back to old name
    fixtures_file = data_dir / 'fixtures.csv'
    if not fixtures_file.exists():
        fixtures_file = data_dir / 'all_fixtures_8_seasons.csv'
    if not fixtures_file.exists():
        raise FileNotFoundError(f"Fixtures file not found in {data_dir}")
    
    df = pd.read_csv(fixtures_file)
    df = df.rename(columns=COLUMN_MAP)
    
    # Remove duplicate fixtures (same match_id)
    n_before = len(df)
    df = df.drop_duplicates(subset=['match_id'], keep='first')
    n_dupes = n_before - len(df)
    
    if verbose:
        print(f"Loaded {len(df):,} fixtures")
        if n_dupes > 0:
            print(f"  Removed {n_dupes:,} duplicate fixtures")
    
    return df


def merge_fixtures(player_df: pd.DataFrame, fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Merge player stats with fixture info to get home/away and opponent."""
    # Create home/away lookup
    home_games = fixtures_df[['match_id', 'home_team', 'away_team']].copy()
    home_games['is_home'] = 1
    home_games = home_games.rename(columns={'home_team': 'team_fixture', 'away_team': 'opponent'})
    
    away_games = fixtures_df[['match_id', 'home_team', 'away_team']].copy()
    away_games['is_home'] = 0
    away_games = away_games.rename(columns={'away_team': 'team_fixture', 'home_team': 'opponent'})
    
    fixture_lookup = pd.concat([home_games, away_games], ignore_index=True)
    
    # Team name mapping to handle variations (FotMob vs FPL vs fixtures)
    TEAM_ALIASES = {
        'bournemouth': 'afc_bournemouth',
        'brighton': 'brighton_&_hove_albion',
        'brighton_and_hove_albion': 'brighton_&_hove_albion',
        'man_city': 'manchester_city',
        'man_utd': 'manchester_united',
        'man_united': 'manchester_united',
        'newcastle': 'newcastle_united',
        'nottm_forest': 'nottingham_forest',
        "nott'm_forest": 'nottingham_forest',
        'spurs': 'tottenham_hotspur',
        'tottenham': 'tottenham_hotspur',
        'wolves': 'wolverhampton_wanderers',
        'west_ham': 'west_ham_united',
        'leicester': 'leicester_city',
        'leeds': 'leeds_united',
        'sheffield_utd': 'sheffield_united',
        'ipswich': 'ipswich_town',
    }
    
    # Normalize team names for matching
    def normalize_team(name):
        if pd.isna(name):
            return ''
        name_lower = str(name).lower().replace(' ', '_').replace("'", "").strip()
        # Apply aliases
        return TEAM_ALIASES.get(name_lower, name_lower)
    
    fixture_lookup['team_norm'] = fixture_lookup['team_fixture'].apply(normalize_team)
    player_df = player_df.copy()
    player_df['team_norm'] = player_df['team'].apply(normalize_team)
    
    # Merge
    merged = player_df.merge(
        fixture_lookup[['match_id', 'team_norm', 'opponent', 'is_home']],
        on=['match_id', 'team_norm'],
        how='left'
    )
    
    merged = merged.drop(columns=['team_norm'])
    merged['is_home'] = merged['is_home'].fillna(0).astype(int)
    merged['opponent'] = merged['opponent'].fillna('Unknown')
    
    return merged


def get_fpl_positions() -> dict:
    """Fetch FPL positions from API."""
    import requests
    try:
        bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
        POS_MAP = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        
        positions = {}
        for p in bootstrap['elements']:
            names = [p['web_name'], f"{p['first_name']} {p['second_name']}", p['second_name']]
            pos = POS_MAP.get(p['element_type'], 'MID')
            for name in names:
                positions[name.lower()] = pos
        return positions
    except:
        return {}


def map_fpl_position(position_code, player_name: str = None, fpl_positions: dict = None) -> str:
    """Map FotMob position code to FPL position."""
    # Try FPL API first
    if fpl_positions and player_name:
        name_lower = player_name.lower()
        if name_lower in fpl_positions:
            return fpl_positions[name_lower]
        # Try last name
        parts = player_name.split()
        if len(parts) > 1 and parts[-1].lower() in fpl_positions:
            return fpl_positions[parts[-1].lower()]
    
    # Fall back to FotMob position code
    try:
        code = int(position_code)
        return {0: 'GK', 1: 'DEF', 2: 'MID', 3: 'FWD'}.get(code, 'MID')
    except:
        return 'MID'


def get_fpl_availability() -> dict:
    """Fetch player availability from FPL API.
    
    Returns dict: player_name (lowercase) -> {
        'chance_of_playing': 0-100 or None,
        'status': 'a' (available), 'i' (injured), 'd' (doubtful), 's' (suspended), 'u' (unavailable)
        'news': injury/suspension description
    }
    """
    import requests
    try:
        bootstrap = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10).json()
        
        availability = {}
        for p in bootstrap['elements']:
            names = [
                p['web_name'].lower(),
                f"{p['first_name']} {p['second_name']}".lower(),
                p['second_name'].lower()
            ]
            
            info = {
                'chance_of_playing': p.get('chance_of_playing_next_round'),
                'status': p.get('status', 'a'),  # a=available, i=injured, d=doubtful, s=suspended
                'news': p.get('news', ''),
                'fpl_id': p['id'],
            }
            
            for name in names:
                availability[name] = info
        
        return availability
    except Exception as e:
        print(f"Warning: Could not fetch FPL availability: {e}")
        return {}


def fetch_fpl_actual_points(gameweeks: list = None, cache_dir: str = None,
                            verbose: bool = True) -> pd.DataFrame:
    """Fetch actual FPL points from the FPL API for all completed gameweeks.

    Uses the event/{gw}/live/ endpoint for each gameweek and bootstrap-static
    for player/team metadata.

    Args:
        gameweeks: Specific GWs to fetch. If None, fetches all completed GWs.
        cache_dir: If provided, caches results as CSV to avoid re-fetching.
        verbose: Print progress.

    Returns:
        DataFrame with columns: fpl_id, player_name, web_name, team, gameweek,
        actual_total_points, minutes, goals_scored, assists, bonus, clean_sheets,
        yellow_cards, red_cards, saves, goals_conceded
    """
    import requests

    cache_path = Path(cache_dir) / 'fpl_actual_points.csv' if cache_dir else None

    # Fetch bootstrap for player/team metadata
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=15
    ).json()

    teams_map = {t['id']: t['name'] for t in bootstrap['teams']}

    # Build player lookup
    player_lookup = {}
    for p in bootstrap['elements']:
        player_lookup[p['id']] = {
            'full_name': f"{p['first_name']} {p['second_name']}",
            'web_name': p['web_name'],
            'team': teams_map.get(p['team'], ''),
            'team_id': p['team'],
            'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(p['element_type'], 'MID'),
        }

    # Determine which GWs to fetch
    if gameweeks is None:
        gameweeks = [ev['id'] for ev in bootstrap['events'] if ev.get('finished')]

    if verbose:
        print(f"Fetching actual FPL points for GW {min(gameweeks)}-{max(gameweeks)} ({len(gameweeks)} GWs)...")

    # Check cache for already-fetched GWs
    cached_df = None
    cached_gws = set()
    if cache_path and cache_path.exists():
        cached_df = pd.read_csv(cache_path)
        cached_gws = set(cached_df['gameweek'].unique())
        if verbose:
            print(f"  Cache has {len(cached_gws)} GWs already")

    gws_to_fetch = [gw for gw in gameweeks if gw not in cached_gws]

    rows = []
    for gw in gws_to_fetch:
        try:
            live = requests.get(
                f"https://fantasy.premierleague.com/api/event/{gw}/live/", timeout=15
            ).json()
            for el in live['elements']:
                pid = el['id']
                stats = el['stats']
                pinfo = player_lookup.get(pid, {})
                rows.append({
                    'fpl_id': pid,
                    'player_name': pinfo.get('full_name', ''),
                    'web_name': pinfo.get('web_name', ''),
                    'team': pinfo.get('team', ''),
                    'fpl_position': pinfo.get('position', 'MID'),
                    'gameweek': gw,
                    'actual_total_points': stats.get('total_points', 0),
                    'minutes': stats.get('minutes', 0),
                    'goals_scored': stats.get('goals_scored', 0),
                    'assists': stats.get('assists', 0),
                    'bonus': stats.get('bonus', 0),
                    'clean_sheets': stats.get('clean_sheets', 0),
                    'yellow_cards': stats.get('yellow_cards', 0),
                    'red_cards': stats.get('red_cards', 0),
                    'saves': stats.get('saves', 0),
                    'goals_conceded': stats.get('goals_conceded', 0),
                })
            if verbose:
                print(f"  GW{gw}: {sum(1 for el in live['elements'] if el['stats']['minutes'] > 0)} players played")
            time.sleep(0.3)  # Rate limit
        except Exception as e:
            if verbose:
                print(f"  GW{gw}: FAILED ({e})")

    new_df = pd.DataFrame(rows)

    # Merge with cache
    if cached_df is not None and len(cached_df) > 0:
        result = pd.concat([cached_df, new_df], ignore_index=True)
    else:
        result = new_df

    # Save cache
    if cache_path and len(result) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        if verbose:
            print(f"  Cached to {cache_path}")

    if verbose:
        print(f"  Total: {len(result):,} player-gameweek records")

    return result
