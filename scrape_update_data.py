"""
FotMob Incremental Data Updater
Updates specific gameweeks without re-scraping everything.

Usage:
    # Update single gameweek
    python scrape_update_data.py --gameweek 23
    
    # Update multiple gameweeks
    python scrape_update_data.py --gameweek 22 23
    
    # Update gameweek range
    python scrape_update_data.py --gameweek 20-23
    
    # Update specific season (current by default)
    python scrape_update_data.py --gameweek 23 --season 2025/2026
    
    # Update all new matches (auto-detect what's missing)
    python scrape_update_data.py --auto
"""

import requests
import pandas as pd
import os
import time
import random
import argparse
from datetime import datetime
from pathlib import Path
from glob import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

PREMIER_LEAGUE_ID = 47
DATA_DIR = Path("data")
MIN_DELAY = 1.5
MAX_DELAY = 3.5

# =============================================================================
# HELPER FUNCTIONS (from scrape_historical.py)
# =============================================================================

def random_delay():
    """Random delay between requests."""
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)
    return delay


def extract_match_data(match_id):
    """Fetch and extract all player stats from a FotMob match."""
    url = f"https://www.fotmob.com/api/matchDetails?matchId={match_id}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return None, None, None
    
    match = response.json()
    content = match.get('content', {})
    general = match.get('general', {})
    
    match_info = {
        'match_id': match_id,
        'home_team': general.get('homeTeam', {}).get('name'),
        'away_team': general.get('awayTeam', {}).get('name'),
        'home_score': general.get('homeTeam', {}).get('score'),
        'away_score': general.get('awayTeam', {}).get('score'),
        'match_date': general.get('matchTimeUTCDate'),
    }
    
    players_data = []
    player_stats = content.get('playerStats', {})
    
    for player_id, player_data in player_stats.items():
        stats_list = player_data.get('stats', [])
        if not stats_list:
            continue
        
        player_row = {
            'match_id': match_id,
            'player_id': player_id,
            'name': player_data.get('name', 'Unknown'),
            'team': player_data.get('teamName', ''),
            'position': player_data.get('usualPosition', ''),
            'shirt_number': player_data.get('shirtNumber', ''),
        }
        
        for category in stats_list:
            category_stats = category.get('stats', {})
            if isinstance(category_stats, dict):
                for stat_name, stat_data in category_stats.items():
                    stat_obj = stat_data.get('stat', {})
                    value = stat_obj.get('value')
                    col_name = stat_name.replace(' ', '_').lower()
                    player_row[col_name] = value
        
        players_data.append(player_row)
    
    shotmap = content.get('shotmap', {}).get('shots', [])
    shots_data = []
    for shot in shotmap:
        shots_data.append({
            'match_id': match_id,
            'shot_id': shot.get('id'),
            'player_id': shot.get('playerId'),
            'player_name': shot.get('playerName'),
            'team_id': shot.get('teamId'),
            'minute': shot.get('min'),
            'x': shot.get('x'),
            'y': shot.get('y'),
            'xG': shot.get('expectedGoals'),
            'xGOT': shot.get('expectedGoalsOnTarget'),
            'event_type': shot.get('eventType'),
            'shot_type': shot.get('shotType'),
            'situation': shot.get('situation'),
            'is_on_target': shot.get('isOnTarget'),
            'is_blocked': shot.get('isBlocked'),
            'is_from_inside_box': shot.get('isFromInsideBox'),
        })
    
    return match_info, players_data, shots_data


def get_season_fixtures(season):
    """Get all fixtures for a specific season."""
    url = f"https://www.fotmob.com/api/leagues?id={PREMIER_LEAGUE_ID}&season={season}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Failed to fetch fixtures for {season}")
        return []
    
    data = response.json()
    fixtures_data = data.get('fixtures', {})
    matches = fixtures_data.get('allMatches', [])
    
    completed = []
    for m in matches:
        status = m.get('status', {})
        is_finished = status.get('finished', False) if isinstance(status, dict) else False
        
        if is_finished:
            completed.append({
                'match_id': int(m.get('id')),  # Ensure int for consistent comparison
                'season': season,
                'round': m.get('round') or m.get('roundName'),
                'home_team': m.get('home', {}).get('name'),
                'away_team': m.get('away', {}).get('name'),
                'home_score': m.get('home', {}).get('score'),
                'away_score': m.get('away', {}).get('score'),
            })
    
    return completed


def get_current_season():
    """Determine current season from FotMob."""
    url = f"https://www.fotmob.com/api/leagues?id={PREMIER_LEAGUE_ID}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    seasons = data.get('allAvailableSeasons', [])
    return seasons[0] if seasons else '2025/2026'


def get_latest_player_file():
    """Find the most recent player stats file."""
    player_files = list((DATA_DIR / 'players').glob('player_stats_*.csv'))
    if not player_files:
        return None
    return sorted(player_files)[-1]


def get_existing_match_ids():
    """Get set of match_ids already in our data."""
    player_file = get_latest_player_file()
    if player_file is None:
        return set()
    
    print(f"Checking existing data in: {player_file.name}")
    df = pd.read_csv(player_file, usecols=['match_id'], on_bad_lines='skip')
    # Convert to int to ensure consistent comparison with FotMob API
    match_ids = set(int(mid) for mid in df['match_id'].unique() if pd.notna(mid))
    return match_ids


def parse_gameweeks(gw_args):
    """Parse gameweek arguments into list of ints.
    
    Handles: 23, "22 23", "20-23", ["22", "23"]
    """
    gameweeks = []
    for arg in gw_args:
        if '-' in str(arg) and not str(arg).startswith('-'):
            # Range like "20-23"
            start, end = arg.split('-')
            gameweeks.extend(range(int(start), int(end) + 1))
        else:
            gameweeks.append(int(arg))
    return sorted(set(gameweeks))


def safe_int(val):
    """Safely convert value to int, returns None if not possible."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# =============================================================================
# MAIN UPDATE FUNCTION
# =============================================================================

def update_data(gameweeks=None, season=None, auto=False, force=False, verbose=True):
    """
    Update data for specific gameweeks.
    
    Args:
        gameweeks: List of gameweek numbers to update, or None for auto
        season: Season string like '2025/2026', or None for current
        auto: If True, automatically detect and scrape missing matches
        force: If True, re-scrape even if matches already exist in data
        verbose: Print progress
    """
    print("=" * 60)
    print("FotMob Incremental Data Updater")
    print("=" * 60)
    
    # Determine season
    if season is None:
        season = get_current_season()
    print(f"\nSeason: {season}")
    
    # Get all completed fixtures for the season
    print("Fetching fixture list...")
    all_fixtures = get_season_fixtures(season)
    print(f"Found {len(all_fixtures)} completed matches in {season}")
    
    if not all_fixtures:
        print("No completed fixtures found!")
        return
    
    # Get existing match IDs to avoid re-scraping (unless force=True)
    existing_ids = get_existing_match_ids() if not force else set()
    if existing_ids and verbose:
        print(f"Found {len(existing_ids)} existing match IDs in data")
    
    # Filter to requested gameweeks
    if auto:
        # Auto mode: find matches we don't have
        matches_to_scrape = [f for f in all_fixtures if f['match_id'] not in existing_ids]
        print(f"Auto mode: {len(matches_to_scrape)} new matches to scrape")
    elif gameweeks:
        gw_matches = [f for f in all_fixtures if safe_int(f['round']) in gameweeks]
        # Filter out matches we already have (unless force=True)
        if force:
            matches_to_scrape = gw_matches
            print(f"Filtering to gameweeks {gameweeks}: {len(gw_matches)} matches")
            print(f"  --force: will re-scrape all matches")
        else:
            matches_to_scrape = [f for f in gw_matches if f['match_id'] not in existing_ids]
            n_skipped = len(gw_matches) - len(matches_to_scrape)
            print(f"Filtering to gameweeks {gameweeks}: {len(gw_matches)} matches")
            if n_skipped > 0:
                print(f"  Skipping {n_skipped} matches already in data")
            print(f"  {len(matches_to_scrape)} new matches to scrape")
    else:
        print("No gameweeks specified and --auto not set. Nothing to do.")
        return
    
    if not matches_to_scrape:
        print("No matches to scrape!")
        return
    
    # Show what we're scraping
    gw_counts = {}
    for m in matches_to_scrape:
        gw = safe_int(m['round']) or m['round']
        gw_counts[gw] = gw_counts.get(gw, 0) + 1
    print(f"Matches per gameweek: {dict(sorted(gw_counts.items()))}")
    
    # Scrape matches
    all_players = []
    all_shots = []
    all_match_details = []
    failed = []
    
    for i, match in enumerate(matches_to_scrape):
        match_id = match['match_id']
        home = match.get('home_team', '?')
        away = match.get('away_team', '?')
        gw = match.get('round', '?')
        
        try:
            match_info, players_data, shots_data = extract_match_data(match_id)
            
            if match_info:
                match_info['season'] = season
                all_match_details.append(match_info)
                
                for p in players_data:
                    p['season'] = season
                all_players.extend(players_data)
                
                for s in shots_data:
                    s['season'] = season
                all_shots.extend(shots_data)
                
                if verbose:
                    print(f"[{i+1}/{len(matches_to_scrape)}] GW{gw}: {home} vs {away} ({len(players_data)} players)")
            else:
                failed.append(match)
                print(f"[{i+1}/{len(matches_to_scrape)}] FAILED: GW{gw} {home} vs {away}")
        
        except Exception as e:
            failed.append(match)
            print(f"[{i+1}/{len(matches_to_scrape)}] ERROR: {str(e)[:50]}")
        
        if i < len(matches_to_scrape) - 1:
            random_delay()
    
    # Save/append data
    if not all_players:
        print("\nNo data scraped!")
        return
    
    print("\n" + "=" * 60)
    print("Saving data...")
    
    # Create dataframes
    df_players_new = pd.DataFrame(all_players)
    df_shots_new = pd.DataFrame(all_shots)
    df_matches_new = pd.DataFrame(all_match_details)
    
    # Get existing files
    player_file = get_latest_player_file()
    
    if player_file:
        # Append to existing data
        print(f"Appending to existing file: {player_file.name}")
        
        df_players_old = pd.read_csv(player_file, on_bad_lines='skip')
        df_players_combined = pd.concat([df_players_old, df_players_new], ignore_index=True)
        df_players_combined = df_players_combined.drop_duplicates(subset=['match_id', 'player_id'], keep='last')
        
        # Save with new timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_player_file = DATA_DIR / 'players' / f'player_stats_8seasons_{timestamp}.csv'
        df_players_combined.to_csv(new_player_file, index=False)
        print(f"Saved: {new_player_file.name} ({len(df_players_combined)} total records)")
        
        # Also update shotmap
        shot_files = list((DATA_DIR / 'matches').glob('shotmap_*.csv'))
        if shot_files:
            shot_file = sorted(shot_files)[-1]
            df_shots_old = pd.read_csv(shot_file, on_bad_lines='skip')
            df_shots_combined = pd.concat([df_shots_old, df_shots_new], ignore_index=True)
            df_shots_combined = df_shots_combined.drop_duplicates(subset=['match_id', 'shot_id'], keep='last')
            new_shot_file = DATA_DIR / 'matches' / f'shotmap_8seasons_{timestamp}.csv'
            df_shots_combined.to_csv(new_shot_file, index=False)
            print(f"Saved: {new_shot_file.name} ({len(df_shots_combined)} total shots)")
        
        # Update match details
        match_files = list((DATA_DIR / 'matches').glob('match_details_*.csv'))
        if match_files:
            match_file = sorted(match_files)[-1]
            df_matches_old = pd.read_csv(match_file, on_bad_lines='skip')
            df_matches_combined = pd.concat([df_matches_old, df_matches_new], ignore_index=True)
            df_matches_combined = df_matches_combined.drop_duplicates(subset=['match_id'], keep='last')
            new_match_file = DATA_DIR / 'matches' / f'match_details_8seasons_{timestamp}.csv'
            df_matches_combined.to_csv(new_match_file, index=False)
            print(f"Saved: {new_match_file.name} ({len(df_matches_combined)} total matches)")
    else:
        # No existing data, create new files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        (DATA_DIR / 'players').mkdir(parents=True, exist_ok=True)
        (DATA_DIR / 'matches').mkdir(parents=True, exist_ok=True)
        
        df_players_new.to_csv(DATA_DIR / 'players' / f'player_stats_8seasons_{timestamp}.csv', index=False)
        df_shots_new.to_csv(DATA_DIR / 'matches' / f'shotmap_8seasons_{timestamp}.csv', index=False)
        df_matches_new.to_csv(DATA_DIR / 'matches' / f'match_details_8seasons_{timestamp}.csv', index=False)
        print(f"Created new data files with timestamp {timestamp}")
    
    # Update fixtures file
    fixtures_file = DATA_DIR / 'all_fixtures_8_seasons.csv'
    if fixtures_file.exists():
        df_fix_old = pd.read_csv(fixtures_file, on_bad_lines='skip')
        df_fix_new = pd.DataFrame(all_fixtures)
        df_fix_combined = pd.concat([df_fix_old, df_fix_new], ignore_index=True)
        df_fix_combined = df_fix_combined.drop_duplicates(subset=['match_id'], keep='last')
        df_fix_combined.to_csv(fixtures_file, index=False)
        print(f"Updated fixtures file ({len(df_fix_combined)} total fixtures)")
    
    print("=" * 60)
    print("UPDATE COMPLETE!")
    print(f"Added {len(df_players_new)} player records from {len(matches_to_scrape)} matches")
    if failed:
        print(f"Failed matches: {len(failed)}")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Update FotMob data for specific gameweeks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scrape_update_data.py --gameweek 23
  python scrape_update_data.py --gameweek 22 23
  python scrape_update_data.py --gameweek 20-23
  python scrape_update_data.py --auto
  python scrape_update_data.py --auto --season 2024/2025
        """
    )
    
    parser.add_argument(
        '--gameweek', '-g',
        nargs='+',
        help='Gameweek(s) to update. Can be single (23), multiple (22 23), or range (20-23)'
    )
    
    parser.add_argument(
        '--season', '-s',
        type=str,
        default=None,
        help='Season to update (e.g., 2025/2026). Defaults to current season.'
    )
    
    parser.add_argument(
        '--auto', '-a',
        action='store_true',
        help='Auto-detect and scrape all missing matches'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-scrape even if matches already exist (will overwrite)'
    )
    
    args = parser.parse_args()
    
    if not args.gameweek and not args.auto:
        parser.print_help()
        print("\nError: Must specify --gameweek or --auto")
        return
    
    gameweeks = parse_gameweeks(args.gameweek) if args.gameweek else None
    
    update_data(
        gameweeks=gameweeks,
        season=args.season,
        auto=args.auto,
        force=args.force
    )


if __name__ == "__main__":
    main()
