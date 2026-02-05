"""
FotMob Historical Data Scraper
Scrapes 8 seasons of Premier League player stats (2024-25 back to 2017-18)
"""

import requests
import pandas as pd
import json
import os
import time
import random
import sys
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

PREMIER_LEAGUE_ID = 47  # FotMob league ID
DATA_DIR = "data"
MIN_DELAY = 1.5  # Minimum seconds between requests
MAX_DELAY = 3.5  # Maximum seconds between requests

# Create data folders
os.makedirs(f"{DATA_DIR}/matches", exist_ok=True)
os.makedirs(f"{DATA_DIR}/players", exist_ok=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def random_delay():
    """Random delay between requests to be respectful."""
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
    
    # Match metadata
    match_info = {
        'match_id': match_id,
        'home_team': general.get('homeTeam', {}).get('name'),
        'away_team': general.get('awayTeam', {}).get('name'),
        'home_score': general.get('homeTeam', {}).get('score'),
        'away_score': general.get('awayTeam', {}).get('score'),
        'match_date': general.get('matchTimeUTCDate'),
    }
    
    # Extract player stats
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
        
        # Parse all stat categories
        for category in stats_list:
            category_stats = category.get('stats', {})
            if isinstance(category_stats, dict):
                for stat_name, stat_data in category_stats.items():
                    stat_obj = stat_data.get('stat', {})
                    value = stat_obj.get('value')
                    col_name = stat_name.replace(' ', '_').lower()
                    player_row[col_name] = value
        
        players_data.append(player_row)
    
    # Extract shotmap for detailed xG
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


def get_available_seasons():
    """Get list of available seasons from FotMob."""
    print("Fetching available seasons...")
    url = f"https://www.fotmob.com/api/leagues?id={PREMIER_LEAGUE_ID}"
    response = requests.get(url, headers=HEADERS)
    data = response.json()
    
    seasons = data.get('allAvailableSeasons', [])
    print(f"Found {len(seasons)} available seasons")
    return seasons[:8]  # Last 8 seasons


def get_season_fixtures(season):
    """Get all fixtures for a specific season."""
    url = f"https://www.fotmob.com/api/leagues?id={PREMIER_LEAGUE_ID}&season={season}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        return []
    
    data = response.json()
    fixtures_data = data.get('fixtures', {})
    matches = fixtures_data.get('allMatches', [])
    
    # Filter completed matches
    completed = []
    for m in matches:
        status = m.get('status', {})
        is_finished = status.get('finished', False) if isinstance(status, dict) else False
        
        if is_finished:
            completed.append({
                'match_id': m.get('id'),
                'season': season,
                'round': m.get('round') or m.get('roundName'),
                'home_team': m.get('home', {}).get('name'),
                'away_team': m.get('away', {}).get('name'),
                'home_score': m.get('home', {}).get('score'),
                'away_score': m.get('away', {}).get('score'),
            })
    
    return completed


# =============================================================================
# MAIN SCRAPER
# =============================================================================

def main():
    print("=" * 60)
    print("FotMob Historical Data Scraper")
    print("8 Seasons of Premier League Data")
    print("=" * 60)
    
    # Get seasons
    seasons = get_available_seasons()
    print(f"\nSeasons to scrape: {seasons}")
    
    # Get all fixtures
    all_completed_matches = []
    for season in seasons:
        print(f"\nFetching fixtures for {season}...")
        matches = get_season_fixtures(season)
        all_completed_matches.extend(matches)
        print(f"  Found {len(matches)} completed matches")
        random_delay()
    
    # Save fixtures
    df_fixtures = pd.DataFrame(all_completed_matches)
    df_fixtures.to_csv(f"{DATA_DIR}/all_fixtures_8_seasons.csv", index=False)
    print(f"\nTotal matches to scrape: {len(all_completed_matches)}")
    
    # Estimate time
    avg_delay = (MIN_DELAY + MAX_DELAY) / 2
    estimated_minutes = len(all_completed_matches) * avg_delay / 60
    print(f"Estimated time: ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")
    print("=" * 60)
    
    # Scrape all matches
    all_players = []
    all_shots = []
    all_match_details = []
    failed_matches = []
    
    for i, match in enumerate(all_completed_matches):
        match_id = match.get('match_id')
        season = match.get('season', '?')
        home = match.get('home_team', '?')
        away = match.get('away_team', '?')
        
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
                
                print(f"[{i+1}/{len(all_completed_matches)}] OK {season}: {home} vs {away} ({len(players_data)} players)")
            else:
                failed_matches.append({'match_id': match_id, 'season': season})
                print(f"[{i+1}/{len(all_completed_matches)}] FAIL {season}: {home} vs {away} - Failed")
        
        except Exception as e:
            failed_matches.append({'match_id': match_id, 'season': season, 'error': str(e)})
            print(f"[{i+1}/{len(all_completed_matches)}] ERROR: {str(e)[:40]}")
        
        # Save progress every 100 matches
        if (i + 1) % 100 == 0:
            print(f"\n--- Checkpoint: {i+1} matches ---")
            pd.DataFrame(all_players).to_csv(f"{DATA_DIR}/players/player_stats_progress.csv", index=False)
            pd.DataFrame(all_shots).to_csv(f"{DATA_DIR}/matches/shotmap_progress.csv", index=False)
            print(f"--- Progress saved ---\n")
        
        # Random delay
        if i < len(all_completed_matches) - 1:
            random_delay()
    
    # Save final data
    print("\n" + "=" * 60)
    print("Saving final data...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    df_players = pd.DataFrame(all_players)
    df_players.to_csv(f"{DATA_DIR}/players/player_stats_8seasons_{timestamp}.csv", index=False)
    
    df_shots = pd.DataFrame(all_shots)
    df_shots.to_csv(f"{DATA_DIR}/matches/shotmap_8seasons_{timestamp}.csv", index=False)
    
    df_matches = pd.DataFrame(all_match_details)
    df_matches.to_csv(f"{DATA_DIR}/matches/match_details_8seasons_{timestamp}.csv", index=False)
    
    if failed_matches:
        pd.DataFrame(failed_matches).to_csv(f"{DATA_DIR}/failed_matches.csv", index=False)
    
    print("=" * 60)
    print("SCRAPING COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved to {DATA_DIR}/:")
    print(f"  - players/player_stats_8seasons_{timestamp}.csv ({len(df_players)} records)")
    print(f"  - matches/shotmap_8seasons_{timestamp}.csv ({len(df_shots)} shots)")
    print(f"  - matches/match_details_8seasons_{timestamp}.csv ({len(df_matches)} matches)")
    print(f"  - all_fixtures_8_seasons.csv")
    print(f"\nFailed matches: {len(failed_matches)}")
    
    # Summary by season
    if len(df_players) > 0:
        print("\nRecords by season:")
        print(df_players.groupby('season').size().sort_index(ascending=False))


if __name__ == "__main__":
    main()
