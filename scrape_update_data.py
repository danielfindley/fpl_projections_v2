"""
FotMob Incremental Data Updater
Updates specific gameweeks without re-scraping everything.
Uses Playwright to bypass Cloudflare Turnstile protection.
Uses FPL API for accurate gameweek-to-match mapping (handles double GWs).

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
import json
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright

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

# Canonical file paths (no more timestamped copies)
PLAYER_STATS_FILE = DATA_DIR / 'players' / 'player_stats.csv'
SHOTMAP_FILE = DATA_DIR / 'matches' / 'shotmap.csv'
MATCH_DETAILS_FILE = DATA_DIR / 'matches' / 'match_details.csv'
FIXTURES_FILE = DATA_DIR / 'fixtures.csv'

# FPL short name -> FotMob full name mapping
FPL_TO_FOTMOB_TEAMS = {
    'arsenal': 'arsenal',
    'aston villa': 'aston villa',
    'bournemouth': 'afc bournemouth',
    'brentford': 'brentford',
    'brighton': 'brighton & hove albion',
    'burnley': 'burnley',
    'chelsea': 'chelsea',
    'crystal palace': 'crystal palace',
    'everton': 'everton',
    'fulham': 'fulham',
    'leeds': 'leeds united',
    'liverpool': 'liverpool',
    'man city': 'manchester city',
    'man utd': 'manchester united',
    'newcastle': 'newcastle united',
    "nott'm forest": 'nottingham forest',
    'sunderland': 'sunderland',
    'spurs': 'tottenham hotspur',
    'west ham': 'west ham united',
    'wolves': 'wolverhampton wanderers',
    'leicester': 'leicester city',
    'ipswich': 'ipswich town',
    'sheffield utd': 'sheffield united',
    'southampton': 'southampton',
    'luton': 'luton town',
}

# =============================================================================
# PLAYWRIGHT BROWSER FOR FOTMOB
# =============================================================================

class FotMobBrowser:
    """Context manager that uses Playwright to get past Cloudflare Turnstile,
    then uses the session cookies for fast API requests."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._session = None  # requests.Session with browser cookies

    def __enter__(self):
        print("Launching browser to solve Cloudflare challenge...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled'],
        )
        self._context = self._browser.new_context(
            viewport={'width': 1920, 'height': 1080},
        )
        self._page = self._context.new_page()

        # Visit homepage to trigger and pass Turnstile challenge
        self._page.goto("https://www.fotmob.com", wait_until="domcontentloaded", timeout=30000)
        self._page.wait_for_timeout(5000)

        # Extract cookies and user-agent for use with requests
        cookies = self._context.cookies()
        user_agent = self._page.evaluate('() => navigator.userAgent')

        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': user_agent,
            'Referer': 'https://www.fotmob.com/',
        })
        for c in cookies:
            self._session.cookies.set(c['name'], c['value'], domain=c.get('domain', '.fotmob.com'))

        # Verify cookies work
        test = self._session.get(f"https://www.fotmob.com/api/matchDetails?matchId=4813627")
        if test.status_code == 200:
            print(f"Session established! ({len(cookies)} cookies)")
        else:
            print(f"Warning: Cookie test returned {test.status_code}, scraping may fail")

        # Close browser - we only needed it for cookies
        self._context.close()
        self._browser.close()
        self._browser = None
        self._context = None
        self._page = None

        return self

    def fetch_match_json(self, match_id):
        """Fetch match data using the authenticated session cookies."""
        try:
            r = self._session.get(
                f"https://www.fotmob.com/api/matchDetails?matchId={match_id}",
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
            else:
                print(f"  API returned {r.status_code} for match {match_id}")
                return None
        except Exception as e:
            print(f"  Request error for match {match_id}: {e}")
            return None

    def __exit__(self, *args):
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        print("Session closed.")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def random_delay():
    """Random delay between requests."""
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(delay)
    return delay


def extract_match_data(match_id, browser=None):
    """Fetch and extract all player stats from a FotMob match.

    Uses Playwright browser if provided, otherwise falls back to requests.
    """
    if browser:
        match = browser.fetch_match_json(match_id)
        if not match:
            return None, None, None
    else:
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
                'match_id': int(m.get('id')),
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


def migrate_to_canonical_files():
    """One-time migration: rename latest timestamped files to canonical names."""
    migrations = [
        ((DATA_DIR / 'players').glob('player_stats_8seasons_*.csv'), PLAYER_STATS_FILE),
        ((DATA_DIR / 'matches').glob('shotmap_8seasons_*.csv'), SHOTMAP_FILE),
        ((DATA_DIR / 'matches').glob('match_details_8seasons_*.csv'), MATCH_DETAILS_FILE),
    ]

    for file_glob, canonical_path in migrations:
        if canonical_path.exists():
            continue
        files = sorted(list(file_glob))
        if files:
            latest = files[-1]
            print(f"Migrating {latest.name} -> {canonical_path.name}")
            latest.rename(canonical_path)

    # Also migrate fixtures
    old_fixtures = DATA_DIR / 'all_fixtures_8_seasons.csv'
    if old_fixtures.exists() and not FIXTURES_FILE.exists():
        print(f"Migrating {old_fixtures.name} -> {FIXTURES_FILE.name}")
        old_fixtures.rename(FIXTURES_FILE)


def get_existing_match_ids():
    """Get set of match_ids already in our data."""
    if not PLAYER_STATS_FILE.exists():
        return set()

    print(f"Checking existing data in: {PLAYER_STATS_FILE.name}")
    df = pd.read_csv(PLAYER_STATS_FILE, usecols=['match_id'], on_bad_lines='skip')
    match_ids = set(int(mid) for mid in df['match_id'].unique() if pd.notna(mid))
    return match_ids


def parse_gameweeks(gw_args):
    """Parse gameweek arguments into list of ints.

    Handles: 23, "22 23", "20-23", ["22", "23"]
    """
    gameweeks = []
    for arg in gw_args:
        if '-' in str(arg) and not str(arg).startswith('-'):
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
# FPL GAMEWEEK RESOLUTION (handles double gameweeks)
# =============================================================================

def get_fpl_gameweek_matches(gameweeks):
    """Get the actual matches for FPL gameweeks from the official FPL API.

    Returns list of (home_team_lower, away_team_lower) tuples using FotMob-style names.
    This correctly handles double gameweeks where a team plays twice.
    """
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10
    ).json()
    fpl_teams = {t['id']: t['name'].lower() for t in bootstrap['teams']}

    matches = []
    for gw in gameweeks:
        fixtures = requests.get(
            f"https://fantasy.premierleague.com/api/fixtures/?event={gw}", timeout=10
        ).json()
        for f in fixtures:
            home_fpl = fpl_teams.get(f['team_h'], '')
            away_fpl = fpl_teams.get(f['team_a'], '')
            # Convert FPL names to FotMob names
            home_fotmob = FPL_TO_FOTMOB_TEAMS.get(home_fpl, home_fpl)
            away_fotmob = FPL_TO_FOTMOB_TEAMS.get(away_fpl, away_fpl)
            matches.append((home_fotmob, away_fotmob))

    return matches


def resolve_fotmob_match_ids(fpl_matches, fotmob_fixtures):
    """Find FotMob match IDs that correspond to FPL gameweek matches.

    Cross-references by team names (case-insensitive).
    """
    matched = []
    for home_fpl, away_fpl in fpl_matches:
        for fix in fotmob_fixtures:
            home_fm = fix['home_team'].lower()
            away_fm = fix['away_team'].lower()
            if home_fm == home_fpl and away_fm == away_fpl:
                matched.append(fix)
                break
        else:
            print(f"  Warning: Could not find FotMob match for {home_fpl} vs {away_fpl}")

    return matched


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

    # Migrate old timestamped files to canonical names (one-time)
    migrate_to_canonical_files()

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

    # Filter to requested matches
    if auto:
        matches_to_scrape = [f for f in all_fixtures if f['match_id'] not in existing_ids]
        print(f"Auto mode: {len(matches_to_scrape)} new matches to scrape")
    elif gameweeks:
        # Use FPL API to resolve actual matches (handles double GWs)
        print(f"Resolving FPL gameweeks {gameweeks} via FPL API...")
        fpl_matches = get_fpl_gameweek_matches(gameweeks)
        print(f"  FPL API returned {len(fpl_matches)} fixtures for GW {gameweeks}")

        gw_matches = resolve_fotmob_match_ids(fpl_matches, all_fixtures)
        print(f"  Matched {len(gw_matches)} FotMob fixtures")

        if force:
            matches_to_scrape = gw_matches
        else:
            matches_to_scrape = [f for f in gw_matches if f['match_id'] not in existing_ids]
            n_skipped = len(gw_matches) - len(matches_to_scrape)
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
    print(f"Matches per matchday: {dict(sorted(gw_counts.items()))}")

    # Scrape matches using Playwright browser
    all_players = []
    all_shots = []
    all_match_details = []
    failed = []

    with FotMobBrowser() as browser:
        for i, match in enumerate(matches_to_scrape):
            match_id = match['match_id']
            home = match.get('home_team', '?')
            away = match.get('away_team', '?')
            gw = match.get('round', '?')

            try:
                match_info, players_data, shots_data = extract_match_data(match_id, browser=browser)

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
                        print(f"[{i+1}/{len(matches_to_scrape)}] MD{gw}: {home} vs {away} ({len(players_data)} players)")
                else:
                    failed.append(match)
                    print(f"[{i+1}/{len(matches_to_scrape)}] FAILED: MD{gw} {home} vs {away}")

            except Exception as e:
                failed.append(match)
                print(f"[{i+1}/{len(matches_to_scrape)}] ERROR: {str(e)[:80]}")

            if i < len(matches_to_scrape) - 1:
                random_delay()

    # Save/append data
    if not all_players:
        print("\nNo data scraped!")
        return

    print("\n" + "=" * 60)
    print("Saving data...")

    # Ensure directories exist
    (DATA_DIR / 'players').mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'matches').mkdir(parents=True, exist_ok=True)

    # Create dataframes from new data
    df_players_new = pd.DataFrame(all_players)
    df_shots_new = pd.DataFrame(all_shots)
    df_matches_new = pd.DataFrame(all_match_details)

    # --- Player stats ---
    if PLAYER_STATS_FILE.exists():
        df_players_old = pd.read_csv(PLAYER_STATS_FILE, on_bad_lines='skip')
        df_players = pd.concat([df_players_old, df_players_new], ignore_index=True)
        df_players = df_players.drop_duplicates(subset=['match_id', 'player_id'], keep='last')
    else:
        df_players = df_players_new
    df_players.to_csv(PLAYER_STATS_FILE, index=False)
    print(f"Saved: {PLAYER_STATS_FILE} ({len(df_players)} total records)")

    # --- Shotmap ---
    if SHOTMAP_FILE.exists():
        df_shots_old = pd.read_csv(SHOTMAP_FILE, on_bad_lines='skip')
        df_shots = pd.concat([df_shots_old, df_shots_new], ignore_index=True)
        df_shots = df_shots.drop_duplicates(subset=['match_id', 'shot_id'], keep='last')
    else:
        df_shots = df_shots_new
    df_shots.to_csv(SHOTMAP_FILE, index=False)
    print(f"Saved: {SHOTMAP_FILE} ({len(df_shots)} total shots)")

    # --- Match details ---
    if MATCH_DETAILS_FILE.exists():
        df_matches_old = pd.read_csv(MATCH_DETAILS_FILE, on_bad_lines='skip')
        df_matches = pd.concat([df_matches_old, df_matches_new], ignore_index=True)
        df_matches = df_matches.drop_duplicates(subset=['match_id'], keep='last')
    else:
        df_matches = df_matches_new
    df_matches.to_csv(MATCH_DETAILS_FILE, index=False)
    print(f"Saved: {MATCH_DETAILS_FILE} ({len(df_matches)} total matches)")

    # --- Fixtures ---
    df_fix_new = pd.DataFrame(all_fixtures)
    if FIXTURES_FILE.exists():
        df_fix_old = pd.read_csv(FIXTURES_FILE, on_bad_lines='skip')
        df_fix = pd.concat([df_fix_old, df_fix_new], ignore_index=True)
        df_fix = df_fix.drop_duplicates(subset=['match_id'], keep='last')
    else:
        df_fix = df_fix_new
    df_fix.to_csv(FIXTURES_FILE, index=False)
    print(f"Saved: {FIXTURES_FILE} ({len(df_fix)} total fixtures)")

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
        help='FPL gameweek(s) to update. Can be single (23), multiple (22 23), or range (20-23). Handles double GWs automatically.'
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
