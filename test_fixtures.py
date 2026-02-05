"""Test fixture fetching."""
import sys
sys.path.insert(0, '.')
from src.pipeline import FPLPipeline, normalize_team_name

# Test normalization
test_names = ['Burnley', 'Man City', 'Spurs', 'Newcastle', 'Brighton', 'Wolves', "Nott'm Forest"]
print('Normalization test:')
for name in test_names:
    norm = normalize_team_name(name)
    print(f'  {name!r:25} -> {norm!r}')

# Test fixture fetching
pipeline = FPLPipeline(data_dir='data')
fixtures = pipeline._get_gw_fixtures(23, '2025/2026')
print(f'\nGW23 fixtures from pipeline ({len(fixtures)} fixtures):')
print(fixtures)

# Also test matching
print('\nTeam name matching test:')
player_teams = ['Burnley', 'Manchester City', 'Tottenham Hotspur', 'Newcastle United', 
                'Brighton and Hove Albion', 'Wolverhampton Wanderers', 'Nottingham Forest']

for player_team in player_teams:
    player_norm = normalize_team_name(player_team)
    matched = False
    for _, fix in fixtures.iterrows():
        home_norm = normalize_team_name(fix['home_team'])
        away_norm = normalize_team_name(fix['away_team'])
        
        if player_norm == home_norm or player_norm in home_norm or home_norm in player_norm:
            print(f'  {player_team:30} -> HOME vs {fix["away_team"]}')
            matched = True
            break
        elif player_norm == away_norm or player_norm in away_norm or away_norm in player_norm:
            print(f'  {player_team:30} -> AWAY vs {fix["home_team"]}')
            matched = True
            break
    
    if not matched:
        print(f'  {player_team:30} -> NO MATCH (norm: {player_norm})')





