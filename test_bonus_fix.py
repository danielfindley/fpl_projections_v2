"""Test bonus fix for per90 inflation."""
import pandas as pd

df = pd.read_csv('data/predictions/gw23_2025-2026.csv')

# Look at the Man City vs Wolves match
match_df = df[(df['team'].str.contains('City|Wolves|Wolverhampton', case=False, na=False)) | 
              (df['opponent'].str.contains('City|Wolves|Wolverhampton', case=False, na=False))]

# Sort by predicted bonus
match_df = match_df.sort_values('pred_bonus', ascending=False)

print('Man City vs Wolves - Top 15 by pred_bonus (AFTER FIX):')
cols = ['player_name', 'team', 'fpl_position', 'pred_bonus', 'pred_minutes', 'pred_exp_goals']
print(match_df[cols].head(15).to_string())

print()
print('Tchatchoua specifically:')
tch = df[df['player_name'].str.contains('Tchatchoua', case=False, na=False)]
print(f"  pred_bonus: {tch['pred_bonus'].values[0]:.3f}")
print(f"  tackles_per90_roll5: {tch['tackles_per90_roll5'].values[0]:.2f}")
print(f"  clearances_per90_roll5: {tch['clearances_per90_roll5'].values[0]:.2f}")
print(f"  minutes_roll5: {tch['minutes_roll5'].values[0]:.1f}")

print()
print('Haaland specifically:')
haaland = df[df['player_name'].str.contains('Haaland', case=False, na=False)]
print(f"  pred_bonus: {haaland['pred_bonus'].values[0]:.3f}")
print(f"  pred_exp_goals: {haaland['pred_exp_goals'].values[0]:.3f}")





