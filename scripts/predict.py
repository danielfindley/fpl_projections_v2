#!/usr/bin/env python
"""Generate FPL predictions for a gameweek."""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import FPLPipeline


def main():
    parser = argparse.ArgumentParser(description='Generate FPL predictions')
    parser.add_argument('--gameweek', '-g', type=int, required=True, help='Target gameweek')
    parser.add_argument('--season', '-s', type=str, default='2025/2026', help='Season (default: 2025/2026)')
    parser.add_argument('--top', '-n', type=int, default=30, help='Show top N players (default: 30)')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"FPL PREDICTIONS - GW{args.gameweek} ({args.season})")
    print("=" * 70)
    
    # Run pipeline
    pipeline = FPLPipeline(data_dir='data')
    pipeline.load_data()
    pipeline.compute_features()
    pipeline.train(verbose=False)
    
    predictions = pipeline.predict(gameweek=args.gameweek, season=args.season)
    
    # Display top players
    print("\n" + "=" * 70)
    print(f"TOP {args.top} PLAYERS BY EXPECTED POINTS")
    print("=" * 70)
    
    top = pipeline.get_top_players(predictions, args.top)
    
    print(f"\n{'Player':<25} {'Team':<15} {'Pos':<4} {'Opp':<12} {'H/A':<3} {'Mins':<5} {'xG':<5} {'xA':<5} {'CS%':<5} {'xPts':<6}")
    print("-" * 95)
    
    for _, row in top.iterrows():
        ha = 'H' if row.get('is_home', 0) == 1 else 'A'
        print(f"{str(row.get('player_name', ''))[:24]:<25} "
              f"{str(row.get('team', ''))[:14]:<15} "
              f"{str(row.get('fpl_position', 'MID')):<4} "
              f"{str(row.get('opponent', ''))[:11]:<12} "
              f"{ha:<3} "
              f"{row.get('pred_minutes', 0):>4.0f} "
              f"{row.get('pred_exp_goals', 0):>5.2f} "
              f"{row.get('pred_exp_assists', 0):>5.2f} "
              f"{row.get('pred_cs_prob', 0)*100:>4.0f}% "
              f"{row.get('exp_total_pts', 0):>5.2f}")
    
    # Top by position
    print("\n" + "=" * 70)
    print("TOP 5 BY POSITION")
    print("=" * 70)
    
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        pos_df = predictions[predictions['fpl_position'] == pos].nlargest(5, 'exp_total_pts')
        print(f"\n{pos}:")
        for _, row in pos_df.iterrows():
            ha = '(H)' if row.get('is_home', 0) == 1 else '(A)'
            print(f"  {str(row.get('player_name', ''))[:25]:<25} vs {str(row.get('opponent', ''))[:12]:<12} {ha} | xPts: {row.get('exp_total_pts', 0):.2f}")


if __name__ == '__main__':
    main()
