"""Backfill raw match-details JSON for every match in match_details.csv.

Saves to data/matches/raw/{match_id}.json.gz. Skips IDs that already have a file.

Usage:
    python backfill_raw.py             # backfill all missing
    python backfill_raw.py --limit 50  # cap iterations (testing)
"""

import argparse
import sys
import time

import pandas as pd

from scrape_update_data import (
    FotMobBrowser,
    MATCH_DETAILS_FILE,
    PREMIER_LEAGUE_ID,
    RAW_DIR,
    random_delay,
    save_raw_match,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not MATCH_DETAILS_FILE.exists():
        print(f"Missing {MATCH_DETAILS_FILE}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(MATCH_DETAILS_FILE)
    df["season"] = df["season"].astype(str)
    df = df.sort_values(["season", "match_id"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    by_season = {}
    for season, sub in df.groupby("season"):
        ids = sorted(set(int(m) for m in sub["match_id"].dropna()))
        ids = [m for m in ids if not (RAW_DIR / f"{m}.json.gz").exists()]
        if ids:
            by_season[season] = ids

    total_todo = sum(len(v) for v in by_season.values())
    print(f"Total matches in CSV: {len(df)}  |  to fetch: {total_todo}")
    for s, ids in by_season.items():
        print(f"  {s}: {len(ids)} missing")

    if args.limit is not None:
        kept = {}
        remaining = args.limit
        for s, ids in by_season.items():
            if remaining <= 0:
                break
            take = min(len(ids), remaining)
            kept[s] = ids[:take]
            remaining -= take
        by_season = kept
        total_todo = sum(len(v) for v in by_season.values())
        print(f"Limited to {total_todo} this run")

    if not by_season:
        print("Nothing to backfill.")
        return

    failed = []
    t0 = time.time()
    done = 0
    with FotMobBrowser() as browser:
        for season, todo in by_season.items():
            print(f"\n=== Season {season}: warming with league fetch... ===")
            league = browser.fetch_json(
                f"/api/data/leagues?id={PREMIER_LEAGUE_ID}&season={season}"
            )
            if not league:
                print(f"  WARN: league fetch returned empty for {season}")
            random_delay()

            for match_id in todo:
                done += 1
                try:
                    payload = browser.fetch_match_json(match_id)
                    if not payload:
                        failed.append(match_id)
                        print(f"  [{done}/{total_todo}] {match_id} EMPTY")
                    else:
                        save_raw_match(match_id, payload)
                        if done % 25 == 0 or done <= 3:
                            elapsed = time.time() - t0
                            rate = done / elapsed if elapsed else 0
                            eta = (total_todo - done) / rate if rate else 0
                            print(
                                f"  [{done}/{total_todo}] {match_id} OK"
                                f"  ({rate:.2f}/s, ETA {eta/60:.1f} min)"
                            )
                except Exception as e:
                    failed.append(match_id)
                    print(f"  [{done}/{total_todo}] {match_id} ERROR: {e}")
                random_delay()

    print(
        f"\nDone. Saved {total_todo - len(failed)} / {total_todo}.  Failed: {len(failed)}"
    )
    if failed:
        print(
            "Failed match IDs:",
            failed[:20],
            "..." if len(failed) > 20 else "",
        )


if __name__ == "__main__":
    main()
