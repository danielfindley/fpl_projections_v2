"""Predicted lineups from RotoWire.

The pipeline's weakest moment is a season opener: every feature is three months
stale, so the minutes model cannot tell a nailed starter from a squad player and
hedges on both. A lineup feed supplies exactly the missing fact — who is in the XI —
and slots into the one place the model is guessing, ``p_start`` in
``MinutesModel._blend``.

Used at inference only. Nothing here touches training, so no model needs retuning.
Scrapes are cached to data/lineups/ because RotoWire serves only the current slate:
per-gameweek history cannot be recovered later, and it is what lets us measure the
feed's hit rate and eventually train on it as a feature.
"""
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import normalize_player_name

LINEUPS_URL = "https://www.rotowire.com/soccer/lineups.php"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Statuses that mean the player cannot feature at all, whatever the XI says.
UNAVAILABLE = {'OUT', 'SUS'}

# normalize_player_name strips combining accents via NFD, which handles é/ã/ü but not
# letters that are their own codepoint and never decompose. RotoWire writes plain
# ASCII, so without these Odegaard never matches Ødegaard and Gross never matches Groß.
_LETTER_FOLD = str.maketrans({
    'ø': 'o', 'æ': 'ae', 'å': 'a', 'ß': 'ss', 'đ': 'd', 'ð': 'd',
    'ł': 'l', 'þ': 'th', 'ħ': 'h', 'ı': 'i', 'œ': 'oe',
})


def _fold(name) -> str:
    """Normalised name with non-decomposing letters folded to ASCII."""
    return normalize_player_name(name).translate(_LETTER_FOLD)

_BOX = re.compile(r'lineup__box"(.*?)(?=lineup__box"|\Z)', re.S)
_TEAM_HOME = re.compile(r'lineup__mteam is-home">\s*([^<]+?)\s*<', re.S)
_TEAM_AWAY = re.compile(r'lineup__mteam is-visit">\s*([^<]+?)\s*<', re.S)
_LIST = re.compile(r'<ul class="lineup__list is-(home|visit)">(.*?)</ul>', re.S)
_STATUS = re.compile(r'lineup__status is-(\w+)')
_INJ_SPLIT = re.compile(r'<li class="lineup__title is-middle">Injuries</li>', re.S)
_PLAYER = re.compile(
    r'<li class="lineup__player">\s*'
    r'<div class="lineup__pos[^"]*">([^<]*)</div>\s*'
    r'<a title="([^"]*)" href="/soccer/player/[^"]*?-(\d+)">([^<]*)</a>'
    r'(?:\s*<span class="lineup__inj">([A-Z]+)</span>)?',
    re.S)


def fetch_lineups(url: str = LINEUPS_URL, timeout: int = 25) -> str:
    """Fetch the lineups page. One request per deploy — do not poll."""
    import requests
    resp = requests.get(url, headers={'User-Agent': _UA}, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def parse_lineups(html: str) -> pd.DataFrame:
    """Parse the lineups page into one row per listed player.

    The display name is often abbreviated ('R. Calafiori'), but every row carries
    the full name in the anchor's title attribute and a stable RotoWire id in its
    href, so matching never has to work from initials.

    Returns columns: fixture, team, is_home, confirmed, position_code, player_name,
    display_name, rw_id, status, is_predicted_starter.
    """
    rows = []
    for fi, box in enumerate(_BOX.findall(html)):
        home = _TEAM_HOME.search(box)
        away = _TEAM_AWAY.search(box)
        teams = {'home': home.group(1).strip() if home else None,
                 'visit': away.group(1).strip() if away else None}
        for side, body in _LIST.findall(box):
            status_m = _STATUS.search(body)
            confirmed = bool(status_m and status_m.group(1) == 'confirmed')
            # Everything before the 'Injuries' divider is the XI; after it is the
            # unavailable list, which repeats some XI names with a status.
            parts = _INJ_SPLIT.split(body)
            for is_xi, chunk in ((True, parts[0]),
                                 (False, parts[1] if len(parts) > 1 else '')):
                for pos, full, rw_id, disp, inj in _PLAYER.findall(chunk):
                    rows.append({
                        'fixture': fi,
                        'team': teams[side],
                        'is_home': side == 'home',
                        'confirmed': confirmed,
                        'position_code': pos.strip(),
                        'player_name': full.strip(),
                        'display_name': disp.strip(),
                        'rw_id': int(rw_id),
                        'status': inj.strip() or None,
                        'is_predicted_starter': is_xi,
                    })
    return pd.DataFrame(rows)


def save_lineups(lineups: pd.DataFrame, gameweek: int, season: str,
                 data_dir: str = 'data', verbose: bool = True) -> Path:
    """Persist a scrape. RotoWire only ever serves the current slate, so a snapshot
    not taken now is gone — same constraint as prices."""
    out_dir = Path(data_dir) / 'lineups'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = out_dir / f'gw{gameweek}_{ts}.csv'
    snap = lineups.copy()
    snap.insert(0, 'season', season)
    snap.insert(1, 'gameweek', gameweek)
    snap.insert(2, 'scraped_at', datetime.now().isoformat(timespec='seconds'))
    snap.to_csv(path, index=False)
    if verbose:
        print(f"  Lineup snapshot: {len(snap)} rows -> {path}")
    return path


def match_lineups(lineups: pd.DataFrame, predictions: pd.DataFrame,
                  verbose: bool = True) -> pd.DataFrame:
    """Attach our player_name to each lineup row, matching within club.

    Scoping by club is what makes this safe: disambiguation runs over ~25 squad
    members rather than every player in the league. Cascade is exact normalised
    full name, then a token-subset pass for middle names, mirroring the FPL price
    match.

    An unmatched *starter* is the dangerous case — silently treated as benched it
    would wreck that player's projection — so unmatched names are reported and the
    caller leaves them on the model's own estimate.
    """
    # imported here, not at module scope: pipeline pulls in lineups lazily and a
    # top-level import would close the cycle
    from .pipeline import normalize_team_name

    preds = predictions[['player_name', 'team']].copy()
    preds['_key'] = preds['player_name'].map(_fold)
    preds['_team'] = preds['team'].map(normalize_team_name)

    by_team = {}
    for team, grp in preds.groupby('_team'):
        by_team[team] = dict(zip(grp['_key'], grp['player_name']))

    def _match(row):
        team = normalize_team_name(row['team']) if row['team'] else None
        pool = by_team.get(team, {})
        key = _fold(row['player_name'])
        if not key:
            return None
        if key in pool:
            return pool[key]
        tokens = set(key.split())
        if len(tokens) >= 2:
            hits = [v for k, v in pool.items() if tokens <= set(k.split())]
            if len(hits) == 1:
                return hits[0]
            # surname alone, but only when unique within this club
            last = key.split()[-1]
            hits = [v for k, v in pool.items() if last in k.split()]
            if len(hits) == 1:
                return hits[0]
        else:
            # mononyms ('Alisson') against our fuller record ('Alisson Becker')
            hits = [v for k, v in pool.items() if key in k.split()]
            if len(hits) == 1:
                return hits[0]
        return None

    out = lineups.copy()
    out['matched_name'] = out.apply(_match, axis=1)

    if verbose:
        xi = out[out['is_predicted_starter']]
        n = xi['matched_name'].notna().sum()
        print(f"  Lineup match: {n}/{len(xi)} predicted starters ({n / max(len(xi), 1):.1%})")
        missing = xi.loc[xi['matched_name'].isna(), ['team', 'player_name']]
        if len(missing):
            preview = ', '.join(f"{r.player_name} ({r.team})"
                                for r in missing.head(8).itertuples())
            print(f"    unmatched starters (left on model estimate): {preview}"
                  + (' …' if len(missing) > 8 else ''))
    return out


def build_overrides(matched: pd.DataFrame, predictions: pd.DataFrame,
                    p_in: float = 0.95, p_out: float = 0.10,
                    verbose: bool = True) -> pd.DataFrame:
    """Turn matched lineup rows into per-player overrides aligned to predictions.

    Returns a frame indexed like ``predictions`` with:
      lineup_p_start   P(starts) asserted by the feed, NaN where it says nothing
      lineup_available 0.0 for OUT/SUS, else NaN

    Players the feed does not mention at all keep NaN and fall through to the model,
    since absence from the page is not evidence of being dropped.
    """
    from .pipeline import normalize_team_name

    ok = matched[matched['matched_name'].notna()]
    xi = ok[ok['is_predicted_starter']]
    listed = ok[~ok['is_predicted_starter']]

    starters = set(xi['matched_name'])

    # Benching is inferred from absence, so it is only safe where the whole XI was
    # matched. With even one starter unmatched we cannot tell "benched" from "we
    # failed to recognise the name", and guessing wrong turns a nailed starter into
    # a 20-minute sub. Clubs with an incomplete match still get their positive
    # starter overrides; everyone else there falls through to the model.
    all_xi = matched[matched['is_predicted_starter']]
    per_team = all_xi.groupby('team')['matched_name'].agg(['size', 'count'])
    complete = {normalize_team_name(t) for t, r in per_team.iterrows()
                if r['size'] == r['count']}

    unavailable = set(listed.loc[listed['status'].isin(UNAVAILABLE), 'matched_name'])
    unavailable |= set(xi.loc[xi['status'].isin(UNAVAILABLE), 'matched_name'])

    p_start = np.full(len(predictions), np.nan)
    available = np.full(len(predictions), np.nan)
    teams = predictions['team'].map(normalize_team_name)

    for i, (name, team) in enumerate(zip(predictions['player_name'], teams)):
        if name in unavailable:
            available[i] = 0.0
            p_start[i] = p_out
        elif name in starters:
            p_start[i] = p_in
        elif team in complete:
            p_start[i] = p_out

    if verbose:
        n_partial = len(set(per_team.index.map(normalize_team_name)) - complete)
        print(f"  Lineup overrides: {int(np.sum(p_start == p_in))} starters, "
              f"{int(np.sum((p_start == p_out) & (available != 0.0)))} benched, "
              f"{int(np.nansum(available == 0.0))} unavailable, "
              f"{int(np.sum(np.isnan(p_start)))} left to the model")
        if n_partial:
            print(f"    {n_partial} club(s) had an incomplete XI match — "
                  f"their non-starters were NOT benched")
    return pd.DataFrame({'lineup_p_start': p_start, 'lineup_available': available},
                        index=predictions.index)
