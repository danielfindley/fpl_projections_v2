"""FotMob <-> FPL name matching.

A current player predict() cannot find in the FPL squads is dropped as departed,
so every miss silently removes a real player from the projections (GW10 2026/27
lost Josh King, Joe Willock, Eddie Nketiah and eight others this way).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import match_within_club, normalize_player_name
from src.pipeline import FPLPipeline


def _roster(*names):
    return [normalize_player_name(n) for n in names]


@pytest.mark.parametrize('fotmob, fpl_squad, expected', [
    # nickname vs given name, with the surname shared by another club's player
    ('Joshua King', ['Josh King', 'Raúl Jiménez'], 'Josh King'),
    ('Joseph Willock', ['Joe Willock', 'Lewis Hall'], 'Joe Willock'),
    ('Edward Nketiah', ['Eddie Nketiah', 'Adam Wharton'], 'Eddie Nketiah'),
    ('Konstantinos Tsimikas', ['Kostas Tsimikas', 'Joe Gomez'], 'Kostas Tsimikas'),
    # hyphenated FotMob surname, FPL keeps only the first part
    ('Jaden Philogene-Bidace', ['Jaden Philogene'], 'Jaden Philogene'),
    ('Arnaud Kalimuendo-Muinga', ['Arnaud Kalimuendo'], 'Arnaud Kalimuendo'),
    # mononym plus surname vs FPL's full legal name
    ('Chema Andrés', ['José María Andrés Baixauli', 'Danny Welbeck'], 'José María Andrés Baixauli'),
    # FPL name inside ours, where the surname alone is ambiguous in the squad
    ('Juan Riquelme Angulo', ['Juan Angulo', 'Nilson Angulo'], 'Juan Angulo'),
    # transliteration
    ('Timur Tuterov', ['Tymur Tutierov', 'Granit Xhaka'], 'Tymur Tutierov'),
])
def test_within_club_resolves_known_misses(fotmob, fpl_squad, expected):
    assert match_within_club(fotmob, _roster(*fpl_squad)) == normalize_player_name(expected)


def test_within_club_refuses_ambiguous_or_claimed_matches():
    # two squad members share the surname and neither name is inside ours
    assert match_within_club('Sam Angulo', _roster('Juan Angulo', 'Nilson Angulo')) is None
    # the only candidate is already someone else's match
    roster = _roster('Josh King')
    assert match_within_club('Joshua King', roster, claimed=set(roster)) is None
    # nothing in the club resembles the name
    assert match_within_club('Allan', _roster('Carlos Alcaraz Durán', 'Jordan Pickford')) is None


def test_review_matcher_retries_misses_within_club():
    act = pd.DataFrame({
        'player_name': ['Vitalii Mykolenko', 'Tom King', 'Josh King'],
        'web_name': ['Mykolenko', 'King', 'King'],
        'team': ['Everton', 'Everton', 'Fulham'],
    })
    names = ['Vitaliy Mykolenko', 'Joshua King']
    assert FPLPipeline._match_fpl_actuals(names, act) == [None, None]
    assert FPLPipeline._match_fpl_actuals(names, act, ['Everton', 'Fulham']) == [0, 2]


def _bootstrap(players):
    teams = sorted({p[1] for p in players})
    ids = {t: i + 1 for i, t in enumerate(teams)}
    return {'teams': [{'id': i, 'name': t} for t, i in ids.items()],
            'elements': [{'first_name': f, 'second_name': s, 'web_name': s, 'team': ids[t],
                          'minutes': m, 'status': st}
                         for (f, s), t, m, st in players]}


def test_coverage_audit_flags_fit_players_without_predictions(tmp_path):
    boot = _bootstrap([
        (('Josh', 'King'), 'Fulham', 735, 'a'),        # predicted under another spelling
        (('Raúl', 'Jiménez'), 'Fulham', 600, 'a'),     # genuinely missing
        (('Harry', 'Wilson'), 'Fulham', 300, 'i'),     # injured: not expected
        (('Ben', 'Reserve'), 'Fulham', 0, 'a'),        # never played: not expected
        (('Erling', 'Haaland'), 'Man City', 800, 'a'),  # club has no fixture
    ])
    preds = pd.DataFrame({'player_name': ['Joshua King'], 'team': ['Fulham']})
    missing = FPLPipeline(str(tmp_path)).audit_fpl_coverage(preds, bootstrap=boot, verbose=False)
    assert missing['player_name'].tolist() == ['Raúl Jiménez']


def test_latest_run_covers_every_fit_fpl_player():
    """Live check on the newest saved run: run before every deploy push."""
    runs = [d for d in Path('data/runs').glob('gw*') if (d / 'predictions.csv').exists()
            and (d / 'meta.json').exists()]
    if not runs:
        pytest.skip('no saved runs')
    latest = max(runs, key=lambda d: (d / 'predictions.csv').stat().st_mtime)
    requests = pytest.importorskip('requests')
    try:
        boot = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/',
                            timeout=20).json()
    except Exception as exc:
        pytest.skip(f'FPL API unreachable: {exc}')
    season = json.loads((latest / 'meta.json').read_text(encoding='utf-8')).get('season')
    events = boot.get('events') or []
    d = events[0]['deadline_time'] if events else ''
    if d and season != (f"{int(d[:4])}/{int(d[:4]) + 1}" if int(d[5:7]) >= 6
                        else f"{int(d[:4]) - 1}/{int(d[:4])}"):
        pytest.skip(f'latest run {latest.name} is not from the live season')
    preds = pd.read_csv(latest / 'predictions.csv')
    missing = FPLPipeline('data').audit_fpl_coverage(preds, bootstrap=boot, verbose=False)
    assert missing.empty, (f"{latest.name} is missing fit FPL players (name-match misses):\n"
                           + missing.to_string(index=False))


def test_fpl_export_keys_every_player_with_a_fixture_by_fpl_id(tmp_path):
    boot = _bootstrap([
        (('Josh', 'King'), 'Fulham', 735, 'a'),
        (('Harry', 'Wilson'), 'Fulham', 300, 'i'),       # no prediction row: written as 0
        (('Erling', 'Haaland'), 'Man City', 800, 'a'),   # club has no fixture: left out
    ])
    for i, e in enumerate(boot['elements']):
        e.update(id=i + 1, element_type=3)
    boot['element_types'] = [{'id': 3, 'singular_name_short': 'MID'}]
    preds = pd.DataFrame({'player_id': [1], 'player_name': ['Joshua King'], 'team': ['Fulham'],
                          'opponent': ['Newcastle'], 'exp_total_pts': [3.9],
                          'pred_minutes_uncond': [74.8], 'pred_appear_prob': [0.99]})
    out = FPLPipeline(str(tmp_path)).export_fpl_predictions(
        preds, 10, output_path=tmp_path / 'out.csv', bootstrap=boot)
    assert out.set_index('fpl_id')['expected_points'].to_dict() == {1: 3.9, 2: 0.0}
