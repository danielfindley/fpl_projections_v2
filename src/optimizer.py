"""Budget-constrained FPL squad selection.

Picks the 15 that maximise expected points under the £100.0m budget and FPL's squad
rules, solved exactly as a MILP via scipy. The bench is valued by how likely it is to
actually score: a bench slot only pays out when a starter fails to appear, so its
weight comes from the appearance model rather than a hand-set discount.

Prices come from the FPL API, which is the only source for them — nothing in the repo
stores a price. ``snapshot_prices`` appends each fetch to data/fpl_prices.csv because
per-gameweek price history is unrecoverable once a season rolls over.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import milp, LinearConstraint, Bounds

from .data_loader import normalize_player_name

# FPL squad rules
SQUAD_SIZE = 15
BUDGET = 100.0
SQUAD_QUOTA = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
XI_MIN = {'GK': 1, 'DEF': 3, 'MID': 2, 'FWD': 1}
XI_MAX = {'GK': 1, 'DEF': 5, 'MID': 5, 'FWD': 3}
XI_SIZE = 11
MAX_PER_TEAM = 3
POSITIONS = ['GK', 'DEF', 'MID', 'FWD']


def fetch_fpl_prices(verbose: bool = True) -> pd.DataFrame:
    """Current price for every FPL player.

    Returns fpl_id, player_name, web_name, team, position, price (in £m).
    """
    import requests
    bootstrap = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=15
    ).json()
    teams = {t['id']: t['name'] for t in bootstrap['teams']}
    pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    rows = [{
        'fpl_id': e['id'],
        'player_name': f"{e['first_name']} {e['second_name']}",
        'web_name': e['web_name'],
        'second_name': e['second_name'],
        'team': teams.get(e['team'], ''),
        'position': pos_map.get(e['element_type'], 'MID'),
        'price': e['now_cost'] / 10.0,
        'status': e['status'],
    } for e in bootstrap['elements']]

    df = pd.DataFrame(rows)
    if verbose:
        print(f"  Fetched prices for {len(df)} FPL players "
              f"(£{df['price'].min():.1f}m–£{df['price'].max():.1f}m)")
    return df


def snapshot_prices(prices: pd.DataFrame, gameweek: int, season: str,
                    data_dir: str = 'data', verbose: bool = True) -> Path:
    """Append this week's prices to data/fpl_prices.csv.

    The FPL API only serves per-gameweek price history for the current season and
    drops it at rollover, keeping just start_cost/end_cost per player per season. So
    a snapshot not taken now cannot be reconstructed later.
    """
    path = Path(data_dir) / 'fpl_prices.csv'
    snap = prices[['fpl_id', 'player_name', 'web_name', 'team', 'position', 'price']].copy()
    snap.insert(0, 'season', season)
    snap.insert(1, 'gameweek', gameweek)

    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, snap], ignore_index=True)
        combined = combined.drop_duplicates(subset=['season', 'gameweek', 'fpl_id'], keep='last')
    else:
        combined = snap

    combined.to_csv(path, index=False)
    if verbose:
        print(f"  Price snapshot: GW{gameweek} {season} -> {path} ({len(combined):,} rows total)")
    return path


def attach_prices(predictions: pd.DataFrame, prices: pd.DataFrame = None,
                  verbose: bool = True) -> pd.DataFrame:
    """Join FPL prices onto predictions by name.

    Same problem the FPL card merge has: FotMob and FPL spell names differently. Uses
    exact full name, then web_name/second_name, then a token-subset match for FPL's
    middle names ('Bruno Fernandes' -> 'Bruno Borges Fernandes'). Only unambiguous
    matches are taken; players left unpriced are dropped by the optimizer and reported.
    """
    if prices is None:
        prices = fetch_fpl_prices(verbose=verbose)

    full, web, second = {}, {}, {}
    for _, r in prices.iterrows():
        rec = (r['price'], r['position'], r['team'], r['status'], r['web_name'])
        full.setdefault(normalize_player_name(r['player_name']), []).append(rec)
        web.setdefault(normalize_player_name(r['web_name']), []).append(rec)
        second.setdefault(normalize_player_name(r['second_name']), []).append(rec)

    def lookup(name):
        key = normalize_player_name(name)
        if not key:
            return None
        for table in (full, web, second):
            if len(table.get(key, [])) == 1:
                return table[key][0]
        tokens = set(key.split())
        if len(tokens) >= 2:
            hits = [v[0] for fk, v in full.items()
                    if tokens <= set(fk.split()) and len(v) == 1]
            if len(hits) == 1:
                return hits[0]
        return None

    out = predictions.copy()
    matched = out['player_name'].apply(lookup)
    out['price'] = [m[0] if m else np.nan for m in matched]
    out['fpl_status'] = [m[3] if m else None for m in matched]
    # FPL's own short name — what the pitch view labels each dot with
    out['web_name'] = [m[4] if m else None for m in matched]

    if verbose:
        n = out['price'].notna().sum()
        print(f"  Price match: {n}/{len(out)} ({n / len(out):.1%})")
        missing = out.loc[out['price'].isna(), 'player_name'].tolist()
        if missing:
            preview = ', '.join(missing[:6]) + (' …' if len(missing) > 6 else '')
            print(f"    unpriced (excluded): {preview}")
    return out


def _bench_weights(p_appear_xi: np.ndarray, n_bench: int = 3) -> np.ndarray:
    """P(at least k starters fail to appear), k = 1..n_bench.

    An outfield bench slot scores only via autosub, which needs a starter to blank.
    The count of blanking starters is Poisson-binomial over the XI, computed exactly
    by convolving the per-player Bernoullis.
    """
    dist = np.array([1.0])
    for p in p_appear_xi:
        blank = 1.0 - p
        nxt = np.zeros(len(dist) + 1)
        nxt[:-1] += dist * p
        nxt[1:] += dist * blank
        dist = nxt
    tail = 1.0 - np.cumsum(dist)          # tail[k-1] = P(blanks > k-1) = P(blanks >= k)
    return np.array([tail[k - 1] if k - 1 < len(tail) else 0.0
                     for k in range(1, n_bench + 1)])


def optimize_squad(predictions: pd.DataFrame, budget: float = BUDGET,
                   points_col: str = 'exp_total_pts', verbose: bool = True,
                   max_iter: int = 4) -> dict:
    """Select the 15 maximising XI points plus probability-weighted bench points.

    Bench weights depend on the XI's appearance probabilities, and the XI depends on
    the weights, so the MILP is solved repeatedly with weights recomputed from the
    previous solution until the squad stops changing (usually 2-3 rounds).

    Returns a dict with the squad frame, the XI, the bench in autosub order, the
    formation, cost, and the bench weights that were used.
    """
    df = predictions.copy()
    df = df[df['price'].notna()]
    df = df[df['fpl_position'].isin(POSITIONS)]
    # Suspended / injured / unavailable can't be picked
    if 'fpl_status' in df.columns:
        df = df[~df['fpl_status'].isin(['s', 'u'])]
    df = df.reset_index(drop=True)

    if 'pred_appear_prob' in df.columns and df['pred_appear_prob'].notna().any():
        p_appear = df['pred_appear_prob'].fillna(0.75).clip(0, 1).values
    else:
        raise ValueError(
            "optimize_squad needs pred_appear_prob — train the pipeline so "
            "MinutesModel fits its AppearClassifier"
        )

    # exp_total_pts is built from pred_minutes, which is E[minutes | appears] — so it is
    # E[points | appears] and overstates a fringe player by 1/p_appear. Scale it to an
    # unconditional expectation before optimising. Applied to starters and bench alike;
    # the bench is then discounted a second time by P(its slot is used), which is a
    # different event and not double counting.
    ep = df[points_col].values.astype(float) * p_appear
    df['exp_pts_uncond'] = ep
    price = df['price'].values.astype(float)
    pos = df['fpl_position'].values
    team = df['team'].values
    n = len(df)

    pos_idx = {p: np.where(pos == p)[0] for p in POSITIONS}
    for p, need in SQUAD_QUOTA.items():
        if len(pos_idx[p]) < need:
            raise ValueError(f"Only {len(pos_idx[p])} priced {p} available, need {need}")

    # Variables: [squad_0..squad_n-1, start_0..start_n-1]
    S, X = slice(0, n), slice(n, 2 * n)

    constraints = []

    def row(squad_coef=None, start_coef=None):
        r = np.zeros(2 * n)
        if squad_coef is not None:
            r[S] = squad_coef
        if start_coef is not None:
            r[X] = start_coef
        return r

    constraints.append(LinearConstraint(row(squad_coef=np.ones(n)), SQUAD_SIZE, SQUAD_SIZE))
    constraints.append(LinearConstraint(row(start_coef=np.ones(n)), XI_SIZE, XI_SIZE))
    constraints.append(LinearConstraint(row(squad_coef=price), 0, budget))

    for p in POSITIONS:
        sel = np.zeros(n)
        sel[pos_idx[p]] = 1
        constraints.append(LinearConstraint(row(squad_coef=sel), SQUAD_QUOTA[p], SQUAD_QUOTA[p]))
        constraints.append(LinearConstraint(row(start_coef=sel), XI_MIN[p], XI_MAX[p]))

    for t in pd.unique(team):
        sel = (team == t).astype(float)
        constraints.append(LinearConstraint(row(squad_coef=sel), 0, MAX_PER_TEAM))

    # A starter must be in the squad: start_i - squad_i <= 0
    link = np.zeros((n, 2 * n))
    link[np.arange(n), np.arange(n)] = -1.0
    link[np.arange(n), n + np.arange(n)] = 1.0
    constraints.append(LinearConstraint(link, -np.inf, 0))

    integrality = np.ones(2 * n)
    bounds = Bounds(np.zeros(2 * n), np.ones(2 * n))

    # Seed weights: bench value before any XI is known. Deliberately small.
    weights = np.array([0.18, 0.09, 0.04])
    gk_bench_weight = 0.10
    squad_idx = None

    for it in range(max_iter):
        # Bench contribution is (squad - start); each position's bench weight is the
        # average slot weight, since which bench slot a player lands in is decided
        # after selection by expected points order.
        w_out = float(np.mean(weights))
        bench_w = np.where(pos == 'GK', gk_bench_weight, w_out)
        bench_value = ep * bench_w

        c = np.zeros(2 * n)
        c[S] = -bench_value                     # counted for everyone in the squad…
        c[X] = -(ep - bench_value)              # …upgraded to full value when starting
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if not res.success:
            raise RuntimeError(f"MILP failed: {res.message}")

        new_squad = np.where(res.x[S] > 0.5)[0]
        new_start = np.where(res.x[X] > 0.5)[0]

        outfield_xi = [i for i in new_start if pos[i] != 'GK']
        weights = _bench_weights(p_appear[outfield_xi], n_bench=3)
        gk_start = [i for i in new_start if pos[i] == 'GK']
        gk_bench_weight = float(1.0 - p_appear[gk_start[0]]) if gk_start else 0.10

        if squad_idx is not None and set(new_squad) == set(squad_idx) and set(new_start) == set(start_idx):
            squad_idx, start_idx = new_squad, new_start
            if verbose:
                print(f"  Converged after {it + 1} iteration(s)")
            break
        squad_idx, start_idx = new_squad, new_start

    squad = df.iloc[squad_idx].copy()
    squad['is_starter'] = squad.index.isin(df.index[start_idx])

    xi = squad[squad['is_starter']].sort_values('exp_pts_uncond', ascending=False)
    bench = squad[~squad['is_starter']].copy()
    # Autosub order: GK first (only replaces the GK), then outfield by expected points
    bench_gk = bench[bench['fpl_position'] == 'GK']
    bench_out = bench[bench['fpl_position'] != 'GK'].sort_values('exp_pts_uncond', ascending=False)
    bench = pd.concat([bench_gk, bench_out])

    formation = {p: int((xi['fpl_position'] == p).sum()) for p in POSITIONS}
    formation_str = f"{formation['DEF']}-{formation['MID']}-{formation['FWD']}"

    result = {
        'squad': squad,
        'xi': xi,
        'bench': bench,
        'formation': formation_str,
        'total_cost': float(squad['price'].sum()),
        'budget_left': float(budget - squad['price'].sum()),
        'xi_points': float(xi['exp_pts_uncond'].sum()),
        'bench_cost': float(bench['price'].sum()),
        'bench_weights': weights.tolist(),
        'gk_bench_weight': gk_bench_weight,
        'captain': xi.iloc[0]['player_name'] if len(xi) else None,
    }

    if verbose:
        _print_squad(result)
    return result


def _print_squad(result: dict):
    xi, bench = result['xi'], result['bench']
    print(f"\n  Formation {result['formation']} | "
          f"cost £{result['total_cost']:.1f}m (£{result['budget_left']:.1f}m left) | "
          f"XI E[pts] {result['xi_points']:.1f}")
    print(f"  Captain: {result['captain']}")
    print(f"  {'':2} {'Player':<24}{'Pos':<5}{'Team':<18}{'£':>6}{'E[pts]':>8}{'P(app)':>8}")
    for _, r in xi.iterrows():
        print(f"  {'':2} {r['player_name'][:23]:<24}{r['fpl_position']:<5}{str(r['team'])[:17]:<18}"
              f"{r['price']:>6.1f}{r['exp_pts_uncond']:>8.2f}{r.get('pred_appear_prob', float('nan')):>8.2f}")
    print(f"  bench (£{result['bench_cost']:.1f}m, slot weights "
          f"{[round(w, 3) for w in result['bench_weights']]}, GK {result['gk_bench_weight']:.3f})")
    for i, (_, r) in enumerate(bench.iterrows(), 1):
        print(f"  {i:>2} {r['player_name'][:23]:<24}{r['fpl_position']:<5}{str(r['team'])[:17]:<18}"
              f"{r['price']:>6.1f}{r['exp_pts_uncond']:>8.2f}{r.get('pred_appear_prob', float('nan')):>8.2f}")
