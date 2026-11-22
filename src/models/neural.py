"""Pure deep-learning backend (``MODEL_BACKEND = 'neural'`` in pipeline.py).

Every component the pipeline calls is a neural network; no XGBoost output is
used anywhere. Each network takes the row's engineered features (quantile
clipped, NaN flags, optional piecewise-linear encoding) plus history sequences,
each encoded by a transformer/GRU, attention-pooled by the static features, and
concatenated with its last token:

* appearances: the player's last 20 raw per-match stat lines;
* calendar: the club's last fixtures for that player, including weeks he did not
  play (minutes 0), so absences are visible;
* squad: up to 30 squad-mates eligible for the same fixture, pre-match state;
* team / opponent: the last 10 team-matches of each side.

Every token is restricted to results known before the row's forecast deadline.

Components (the recipe frozen from the 2025/26 GW29-38 and 2026/27 GW1-10 holdouts):

* minutes: a split P(60+ | appears) classifier plus start/sub minute regressors
  (GRU), with P(60+) averaged with a 3-class grid net (did not play / 1-59 / 60+)
  trained on every eligible fixture, which also gives P(appear);
* events: per-90 Poisson rates on a log(minutes/90) exposure offset. Goals and
  defcon average a single-task and a multi-task net; assists, fouls, yellow cards
  and baseline BPS come from a second multi-task net; saves is single-task.
  The defcon Negative Binomial ``r`` comes from early-stopped nets scored on
  their own validation slice (the refit nets are in-sample: r 23.8 vs 18.5);
* goals conceded: a no-offset team net, 10 seeds.

``NeuralSystem.models()`` returns objects with the XGBoost models' interfaces,
so predict(), the holdout evaluation, the simulation and the five-week forecast
run unchanged. Training needs a CUDA GPU (~20 minutes on an RTX 4070 SUPER).
"""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .assists import AssistsModel
from .bonus import BaselineBPSModel, BonusModel, normalize_team_name
from .cards import CardsModel, RED_PER_FOUL
from .clean_sheet import CleanSheetModel
from .defcon import DefconModel
from .goals import GoalsModel
from .minutes import ALL_FEATURES as MIN_FEATS, MinutesModel
from .saves import SavesModel
from .sequence import baseline_bps_target


def _device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "MODEL_BACKEND='neural' needs a CUDA GPU (torch reports none; CPU training "
            "takes hours). Install a CUDA torch build, or set MODEL_BACKEND='xgboost' "
            "in src/pipeline.py to use the XGBoost models deliberately.")
    return torch.device('cuda')


# ---------------------------------------------------------------- inputs
class StaticScaler:
    def __init__(self, cols, ple=0):
        self.cols, self.ple = cols, ple

    def fit(self, frame):
        v = frame.reindex(columns=self.cols).apply(pd.to_numeric, errors='coerce').to_numpy(float)
        self.lo = np.nan_to_num(np.nanpercentile(v, 0.5, axis=0))
        self.hi = np.nan_to_num(np.nanpercentile(v, 99.5, axis=0))
        c = np.clip(v, self.lo, self.hi)
        self.med = np.nan_to_num(np.nanmedian(c, axis=0))
        c = np.where(np.isfinite(c), c, self.med)
        self.mu, self.sd = c.mean(0), c.std(0) + 1e-6
        self.nan_cols = np.flatnonzero(np.isnan(v).mean(0) > 0.01)
        if self.ple:
            q = np.nanquantile(c, np.linspace(0, 1, self.ple + 1), axis=0).T
            self.edges = [np.unique(e) for e in q]
        return self

    def _ple(self, c):
        out = []
        for j, e in enumerate(self.edges):
            if len(e) < 3:
                continue
            lo, hi = e[:-1], e[1:]
            out.append(np.clip((c[:, j:j + 1] - lo) / (hi - lo), 0, 1))
        return np.concatenate(out, 1).astype(np.float32)

    def transform(self, frame):
        v = frame.reindex(columns=self.cols).apply(pd.to_numeric, errors='coerce').to_numpy(float)
        miss = np.isnan(v[:, self.nan_cols]).astype(np.float32)
        c = np.clip(v, self.lo, self.hi)
        c = np.where(np.isfinite(c), c, self.med)
        parts = [((c - self.mu) / self.sd).astype(np.float32), miss]
        if self.ple:
            parts.append(self._ple(c))
        return np.concatenate(parts, 1)


def history(pool, rows, cols, length, key, row_key=None, same=None):
    """Right-aligned (n, length, len(cols)+2) tokens strictly before each row's deadline."""
    row_key = row_key or key
    pool = pool.sort_values([key, 'result_time'], kind='stable')
    base = np.column_stack([pd.to_numeric(pool[c], errors='coerce').fillna(0).to_numpy(np.float32)
                            if c in pool else np.zeros(len(pool), np.float32) for c in cols])
    t = pool['result_time'].to_numpy('datetime64[ns]')
    ids = pool[key].to_numpy()
    edges = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    spans = {ids[a]: (a, b) for a, b in zip(edges[:-1], edges[1:])}
    psame = pool[same].astype(str).to_numpy() if same else None
    rsame = rows[same].astype(str).to_numpy() if same else None
    dl = rows['forecast_time'].to_numpy('datetime64[ns]')
    tok = np.zeros((len(rows), length, len(cols) + 2), np.float32)
    mask = np.ones((len(rows), length), bool)
    for i, rid in enumerate(rows[row_key].to_numpy()):
        s = spans.get(rid)
        if s is None:
            continue
        lo, hi = s
        end = lo + int(np.searchsorted(t[lo:hi], dl[i], side='left'))
        b = max(lo, end - length)
        k = end - b
        if k == 0:
            continue
        tok[i, -k:, :-2] = base[b:end]
        tok[i, -k:, -2] = np.log1p(np.maximum((dl[i] - t[b:end]) / np.timedelta64(1, 'D'), 0))
        if same:
            tok[i, -k:, -1] = psame[b:end] == rsame[i]
        mask[i, -k:] = False
    return tok, mask


class TokenScaler:
    def fit(self, tok, mask):
        x = tok[~mask]
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        return self

    def transform(self, tok, mask):
        out = np.clip((tok - self.mu) / self.sd, -10, 10)
        return np.where(mask[..., None], 0, out).astype(np.float32)


# ---------------------------------------------------------------- network
class SeqEncoder(nn.Module):
    def __init__(self, n_in, d, length, kind, heads, layers, dropout):
        super().__init__()
        self.kind = kind
        self.embed = nn.Linear(n_in, d)
        self.start = nn.Parameter(torch.zeros(1, 1, d))
        if kind == 'transformer':
            self.pos = nn.Parameter(torch.zeros(1, length + 1, d))
            layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True,
                                               norm_first=True, activation='gelu')
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            rnn = nn.LSTM if kind == 'lstm' else nn.GRU
            self.enc = rnn(d, d, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0)
        self.pool = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, tok, mask, query):
        b = tok.shape[0]
        x = torch.cat([self.start.expand(b, -1, -1), self.embed(tok)], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=mask.device), mask], 1)
        if self.kind == 'transformer':
            x = self.enc(x + self.pos, src_key_padding_mask=pad)
        else:
            # Right-aligned padding: zeroed tokens precede the real history.
            x, _ = self.enc(x)
        x = self.norm(x)
        pooled, _ = self.pool(query[:, None], x, x, key_padding_mask=pad)
        return torch.cat([pooled[:, 0], x[:, -1]], 1)  # attention pool + most recent match


class Net(nn.Module):
    def __init__(self, n_static, seq_dims, n_out, cfg):
        super().__init__()
        d, dr = cfg['d'], cfg['dropout']
        self.static = nn.Sequential(nn.Linear(n_static, cfg['hidden']), nn.GELU(), nn.Dropout(dr),
                                    nn.Linear(cfg['hidden'], d), nn.LayerNorm(d))
        self.seqs = nn.ModuleList([SeqEncoder(n, d, L, cfg['kind'], cfg['heads'], cfg['layers'], dr)
                                   for n, L in seq_dims])
        width = d + 2 * d * len(seq_dims)
        self.head = nn.Sequential(nn.Linear(width, cfg['hidden']), nn.GELU(), nn.Dropout(dr),
                                  nn.Linear(cfg['hidden'], cfg['hidden'] // 2), nn.GELU(),
                                  nn.Linear(cfg['hidden'] // 2, n_out))
        if cfg.get('bias_init') is not None:
            with torch.no_grad():
                self.head[-1].bias.copy_(torch.tensor(cfg['bias_init'], dtype=torch.float32))

    def forward(self, static, *seqs):
        q = self.static(static)
        parts = [q] + [enc(seqs[2 * i], seqs[2 * i + 1], q) for i, enc in enumerate(self.seqs)]
        return self.head(torch.cat(parts, 1))


DEFAULT = dict(d=64, hidden=256, heads=4, layers=2, dropout=0.15, kind='transformer',
               lr=1e-3, wd=1e-4, batch=512, epochs=60, patience=6, seeds=5, refit=False)


def _tensors(X, seqs):
    return [torch.tensor(X)] + [torch.tensor(a) for s in seqs for a in s]


def train_net(X, seqs, T, loss_fn, cfg, valid_mask, label='', verbose=True):
    """Early-stop on the latest 10% of deadlines; with ``refit``, retrain on all rows
    for the best epoch count. Returns (nets, early-stopped nets or [])."""
    dev = _device()
    cfg = {**DEFAULT, **cfg}
    data = [d.to(dev) for d in _tensors(X, seqs)]
    T = torch.tensor(T, dtype=torch.float32, device=dev)
    fit_idx = torch.tensor(np.flatnonzero(~valid_mask), device=dev)
    val_idx = torch.tensor(np.flatnonzero(valid_mask), device=dev)
    seq_dims = [(s[0].shape[2], s[0].shape[1]) for s in seqs]
    nets, early = [], []

    def run(net, idx, train, opt=None):
        tot, n = 0., 0
        order = idx[torch.randperm(len(idx), device=dev)] if train else idx
        bs = cfg['batch'] if train else 8192
        for a in range(0, len(order), bs):
            j = order[a:a + bs]
            loss = loss_fn(net(*(d[j] for d in data)), T[j])
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            tot += float(loss) * len(j)
            n += len(j)
        return tot / max(n, 1)

    for seed in range(cfg['seeds']):
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = Net(X.shape[1], seq_dims, cfg['n_out'], cfg).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        best, state, best_ep, stale = np.inf, None, 0, 0
        for ep in range(cfg['epochs']):
            net.train()
            run(net, fit_idx, True, opt)
            net.eval()
            with torch.no_grad():
                v = run(net, val_idx, False)
            if v < best - 1e-6:
                best, best_ep, stale = v, ep, 0
                state = {k: x.detach().clone() for k, x in net.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg['patience']:
                    break
        if cfg['refit']:
            es = Net(X.shape[1], seq_dims, cfg['n_out'], cfg).to(dev)
            es.load_state_dict(state)
            es.eval()
            early.append(es)
            torch.manual_seed(100 + seed)
            net = Net(X.shape[1], seq_dims, cfg['n_out'], cfg).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
            all_idx = torch.arange(len(X), device=dev)
            for ep in range(best_ep + 1):
                net.train()
                run(net, all_idx, True, opt)
        else:
            net.load_state_dict(state)
        net.eval()
        nets.append(net)
        if verbose:
            print(f'  {label} seed {seed}: best ep {best_ep + 1} val {best:.4f}', flush=True)
    return nets, early


def predict_nets(nets, X, seqs):
    dev = _device()
    data = _tensors(X, seqs)
    outs = []
    with torch.no_grad():
        for net in nets:
            o = [net(*(d[a:a + 8192].to(dev) for d in data)).float().cpu().numpy()
                 for a in range(0, len(X), 8192)]
            outs.append(np.concatenate(o))
    return np.stack(outs)  # (seeds, n, k)


def valid_split(frame, frac=0.1):
    dls = np.sort(frame['forecast_time'].unique())
    return (frame['forecast_time'] >= dls[int(len(dls) * (1 - frac))]).to_numpy()


def poisson_dev(log_mu, y, w=None):
    loss = torch.exp(log_mu) - y + torch.xlogy(y, y) - y * log_mu
    if w is None:
        return 2 * loss.mean()
    return 2 * (loss * w).sum() / w.sum().clamp(min=1)


def huber(pred, y, w=None, delta=10.):
    l = F.huber_loss(pred, y, delta=delta, reduction='none')
    if w is None:
        return l.mean()
    return (l * w).sum() / w.sum().clamp(min=1)


# ---------------------------------------------------------------- data
REST = ['rest_team', 'rest_player', 'rest_excess', 'season_opener']
EXCLUDE = {'pred_minutes', 'pred_team_goals', 'pred_goals_against'}
POSITIONS = ['is_gk', 'is_def', 'is_mid', 'is_fwd']
PLAYER_STATIC = [c for c in dict.fromkeys(
    MIN_FEATS + GoalsModel.FEATURES + AssistsModel.FEATURES + DefconModel.FEATURES
    + SavesModel.FEATURES + CardsModel.FEATURES + BaselineBPSModel.FEATURES
    + ['is_home'] + POSITIONS) if c not in EXCLUDE]
APP_TOKENS = ['minutes', 'goals', 'assists', 'xg', 'npxg', 'xa', 'shots', 'key_passes',
              'penalty_goals', 'touches_in_opposition_box', 'fotmob_rating', 'is_home',
              'defcon', 'saves', 'fouls_committed', 'yellow_cards', 'bps_target']
# Raw per-match stats. Shots keep the on/off/blocked breakdown (total_shots was
# deprecated mid-season by FotMob; blocked NaN = 0, which history() fills).
APP_TOKENS_WIDE = APP_TOKENS + [
    'shots_on_target', 'shots_off_target', 'blocked_shots', 'big_chances_missed',
    'big_chances_created', 'expected_goals_on_target_(xgot)', 'touches', 'accurate_passes',
    'passes_attempted', 'passes_into_final_third', 'accurate_crosses', 'accurate_long_balls',
    'successful_dribbles', 'dispossessed', 'tackles', 'blocks', 'clearances', 'interceptions',
    'recoveries', 'dribbled_past', 'duels_won', 'duels_lost', 'aerial_duels_won', 'was_fouled',
    'offsides', 'goals_conceded', 'xgot_faced', 'saves_inside_box', 'penalties_won',
    'missed_penalty', 'hit_woodwork', 'corners', 'red_cards']
CAL_TOKENS = ['minutes', 'appeared', 'started', 'full90', 'is_home']
SQUAD_TOKENS = POSITIONS + ['last_minutes', 'last_was_starter', 'gw_gap_since_last_appearance',
                            'minutes_roll1', 'minutes_roll3', 'minutes_roll5', 'minutes_roll10',
                            'starter_rate_roll3', 'starter_rate_roll5', 'starter_rate_roll10',
                            'full90_rate_roll5']
TEAM_TOKENS = ['goals', 'goals_conceded', 'xg', 'xga', 'shots_on_target', 'key_passes',
               'touches', 'is_home']
TEAM_STATIC = list(dict.fromkeys(CleanSheetModel.FEATURES + ['prior_lambda', 'is_home']))
GRID_STATIC = ['last_minutes', 'last_was_starter', 'last_was_full_90', 'gw_gap_since_last_appearance',
               'last_app_prev_season', 'minutes_roll1', 'minutes_roll2', 'minutes_roll3', 'minutes_roll5',
               'minutes_roll7', 'minutes_roll10', 'starter_rate_roll1', 'starter_rate_roll2',
               'starter_rate_roll3', 'starter_rate_roll5', 'starter_rate_roll7', 'starter_rate_roll10',
               'full90_rate_roll1', 'full90_rate_roll2', 'full90_rate_roll3', 'full90_rate_roll5',
               'full90_rate_roll7', 'full90_rate_roll10', 'is_home', 'gameweek'] + POSITIONS


def num(f, c, default=np.nan):
    return pd.to_numeric(f[c], errors='coerce') if c in f else pd.Series(default, index=f.index)


def _naive(v):
    return pd.to_datetime(v, utc=True).dt.tz_localize(None).astype('datetime64[ns]')


class Data:
    """History pools built once from the full frame; tokens are cut at each row's deadline."""

    def __init__(self, df, teams):
        from ..features import _compute_calendar_minutes_features
        app = df[df['minutes'] >= 1].copy()
        app['fotmob_rating'] = num(app, 'fotmob_rating') - 6.8
        app['minutes'] = app['minutes'] / 90
        bps, _ = baseline_bps_target(df[df['minutes'] >= 1])
        app['bps_target'] = bps.reindex(app.index).fillna(0) / 10
        self.app = app
        cal = _compute_calendar_minutes_features(df, include_appeared=True, per_fixture=True)
        m = df[df['minutes'] >= 1][['player_id', 'match_id', 'minutes']]
        cal = cal.drop(columns=[c for c in ('minutes',) if c in cal]).merge(
            m.drop_duplicates(['player_id', 'match_id']), on=['player_id', 'match_id'], how='left')
        cal['minutes'] = cal['minutes'].fillna(0)
        cal['appeared'] = (cal['minutes'] > 0).astype(float)
        cal['full90'] = (cal['minutes'] >= 89).astype(float)
        cal['started'] = (cal['minutes'] >= 60).astype(float)
        cal['minutes'] = cal['minutes'] / 90
        home = df.drop_duplicates(['match_id', 'team']).set_index(['match_id', 'team'])['is_home']
        cal['is_home'] = home.reindex(pd.MultiIndex.from_frame(cal[['match_id', 'team']])).to_numpy()
        self.positions = app.sort_values('result_time').groupby('player_id')[POSITIONS].last()
        cal = cal.drop(columns=[k for k in POSITIONS if k in cal]).join(self.positions, on='player_id')
        opp = app.drop_duplicates(['match_id', 'team']).set_index(['match_id', 'team'])['opponent']
        cal['opponent'] = opp.reindex(pd.MultiIndex.from_frame(cal[['match_id', 'team']])).to_numpy()
        self.cal = cal
        self.teams = teams.copy()
        self.teams['_team'] = self.teams['team'].map(normalize_team_name)
        self.teams['_opp'] = self.teams['opponent'].map(normalize_team_name)
        self.team_pool = self.teams[self.teams['goals_conceded'].notna()]
        tt = app[['team', 'result_time']].drop_duplicates()
        self._team_times = tt.assign(_t_team=_naive(tt['result_time']))[['team', '_t_team']].sort_values('_t_team')
        pt = app[['player_id', 'result_time']]
        self._player_times = pt.assign(_t_pl=_naive(pt['result_time']))[['player_id', '_t_pl']].sort_values('_t_pl')
        self._augmented = set()
        self._build_squads()

    def add_rest(self, rows):
        """log days since the club's and the player's last result before the deadline.

        A season opener reads as a 90-day gap for everyone; only the excess of the
        player's rest over his club's marks an injury or a squad absence."""
        r = rows[['team', 'player_id', 'forecast_time']].copy()
        r['_i'] = np.arange(len(r))
        r['forecast_time'] = _naive(r['forecast_time'])
        r = r.sort_values('forecast_time')
        r = pd.merge_asof(r, self._team_times, left_on='forecast_time', right_on='_t_team',
                          by='team', allow_exact_matches=False)
        r = pd.merge_asof(r.sort_values('forecast_time'), self._player_times, left_on='forecast_time',
                          right_on='_t_pl', by='player_id', allow_exact_matches=False).sort_values('_i')
        day = np.timedelta64(1, 'D')
        team_rest = ((r['forecast_time'] - r['_t_team']) / day).to_numpy()
        pl_rest = ((r['forecast_time'] - r['_t_pl']) / day).to_numpy()
        out = rows.copy()
        out['rest_team'] = np.log1p(np.clip(team_rest, 0, None))
        out['rest_player'] = np.log1p(np.clip(pl_rest, 0, None))
        out['rest_excess'] = np.log1p(np.clip(pl_rest - team_rest, 0, None))
        out['season_opener'] = (team_rest > 40).astype(float)
        return out

    def _build_squads(self):
        # No debutants: a player becomes squad context from his first appearance.
        c = self.cal[self.cal['gw_gap_since_last_appearance'].notna()]
        c = c.sort_values('minutes_roll5', ascending=False, na_position='last')
        feats = np.column_stack([pd.to_numeric(c[k], errors='coerce').fillna(0).to_numpy(np.float32)
                                 for k in SQUAD_TOKENS])
        pos = np.argmax(np.nan_to_num(c[POSITIONS].to_numpy(float)), axis=1)
        pids = c['player_id'].to_numpy()
        order = {}
        for i, k in enumerate(zip(c['match_id'].to_numpy(), c['team'].to_numpy())):
            order.setdefault(k, []).append(i)
        self._squad_groups = {k: (pids[idx], feats[idx], pos[idx]) for k, idx in order.items()}

    def squad(self, rows, length):
        """Squad-mates eligible for the same fixture, most-used in the last slot.

        An unplayed fixture has no calendar rows, so its squad is the other
        forecast rows of the same club and fixture."""
        width = len(SQUAD_TOKENS) + 2
        tok = np.zeros((len(rows), length, width), np.float32)
        mask = np.ones((len(rows), length), bool)
        my_pos = self.positions.reindex(rows['player_id']).to_numpy(float)
        my_pos = np.where(np.isnan(my_pos).all(1), -1, np.argmax(np.nan_to_num(my_pos), axis=1))
        live = None
        for i, (m, t, pid) in enumerate(zip(rows['match_id'].to_numpy(), rows['team'].to_numpy(),
                                            rows['player_id'].to_numpy())):
            g = self._squad_groups.get((m, t))
            if g is None:
                live = live or self._live_groups(rows)
                g = live.get((m, t))
                if g is None:
                    continue
            keep = g[0] != pid
            f, ps = g[1][keep][:length], g[2][keep][:length]
            k = len(f)
            if k == 0:
                continue
            tok[i, -k:, :-2] = f[::-1]
            tok[i, -k:, -2] = (ps == my_pos[i])[::-1]
            tok[i, -k:, -1] = np.arange(k)[::-1] / length
            mask[i, -k:] = False
        return tok, mask

    @staticmethod
    def _live_groups(rows):
        r = rows[rows['gw_gap_since_last_appearance'].notna()]
        r = r.assign(_m5=pd.to_numeric(r['minutes_roll5'], errors='coerce').fillna(-1))
        r = r.sort_values('_m5', ascending=False, kind='stable')
        feats = np.column_stack([pd.to_numeric(r[k], errors='coerce').fillna(0).to_numpy(np.float32)
                                 if k in r else np.zeros(len(r), np.float32) for k in SQUAD_TOKENS])
        pos = np.argmax(r[POSITIONS].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(), axis=1)
        pids = r['player_id'].to_numpy()
        groups = {}
        for i, k in enumerate(zip(r['match_id'].to_numpy(), r['team'].to_numpy())):
            groups.setdefault(k, []).append(i)
        return {k: (pids[idx], feats[idx], pos[idx]) for k, idx in groups.items()}

    def augment_unused(self, rows):
        """Squad players with no appearance this season have no calendar rows (a spell
        starts at an appearance), so their history would jump from their last game to
        this fixture and read as ~90% likely to play. Give them their club's unused
        fixtures as history (never as squad context)."""
        cal = self.cal
        add = []
        for season, part in rows.groupby('season'):
            have = set(cal.loc[cal['season'].eq(season), 'player_id'])
            fixtures = cal[cal['season'].eq(season)].drop_duplicates(['match_id', 'team'])
            for pid, team, ft in part[['player_id', 'team', 'forecast_time']].drop_duplicates(
                    'player_id').itertuples(index=False):
                if pid in have or (pid, season) in self._augmented:
                    continue
                f = fixtures[fixtures['team'].eq(team) & (fixtures['result_time'] < ft)].copy()
                self._augmented.add((pid, season))
                if f.empty:
                    continue
                f['player_id'] = pid
                for c in ('minutes', 'appeared', 'started', 'full90'):
                    f[c] = 0.
                f['gw_gap_since_last_appearance'] = np.nan
                add.append(f)
        if add:
            self.cal = pd.concat([cal] + add, ignore_index=True)
            print(f'  neural history: added unused fixtures for {len(add)} squad players '
                  'with no appearance this season', flush=True)

    def player_seqs(self, rows, spec):
        rows = rows.assign(_team=rows['team'].map(normalize_team_name),
                           _opp=rows['opponent'].map(normalize_team_name))
        out = []
        if spec.get('app'):
            cols = APP_TOKENS_WIDE if spec.get('wide') else APP_TOKENS
            out.append(history(self.app, rows, cols, spec['app'], 'player_id', same='team'))
        if spec.get('cal'):
            out.append(history(self.cal, rows, CAL_TOKENS, spec['cal'], 'player_id', same='team'))
        if spec.get('squad'):
            out.append(self.squad(rows, spec['squad']))
        if spec.get('team'):
            out.append(history(self.team_pool, rows, TEAM_TOKENS, spec['team'], '_team'))
            out.append(history(self.team_pool, rows, TEAM_TOKENS, spec['team'], '_team', row_key='_opp'))
        return out


class Inputs:
    def __init__(self, data, static_cols, spec, kind='player'):
        self.data, self.cols, self.spec, self.kind = data, static_cols, spec, kind

    def seqs(self, rows):
        if self.kind == 'team':
            d, L = self.data, self.spec['team']
            return [history(d.team_pool, rows, TEAM_TOKENS, L, '_team'),
                    history(d.team_pool, rows, TEAM_TOKENS, L, '_team', row_key='_opp')]
        return self.data.player_seqs(rows, self.spec)

    def build(self, rows, fit=False):
        if self.spec.get('rest') and self.kind == 'player':
            rows = self.data.add_rest(rows)
        if fit:
            cols = self.cols + (REST if self.spec.get('rest') else [])
            self.ss = StaticScaler([c for c in cols if c in rows], self.spec.get('ple', 0)).fit(rows)
        X = self.ss.transform(rows)
        raw = self.seqs(rows)
        if fit:
            self.ts = [TokenScaler().fit(t, m) for t, m in raw]
        return X, [(ts.transform(t, m), m) for ts, (t, m) in zip(self.ts, raw)]


# ---------------------------------------------------------------- components
class MinutesNet:
    """P(60+ | appears) classifier, then minutes-if-60+ and minutes-if-1-59 regressors."""

    def __init__(self, data, cfg, spec):
        self.cfg, self.inp = cfg, Inputs(data, PLAYER_STATIC, spec)

    @staticmethod
    def _decode(o):
        return torch.sigmoid(o[:, 0]), 60 + 30 * torch.sigmoid(o[:, 1]), 1 + 58 * torch.sigmoid(o[:, 2])

    def fit(self, train):
        # BCE only for P(60+): a loss on the mixture mean "wins" minutes Huber by
        # inflating P(60+), which wrecks points.
        self.nets_p = self._fit(train, (1., 0., 0.))
        self.nets = self._fit(train, (0., 1., 1.))
        return self

    def _fit(self, train, weights):
        train = train.sort_values('forecast_time', kind='stable')
        X, S = self.inp.build(train, fit=True)
        y = train['minutes'].to_numpy(np.float32)[:, None]
        wb, ws, wu = weights

        def loss(o, t):
            m = t[:, 0]
            p60, start, sub = self._decode(o)
            six = (m >= 60).float()
            return (wb * F.binary_cross_entropy(p60.clamp(1e-6, 1 - 1e-6), six)
                    + ws * huber(start, m, six) / 50 + wu * huber(sub, m, 1 - six) / 50)
        return train_net(X, S, y, loss, {**self.cfg, 'n_out': 3}, valid_split(train), 'minutes')[0]

    def predict(self, rows):
        X, S = self.inp.build(rows)
        o = torch.tensor(predict_nets(self.nets, X, S))
        start = torch.stack([self._decode(o[k])[1] for k in range(len(o))]).mean(0).numpy()
        sub = torch.stack([self._decode(o[k])[2] for k in range(len(o))]).mean(0).numpy()
        op = torch.tensor(predict_nets(self.nets_p, X, S))
        p60 = torch.stack([self._decode(op[k])[0] for k in range(len(op))]).mean(0).numpy()
        return p60, start, sub


class GridMinutesNet:
    """Softmax over {did not play, 1-59, 60+} on every eligible player-fixture."""

    def __init__(self, data, cfg, spec):
        self.cfg, self.data = cfg, data
        self.inp = Inputs(data, GRID_STATIC, spec)

    def fit(self, train):
        c = self.data.cal
        # Season openers stay in: the rest features tell them apart from injuries
        # (dropping each player's first fixture took 60+ log loss .4075 -> .4182).
        c = c[c['match_id'].isin(train['match_id']) & c['opponent'].notna()]
        c = c.sort_values('forecast_time', kind='stable')
        X, S = self.inp.build(c, fit=True)
        m = c['minutes'].to_numpy() * 90
        y = np.where(m <= 0, 0, np.where(m < 60, 1, 2)).astype(np.float32)[:, None]
        self.nets = train_net(X, S, y, lambda o, t: F.cross_entropy(o, t[:, 0].long()),
                              {**self.cfg, 'n_out': 3}, valid_split(c), 'grid3')[0]
        return self

    def predict(self, rows):
        X, S = self.inp.build(rows)
        return torch.softmax(torch.tensor(predict_nets(self.nets, X, S)), -1).mean(0).numpy()


EVENTS = ['goals', 'assists', 'defcon', 'saves', 'fouls', 'yellow', 'bps']


class EventNet:
    """Per-90 rates with a log(actual minutes/90) exposure offset."""

    def __init__(self, data, cfg, spec, tasks=None):
        self.cfg, self.inp = cfg, Inputs(data, PLAYER_STATIC, spec)
        self.tasks = tasks or EVENTS

    @staticmethod
    def _targets(f):
        minutes = num(f, 'minutes').to_numpy()
        bps, auth = baseline_bps_target(f)
        yel = num(f, 'yellow_cards')
        carded = yel.groupby(f['season']).transform('sum').gt(0) & yel.notna()
        expo = minutes / 90
        return np.column_stack([
            expo, num(f, 'goals').fillna(0), num(f, 'assists').fillna(0),
            num(f, 'defcon').fillna(0),
            (f['defcon_position'].isin(['DEF', 'MID']) & num(f, 'defcon').notna()),
            num(f, 'saves').fillna(0), num(f, 'is_gk').fillna(0),
            num(f, 'fouls_committed').fillna(0), yel.clip(0, 1).fillna(0), carded,
            np.nan_to_num(bps.to_numpy() / np.maximum(expo, 1 / 90)),
            np.where(minutes >= 60, expo, 0) * np.where(auth, 5., 1.),
        ]).astype(np.float32)

    def fit(self, train):
        train = train.sort_values('forecast_time', kind='stable')
        X, S = self.inp.build(train, fit=True)
        T = self._targets(train)
        tasks, ex = self.tasks, T[:, 0]
        idx = {t: i for i, t in enumerate(tasks)}
        # Start each head at its pooled per-90 rate (multi-task scaling otherwise
        # crushes defcon), then scale each task by that constant predictor's loss.
        rate = lambda y, w=1: np.log(max((y * w).sum() / (ex * w).sum(), 1e-4))
        init = {'goals': rate(T[:, 1]), 'assists': rate(T[:, 2]), 'defcon': rate(T[:, 3], T[:, 4]),
                'saves': rate(T[:, 5], T[:, 6]), 'fouls': rate(T[:, 7]), 'yellow': rate(T[:, 8], T[:, 9]),
                'bps': float((T[:, 11] * T[:, 10] / 10).sum() / T[:, 11].sum())}
        cfg = {**self.cfg, 'bias_init': [init[t] for t in tasks], 'n_out': len(tasks)}
        state = {'scale': None}

        def loss(o, t):
            le = torch.log(t[:, 0])
            terms = []
            for name in tasks:
                z = o[:, idx[name]]
                if name == 'goals':
                    terms.append(poisson_dev(z + le, t[:, 1]))
                elif name == 'assists':
                    terms.append(poisson_dev(z + le, t[:, 2]))
                elif name == 'defcon':
                    terms.append(poisson_dev(z + le, t[:, 3], t[:, 4]))
                elif name == 'saves':
                    terms.append(poisson_dev(z + le, t[:, 5], t[:, 6]))
                elif name == 'fouls':
                    terms.append(poisson_dev(z + le, t[:, 7]))
                elif name == 'yellow':
                    py = (1 - torch.exp(-torch.exp(z) * t[:, 0])).clamp(1e-6, 1 - 1e-6)
                    terms.append((F.binary_cross_entropy(py, t[:, 8], reduction='none') * t[:, 9]).sum()
                                 / t[:, 9].sum().clamp(min=1))
                elif name == 'bps':
                    w = t[:, 11]
                    terms.append((w * (z - t[:, 10] / 10) ** 2).sum() / w.sum().clamp(min=1))
            st = torch.stack(terms)
            if state['scale'] is None:
                state['scale'] = st.detach().clamp(min=1e-3)
            return (st / state['scale']).sum()
        self.valid = valid_split(train)
        self.nets, self.early_nets = train_net(X, S, T, loss, cfg, self.valid,
                                               'events:' + ','.join(tasks))
        if 'defcon' in tasks:
            # Out-of-sample defcon rates on the validation slice, for the NB dispersion.
            v = self.valid
            nets = self.early_nets or self.nets
            o = predict_nets(nets, X[v], [(t[v], m[v]) for t, m in S])
            self.valid_defcon90 = pd.Series(np.exp(o[:, :, idx['defcon']]).mean(0), index=train.index[v])
        return self

    def predict(self, rows):
        X, S = self.inp.build(rows)
        o = predict_nets(self.nets, X, S)
        out = pd.DataFrame(index=rows.index)
        for i, name in enumerate(self.tasks):
            out[f'{name}90'] = o[:, :, i].mean(0) * 10 if name == 'bps' else np.exp(o[:, :, i]).mean(0)
        return out


class TeamNet:
    def __init__(self, data, cfg, length=20):
        self.cfg, self.inp = cfg, Inputs(data, TEAM_STATIC, dict(team=length), kind='team')

    def fit(self, team_train):
        f = team_train[team_train['goals_conceded'].notna()].sort_values('forecast_time', kind='stable')
        X, S = self.inp.build(f, fit=True)
        T = np.column_stack([f['goals_conceded'].to_numpy(), np.full(len(f), np.log(1.35))]).astype(np.float32)
        self.nets = train_net(X, S, T, lambda o, t: poisson_dev(o[:, 0] + t[:, 1], t[:, 0]),
                              {**self.cfg, 'n_out': 2}, valid_split(f), 'team')[0]
        return self

    def predict(self, rows):
        X, S = self.inp.build(rows)
        return np.exp(predict_nets(self.nets, X, S)[:, :, 0] + np.log(1.35)).mean(0)


def defcon_r(defcon90, rows):
    """NB r by method of moments on 60+ minute DEF/MID rows, at actual minutes."""
    rows = rows[DefconModel._eligible(rows) & rows['defcon'].notna() & (rows['minutes'] >= 60)]
    mu = np.maximum(defcon90.loc[rows.index].to_numpy() * rows['minutes'].to_numpy() / 90, .01)
    y = rows['defcon'].to_numpy()
    phi = np.sum((y - mu) ** 2 / mu) / (len(y) - 1)
    return float(mu.mean() / (phi - 1)) if phi > 1 else None


# ---------------------------------------------------------------- system
GRU = {'kind': 'gru', 'layers': 1}
MINUTES_CFG = {'seeds': 5, 'lr': 3e-4, 'dropout': 0.3, 'wd': 1e-3, 'refit': True, **GRU}
MINUTES_SPEC = {'app': 20, 'cal': 20, 'ple': 8, 'squad': 30, 'wide': True, 'rest': True}
GRID_CFG = {'seeds': 3, 'lr': 3e-4, 'dropout': 0.3, 'wd': 1e-3, 'refit': True}
GRID_SPEC = {'app': 20, 'cal': 20, 'squad': 30, 'ple': 8, 'rest': True}
EVENT_CFG = {'seeds': 3, 'lr': 3e-4, 'refit': True}
EVENT_PLE = {'app': 20, 'cal': 10, 'team': 10, 'ple': 16}
EVENT_RAW = {'app': 20, 'cal': 10, 'team': 10}


class _RowCache:
    """Memoises per-row network outputs keyed on the row's identity and inputs, so
    the many adapter calls per frame (and per row subset) run each net once."""

    def __init__(self, fn, cols):
        self.fn, self.cols, self.store = fn, cols, {}

    def __call__(self, frame):
        cols = [c for c in self.cols if c in frame]
        keys = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy()
        todo = np.array([k not in self.store for k in keys])
        if todo.any():
            sub = frame[todo]
            out = self.fn(sub)
            for k, row in zip(keys[todo], out):
                self.store[k] = row
        return np.stack([self.store[k] for k in keys])


class NeuralSystem:
    def __init__(self, n_sims=5000, verbose=True):
        self.n_sims, self.verbose = n_sims, verbose

    def fit(self, df, train, teams):
        """``df`` is the full feature frame (history pools are cut at deadlines);
        ``train`` the appearance rows to fit on; ``teams`` the prepared team-sides,
        of which those with a match in ``train`` are fitted."""
        t0 = time.time()
        _device()
        d = self.data = Data(df, teams)
        self.minutes = MinutesNet(d, MINUTES_CFG, MINUTES_SPEC).fit(train)
        self.grid = GridMinutesNet(d, GRID_CFG, GRID_SPEC).fit(train)
        self.e_single = EventNet(d, EVENT_CFG, EVENT_PLE).fit(train)
        self.e_multi = EventNet(d, EVENT_CFG, EVENT_RAW).fit(train)
        self.e_goals = EventNet(d, EVENT_CFG, EVENT_PLE, ['goals']).fit(train)
        self.e_defcon = EventNet(d, EVENT_CFG, EVENT_PLE, ['defcon']).fit(train)
        self.e_saves = EventNet(d, {'seeds': 5, 'lr': 3e-4}, EVENT_RAW, ['saves']).fit(train)
        self.team = TeamNet(d, {'seeds': 10, 'lr': 3e-4}, 20).fit(
            d.teams[d.teams['match_id'].isin(train['match_id'])])
        # Dispersion of the averaged defcon rate, from early-stopped nets only.
        rate = (self.e_defcon.valid_defcon90 + self.e_single.valid_defcon90) / 2
        self.dispersion_r = defcon_r(rate, train.loc[rate.index])
        self.p60_given_start = 0.90  # MinutesModel's value without a 'started' column
        self.training_rows = len(train)
        if self.verbose:
            print(f'Neural system trained in {time.time() - t0:.0f}s (defcon r {self.dispersion_r})',
                  flush=True)
        keys = ['player_id', 'match_id', 'team', 'opponent', 'forecast_time']
        self._heads = _RowCache(self._compute_heads, keys + PLAYER_STATIC)
        self._mins = _RowCache(self._compute_minutes, keys + list(dict.fromkeys(PLAYER_STATIC + GRID_STATIC)))
        return self

    def _compute_heads(self, frame):
        self.data.augment_unused(frame)
        s, m, g, dc, sv = (e.predict(frame) for e in
                           (self.e_single, self.e_multi, self.e_goals, self.e_defcon, self.e_saves))
        return np.column_stack([(g['goals90'] + s['goals90']) / 2, m['assists90'],
                                (dc['defcon90'] + s['defcon90']) / 2, sv['saves90'],
                                m['fouls90'], m['yellow90'], m['bps90']])

    def _compute_minutes(self, frame):
        self.data.augment_unused(frame)
        p60, start, sub = self.minutes.predict(frame)
        pr = self.grid.predict(frame)
        p60_grid = pr[:, 2] / np.maximum(pr[:, 1] + pr[:, 2], 1e-6)
        return np.column_stack([(p60 + p60_grid) / 2, start, sub, 1 - pr[:, 0]])

    def heads(self, frame, name):
        return self._heads(frame)[:, ['goals', 'assists', 'defcon', 'saves', 'fouls',
                                      'yellow', 'bps'].index(name)]

    def minutes_part(self, frame, name):
        return self._mins(frame)[:, ['p60', 'start', 'sub', 'appear'].index(name)]

    def models(self):
        """Adapters with the interfaces the pipeline calls on its fitted models."""
        return {'minutes': NeuralMinutes(self), 'goals': NeuralCount(self, 'goals'),
                'assists': NeuralCount(self, 'assists'), 'defcon': NeuralDefcon(self),
                'saves': NeuralSaves(self), 'cards': NeuralCards(self),
                'bonus': NeuralBonus(self), 'clean_sheet': NeuralTeam(self)}


class _Part:
    def __init__(self, fn):
        self.fn = fn

    def predict(self, df):
        return self.fn(df)

    predict_proba = predict


class NeuralMinutes(MinutesModel):
    """MinutesModel's lineup blending, availability and keeper cap, over neural parts."""

    def __init__(self, system):
        self.system = system
        self.is_fitted = self.appear_is_fitted = True
        self.p60_given_start = system.p60_given_start
        self.selected_features = None
        part = lambda name: _Part(lambda df: system.minutes_part(df, name))
        self.classifier, self.starter_model = part('p60'), part('start')
        self.sub_model, self.appear_model = part('sub'), part('appear')
        self.appearance_training_rows = None


class NeuralCount:
    is_fitted = True

    def __init__(self, system, name):
        self.system, self.name, self.TARGET = system, name, name

    def predict(self, df):
        out = self.system.heads(df, self.name) * df['pred_minutes'].to_numpy() / 90
        if self.name == 'defcon':
            out = np.where(DefconModel._eligible(df).to_numpy(), out, 0.)
        return out


class NeuralDefcon(NeuralCount):
    predict_threshold_prob = DefconModel.predict_threshold_prob

    def __init__(self, system):
        super().__init__(system, 'defcon')
        self.dispersion_r = system.dispersion_r


class NeuralSaves:
    is_fitted, TARGET = True, 'saves_per90'

    def __init__(self, system):
        self.system = system

    def predict_per90(self, df):
        return np.clip(self.system.heads(df, 'saves'), 0, 12.)

    def predict_expected_saves(self, df, pred_minutes):
        return self.predict_per90(df) * np.asarray(pred_minutes) / 90


class NeuralCards:
    is_fitted, TARGET = True, 'yellow_cards'

    def __init__(self, system):
        self.system = system

    def predict_yellow_prob(self, df, pred_minutes):
        return 1 - np.exp(-self.system.heads(df, 'yellow') * np.asarray(pred_minutes) / 90)

    def predict_red_prob(self, df, pred_minutes):
        fouls = np.clip(self.system.heads(df, 'fouls'), 0, 6)
        return fouls * np.asarray(pred_minutes) / 90 * RED_PER_FOUL


class NeuralBonus(BonusModel):
    def __init__(self, system):
        super().__init__(n_simulations=system.n_sims)
        self.baseline_model = _Part(lambda df: system.heads(df, 'bps'))
        self.is_fitted = True


class NeuralTeam:
    is_fitted = True

    def __init__(self, system):
        self.system = system

    def prepare_team_features(self, *a, **k):
        return CleanSheetModel().prepare_team_features(*a, **k)

    def predict_goals_against(self, rows):
        rows = rows.assign(_team=rows['team'].map(normalize_team_name),
                           _opp=rows['opponent'].map(normalize_team_name))
        return self.system.team.predict(rows)
