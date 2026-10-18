"""Experimental attention models that stand in for all eight component models.

Three networks share one architecture: a transformer encodes a history of raw
per-match tokens, and the row's engineered feature vector forms the query that
attention-pools that history.

* ``NeuralPlayerModel`` — appearances. Per-90 rates for goals, assists, defcon,
  saves, fouls, yellow-card hazard and baseline BPS, plus P(60+ | appears) and
  minutes-if-starting / minutes-if-subbed. Every event head is a rate, so
  expected counts are rate x predicted minutes and lineup news reaches them
  only through the minutes model.
* ``NeuralAppearModel`` — the appearance grid, P(plays >= 1 minute).
* ``NeuralTeamModel`` — team-matches, goals conceded on a ``prior_lambda`` offset.

``NeuralSuite`` wraps them in the interfaces the pipeline already calls, so the
holdout evaluation, simulation and report run unchanged; it remains an
experiment. Production uses only ``NeuralPlayerModel`` through ``EventBlend``:
its goal, assist, defcon and save rates replace or blend with the XGBoost event
models, while minutes, clean sheets, cards and bonus stay XGBoost. History tokens
are restricted to results known at each row's forecast deadline, the same cutoff
the rolling features use.
"""
import numpy as np
import pandas as pd

from .assists import AssistsModel
from .bonus import BaselineBPSModel, BonusModel, cap_per90_stats
from .cards import CardsModel, RED_PER_FOUL
from .clean_sheet import CleanSheetModel
from .defcon import DefconModel
from .goals import GoalsModel
from .minutes import ALL_FEATURES as MINUTES_FEATURES, APPEARANCE_FEATURES
from .saves import SavesModel

TOKEN_COLUMNS = [
    'minutes', 'goals', 'assists', 'xg', 'npxg', 'xa', 'shots', 'key_passes',
    'penalty_goals', 'touches_in_opposition_box', 'fotmob_rating', 'is_home',
    'defcon', 'saves', 'fouls_committed', 'yellow_cards',
]
TEAM_TOKEN_COLUMNS = ['goals', 'goals_conceded', 'xg', 'xga', 'shots_on_target',
                      'key_passes', 'touches', 'is_home']
PLAYER_STATIC = list(dict.fromkeys(
    MINUTES_FEATURES + GoalsModel.FEATURES + AssistsModel.FEATURES
    + DefconModel.FEATURES + SavesModel.FEATURES + CardsModel.FEATURES
    + BaselineBPSModel.FEATURES + ['pred_team_goals']))
PLAYER_HEADS = ['goals', 'assists', 'defcon', 'saves', 'fouls', 'yellow', 'bps',
                'p60', 'start_minutes', 'sub_minutes']


def _numeric(frame, name):
    return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)),
                         errors='coerce')


def baseline_bps_target(frame):
    """(baseline BPS, is-authoritative) per row, built as BaselineBPSModel.fit does."""
    from .bonus import normalize_team_name
    df = frame.copy()
    stats = df.assign(_team=df['team'].map(normalize_team_name),
                      _og=_numeric(df, 'own_goal').fillna(0))
    totals = stats.groupby(['match_id', '_team'])[['goals', '_og']].sum()
    opposing = pd.MultiIndex.from_arrays([df['match_id'], df['opponent'].map(normalize_team_name)])
    own = pd.MultiIndex.from_arrays([df['match_id'], df['team'].map(normalize_team_name)])
    df['_bps_goals_against'] = (totals['goals'].reindex(opposing).fillna(0).to_numpy()
                                + totals['_og'].reindex(own).fillna(0).to_numpy())
    df = cap_per90_stats(df)
    model = BaselineBPSModel()
    actual = model._compute_baseline_bps(df).where(
        df.get('season', pd.Series('', index=df.index)).astype(str).eq('2026/2027'))
    return actual.where(actual.notna(), model._estimate_baseline_bps(df)), actual.notna()


def _token_matrix(pool, columns):
    out = []
    for name in columns:
        values = _numeric(pool, name)
        if name == 'minutes':
            values = values / 90.
        elif name == 'fotmob_rating':
            # Missing ratings are common in older seasons; centre on a typical 6.8.
            values = values - 6.8
        out.append(values.fillna(0).to_numpy(dtype=np.float32))
    return np.column_stack(out)


def build_history(pool, rows, history=20, key='player_id', row_key=None,
                  columns=TOKEN_COLUMNS, same='team'):
    """Return (tokens, mask) of shape (n, history, n_columns + 2) and (n, history).

    ``pool[key]`` is matched against ``rows[row_key or key]``. Extras per token:
    log days from that match to the row's deadline, and whether ``pool[same]``
    equals ``rows[same]`` (e.g. same club). ``mask`` is True for padding.
    """
    row_key = row_key or key
    pool = pool.sort_values([key, 'result_time'], kind='stable')
    base = _token_matrix(pool, columns)
    pool_time = pool['result_time'].to_numpy('datetime64[ns]')
    pool_same = pool[same].astype(str).to_numpy() if same else None
    ids = pool[key].to_numpy()
    edges = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    spans = {ids[lo]: (lo, hi) for lo, hi in zip(edges[:-1], edges[1:])}

    tokens = np.zeros((len(rows), history, base.shape[1] + 2), dtype=np.float32)
    mask = np.ones((len(rows), history), dtype=bool)
    deadline = rows['forecast_time'].to_numpy('datetime64[ns]')
    row_same = rows[same].astype(str).to_numpy() if same else None
    for i, rid in enumerate(rows[row_key].to_numpy()):
        if rid not in spans:
            continue
        lo, hi = spans[rid]
        end = lo + int(np.searchsorted(pool_time[lo:hi], deadline[i], side='left'))
        begin = max(lo, end - history)
        k = end - begin
        if k == 0:
            continue
        days = (deadline[i] - pool_time[begin:end]) / np.timedelta64(1, 'D')
        # Right-aligned: the most recent match always sits in the last slot.
        tokens[i, -k:, :-2] = base[begin:end]
        tokens[i, -k:, -2] = np.log1p(np.maximum(days, 0))
        if same:
            tokens[i, -k:, -1] = pool_same[begin:end] == row_same[i]
        mask[i, -k:] = False
    return tokens, mask


class _Inputs:
    """Train-fitted scaling for the static vector and the history tokens."""

    def __init__(self, static_features, histories):
        # histories: list of build_history kwargs; their sequences are concatenated.
        self.static_features, self.histories = static_features, histories

    def static(self, frame, fit=False):
        frame = cap_per90_stats(frame)
        values = frame.reindex(columns=self.static_features).apply(
            pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
        if fit:
            median = np.nanmedian(values, axis=0)
            self.median = np.where(np.isfinite(median), median, 0)
            filled = np.where(np.isfinite(values), values, self.median)
            self.mean, self.std = filled.mean(axis=0), filled.std(axis=0) + 1e-6
        filled = np.where(np.isfinite(values), values, self.median)
        return np.clip((filled - self.mean) / self.std, -8, 8).astype(np.float32)

    def tokens(self, pools, frame, fit=False):
        parts = [build_history(pool, frame, **spec) for pool, spec in zip(pools, self.histories)]
        width = max(t.shape[2] for t, _ in parts)
        tokens = np.concatenate([np.pad(t, ((0, 0), (0, 0), (0, width - t.shape[2])))
                                 for t, _ in parts], axis=1)
        mask = np.concatenate([m for _, m in parts], axis=1)
        if fit:
            present = tokens[~mask]
            self.token_mean = present.mean(axis=0)
            self.token_std = present.std(axis=0) + 1e-6
        tokens = np.where(mask[..., None], 0, (tokens - self.token_mean) / self.token_std)
        return tokens.astype(np.float32), mask


def _network(n_token, n_static, n_out, length, d=64, heads=4, layers=2, dropout=0.1):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(net):
            super().__init__()
            net.embed = nn.Linear(n_token, d)
            net.position = nn.Parameter(torch.zeros(length + 1, d))
            # Always-visible start token, so debutants still have one key.
            net.start = nn.Parameter(torch.zeros(1, 1, d))
            layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                               norm_first=True, activation='gelu')
            net.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            net.static = nn.Sequential(nn.Linear(n_static, 256), nn.GELU(),
                                       nn.Dropout(dropout), nn.Linear(256, d))
            net.pool = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
            net.head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(128, n_out))

        def forward(net, tokens, mask, static):
            batch = tokens.shape[0]
            x = torch.cat([net.start.expand(batch, -1, -1), net.embed(tokens)], 1) + net.position
            pad = torch.cat([torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), mask], 1)
            encoded = net.encoder(x, src_key_padding_mask=pad)
            context = net.static(static)
            pooled, _ = net.pool(context[:, None], encoded, encoded, key_padding_mask=pad)
            return net.head(torch.cat([pooled[:, 0], context], 1))

    return Net()


class _SequenceRegressor:
    """Shared fit/predict loop. Subclasses define inputs, targets and the loss."""

    def __init__(self, history=20, n_seeds=3, max_epochs=40, patience=5, lr=1e-3,
                 weight_decay=1e-4, batch_size=512, verbose=True):
        self.history, self.n_seeds, self.max_epochs = history, n_seeds, max_epochs
        self.patience, self.lr, self.weight_decay = patience, lr, weight_decay
        self.batch_size, self.verbose = batch_size, verbose

    def _arrays(self, pools, frame, fit=False):
        tokens, mask = self.inputs.tokens(pools, frame, fit)
        return tokens, mask, self.inputs.static(frame, fit)

    def _train(self, pools, frame, targets, n_out, label):
        import torch
        frame = frame.sort_values('forecast_time', kind='stable')
        tokens, mask, static = self._arrays(pools, frame, fit=True)
        targets = np.asarray(targets(frame), dtype=np.float32)
        # Early stopping on the latest ~10% of deadlines inside the training window.
        deadlines = np.sort(frame['forecast_time'].unique())
        valid = (frame['forecast_time'] >= deadlines[int(len(deadlines) * 0.9)]).to_numpy()
        data = [torch.tensor(a) for a in (tokens, mask, static, targets)]
        fit_idx = torch.tensor(np.flatnonzero(~valid))
        val_idx = torch.tensor(np.flatnonzero(valid))
        self.nets = []
        for seed in range(self.n_seeds):
            torch.manual_seed(seed)
            net = _network(tokens.shape[2], static.shape[1], n_out, tokens.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            best, state, best_epoch, stale = np.inf, None, 0, 0
            for epoch in range(self.max_epochs):
                net.train()
                order = fit_idx[torch.randperm(len(fit_idx))]
                for start in range(0, len(order), self.batch_size):
                    idx = order[start:start + self.batch_size]
                    opt.zero_grad()
                    loss = self._loss(net(*(t[idx] for t in data[:3])), data[3][idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    val = float(self._loss(net(*(t[val_idx] for t in data[:3])), data[3][val_idx]))
                if val < best - 1e-5:
                    best, best_epoch, stale = val, epoch, 0
                    state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            net.load_state_dict(state)
            net.eval()
            self.nets.append(net)
            if self.verbose:
                print(f"  {label} seed {seed}: best epoch {best_epoch + 1}, val loss {best:.4f}",
                      flush=True)
        self._validation = (frame[valid], val_idx)
        return self

    def _raw(self, pools, frame):
        """Seed-averaged raw network outputs, one row per frame row."""
        import torch
        tokens, mask, static = (torch.tensor(a) for a in self._arrays(pools, frame))
        out = np.zeros((len(frame), self.nets[0].head[-1].out_features))
        with torch.no_grad():
            for net in self.nets:
                for start in range(0, len(frame), 4096):
                    sl = slice(start, start + 4096)
                    out[sl] += net(tokens[sl], mask[sl], static[sl]).numpy()
        return out / len(self.nets)


class NeuralPlayerModel(_SequenceRegressor):
    """Multi-task heads over appearances; see ``PLAYER_HEADS``."""

    def fit(self, pool, frame):
        import torch
        import torch.nn.functional as F
        self.pool = pool
        self.inputs = _Inputs([c for c in PLAYER_STATIC if c in frame.columns],
                              [dict(history=self.history)])
        frame = frame[frame['minutes'] >= 1]

        def targets(f):
            minutes = _numeric(f, 'minutes').to_numpy()
            exposure = minutes / 90
            bps, authoritative = baseline_bps_target(f)
            bps, bps_weight = bps.to_numpy(), np.where(authoritative, 5., 1.)
            # Seasons before the FPL card merge carry yellow_cards == 0, not NaN.
            # Train the card head only on seasons that recorded any yellow.
            yellows = _numeric(f, 'yellow_cards')
            carded = yellows.groupby(f['season']).transform('sum').gt(0) & yellows.notna()
            return np.column_stack([
                exposure,
                _numeric(f, 'goals').fillna(0), _numeric(f, 'assists').fillna(0),
                _numeric(f, 'defcon').fillna(0),
                (f['defcon_position'].isin(['DEF', 'MID']) & _numeric(f, 'defcon').notna()),
                _numeric(f, 'saves').fillna(0), _numeric(f, 'is_gk').fillna(0),
                _numeric(f, 'fouls_committed').fillna(0),
                _numeric(f, 'yellow_cards').clip(0, 1).fillna(0),
                np.nan_to_num(bps / np.maximum(exposure, 1 / 90)),
                # BaselineBPSModel trains on 60+ rows, weighted by minutes and 5x
                # for authoritative FPL BPS rows.
                np.where(minutes >= 60, exposure, 0) * bps_weight,
                minutes,
                carded,
            ])

        def poisson(log_rate, y, exposure, weight=None):
            # Unit deviance, not torch's NLL: the NLL drops y*log(y) - y and goes
            # negative for count-heavy targets (defcon, saves), which breaks the
            # per-task scaling below.
            log_mu = log_rate + torch.log(exposure)
            loss = torch.exp(log_mu) - y + torch.xlogy(y, y) - y * log_mu
            if weight is None:
                return loss.mean()
            return (loss * weight).sum() / weight.sum().clamp(min=1)

        def huber(pred, y, weight):
            loss = F.huber_loss(pred, y, delta=10., reduction='none') / 100.
            return (loss * weight).sum() / weight.sum().clamp(min=1)

        def loss(out, t):
            exposure, minutes = t[:, 0], t[:, 11]
            hazard = torch.exp(out[:, 5]) * exposure
            p_yellow = (1 - torch.exp(-hazard)).clamp(1e-6, 1 - 1e-6)
            bps_w = t[:, 10]
            sixty = (minutes >= 60).float()
            terms = [
                poisson(out[:, 0], t[:, 1], exposure),
                poisson(out[:, 1], t[:, 2], exposure),
                poisson(out[:, 2], t[:, 3], exposure, t[:, 4]),
                poisson(out[:, 3], t[:, 5], exposure, t[:, 6]),
                poisson(out[:, 4], t[:, 7], exposure),
                (F.binary_cross_entropy(p_yellow, t[:, 8], reduction='none') * t[:, 12]).sum()
                / t[:, 12].sum().clamp(min=1),
                (bps_w * (out[:, 6] - t[:, 9]) ** 2).sum() / bps_w.sum().clamp(min=1) / 100.,
                F.binary_cross_entropy_with_logits(out[:, 7], sixty),
                huber(60 + 30 * torch.sigmoid(out[:, 8]), minutes, sixty),
                huber(1 + 58 * torch.sigmoid(out[:, 9]), minutes, 1 - sixty),
            ]
            # Each task's loss is divided by its scale at the naive start, so no
            # single head (defcon's large counts, say) dominates the shared encoder.
            stacked = torch.stack(terms)
            if self._scale is None:
                self._scale = stacked.detach().clamp(min=1e-3)
            return (stacked / self._scale).sum()

        self._scale = None
        self._loss = loss
        return self._train([pool], frame, targets, len(PLAYER_HEADS), 'player')

    def predict_heads(self, frame, pool=None):
        raw = self._raw([self.pool if pool is None else pool], frame)
        sig = lambda x: 1 / (1 + np.exp(-x))
        out = pd.DataFrame(index=frame.index)
        for i, name in enumerate(['goals', 'assists', 'defcon', 'saves', 'fouls']):
            out[f'{name}_per90'] = np.exp(raw[:, i])
        out['yellow_hazard90'] = np.exp(raw[:, 5])
        out['bps_per90'] = raw[:, 6]
        out['p60'] = sig(raw[:, 7])
        out['start_minutes'] = 60 + 30 * sig(raw[:, 8])
        out['sub_minutes'] = 1 + 58 * sig(raw[:, 9])
        return out


class NeuralAppearModel(_SequenceRegressor):
    """P(plays >= 1 minute) on the appearance grid, isotonic-calibrated on the
    early-stopping slice (the production classifier calibrates on its latest season)."""

    def fit(self, pool, grid):
        import torch.nn.functional as F
        from sklearn.isotonic import IsotonicRegression
        self.pool = pool
        self.inputs = _Inputs([c for c in APPEARANCE_FEATURES if c in grid.columns],
                              [dict(history=self.history)])
        self._loss = lambda out, t: F.binary_cross_entropy_with_logits(out[:, 0], t[:, 0])
        self._train([pool], grid, lambda f: f[['appeared']].astype(float).to_numpy(), 1, 'appear')
        valid, _ = self._validation
        self.calibrator = IsotonicRegression(out_of_bounds='clip').fit(
            self._uncalibrated(valid), valid['appeared'].astype(int))
        return self

    def _uncalibrated(self, frame):
        return 1 / (1 + np.exp(-self._raw([self.pool], frame)[:, 0]))

    def predict_proba(self, frame):
        return np.clip(self.calibrator.predict(self._uncalibrated(frame)), 0, 1)


class NeuralTeamModel(CleanSheetModel):
    """Goals conceded: this side's history plus the opponent's, on a prior offset."""

    TEAM_HISTORY = [dict(key='_team', same=None, columns=TEAM_TOKEN_COLUMNS),
                    dict(key='_team', row_key='_opponent', same=None, columns=TEAM_TOKEN_COLUMNS)]

    def __init__(self, **kwargs):
        super().__init__()
        self.regressor = _SequenceRegressor(**kwargs)

    def _keyed(self, frame):
        from .bonus import normalize_team_name
        return frame.assign(_team=frame['team'].map(normalize_team_name),
                            _opponent=frame['opponent'].map(normalize_team_name))

    def fit_prepared(self, team_df, verbose=False):
        import torch
        import torch.nn.functional as F
        reg = self.regressor
        team_df = self._keyed(team_df[team_df['goals_conceded'].notna()])
        self.pool = team_df
        reg.inputs = _Inputs([c for c in self.FEATURES if c in team_df.columns],
                             [dict(h, history=reg.history) for h in self.TEAM_HISTORY])

        def targets(f):
            prior = _numeric(f, 'prior_lambda').fillna(1.3).clip(lower=.05)
            return np.column_stack([f['goals_conceded'], np.log(prior)])

        reg._loss = lambda out, t: F.poisson_nll_loss(out[:, 0] + t[:, 1], t[:, 0], log_input=True)
        reg._train([team_df, team_df], team_df, targets, 1, 'team')
        self.is_fitted = True
        return self

    def predict_goals_against(self, df):
        frame = self._keyed(df)
        known = frame[frame['goals_conceded'].notna()] if 'goals_conceded' in frame else frame.iloc[:0]
        pool = pd.concat([self.pool, known[~known.index.isin(self.pool.index)]])
        prior = _numeric(frame, 'prior_lambda').fillna(1.3).clip(lower=.05).to_numpy()
        return np.exp(self.regressor._raw([pool, pool], frame)[:, 0]) * prior


class NeuralSuite:
    """Builds pipeline-compatible adapters for every component model."""

    def __init__(self, pool, **kwargs):
        self.pool, self.kwargs = pool, kwargs
        self._player_key = self.player = self.appear = None

    def fit_player(self, frame):
        key = (len(frame), frame.index.min(), frame.index.max())
        if key != self._player_key:
            from ..features import build_appearance_grid
            self.player = NeuralPlayerModel(**self.kwargs).fit(self.pool, frame)
            grid = build_appearance_grid(frame, verbose=False)
            self.appear = NeuralAppearModel(**self.kwargs).fit(self.pool, grid)
            self.appear_rows = (len(grid), int(grid['appeared'].eq(0).sum()))
            self._player_key = key
        return self

    def heads(self, frame):
        return self.player.predict_heads(frame)

    def model(self, name, frame):
        if name == 'clean_sheet':
            return NeuralTeamModel(**self.kwargs).fit_prepared(frame)
        self.fit_player(frame)
        if name == 'bonus':
            return _NeuralBonus(self)
        return {'minutes': _Minutes, 'goals': _Count, 'assists': _Count,
                'defcon': _Defcon, 'saves': _Saves, 'cards': _Cards}[name](self, name, frame)


def _minutes_of(frame, pred_minutes=None):
    if pred_minutes is not None:
        return np.asarray(pred_minutes, dtype=float)
    return _numeric(frame, 'pred_minutes').fillna(60).to_numpy()


class _Adapter:
    def __init__(self, suite, name, frame=None):
        self.suite, self.name = suite, name
        self.is_fitted = True

    def fit(self, *args, **kwargs):
        return self


class _Minutes(_Adapter):
    TARGET = 'minutes'
    appear_is_fitted = True
    p60_given_start = 0.90  # no 'started' column, same default as MinutesModel

    def __init__(self, suite, name, frame=None):
        super().__init__(suite, name)
        self.appearance_training_rows, self.appearance_training_nonappearances = suite.appear_rows

    def predict_appear_proba(self, df):
        return self.suite.appear.predict_proba(df)

    def predict_distribution(self, df, lineup_p_start=None, lineup_weight=0.7,
                             appear_prob=None, availability=None):
        """Same mixture as MinutesModel.predict_distribution, with neural components."""
        heads = self.suite.heads(df)
        if appear_prob is None:
            appear_prob = self.predict_appear_proba(df)
        appear = np.clip(np.asarray(appear_prob, dtype=float), 0, 1)
        p60_joint = appear * heads['p60'].to_numpy()
        if lineup_p_start is not None:
            lineup = np.asarray(lineup_p_start, dtype=float)
            have = np.isfinite(lineup)
            starts = np.clip(lineup, 0, 1)
            p60_joint = np.where(have, (1 - lineup_weight) * p60_joint
                                 + lineup_weight * starts * self.p60_given_start, p60_joint)
            appear = np.where(have, (1 - lineup_weight) * appear
                              + lineup_weight * np.maximum(appear, starts), appear)
        if availability is not None:
            available = np.clip(np.asarray(availability, dtype=float), 0, 1)
            appear, p60_joint = appear * available, p60_joint * available
        probability = np.column_stack([1 - appear, appear - p60_joint, p60_joint])
        minutes = np.column_stack([np.zeros(len(df)), heads['sub_minutes'], heads['start_minutes']])
        return probability, minutes

    def predict(self, df):
        heads = self.suite.heads(df)
        p60 = heads['p60'].to_numpy()
        return p60 * heads['start_minutes'].to_numpy() + (1 - p60) * heads['sub_minutes'].to_numpy()


class _Count(_Adapter):
    @property
    def TARGET(self):
        return self.name

    def predict(self, df):
        return self.suite.heads(df)[f'{self.name}_per90'].to_numpy() * _minutes_of(df) / 90


class _Defcon(_Adapter):
    TARGET = 'defcon'

    def __init__(self, suite, name, frame):
        super().__init__(suite, name)
        # Pearson dispersion on the training rows, as DefconModel does.
        rows = frame[frame['defcon_position'].isin(['DEF', 'MID']) & frame['defcon'].notna()
                     & (frame['minutes'] >= 1)]
        mu = np.maximum(suite.heads(rows)['defcon_per90'].to_numpy() * rows['minutes'].to_numpy() / 90, .01)
        phi = np.sum((rows['defcon'].to_numpy() - mu) ** 2 / mu) / (len(rows) - 1)
        self.dispersion_r = max(float(mu.mean() / (phi - 1)), .5) if phi > 1 else None

    def predict(self, df):
        expected = self.suite.heads(df)['defcon_per90'].to_numpy() * _minutes_of(df) / 90
        return np.where(df['defcon_position'].isin(['DEF', 'MID']).to_numpy(), expected, 0.)

    predict_threshold_prob = DefconModel.predict_threshold_prob


class _Saves(_Adapter):
    TARGET = 'saves_per90'

    def predict_per90(self, df):
        return np.clip(self.suite.heads(df)['saves_per90'].to_numpy(), 0, 12.)

    def predict_expected_saves(self, df, pred_minutes):
        return self.predict_per90(df) * np.asarray(pred_minutes) / 90


class _Cards(_Adapter):
    TARGET = 'yellow_cards'

    def predict_yellow_prob(self, df, pred_minutes=None):
        hazard = self.suite.heads(df)['yellow_hazard90'].to_numpy()
        return 1 - np.exp(-hazard * _minutes_of(df, pred_minutes) / 90)

    def predict_red_prob(self, df, pred_minutes=None):
        fouls = np.clip(self.suite.heads(df)['fouls_per90'].to_numpy(), 0, 6.)
        return fouls * _minutes_of(df, pred_minutes) / 90 * RED_PER_FOUL


class _NeuralBonus(BonusModel):
    def __init__(self, suite):
        super().__init__()
        self.baseline_model = _BaselineBPS(suite)
        self.is_fitted = True

    def fit(self, *args, **kwargs):
        return self


class _BaselineBPS:
    def __init__(self, suite):
        self.suite = suite

    def predict(self, df):
        return self.suite.heads(df)['bps_per90'].to_numpy()


class EventBlend:
    """``weight`` x neural + (1 - weight) x XGBoost for one event model.

    The neural side is a per-90 rate times ``pred_minutes``, so it shares the
    XGBoost minutes model and lineup handling. Exposes the interface of the
    model it wraps (goals/assists ``predict``, defcon threshold probabilities,
    saves per 90).
    """

    def __init__(self, player, base, name, weight, train):
        self.player, self.base, self.name, self.weight = player, base, name, weight
        self.TARGET, self.is_fitted = base.TARGET, True
        if name == 'defcon':
            # Pearson dispersion of the blended mean, as DefconModel.fit does.
            self._estimate_dispersion(train[DefconModel._eligible(train) & train['defcon'].notna()
                                            & (train['minutes'] >= 1)])

    def _neural(self, df):
        heads = self.player.predict_heads(df)
        if self.name == 'saves':
            return np.clip(heads['saves_per90'].to_numpy(), 0, 12.)
        expected = heads[f'{self.name}_per90'].to_numpy() * _minutes_of(df) / 90
        if self.name == 'defcon':
            expected = np.where(DefconModel._eligible(df).to_numpy(), expected, 0.)
        return expected

    def _blend(self, df, base):
        return self.weight * self._neural(df) + (1 - self.weight) * base

    def predict(self, df):
        return self._blend(df, self.base.predict(df))

    def predict_per90(self, df):
        return self._blend(df, self.base.predict_per90(df))

    def predict_expected_saves(self, df, pred_minutes):
        return self.predict_per90(df) * np.asarray(pred_minutes) / 90

    _estimate_dispersion = DefconModel._estimate_dispersion
    predict_threshold_prob = DefconModel.predict_threshold_prob

    def __getattr__(self, attr):
        # Anything else (feature_importance, selected features) is the XGBoost model's.
        if attr.startswith('__') or 'base' not in self.__dict__:
            raise AttributeError(attr)
        return getattr(self.base, attr)
