"""
Experiment logging for FPL tuning runs.

Persists tuning results (CV scores, test metrics, hyperparams, selected features)
to a SQLite database for cross-session comparison.
"""
import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def _get_db_path(data_dir: str = 'data') -> Path:
    return Path(data_dir) / 'experiments.db'


def _init_db(conn: sqlite3.Connection):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            run_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            description TEXT,
            n_iter INTEGER,
            test_size REAL,
            fpl_points_mae REAL,
            fpl_points_mae_inc_bonus REAL,
            fpl_points_mae_top25_inc_bonus REAL
        );

        CREATE TABLE IF NOT EXISTS model_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            metric_name TEXT,
            cv_score REAL,
            test_score REAL,
            mae REAL,
            n_features INTEGER,
            tuned_params TEXT,
            selected_features TEXT,
            FOREIGN KEY (run_id) REFERENCES experiments(run_id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            gameweek INTEGER NOT NULL,
            season TEXT NOT NULL,
            player_id TEXT,
            player_name TEXT,
            team TEXT,
            opponent TEXT,
            is_home INTEGER,
            fpl_position TEXT,
            pred_minutes REAL,
            pred_exp_goals REAL,
            pred_exp_assists REAL,
            pred_cs_prob REAL,
            pred_goals_against REAL,
            pred_defcon_prob REAL,
            pred_yellow_prob REAL,
            pred_red_prob REAL,
            pred_exp_saves REAL,
            pred_bonus REAL,
            exp_total_pts REAL
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_gw
            ON predictions(season, gameweek);
        CREATE INDEX IF NOT EXISTS idx_predictions_player
            ON predictions(player_id, season);
    """)


def _migrate(conn: sqlite3.Connection):
    """Add columns introduced after initial schema. Idempotent."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()}
    if 'fpl_points_mae_top25_inc_bonus' not in cols:
        conn.execute("ALTER TABLE experiments ADD COLUMN fpl_points_mae_top25_inc_bonus REAL")
        conn.commit()


def _connect(data_dir: str = 'data') -> sqlite3.Connection:
    db_path = _get_db_path(data_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    _init_db(conn)
    _migrate(conn)
    return conn


def log_experiment(
    data_dir: str,
    n_iter: int,
    test_size: float,
    tuned_params: dict,
    cv_scores: dict,
    test_metrics: dict,
    description: str = '',
    fpl_points_mae: float = None,
    fpl_points_mae_inc_bonus: float = None,
    fpl_points_mae_top25_inc_bonus: float = None,
) -> str:
    """Log a complete tuning run to the database.

    Args:
        data_dir: Path to data directory (where experiments.db lives).
        n_iter: Number of Optuna trials per model.
        test_size: Fraction held out for testing.
        tuned_params: Dict of {model_name: {param: value, selected_features: [...]}}
        cv_scores: Dict of {model_name: best_cv_score} from Optuna.
        test_metrics: Dict of {model_name: {metric_name, primary, MAE}} from _evaluate_on_test_set.
        description: Optional description of this run.

    Returns:
        run_id (str)
    """
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    timestamp = datetime.now().isoformat()

    conn = _connect(data_dir)
    try:
        conn.execute(
            """INSERT INTO experiments (run_id, timestamp, description, n_iter, test_size,
               fpl_points_mae, fpl_points_mae_inc_bonus, fpl_points_mae_top25_inc_bonus)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, timestamp, description, n_iter, test_size,
             fpl_points_mae, fpl_points_mae_inc_bonus, fpl_points_mae_top25_inc_bonus),
        )

        all_models = set(tuned_params.keys()) | set(test_metrics.keys())
        for model_name in all_models:
            raw_params = tuned_params.get(model_name, {})
            params_copy = {k: v for k, v in raw_params.items() if k != 'selected_features'}
            selected = raw_params.get('selected_features')

            metrics = test_metrics.get(model_name, {})
            cv = cv_scores.get(model_name)

            conn.execute(
                """INSERT INTO model_results
                   (run_id, model_name, metric_name, cv_score, test_score, mae, n_features, tuned_params, selected_features)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    model_name,
                    metrics.get('metric_name'),
                    cv,
                    metrics.get('primary'),
                    metrics.get('MAE'),
                    len(selected) if selected else None,
                    json.dumps(params_copy),
                    json.dumps(selected) if selected else None,
                ),
            )

        conn.commit()
    finally:
        conn.close()

    return run_id


def get_history(data_dir: str = 'data', model: Optional[str] = None) -> pd.DataFrame:
    """Return experiment history as a DataFrame.

    Args:
        data_dir: Path to data directory.
        model: If provided, filter to a specific model.

    Returns:
        DataFrame with columns: run_id, timestamp, description, model_name,
        metric_name, cv_score, test_score, mae, n_features, n_iter, test_size
    """
    conn = _connect(data_dir)
    try:
        query = """
            SELECT e.run_id, e.timestamp, e.description, e.n_iter, e.test_size,
                   e.fpl_points_mae, e.fpl_points_mae_inc_bonus,
                   m.model_name, m.metric_name, m.cv_score, m.test_score, m.mae, m.n_features
            FROM experiments e
            JOIN model_results m ON e.run_id = m.run_id
        """
        params = []
        if model:
            query += " WHERE m.model_name = ?"
            params.append(model)
        query += " ORDER BY e.timestamp DESC, m.model_name"

        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


def get_best_run(data_dir: str, model_name: str) -> Optional[pd.Series]:
    """Return the best test score row for a given model (lowest test_score).

    Args:
        data_dir: Path to data directory.
        model_name: Model to query.

    Returns:
        Series with best run info, or None if no runs found.
    """
    conn = _connect(data_dir)
    try:
        query = """
            SELECT e.run_id, e.timestamp, e.description, e.n_iter, e.test_size,
                   m.model_name, m.metric_name, m.cv_score, m.test_score, m.mae, m.n_features
            FROM experiments e
            JOIN model_results m ON e.run_id = m.run_id
            WHERE m.model_name = ? AND m.test_score IS NOT NULL
            ORDER BY m.test_score ASC
            LIMIT 1
        """
        df = pd.read_sql_query(query, conn, params=[model_name])
    finally:
        conn.close()

    if df.empty:
        return None
    return df.iloc[0]


def log_predictions(
    data_dir: str,
    predictions: pd.DataFrame,
    gameweek: int,
    season: str,
) -> int:
    """Log gameweek predictions to the database.

    Replaces any existing predictions for the same gameweek/season.

    Returns:
        Number of rows inserted.
    """
    timestamp = datetime.now().isoformat()
    conn = _connect(data_dir)

    try:
        # Clear previous predictions for this GW (allows re-running)
        conn.execute(
            "DELETE FROM predictions WHERE gameweek = ? AND season = ?",
            (gameweek, season),
        )

        rows = []
        for _, r in predictions.iterrows():
            rows.append((
                timestamp,
                gameweek,
                season,
                str(r.get('player_id', '')),
                str(r.get('player_name', '')),
                str(r.get('team', '')),
                str(r.get('opponent', '')),
                int(r.get('is_home', 0)) if not pd.isna(r.get('is_home', 0)) else 0,
                str(r.get('fpl_position', '')),
                float(r.get('pred_minutes', 0)),
                float(r.get('pred_exp_goals', 0)),
                float(r.get('pred_exp_assists', 0)),
                float(r.get('pred_cs_prob', 0)),
                float(r.get('pred_goals_against', 0)),
                float(r.get('pred_defcon_prob', 0)),
                float(r.get('pred_yellow_prob', 0)),
                float(r.get('pred_red_prob', 0)),
                float(r.get('pred_exp_saves', 0)),
                float(r.get('pred_bonus', 0)),
                float(r.get('exp_total_pts', 0)),
            ))

        conn.executemany(
            """INSERT INTO predictions
               (timestamp, gameweek, season, player_id, player_name, team, opponent,
                is_home, fpl_position, pred_minutes, pred_exp_goals, pred_exp_assists,
                pred_cs_prob, pred_goals_against, pred_defcon_prob, pred_yellow_prob,
                pred_red_prob, pred_exp_saves, pred_bonus, exp_total_pts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
    finally:
        conn.close()

    return len(rows)


def clear_experiments(data_dir: str = 'data'):
    """Delete all experiment and model_results rows. Predictions are kept."""
    conn = _connect(data_dir)
    try:
        conn.execute("DELETE FROM model_results")
        conn.execute("DELETE FROM experiments")
        conn.commit()
    finally:
        conn.close()


def get_predictions(
    data_dir: str = 'data',
    gameweek: Optional[int] = None,
    season: Optional[str] = None,
    player_name: Optional[str] = None,
) -> pd.DataFrame:
    """Query stored predictions.

    Args:
        data_dir: Path to data directory.
        gameweek: Filter to a specific gameweek.
        season: Filter to a specific season.
        player_name: Filter to a player (substring match, case-insensitive).

    Returns:
        DataFrame of predictions.
    """
    conn = _connect(data_dir)
    try:
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        if gameweek is not None:
            query += " AND gameweek = ?"
            params.append(gameweek)
        if season is not None:
            query += " AND season = ?"
            params.append(season)
        if player_name is not None:
            query += " AND player_name LIKE ?"
            params.append(f'%{player_name}%')
        query += " ORDER BY exp_total_pts DESC"

        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df
