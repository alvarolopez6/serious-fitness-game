import sqlite3
from datetime import datetime
from pathlib import Path


DATABASE_PATH = Path("data/sessions.db")


def initialize_database() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                exercise TEXT NOT NULL,
                mode TEXT NOT NULL,
                repetitions INTEGER NOT NULL,
                duration REAL NOT NULL,
                score INTEGER NOT NULL
            )
            """
        )


def save_session(
    exercise: str,
    mode: str,
    repetitions: int,
    duration: float,
    score: int,
) -> None:
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO sessions
                (date, exercise, mode, repetitions, duration, score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                exercise,
                mode,
                repetitions,
                duration,
                score,
            ),
        )


def get_mode_statistics(mode: str) -> tuple[int, int, float, float]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        result = connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(repetitions), 0),
                   COALESCE(AVG(duration), 0), COALESCE(AVG(score), 0)
            FROM sessions
            WHERE mode = ?
            """,
            (mode,),
        ).fetchone()

    return result


def get_recent_sessions(mode: str, limit: int = 3) -> list[tuple]:
    with sqlite3.connect(DATABASE_PATH) as connection:
        return connection.execute(
            """
            SELECT date, repetitions, duration, score
            FROM sessions
            WHERE mode = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (mode, limit),
        ).fetchall()
