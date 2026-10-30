"""SQLite persistence for users, detection events and performance samples."""
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

DB = os.environ.get("DB_PATH", str(Path(__file__).resolve().parent / "data" / "visionpro.db"))
Path(DB).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def conn():
    connection = sqlite3.connect(DB, timeout=20)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=20000")
    connection.execute("PRAGMA journal_mode=WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init():
    with conn() as connection:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, pw TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'admin');
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, msg TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, ts REAL NOT NULL, objects INT, in_count INT, out_count INT, fps REAL);
        CREATE INDEX IF NOT EXISTS ix_ev_ts ON events(ts);
        CREATE INDEX IF NOT EXISTS ix_ev_kind_ts ON events(kind, ts);
        CREATE INDEX IF NOT EXISTS ix_sm_ts ON samples(ts);
        """)
        username = os.environ["ADMIN_USER"].strip()
        password = os.environ["ADMIN_PASS"]
        if len(password) < 12:
            raise RuntimeError("ADMIN_PASS must contain at least 12 characters.")
        existing = connection.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if not existing:
            connection.execute("INSERT INTO users(username,pw,role) VALUES(?,?,'admin')", (username, generate_password_hash(password)))


def check_user(username, password):
    with conn() as connection:
        row = connection.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return dict(row) if row and check_password_hash(row["pw"], password) else None


def add_event(msg, kind):
    with conn() as connection:
        connection.execute("INSERT INTO events(ts,kind,msg) VALUES(?,?,?)", (time.time(), kind, str(msg)[:1000]))


def add_sample(stats):
    with conn() as connection:
        connection.execute("INSERT INTO samples(ts,objects,in_count,out_count,fps) VALUES(?,?,?,?,?)",
                           (time.time(), stats.get("objects", 0), stats.get("in_count", 0), stats.get("out_count", 0), stats.get("fps", 0)))


def events(kind="", limit=200):
    query, args = "SELECT ts,kind,msg FROM events", []
    if kind:
        query += " WHERE kind=?"
        args.append(kind)
    with conn() as connection:
        return [dict(row) for row in connection.execute(query + " ORDER BY ts DESC LIMIT ?", args + [limit])]


def samples(minutes=60):
    with conn() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT ts,objects,in_count,out_count,fps FROM samples WHERE ts>? ORDER BY ts", (time.time() - minutes * 60,))]


def summary():
    with conn() as connection:
        counts = {row["kind"]: row["n"] for row in connection.execute("SELECT kind,COUNT(*) n FROM events GROUP BY kind")}
        peak = connection.execute("SELECT COALESCE(MAX(objects),0) m, COALESCE(AVG(fps),0) f FROM samples").fetchone()
    return dict(new=counts.get("new", 0), inn=counts.get("in", 0), out=counts.get("out", 0), peak=peak["m"], avg_fps=round(peak["f"], 1))
