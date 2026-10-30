"""SQLite storage: users, detection events, periodic stat samples."""
import os, sqlite3, time
from werkzeug.security import generate_password_hash, check_password_hash

DB = os.environ.get("DB_PATH", "data/visionpro.db")
os.makedirs(os.path.dirname(DB) or ".", exist_ok=True)


def conn():
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT UNIQUE, pw TEXT, role TEXT);
        CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, msg TEXT);
        CREATE TABLE IF NOT EXISTS samples(id INTEGER PRIMARY KEY, ts REAL, objects INT, in_count INT, out_count INT, fps REAL);
        CREATE INDEX IF NOT EXISTS ix_ev ON events(ts); CREATE INDEX IF NOT EXISTS ix_sm ON samples(ts);""")
        user = os.environ.get("ADMIN_USER", "ritesh")
        pw = os.environ.get("ADMIN_PASS", "vision@123")
        if not c.execute("SELECT 1 FROM users WHERE username=?", (user,)).fetchone():
            c.execute("INSERT INTO users(username,pw,role) VALUES(?,?,'admin')", (user, generate_password_hash(pw)))


def check_user(username, password):
    with conn() as c:
        r = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    return r if r and check_password_hash(r["pw"], password) else None


def add_event(msg, kind):
    with conn() as c:
        c.execute("INSERT INTO events(ts,kind,msg) VALUES(?,?,?)", (time.time(), kind, msg))


def add_sample(s):
    with conn() as c:
        c.execute("INSERT INTO samples(ts,objects,in_count,out_count,fps) VALUES(?,?,?,?,?)",
                  (time.time(), s["objects"], s["in_count"], s["out_count"], s["fps"]))


def events(kind="", limit=200):
    q, a = "SELECT ts,kind,msg FROM events", []
    if kind: q += " WHERE kind=?"; a.append(kind)
    with conn() as c:
        return [dict(r) for r in c.execute(q + " ORDER BY ts DESC LIMIT ?", a + [limit])]


def samples(minutes=60):
    with conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT ts,objects,in_count,out_count,fps FROM samples WHERE ts>? ORDER BY ts", (time.time() - minutes * 60,))]


def summary():
    with conn() as c:
        k = {r["kind"]: r["n"] for r in c.execute("SELECT kind,COUNT(*) n FROM events GROUP BY kind")}
        peak = c.execute("SELECT COALESCE(MAX(objects),0) m, COALESCE(AVG(fps),0) f FROM samples").fetchone()
    return dict(new=k.get("new", 0), inn=k.get("in", 0), out=k.get("out", 0), peak=peak["m"], avg_fps=round(peak["f"], 1))
