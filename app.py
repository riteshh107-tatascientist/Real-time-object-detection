"""VisionPro - real-time AI object detection studio. Author: Ritesh Kumar Singh (Founder & CEO)."""
import csv, io, os, threading, time
from functools import wraps
from flask import (Flask, Response, jsonify, redirect, render_template, request,
                   send_file, session, url_for)
import db
from engine import Engine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
db.init()
eng = Engine()
eng.on_event = db.add_event


def sampler():
    while True:
        time.sleep(5)
        if eng.stats["status"] == "live":
            try: db.add_sample(eng.stats)
            except Exception: pass


threading.Thread(target=sampler, daemon=True).start()


def login_required(f):
    @wraps(f)
    def w(*a, **k):
        if not session.get("user"):
            return (jsonify(error="unauthorized"), 401) if request.path.startswith("/api") else redirect(url_for("login"))
        return f(*a, **k)
    return w


@app.route("/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        u = db.check_user(request.form.get("username", "").strip(), request.form.get("password", ""))
        if u:
            session["user"] = u["username"]; return redirect(url_for("index"))
        time.sleep(1); err = "Wrong username or password."
    return render_template("login.html", err=err)


@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=session["user"])


@app.route("/history")
@login_required
def history():
    return render_template("history.html", user=session["user"])


def stream():
    while True:
        if eng.frame_jpg:
            yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + eng.frame_jpg + b"\r\n"
        time.sleep(0.03)


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=f")


@app.route("/api/stats")
@login_required
def stats():
    return jsonify(stats=eng.stats, history=list(eng.history), events=list(eng.events), cfg=eng.cfg)


@app.route("/api/meta")
@login_required
def meta():
    return jsonify(names=eng.names())


@app.route("/api/config", methods=["POST"])
@login_required
def config():
    eng.set_config(request.get_json(force=True)); return jsonify(ok=True)


@app.route("/api/snapshot")
@login_required
def snapshot():
    p = eng.snapshot()
    return send_file(p, as_attachment=True) if p else (jsonify(error="no frame"), 400)


@app.route("/api/record", methods=["POST"])
@login_required
def record():
    return jsonify(recording=eng.toggle_record())


@app.route("/api/history")
@login_required
def api_history():
    ev = db.events(request.args.get("kind", ""), min(int(request.args.get("limit", 200)), 2000))
    return jsonify(events=ev, summary=db.summary(), samples=db.samples(int(request.args.get("minutes", 60))))


def csv_response(name, rows, cols):
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(cols)
    for r in rows:
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r[c])) if c == "ts" else r[c] for c in cols])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={name}_{time.strftime('%Y%m%d_%H%M')}.csv"})


@app.route("/export/events.csv")
@login_required
def export_events():
    return csv_response("visionpro_events", db.events(request.args.get("kind", ""), 100000), ["ts", "kind", "msg"])


@app.route("/export/samples.csv")
@login_required
def export_samples():
    return csv_response("visionpro_samples", db.samples(60 * 24 * 30), ["ts", "objects", "in_count", "out_count", "fps"])


if __name__ == "__main__":
    print("\n  VisionPro running -> http://127.0.0.1:5000   (default login: see README)\n")
    app.run(host="0.0.0.0", port=5000, threaded=True)
