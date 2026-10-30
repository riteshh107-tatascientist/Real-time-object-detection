"""VisionPro production Flask application. Built by Ritesh Kumar Singh."""
import csv
import io
import os
import secrets
import threading
import time
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, send_file, session, url_for)
from werkzeug.utils import secure_filename

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", BASE_DIR / "videos")).resolve()
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

app = Flask(__name__)
secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    if os.environ.get("RENDER") or os.environ.get("APP_ENV") == "production":
        raise RuntimeError("SECRET_KEY is required in production. Set it in your hosting environment.")
    secret_key = secrets.token_urlsafe(32)
    print("[VisionPro] Generated a temporary development SECRET_KEY.")
app.secret_key = secret_key
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER") or os.environ.get("APP_ENV") == "production"),
    MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024,
)

import db
from engine import Engine

if not os.environ.get("ADMIN_USER") or not os.environ.get("ADMIN_PASS"):
    raise RuntimeError("Set ADMIN_USER and ADMIN_PASS in your .env file or hosting environment before starting VisionPro.")

db.init()
eng = Engine()
eng.on_event = db.add_event


def csrf_token():
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token

app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def protect_post_requests():
    if request.method == "POST":
        supplied = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        expected = session.get("csrf_token")
        if not supplied or not expected or not secrets.compare_digest(supplied, expected):
            abort(403, description="Security token expired. Refresh the page and try again.")


def sampler():
    while True:
        time.sleep(5)
        if eng.stats["status"] == "live":
            try:
                db.add_sample(eng.stats)
            except Exception as exc:
                app.logger.warning("Could not save statistics sample: %s", exc)


threading.Thread(target=sampler, daemon=True, name="visionpro-sampler").start()


def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return (jsonify(error="unauthorized"), 401) if request.path.startswith("/api") else redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapped


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"Video is too large. Maximum upload size is {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."), 413


@app.route("/health")
def health():
    return jsonify(ok=True, service="VisionPro", status=eng.stats.get("status", "starting")), 200


@app.route("/login", methods=["GET", "POST"])
def login():
    err = None
    if request.method == "POST":
        user = db.check_user(request.form.get("username", "").strip(), request.form.get("password", ""))
        if user:
            session.clear()
            session["user"] = user["username"]
            csrf_token()
            return redirect(url_for("index"))
        time.sleep(0.7)
        err = "Wrong username or password."
    return render_template("login.html", err=err)


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    return redirect(url_for("login"))


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
        frame = eng.frame_jpg
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\nCache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.04)


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-store"})


@app.route("/api/stats")
@login_required
def stats():
    return jsonify(stats=eng.stats, history=list(eng.history), events=list(eng.events), cfg=eng.cfg)


@app.route("/api/meta")
@login_required
def meta():
    try:
        return jsonify(names=eng.names())
    except Exception as exc:
        app.logger.exception("Model metadata could not be loaded")
        return jsonify(error=f"Could not load model: {exc}"), 503


@app.route("/api/config", methods=["POST"])
@login_required
def config():
    payload = request.get_json(silent=True) or {}
    if "conf" in payload:
        try:
            payload["conf"] = max(0.05, min(0.95, float(payload["conf"])))
        except (TypeError, ValueError):
            return jsonify(error="Confidence must be a number."), 400
    if "model" in payload and payload["model"] not in {"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}:
        return jsonify(error="Unsupported model. Choose yolo11n.pt, yolo11s.pt, or yolo11m.pt."), 400
    if "classes" in payload and (not isinstance(payload["classes"], list) or any(not isinstance(x, int) or x < 0 for x in payload["classes"])):
        return jsonify(error="Classes must be a list of non-negative class IDs."), 400
    if "source" in payload:
        source = str(payload["source"]).strip()
        if not source:
            return jsonify(error="Source cannot be empty."), 400
        if source.isdigit() and len(source) > 2:
            return jsonify(error="Invalid camera index."), 400
        payload["source"] = source
    eng.set_config(payload)
    return jsonify(ok=True, cfg=eng.cfg)


@app.route("/api/upload-video", methods=["POST"])
@login_required
def upload_video():
    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return jsonify(error="Choose a video file first."), 400
    original = secure_filename(uploaded.filename)
    if not original or "." not in original:
        return jsonify(error="Invalid filename."), 400
    ext = original.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify(error="Unsupported format. Use MP4, AVI, MOV, MKV, WEBM or M4V."), 415
    # A unique name avoids overwriting a previously uploaded video.
    target = VIDEO_DIR / f"{int(time.time())}_{secrets.token_hex(4)}_{original}"
    uploaded.save(target)
    eng.set_config({"source": str(target)})
    eng.log(f"Video uploaded and selected: {original}", "info")
    return jsonify(ok=True, filename=original, source=str(target), message="Upload complete. Detection source updated.")


@app.route("/api/snapshot")
@login_required
def snapshot():
    path = eng.snapshot()
    return send_file(path, as_attachment=True) if path else (jsonify(error="No frame available yet."), 400)


@app.route("/api/record", methods=["POST"])
@login_required
def record():
    return jsonify(recording=eng.toggle_record())


@app.route("/api/history")
@login_required
def api_history():
    kind = request.args.get("kind", "")[:20]
    try:
        limit = max(1, min(int(request.args.get("limit", 200)), 2000))
        minutes = max(1, min(int(request.args.get("minutes", 60)), 60 * 24 * 30))
    except (TypeError, ValueError):
        return jsonify(error="limit and minutes must be integers."), 400
    return jsonify(events=db.events(kind, limit), summary=db.summary(), samples=db.samples(minutes))


def csv_response(name, rows, cols):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for row in rows:
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row[col])) if col == "ts" else row[col] for col in cols])
    return Response("\ufeff" + buf.getvalue(), mimetype="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename={name}_{time.strftime('%Y%m%d_%H%M')}.csv"})


@app.route("/export/events.csv")
@login_required
def export_events():
    return csv_response("visionpro_events", db.events(request.args.get("kind", "")[:20], 100000), ["ts", "kind", "msg"])


@app.route("/export/samples.csv")
@login_required
def export_samples():
    return csv_response("visionpro_samples", db.samples(60 * 24 * 30), ["ts", "objects", "in_count", "out_count", "fps"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\n  VisionPro -> http://127.0.0.1:{port}  |  health: /health\n")
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)
