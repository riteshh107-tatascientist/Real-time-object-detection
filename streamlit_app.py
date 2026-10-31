"""VisionPro - AI Vision Studio (Streamlit edition).
Real-time object detection + tracking + line counting with YOLO11 and ByteTrack.
Author: Ritesh Kumar Singh (Founder & CEO)
"""
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="VisionPro | AI Vision Studio", page_icon="👁️", layout="wide",
                   initial_sidebar_state="expanded")

# Streamlit Cloud "Secrets" -> environment variables (so config.py can read them)
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd

import config
import db
from coco import COCO_NAMES
from engine import Engine

MAX_VIDEO_FRAMES = int(os.environ.get("MAX_VIDEO_FRAMES", "900"))

# ------------------------------------------------------------------ look & feel
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden}
.block-container{padding-top:1.4rem;max-width:1400px}
.hero{display:flex;align-items:center;gap:16px;margin-bottom:6px;animation:rise .8s both}
.logo{width:52px;height:52px;border-radius:14px;background:linear-gradient(135deg,#19e3b1,#6c8cff);display:grid;place-items:center;color:#06101a;font:800 24px sans-serif;box-shadow:0 0 28px rgba(25,227,177,.35)}
.hero h1{margin:0;font-size:2rem;background:linear-gradient(90deg,#19e3b1,#6c8cff,#19e3b1);background-size:200% auto;-webkit-background-clip:text;color:transparent;animation:shine 6s linear infinite}
.hero small{color:#8a98b8}
.scan{height:2px;margin:10px 0 18px;background:linear-gradient(90deg,transparent,#19e3b1,transparent);box-shadow:0 0 14px #19e3b1;animation:glow 3s ease-in-out infinite}
@keyframes shine{to{background-position:200% center}}
@keyframes rise{from{opacity:0;transform:translateY(14px)}}
@keyframes glow{50%{opacity:.35}}
[data-testid="stMetric"]{background:rgba(20,28,44,.72);border:1px solid rgba(120,150,200,.16);border-radius:14px;padding:12px 16px;transition:.25s}
[data-testid="stMetric"]:hover{border-color:#19e3b1;transform:translateY(-2px)}
[data-testid="stMetricValue"]{font-weight:800}
.stTabs [data-baseweb="tab"]{font-weight:600}
.pill{display:inline-block;padding:4px 12px;border-radius:99px;border:1px solid rgba(120,150,200,.25);color:#8a98b8;font-size:.8rem}
.foot{text-align:center;color:#8a98b8;font-size:.8rem;margin-top:30px}
@media(prefers-reduced-motion:reduce){*{animation:none!important}}
</style>""", unsafe_allow_html=True)


def hero(sub="Real-time AI Vision Studio"):
    st.markdown(f"""<div class="hero"><div class="logo">V</div><div><h1>VisionPro</h1>
    <small>{sub} &nbsp;|&nbsp; Ritesh Kumar Singh, Founder &amp; CEO</small></div></div><div class="scan"></div>""",
                unsafe_allow_html=True)


# ------------------------------------------------------------------ configuration check
problems = config.missing_settings()
if problems:
    hero()
    st.error("VisionPro is not configured yet.")
    st.markdown("Set these **secrets / environment variables** (see `.env.example`):\n\n"
                + "\n".join(f"- {p}" for p in problems))
    st.stop()


@st.cache_resource
def boot():
    """Runs once per server process: create folders/tables, sync the admin account, start DB writer."""
    db.init()
    db.ensure_admin(config.ADMIN_USERNAME, config.ADMIN_PASSWORD)
    db.start_writer()
    return {"attempts": [], "lock": threading.Lock()}


_state = boot()


def locked_out(record=False):
    """Global brute-force guard: too many failed logins in a window -> temporary lock."""
    now = time.time()
    with _state["lock"]:
        _state["attempts"] = [t for t in _state["attempts"] if now - t < config.LOGIN_WINDOW_SEC]
        if record:
            _state["attempts"].append(now)
        return len(_state["attempts"]) >= config.LOGIN_MAX_ATTEMPTS * 2


# ------------------------------------------------------------------ login
if not st.session_state.get("user"):
    hero("Sign in to continue")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid.form("login"):
        u = st.text_input("Username", autocomplete="username")
        p = st.text_input("Password", type="password", autocomplete="current-password")
        go = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if go:
        if locked_out():
            mid.error("Too many failed attempts. Please wait a few minutes.")
        else:
            row = db.check_user(u.strip(), p)
            if row:
                st.session_state["user"] = row["username"]
                st.rerun()
            locked_out(record=True)
            mid.error("Wrong username or password.")
    st.markdown('<div class="foot">Built by Ritesh Kumar Singh, Founder &amp; CEO</div>', unsafe_allow_html=True)
    st.stop()


# ------------------------------------------------------------------ per-session engine
if "eng" not in st.session_state:
    _e = Engine()
    _e.on_event = db.add_event
    st.session_state["eng"] = _e
eng: Engine = st.session_state["eng"]

with st.sidebar:
    st.markdown("### Controls")
    model = st.selectbox("Model", config.ALLOWED_MODELS, index=config.ALLOWED_MODELS.index(eng.cfg["model"]))
    conf = st.slider("Confidence", 5, 95, int(eng.cfg["conf"] * 100), format="%d%%")
    picked = st.multiselect("Filter classes (empty = all)", COCO_NAMES)
    trails = st.toggle("Motion trails", eng.cfg["trails"])
    line = st.toggle("Counting line", eng.cfg["line"])
    heat = st.toggle("Heatmap", eng.cfg["heatmap"])
    if st.button("Reset counts", use_container_width=True):
        eng.set_config({"reset": True})
    st.divider()
    st.caption(f"Signed in as **{st.session_state['user']}**")
    if st.button("Log out", use_container_width=True):
        st.session_state.clear()
        st.rerun()

eng.set_config({"model": model, "conf": conf / 100, "classes": [COCO_NAMES.index(c) for c in picked],
                "trails": trails, "line": line, "heatmap": heat})


def maybe_sample():
    if time.time() - st.session_state.get("_last_sample", 0) > 5 and eng.frame_id:
        st.session_state["_last_sample"] = time.time()
        db.add_sample(eng.stats)


def show_metrics(container):
    s = eng.stats
    with container.container():
        cols = st.columns(6)
        for col, (label, val) in zip(cols, [("Objects in frame", s["objects"]), ("Unique tracked", s["unique"]),
                                            ("Crossed IN", s["in_count"]), ("Crossed OUT", s["out_count"]),
                                            ("FPS", s["fps"]), ("Latency (ms)", s["latency"])]):
            col.metric(label, val)


def to_h264(src: Path) -> Path:
    """Browsers cannot play OpenCV's mp4v codec, so transcode to H.264 when ffmpeg is available."""
    try:
        import imageio_ffmpeg
        dst = src.with_name("annotated_h264.mp4")
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", str(dst)],
                           capture_output=True, timeout=600)
        return dst if r.returncode == 0 and dst.exists() and dst.stat().st_size > 0 else src
    except Exception:
        return src


def process_video(path, preview, bar, metrics_box):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("That file could not be opened as a video.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_VIDEO_FRAMES
    target = max(1, min(total, MAX_VIDEO_FRAMES))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    eng.set_config({"reset": True})                       # fresh tracker + counters for this video
    raw = Path(tempfile.mkdtemp(prefix="vp_")) / "annotated_raw.mp4"
    writer, i = None, 0
    try:
        while i < target:
            ok, frame = cap.read()
            if not ok:
                break
            eng.step(frame)
            out = eng.last_frame
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
            i += 1
            if i == 1 or i % 3 == 0:
                preview.image(out, channels="BGR", use_container_width=True)
                bar.progress(min(i / target, 1.0), text=f"Processing frame {i} / {target}")
                show_metrics(metrics_box)
            maybe_sample()
    finally:                                              # always release OpenCV resources
        cap.release()
        if writer is not None:
            writer.release()
    if i == 0:
        raise ValueError("The video contains no readable frames.")
    show_metrics(metrics_box)
    bar.progress(1.0, text=f"Done. {i} frames processed.")
    final = to_h264(raw)
    return dict(data=final.read_bytes(), frames=i, truncated=i >= MAX_VIDEO_FRAMES and total > MAX_VIDEO_FRAMES,
                totals=dict(eng.stats["totals"]), unique=eng.stats["unique"],
                inn=eng.stats["in_count"], out=eng.stats["out_count"], h264=final.name.endswith("h264.mp4"))


def csv_bytes(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    if "ts" in df:
        df["ts"] = pd.to_datetime(df["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    if "msg" in df:                                       # block spreadsheet formula injection
        df["msg"] = df["msg"].map(lambda v: "'" + v if v[:1] in ("=", "+", "-", "@") else v)
    return df.to_csv(index=False).encode()


def rtc_configuration():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if os.environ.get("TURN_URL"):                        # optional TURN relay for strict networks
        servers.append({"urls": [os.environ["TURN_URL"]], "username": os.environ.get("TURN_USERNAME", ""),
                        "credential": os.environ.get("TURN_CREDENTIAL", "")})
    return {"iceServers": servers}


# ------------------------------------------------------------------ pages
hero()
tab_live, tab_video, tab_photo, tab_hist = st.tabs(["🎥 Live webcam", "📹 Video studio", "📸 Photo detect", "📊 History"])

live_ctx = None
with tab_live:
    try:
        import av
        from streamlit_webrtc import WebRtcMode, webrtc_streamer

        def on_frame(frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                eng.step(img)
                img = eng.last_frame
            except Exception as e:                        # never kill the stream on one bad frame
                eng.log(f"Frame error: {e}", "warn")
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.caption("Click **START**, allow camera access in your browser, and detection runs on the live stream. "
                   "Needs HTTPS (or localhost).")
        live_ctx = webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, video_frame_callback=on_frame,
                                   rtc_configuration=rtc_configuration(), async_processing=True,
                                   media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False})
        live_metrics = st.empty()
        c1, c2, _ = st.columns([1, 1, 3])
        if c1.button("📸 Capture snapshot"):
            p = eng.snapshot()
            st.session_state["snap"] = p.read_bytes() if p else None
            if not p:
                st.warning("Start the webcam first.")
        if st.session_state.get("snap"):
            c1.download_button("Download snapshot", st.session_state["snap"], "snapshot.jpg", "image/jpeg")
        rec_label = "⏹ Stop recording" if eng.stats["recording"] else "⏺ Record"
        if c2.button(rec_label):
            recording, name = eng.toggle_record()
            if name:
                st.session_state["rec"] = (config.RECORD_DIR / name).read_bytes()
                st.session_state["rec_name"] = name
            st.rerun()
        if st.session_state.get("rec"):
            c2.download_button("Download recording", st.session_state["rec"],
                               st.session_state.get("rec_name", "recording.mp4"), "video/mp4")
    except ImportError:
        st.warning("Live webcam needs `streamlit-webrtc` and `av` (see requirements.txt). "
                   "Use Video studio or Photo detect meanwhile.")

with tab_video:
    st.caption(f"Upload a video (max {st.get_option('server.maxUploadSize')} MB, first {MAX_VIDEO_FRAMES} frames are processed). "
               "You get an annotated video, IN/OUT counts and a class summary.")
    up = st.file_uploader("Video file", type=[e.strip(".") for e in sorted(config.VIDEO_EXTS)])
    if up and st.button("▶ Process video", type="primary"):
        suffix = Path(up.name).suffix.lower()
        tmp = Path(tempfile.mkdtemp(prefix="vp_up_")) / f"input{suffix}"
        tmp.write_bytes(up.getvalue())
        bar = st.progress(0.0, text="Starting...")
        v_metrics, preview = st.empty(), st.empty()
        try:
            with st.spinner("Loading model and processing..."):
                st.session_state["video_result"] = process_video(tmp, preview, bar, v_metrics)
        except Exception as e:
            st.session_state.pop("video_result", None)
            st.error(f"Could not process this video: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    res = st.session_state.get("video_result")
    if res:
        st.success(f"Processed {res['frames']} frames."
                   + (" (Truncated to the frame limit.)" if res["truncated"] else ""))
        a, b, c = st.columns(3)
        a.metric("Unique objects tracked", res["unique"]); b.metric("Crossed IN", res["inn"]); c.metric("Crossed OUT", res["out"])
        if res["totals"]:
            st.bar_chart(pd.Series(res["totals"], name="objects").sort_values(ascending=False))
        if res["h264"]:
            st.video(res["data"])
        else:
            st.info("Preview unavailable (no H.264 encoder). Download the file to watch it.")
        st.download_button("⬇ Download annotated video", res["data"], "visionpro_annotated.mp4", "video/mp4")

with tab_photo:
    st.caption("Take a photo or upload an image for one-shot detection.")
    left, right = st.columns(2)
    shot = left.camera_input("Camera")
    img_up = right.file_uploader("...or upload an image", type=["jpg", "jpeg", "png", "webp"])
    data = shot.getvalue() if shot else (img_up.getvalue() if img_up else None)
    if data:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("That image could not be read.")
        else:
            with st.spinner("Detecting..."):
                eng.new_scene()
                eng.step(frame)
            st.image(eng.last_frame, channels="BGR", use_container_width=True)
            st.write({k: v for k, v in eng.stats["per_class"].items()} or "No objects found at this confidence.")
            st.download_button("⬇ Download annotated image", eng.frame_jpg, "visionpro_detection.jpg", "image/jpeg")
            maybe_sample()

with tab_hist:
    if not config.PERSISTENT_STORAGE:
        st.warning("Storage on this deployment is temporary: history resets when the app restarts or redeploys. "
                   "Attach persistent storage and set PERSISTENT_STORAGE=true to keep it.")
    f1, f2, f3 = st.columns([1, 1, 2])
    kind = f1.selectbox("Event type", ["all", "new", "in", "out", "info", "warn"])
    mins = f2.selectbox("Chart window", [60, 360, 1440], format_func=lambda m: f"Last {m // 60} h")
    if f3.button("🔄 Refresh"):
        st.rerun()
    summ = db.summary()
    m = st.columns(5)
    for col, (label, val) in zip(m, [("Objects tracked", summ["new"]), ("Crossed IN", summ["inn"]),
                                     ("Crossed OUT", summ["out"]), ("Peak in frame", summ["peak"]),
                                     ("Average FPS", summ["avg_fps"])]):
        col.metric(label, val)
    samples = db.samples(mins)
    if samples:
        sdf = pd.DataFrame(samples)
        sdf["time"] = pd.to_datetime(sdf["ts"], unit="s")
        st.line_chart(sdf.set_index("time")["objects"], height=220)
    events = db.events("" if kind == "all" else kind, 300)
    if events:
        edf = pd.DataFrame(events)
        edf["time"] = pd.to_datetime(edf["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(edf[["time", "kind", "msg"]], use_container_width=True, hide_index=True, height=300)
    else:
        st.info("No events yet. Run a video or the webcam and they will appear here.")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Export events CSV", csv_bytes(db.events("" if kind == "all" else kind, 100000), ["ts", "kind", "msg"]),
                       "visionpro_events.csv", "text/csv")
    d2.download_button("⬇ Export stats CSV", csv_bytes(db.samples(60 * 24 * 30), ["ts", "objects", "in_count", "out_count", "fps"]),
                       "visionpro_stats.csv", "text/csv")

st.markdown('<div class="foot">&copy; 2026 VisionPro by <b>Ritesh Kumar Singh</b>, Owner &amp; CEO</div>',
            unsafe_allow_html=True)

# Live metrics loop runs last so every tab above is already drawn.
if live_ctx is not None:
    while live_ctx.state.playing:
        show_metrics(live_metrics)
        maybe_sample()
        time.sleep(0.5)
