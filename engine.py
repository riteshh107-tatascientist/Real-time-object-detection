"""VisionPro detection engine: YOLO11 + ByteTrack, line counting, trails, heatmap, recording.

Two ways to feed frames:
  * server sources (uploaded video, video file, RTSP/HTTP URL, server camera) -> background loop()
  * browser webcam -> the browser POSTs JPEG frames to /api/frame -> push_frame()
Both paths use step(), guarded by one lock, so the model and tracker are never used concurrently.
"""
import atexit
import ipaddress
import logging
import os
import socket
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

import config

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")
log = logging.getLogger("visionpro.engine")

PALETTE = [(255, 170, 0), (0, 220, 150), (255, 90, 120), (120, 140, 255),
           (0, 200, 255), (200, 120, 255), (90, 255, 120), (255, 210, 80)]


class SourceError(ValueError):
    pass


def validate_source(src):
    """Only allow safe sources: browser, (optional) server camera, remote streams, or files we control."""
    src = str(src).strip()
    if src == "browser":
        return src
    if src.isdigit():
        if not config.ALLOW_SERVER_CAMERA:
            raise SourceError("Server cameras are disabled here. Use the browser webcam, an uploaded video or an RTSP URL.")
        return src
    parsed = urlparse(src)
    if parsed.scheme in ("rtsp", "rtsps", "http", "https"):
        if not parsed.hostname:
            raise SourceError("The URL has no host name.")
        if not config.ALLOW_PRIVATE_URLS:
            try:
                infos = socket.getaddrinfo(parsed.hostname, None)
            except socket.gaierror:
                raise SourceError("Cannot resolve that host name.")
            for info in infos:
                ip = ipaddress.ip_address(info[4][0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                    raise SourceError("Private or local network addresses are not allowed on this deployment.")
        return src
    path = Path(src).expanduser()
    if not path.is_absolute():
        path = config.BASE / path
    path = path.resolve()
    for root in (config.UPLOAD_DIR.resolve(), config.VIDEO_DIR.resolve()):
        if root in path.parents and path.is_file() and path.suffix.lower() in config.VIDEO_EXTS:
            return str(path)
    raise SourceError("Video files must be uploaded in the app or placed in the videos/ folder.")


class Engine:
    def __init__(self):
        self.cfg = dict(model=config.DEFAULT_MODEL, conf=0.40, source="browser", classes=[],
                        trails=True, line=True, heatmap=False, paused=False)
        self.model, self.model_name = None, None
        self.frame_jpg, self.frame_id, self.last_frame = None, 0, None
        self.stats = dict(fps=0.0, latency=0.0, objects=0, per_class={}, in_count=0, out_count=0,
                          unique=0, status="starting", recording=False, totals={})
        self.history = deque(maxlen=60)
        self.events = deque(maxlen=40)
        self.trails = defaultdict(lambda: deque(maxlen=30))
        self.side, self.seen, self.last_seen = {}, set(), {}
        self.totals = defaultdict(int)
        self.heat = None
        self.writer, self.writer_size, self.writer_path = None, None, None
        self.on_event = None
        self._reload = True
        self._frames = 0
        self._t_prev, self._fps = time.time(), 0.0
        self._last_push = 0.0
        self._last_log = {}
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.RLock()

    # ---------- lifecycle ----------
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self.loop, name="engine", daemon=True)
            self._thread.start()
            atexit.register(self.shutdown)

    def shutdown(self):
        self._stop.set()
        with self.lock:
            self._close_writer()

    # ---------- config ----------
    def set_config(self, new):
        """Validate and apply settings. Returns an error string, or None on success."""
        with self.lock:
            if "model" in new:
                if new["model"] not in config.ALLOWED_MODELS:
                    return "Unknown model."
                self.cfg["model"] = new["model"]
            if "conf" in new:
                try:
                    self.cfg["conf"] = min(0.95, max(0.05, float(new["conf"])))
                except (TypeError, ValueError):
                    return "Invalid confidence."
            if "classes" in new:
                try:
                    cls = sorted({int(c) for c in new["classes"]})
                except (TypeError, ValueError):
                    return "Invalid class list."
                if any(c < 0 or c > 79 for c in cls):
                    return "Invalid class id."
                self.cfg["classes"] = cls
            for k in ("trails", "line", "heatmap", "paused"):
                if k in new:
                    self.cfg[k] = bool(new[k])
            if "source" in new:
                try:
                    src = validate_source(new["source"])
                except SourceError as e:
                    return str(e)
                if src != self.cfg["source"]:
                    self.cfg["source"] = src
                    self._reload = True
                    self._reset_tracking(counts=True)
            if new.get("reset"):
                self._reset_tracking(counts=True)
        return None

    def new_scene(self):
        """Forget tracker state (use between unrelated images/videos) without touching the counters."""
        with self.lock:
            self._reset_tracking(counts=False)

    def public_cfg(self):
        c = dict(self.cfg)
        src = c["source"]
        if src not in ("browser",) and not src.isdigit() and "://" not in src:
            c["source"] = "upload:" + Path(src).name
        return c

    def _reset_tracking(self, counts):
        self.trails.clear(); self.side.clear(); self.last_seen.clear(); self.heat = None
        if self.model is not None:
            self.model.predictor = None          # forces ByteTrack to start fresh
        if counts:
            self.stats.update(in_count=0, out_count=0, unique=0, objects=0, per_class={}, totals={})
            self.totals.clear(); self.seen.clear(); self.events.clear(); self.history.clear()

    def log(self, msg, kind="info"):
        now = time.time()
        if now - self._last_log.get(msg, 0) < 5 and kind in ("info", "warn"):
            return                                # de-duplicate repeated system messages
        self._last_log[msg] = now
        if len(self._last_log) > 200:
            self._last_log.clear()
        self.events.appendleft(dict(t=time.strftime("%H:%M:%S"), msg=msg, kind=kind))
        if self.on_event:
            try:
                self.on_event(msg, kind)
            except Exception:
                pass

    # ---------- model ----------
    def _ensure_model(self):
        name = self.cfg["model"]
        if self.model is not None and self.model_name == name:
            return
        self.stats["status"] = "loading model"
        try:
            import torch
            torch.set_num_threads(config.TORCH_THREADS)
            from ultralytics import YOLO
            config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self.model = YOLO(str(config.MODEL_DIR / name))
            self.model_name = name
            self.log(f"Model loaded: {name}")
        except Exception as e:
            self.cfg["model"] = self.model_name or config.DEFAULT_MODEL
            self.log(f"Could not load {name}: {e}", "warn")
            if self.model is None:
                raise

    # ---------- inputs ----------
    def push_frame(self, data):
        """Process one JPEG frame from the browser. Returns annotated JPEG bytes (None while paused)."""
        if self.cfg["source"] != "browser":
            raise SourceError("Browser webcam is not the active source.")
        if self.cfg["paused"]:
            return None
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or max(frame.shape[:2]) > 4096:
            raise ValueError("Invalid image.")
        self._last_push = time.time()
        return self.step(frame)

    def _open(self, src):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if "://" in src:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if cap.isOpened():
            self.log("Source connected" if "://" in src or src.isdigit() else f"Video loaded: {Path(src).name}")
            return cap
        cap.release()
        self.log("Cannot open the selected source", "warn")
        self.stats["status"] = "no source"
        return None

    def loop(self):
        cap, fails, is_file = None, 0, False
        while not self._stop.is_set():
            try:
                src = self.cfg["source"]
                if src == "browser":
                    if cap is not None:
                        cap.release(); cap = None
                    self.stats["status"] = "live" if time.time() - self._last_push < 3 else "waiting for webcam"
                    time.sleep(0.25); continue
                if self.cfg["paused"]:
                    self.stats["status"] = "paused"; time.sleep(0.1); continue
                if self._reload or cap is None:
                    if cap is not None:
                        cap.release(); cap = None
                    self._reload, fails = False, 0
                    is_file = not (src.isdigit() or "://" in src)
                    cap = self._open(src)
                    if cap is None:
                        time.sleep(2); self._reload = True; continue
                t0 = time.time()
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if is_file and fails <= 3:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)      # loop uploaded videos
                        self._reset_tracking(counts=False); continue
                    cap.release(); cap = None
                    self.stats["status"] = "no source"
                    self.log("Source stopped delivering frames", "warn")
                    time.sleep(2); self._reload = True; continue
                fails = 0
                self.stats["status"] = "live"
                self.step(frame)
                if is_file:                                        # play files at their natural speed
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    time.sleep(max(0.0, 1.0 / min(fps, 60) - (time.time() - t0)))
            except Exception as e:
                log.exception("engine loop error")
                self.stats["status"] = "error"
                self.log(f"Processing error: {e}", "warn")
                if cap is not None:
                    cap.release(); cap = None
                self._reload = True
                time.sleep(1)
        if cap is not None:
            cap.release()

    # ---------- one frame ----------
    def step(self, frame):
        with self.lock:
            self._ensure_model()
            t0 = time.time()
            h0, w0 = frame.shape[:2]
            if w0 != config.PROCESS_WIDTH:
                frame = cv2.resize(frame, (config.PROCESS_WIDTH, int(h0 * config.PROCESS_WIDTH / w0)))
            out = self.process(frame)
            self.stats["latency"] = round((time.time() - t0) * 1000, 1)
            now = time.time()
            self._fps = 0.9 * self._fps + 0.1 / max(now - self._t_prev, 1e-3) if self._frames else 0.0
            self._t_prev = now
            self.stats["fps"] = round(self._fps, 1)
            self._hud(out)
            self.last_frame = out
            if self.writer is not None:
                h, w = out.shape[:2]
                self.writer.write(out if (w, h) == self.writer_size else cv2.resize(out, self.writer_size))
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.frame_jpg = buf.tobytes()
                self.frame_id += 1
            return self.frame_jpg

    def process(self, frame):
        cfg = self.cfg
        h, w = frame.shape[:2]
        r = self.model.track(frame, persist=True, conf=cfg["conf"], imgsz=config.IMGSZ, verbose=False,
                             classes=cfg["classes"] or None, tracker="bytetrack.yaml")[0]
        line_y = int(h * 0.55)
        if self.heat is None or self.heat.shape[:2] != (h, w):
            self.heat = np.zeros((h, w), np.float32)
        self.heat *= 0.985
        self._frames += 1
        per_class, n = defaultdict(int), 0
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for box, tid in zip(r.boxes, ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls, conf = int(box.cls[0]), float(box.conf[0])
                name = self.model.names[cls]
                col = PALETTE[cls % len(PALETTE)]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                per_class[name] += 1
                n += 1
                cv2.circle(self.heat, (cx, cy), 28, 1.0, -1)
                if tid is not None:
                    self.last_seen[tid] = self._frames
                    if tid not in self.seen:
                        self.seen.add(tid)
                        self.totals[name] += 1
                        self.log(f"New {name} #{tid}", "new")
                    tr = self.trails[tid]
                    tr.append((cx, cy))
                    if cfg["trails"] and len(tr) > 1:
                        for i in range(1, len(tr)):
                            cv2.line(frame, tr[i - 1], tr[i], col, max(1, i // 8))
                    below = cy > line_y
                    if cfg["line"] and tid in self.side and self.side[tid] != below:
                        self.stats["in_count" if below else "out_count"] += 1
                        self.log(f"{name} #{tid} crossed {'IN' if below else 'OUT'}", "in" if below else "out")
                    self.side[tid] = below
                self._draw_box(frame, x1, y1, x2, y2, col,
                               f"{name}{' #' + str(tid) if tid is not None else ''} {conf:.0%}")
        if self._frames % 300 == 0:                      # drop tracks that disappeared (bounded memory)
            for tid in [t for t, f in self.last_seen.items() if self._frames - f > 300]:
                self.last_seen.pop(tid, None); self.trails.pop(tid, None); self.side.pop(tid, None)
        if cfg["heatmap"]:
            hm = cv2.applyColorMap(np.clip(self.heat * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (self.heat > 0.05)[..., None]
            frame[:] = np.where(mask, cv2.addWeighted(frame, 0.55, hm, 0.45, 0), frame)
        if cfg["line"]:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 220, 150), 2)
            cv2.putText(frame, "COUNT LINE", (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 150), 1)
        self.stats.update(objects=n, per_class=dict(per_class), unique=len(self.seen), totals=dict(self.totals))
        self.history.append(n)
        return frame

    @staticmethod
    def _draw_box(img, x1, y1, x2, y2, col, label):
        L = max(12, min(x2 - x1, y2 - y1) // 4)
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(img, (px, py), (px + dx * L, py), col, 3)
            cv2.line(img, (px, py), (px, py + dy * L), col, 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), col, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 20), 1, cv2.LINE_AA)

    @staticmethod
    def _hud(img):
        cv2.putText(img, "VisionPro | Ritesh Kumar Singh", (12, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # ---------- snapshot / recording ----------
    @staticmethod
    def _keep_latest(folder, pattern, keep):
        files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
            except OSError:
                pass

    def snapshot(self):
        if self.frame_jpg is None:
            return None
        config.SNAP_DIR.mkdir(parents=True, exist_ok=True)
        path = config.SNAP_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        path.write_bytes(self.frame_jpg)
        self._keep_latest(config.SNAP_DIR, "snapshot_*.jpg", 30)
        self.log("Snapshot saved")
        return path

    def _close_writer(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = self.writer_size = None
        self.stats["recording"] = False

    def toggle_record(self):
        """Returns (recording, filename). Filename is set when a recording was just finished."""
        with self.lock:
            if self.writer is not None:
                name = self.writer_path.name
                self._close_writer()
                self.log("Recording stopped")
                return False, name
            if self.frame_jpg is None:
                return False, None
            frame = cv2.imdecode(np.frombuffer(self.frame_jpg, np.uint8), cv2.IMREAD_COLOR)
            h, w = frame.shape[:2]
            config.RECORD_DIR.mkdir(parents=True, exist_ok=True)
            self.writer_path = config.RECORD_DIR / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fps = max(5, min(30, round(self._fps or 15)))
            writer = cv2.VideoWriter(str(self.writer_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not writer.isOpened():
                self.log("Recording could not start", "warn")
                return False, None
            self.writer, self.writer_size = writer, (w, h)
            self.stats["recording"] = True
            self._keep_latest(config.RECORD_DIR, "recording_*.mp4", 5)
            self.log("Recording started")
            return True, None
