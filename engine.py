"""VisionPro detection engine: YOLO + ByteTrack, line counting, trails, heatmap, recording."""
import os, time, threading
from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO

os.makedirs("snapshots", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

PALETTE = [(255, 170, 0), (0, 220, 150), (255, 90, 120), (120, 140, 255),
           (0, 200, 255), (200, 120, 255), (90, 255, 120), (255, 210, 80)]


class Engine:
    def __init__(self):
        self.cfg = dict(model="yolo11n.pt", conf=0.40, source="0", classes=[],
                        trails=True, line=True, heatmap=False, paused=False)
        self.model, self.model_name = None, None
        self.frame_jpg = None
        self.raw = None
        self.stats = dict(fps=0.0, latency=0.0, objects=0, per_class={}, in_count=0,
                          out_count=0, unique=0, source="0", status="starting", recording=False)
        self.history = deque(maxlen=60)
        self.events = deque(maxlen=40)
        self.trails = defaultdict(lambda: deque(maxlen=30))
        self.side, self.seen = {}, set()
        self.heat = None
        self.writer = None
        self.reload_src = True
        self.lock = threading.Lock()
        self.on_event = None
        threading.Thread(target=self.loop, daemon=True).start()

    # ---------- config ----------
    def set_config(self, new):
        with self.lock:
            for k, v in new.items():
                if k in self.cfg:
                    if k == "source" and str(v) != str(self.cfg[k]):
                        self.reload_src = True
                    self.cfg[k] = v
            if new.get("reset"):
                self.stats["in_count"] = self.stats["out_count"] = 0
                self.seen.clear(); self.events.clear(); self.side.clear()
                self.heat = None

    def names(self):
        if self.model is None:
            self.model = YOLO(self.cfg["model"]); self.model_name = self.cfg["model"]
        return list(self.model.names.values())

    # ---------- recording / snapshot ----------
    def snapshot(self):
        if self.raw is None:
            return None
        path = f"snapshots/snap_{int(time.time())}.jpg"
        cv2.imwrite(path, self.raw)
        self.log("Snapshot saved", "info")
        return path

    def toggle_record(self):
        if self.writer:
            self.writer.release(); self.writer = None
            self.stats["recording"] = False; self.log("Recording stopped", "info")
        elif self.raw is not None:
            h, w = self.raw.shape[:2]
            path = f"recordings/rec_{int(time.time())}.mp4"
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            self.stats["recording"] = True; self.log("Recording started", "info")
        return self.stats["recording"]

    def log(self, msg, kind="info"):
        self.events.appendleft(dict(t=time.strftime("%H:%M:%S"), msg=msg, kind=kind))
        if self.on_event:
            try: self.on_event(msg, kind)
            except Exception: pass

    # ---------- main loop ----------
    def open_source(self):
        src = str(self.cfg["source"]).strip()
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        self.stats["source"] = src
        self.stats["status"] = "live" if cap.isOpened() else "no source"
        self.log(f"Source: {src}" if cap.isOpened() else f"Cannot open source '{src}'",
                 "info" if cap.isOpened() else "warn")
        return cap

    def loop(self):
        cap, t_prev, fps = None, time.time(), 0.0
        while True:
            try:
                if self.cfg["model"] != self.model_name:
                    self.stats["status"] = "loading model"
                    self.model = YOLO(self.cfg["model"]); self.model_name = self.cfg["model"]
                    self.log(f"Model loaded: {self.model_name}", "info")
                if self.reload_src or cap is None:
                    if cap: cap.release()
                    self.reload_src = False
                    cap = self.open_source()
                if self.cfg["paused"]:
                    self.stats["status"] = "paused"; time.sleep(0.1); continue
                ok, frame = cap.read()
                if not ok:
                    if str(self.cfg["source"]).isdigit():
                        self.stats["status"] = "no source"; time.sleep(0.5)
                        self.reload_src = True
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video files
                    continue
                self.stats["status"] = "live"
                frame = cv2.resize(frame, (960, int(frame.shape[0] * 960 / frame.shape[1])))
                self.raw = frame.copy()
                t0 = time.time()
                out = self.process(frame)
                self.stats["latency"] = round((time.time() - t0) * 1000, 1)
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-3)); t_prev = now
                self.stats["fps"] = round(fps, 1)
                self.hud(out)
                if self.writer: self.writer.write(out)
                self.frame_jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
            except Exception as e:
                self.stats["status"] = "error"; self.log(f"Error: {e}", "warn"); time.sleep(1)

    def process(self, frame):
        cfg = self.cfg
        h, w = frame.shape[:2]
        r = self.model.track(frame, persist=True, conf=cfg["conf"], verbose=False,
                             classes=cfg["classes"] or None, tracker="bytetrack.yaml")[0]
        line_y = int(h * 0.55)
        if self.heat is None or self.heat.shape[:2] != (h, w):
            self.heat = np.zeros((h, w), np.float32)
        self.heat *= 0.985
        per_class, n = defaultdict(int), 0
        if r.boxes is not None and len(r.boxes):
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for box, tid in zip(r.boxes, ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls, conf = int(box.cls[0]), float(box.conf[0])
                name = self.model.names[cls]
                col = PALETTE[cls % len(PALETTE)]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                per_class[name] += 1; n += 1
                cv2.circle(self.heat, (cx, cy), 28, 1.0, -1)
                if tid is not None:
                    if tid not in self.seen:
                        self.seen.add(tid); self.log(f"New {name} #{tid}", "new")
                    tr = self.trails[tid]; tr.append((cx, cy))
                    if cfg["trails"] and len(tr) > 1:
                        for i in range(1, len(tr)):
                            cv2.line(frame, tr[i - 1], tr[i], col, max(1, i // 8))
                    side = cy > line_y
                    if cfg["line"] and tid in self.side and self.side[tid] != side:
                        if side: self.stats["in_count"] += 1
                        else: self.stats["out_count"] += 1
                        self.log(f"{name} #{tid} crossed {'IN' if side else 'OUT'}", "in" if side else "out")
                    self.side[tid] = side
                self.draw_box(frame, x1, y1, x2, y2, col,
                              f"{name}{' #'+str(tid) if tid is not None else ''} {conf:.0%}")
        if cfg["heatmap"]:
            hm = cv2.applyColorMap(np.clip(self.heat * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            mask = (self.heat > 0.05)[..., None]
            frame[:] = np.where(mask, cv2.addWeighted(frame, 0.55, hm, 0.45, 0), frame)
        if cfg["line"]:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 220, 150), 2)
            cv2.putText(frame, "COUNT LINE", (10, line_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 150), 1)
        self.stats.update(objects=n, per_class=dict(per_class), unique=len(self.seen))
        self.history.append(n)
        return frame

    @staticmethod
    def draw_box(img, x1, y1, x2, y2, col, label):
        L = max(12, min(x2 - x1, y2 - y1) // 4)
        for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(img, (px, py), (px + dx * L, py), col, 3)
            cv2.line(img, (px, py), (px, py + dy * L), col, 3)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), col, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 20), 1, cv2.LINE_AA)

    def hud(self, img):
        cv2.putText(img, "VisionPro | Ritesh Kumar Singh", (12, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
