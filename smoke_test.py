"""Engine + database smoke test with the REAL YOLO11 model (no mocks).
Run:  python tests/smoke_test.py      (needs internet once to fetch yolo11n.pt and a sample image)
"""
import os, sys, tempfile, time, urllib.request
from pathlib import Path

tmp = tempfile.mkdtemp(prefix="vp_test_")
os.environ.update(DATA_DIR=tmp, ADMIN_USERNAME="tester", ADMIN_PASSWORD="test-pass-123", PROCESS_WIDTH="640")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2, numpy as np
import config, db
from engine import Engine, SourceError, validate_source

results = []
def check(name, cond, extra=""):
    results.append(cond); print(("PASS" if cond else "FAIL"), "-", name, extra)

# ---- sample image (person-rich) ----
img_path = Path(tmp) / "bus.jpg"
for url in ("https://ultralytics.com/images/bus.jpg",
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"):
    try:
        urllib.request.urlretrieve(url, img_path); break
    except Exception:
        pass
bus = cv2.imread(str(img_path))
check("sample image downloaded", bus is not None)

db.init(); db.ensure_admin("tester", "test-pass-123"); db.start_writer()
eng = Engine(); eng.on_event = db.add_event

# ---- detection ----
eng.step(bus)
check("YOLO11 detects objects", eng.stats["objects"] >= 3, f"({eng.stats['per_class']})")
check("detects a bus and persons", "bus" in eng.stats["per_class"] and "person" in eng.stats["per_class"])
check("annotated frame + jpeg produced", eng.last_frame is not None and eng.frame_jpg[:2] == b"\xff\xd8")

# ---- class filter / confidence validation ----
check("rejects unknown model", eng.set_config({"model": "../../etc/passwd"}) is not None)
check("rejects bad class id", eng.set_config({"classes": [999]}) is not None)
check("accepts valid class filter", eng.set_config({"classes": [5]}) is None)
eng.new_scene(); eng.step(bus)
check("class filter works (bus only)", set(eng.stats["per_class"]) == {"bus"}, f"({eng.stats['per_class']})")
eng.set_config({"classes": [0]})

# ---- tracking + IN/OUT counting on a synthetic video ----
res = eng.model(bus, verbose=False, classes=[0])[0]
boxes = sorted(res.boxes.xyxy.tolist(), key=lambda b: -(b[2]-b[0])*(b[3]-b[1]))
x1, y1, x2, y2 = map(int, boxes[0]); person = bus[y1:y2, x1:x2]
person = cv2.resize(person, (120, int(120 * person.shape[0] / person.shape[1])))
W, H = 640, 480
frames_list = []
n_frames = 36
for i in range(n_frames):                               # person walks top -> bottom across the 55% line
    f = np.full((H, W, 3), 110, np.uint8)
    top = int(10 + i * ((H - person.shape[0] - 20) / (n_frames - 1)))
    f[top:top + person.shape[0], 260:380] = person
    frames_list.append(f)
vid = Path(tmp) / "walk.mp4"
vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
for f in frames_list: vw.write(f)
vw.release()

eng.set_config({"reset": True, "line": True, "conf": 0.3})
for f in frames_list: eng.step(f)
check("tracking: exactly 1 unique object (raw frames)", eng.stats["unique"] == 1, f"(unique={eng.stats['unique']})")
check("counting: exactly 1 IN, 0 OUT (raw frames)", (eng.stats["in_count"], eng.stats["out_count"]) == (1, 0),
      f"(in={eng.stats['in_count']}, out={eng.stats['out_count']})")
check("per-class totals recorded", eng.stats["totals"].get("person", 0) == 1)

eng.set_config({"reset": True})                         # same walk, but through a lossy MP4 file
cap = cv2.VideoCapture(str(vid)); frames = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    eng.step(fr); frames += 1
cap.release()
check("MP4 file processed frame by frame", frames == n_frames, f"({frames} frames)")
check("MP4 file: crossing detected", eng.stats["in_count"] >= 1,
      f"(unique={eng.stats['unique']}, in={eng.stats['in_count']}; compression can split a track)")

# ---- snapshot + recording ----
snap = eng.snapshot()
check("snapshot saved", snap is not None and snap.exists() and snap.stat().st_size > 1000)
started, _ = eng.toggle_record()
for _ in range(5): eng.step(bus)
stopped, name = eng.toggle_record()
rec = config.RECORD_DIR / (name or "x")
check("recording written", started and not stopped and rec.exists() and rec.stat().st_size > 1000)
rc = cv2.VideoCapture(str(rec)); ok, _ = rc.read(); rc.release()
check("recording readable by OpenCV", ok)

# ---- source validation ----
for bad in ("/etc/passwd", "../../etc/passwd", "file:///etc/passwd", "ftp://x/y"):
    try: validate_source(bad); check(f"blocks source {bad}", False)
    except SourceError: check(f"blocks source {bad}", True)
os.environ["X"] = "1"
try: validate_source("http://127.0.0.1/stream"); check("blocks private URL on cloud", config.ALLOW_PRIVATE_URLS)
except SourceError: check("blocks private URL on cloud", True)
check("accepts public-style rtsp URL syntax", validate_source("rtsp://203.0.113.5/live") .startswith("rtsp://"))

# ---- database ----
time.sleep(1.5)
db._flush()
ev = db.events("", 500)
check("events persisted in SQLite", len(ev) > 0, f"({len(ev)} rows)")
check("IN events persisted", any(e["kind"] == "in" for e in ev))
db.add_sample(eng.stats); db._flush()
check("samples persisted", len(db.samples(60)) == 1)
check("admin login works", db.check_user("tester", "test-pass-123") is not None)
check("wrong password rejected", db.check_user("tester", "nope") is None)
db.ensure_admin("tester", "new-pass-456")
check("password change via env applied", db.check_user("tester", "new-pass-456") is not None and db.check_user("tester", "test-pass-123") is None)
os.remove(config.DB_PATH); db.init()
check("DB auto-recreated if file missing", db.summary()["new"] == 0)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
