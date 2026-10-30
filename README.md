# VisionPro - Real-Time AI Object Detection Studio
Built by **Ritesh Kumar Singh** (Founder & CEO)

## Run
```
python -m venv venv && venv\Scripts\activate      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 (model weights download automatically on first run).

## Login
Default: username `ritesh`, password `vision@123`. Change it before showing/deploying:
`ADMIN_USER=you ADMIN_PASS=strong SECRET_KEY=long-random python app.py`

## Docker
```
docker compose up --build
```
Open http://localhost:5000. Webcams do not pass into Docker on Windows/Mac, so use a video file
(put it in `videos/`, set source to `videos/demo.mp4`) or an RTSP URL. On Linux, uncomment the `devices` line.

## Features
- Secure login (hashed passwords, session cookies)
- SQLite history of every detection event and per-5s stats, with a History dashboard
- CSV export of events and stats
- YOLO11 detection + ByteTrack multi-object tracking (persistent IDs)
- Line crossing counter (IN / OUT), unique object count
- Motion trails, live heatmap, class filter, confidence slider
- Switch models live (nano / small / medium), webcam or video file / RTSP URL source
- Live FPS, latency, chart, class breakdown, event log
- Snapshot + video recording, pause, keyboard shortcuts (Space pause, S snapshot, R record)

## Interview talking points
Threaded capture/inference pipeline, MJPEG streaming over Flask, REST config API,
tracker-based counting (not per-frame counting), exportable to ONNX/TensorRT for edge deploy.
