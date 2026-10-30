import io
import os
import tempfile
import time
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="VisionPro | AI Detection Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(circle at 10% 0%, #17263c 0, #0b1220 38%, #080d17 100%); }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
[data-testid="stMetric"] { background: #111d2e; border: 1px solid #26364d; padding: 16px; border-radius: 14px; }
section[data-testid="stSidebar"] { background: #0d1726; border-right: 1px solid #24354c; }
div.stButton > button, div.stDownloadButton > button { border-radius: 10px; font-weight: 650; }
.hero { padding: 22px 24px; border: 1px solid #263b56; border-radius: 18px;
        background: linear-gradient(110deg, rgba(22,45,70,.95), rgba(15,25,43,.92)); }
.hero h1 { margin: 0; font-size: 2.2rem; }
.muted { color: #9eb0c9; }
.small-note { color: #9eb0c9; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return YOLO(model_name)

def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def draw_boxes(frame, result, names, show_labels=True):
    annotated = frame.copy()
    if result.boxes is None:
        return annotated, []
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        label_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        track_id = None
        if box.id is not None:
            try:
                track_id = int(box.id[0].item())
            except Exception:
                pass
        color = (40 + (cls_id * 53) % 180, 180, 240)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{label_name} {conf:.0%}" + (f" | ID {track_id}" if track_id is not None else "")
        if show_labels:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55, 2)
            top = max(0, y1 - th - 10)
            cv2.rectangle(annotated, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
            cv2.putText(annotated, label, (x1 + 4, top + th + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (8, 14, 22), 2, cv2.LINE_AA)
        rows.append({
            "class": label_name, "confidence": round(conf, 4),
            "track_id": track_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })
    return annotated, rows

def image_bytes_rgb(rgb):
    return Image.fromarray(rgb)

def process_image(upload, model, conf, imgsz, selected_classes, show_labels):
    data = np.frombuffer(upload.getvalue(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode this image. Try JPG, JPEG or PNG.")
    class_filter = selected_classes if selected_classes else None
    result = model.predict(frame, conf=conf, imgsz=imgsz, classes=class_filter, verbose=False)[0]
    annotated, rows = draw_boxes(frame, result, model.names, show_labels)
    return frame, annotated, rows

def process_video(upload, model, conf, imgsz, selected_classes, show_labels,
                  use_tracking, max_frames, progress, status):
    suffix = Path(upload.name).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(upload.getbuffer())
        input_path = tmp_in.name

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        os.unlink(input_path)
        raise ValueError("OpenCV could not open this video. Try MP4 (H.264) format.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        cap.release()
        os.unlink(input_path)
        raise ValueError("Video dimensions could not be read.")

    # Bound output size and processing time for free cloud instances.
    scale = min(1.0, 1280 / max(width, height))
    out_w, out_h = max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_out:
        output_path = tmp_out.name
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        os.unlink(input_path)
        os.unlink(output_path)
        raise RuntimeError("Could not initialize MP4 writer on this host.")

    counts = Counter()
    detections = []
    processed = 0
    seen_track_ids = set()
    class_filter = selected_classes if selected_classes else None
    started = time.time()
    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if use_tracking:
                result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                     conf=conf, imgsz=imgsz, classes=class_filter,
                                     verbose=False)[0]
            else:
                result = model.predict(frame, conf=conf, imgsz=imgsz,
                                       classes=class_filter, verbose=False)[0]
            annotated, rows = draw_boxes(frame, result, model.names, show_labels)
            writer.write(annotated)
            for row in rows:
                counts[row["class"]] += 1
                if row["track_id"] is not None:
                    seen_track_ids.add(row["track_id"])
                if len(detections) < 10000:
                    detections.append({"frame": processed + 1, "time_seconds": round(processed / fps, 2), **row})
            processed += 1
            if processed % 3 == 0 or processed == 1:
                pct = min(1.0, processed / max(1, min(total or max_frames, max_frames)))
                progress.progress(pct)
                status.caption(f"Processed {processed:,} frames · {time.time()-started:.1f}s elapsed")
    finally:
        cap.release()
        writer.release()
        try:
            os.unlink(input_path)
        except OSError:
            pass

    if processed == 0:
        os.unlink(output_path)
        raise ValueError("No readable frames found in this video.")
    progress.progress(1.0)
    return output_path, processed, fps, counts, detections, len(seen_track_ids), total

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🎯 VisionPro <span style="color:#35d6c0">AI Studio</span></h1>
  <div class="muted">Computer vision workspace · Object detection · Video analytics · ByteTrack</div>
</div>
""", unsafe_allow_html=True)
st.write("")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.markdown("## ⚙️ Detection settings")
    model_choice = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                                help="Nano is fastest and most suitable for free cloud hosting.")
    confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.40, 0.05)
    image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768], value=640)
    show_labels = st.toggle("Show class labels", True)
    tracking = st.toggle("Enable ByteTrack for videos", True)
    st.divider()
    st.markdown("### 🧩 Class filter")
    st.caption("Leave empty to detect all classes.")
    st.markdown("### ☁️ Runtime")
    st.caption("CPU mode is used by default. Processing time depends on video length and hosting resources.")

# ---------- Load model lazily ----------
if "model_obj" not in st.session_state or st.session_state.get("model_name") != model_choice:
    try:
        with st.spinner(f"Loading {model_choice} (first run downloads model weights)..."):
            st.session_state.model_obj = load_model(model_choice)
            st.session_state.model_name = model_choice
            st.session_state.class_names = st.session_state.model_obj.names
    except Exception as exc:
        st.error(f"Model could not be loaded: {exc}")
        st.info("Check that the host can download model weights and that ultralytics is installed.")
        st.stop()

model = st.session_state.model_obj
class_names = model.names
class_options = [f"{i}: {name}" for i, name in class_names.items()]
selected_labels = st.sidebar.multiselect("Classes to detect", class_options, default=[])
selected_classes = [int(x.split(":", 1)[0]) for x in selected_labels]

# ---------- Input tabs ----------
tab_image, tab_video, tab_camera, tab_about = st.tabs(
    ["🖼️ Image detection", "🎞️ Video analytics", "📷 Camera snapshot", "ℹ️ About"]
)

with tab_image:
    st.subheader("Image object detection")
    st.caption("Upload an image, run YOLO inference, inspect detections and download the annotated result.")
    image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
    if image_file:
        if st.button("🚀 Detect objects in image", type="primary", key="run_image"):
            try:
                with st.spinner("Running object detection..."):
                    original, annotated, rows = process_image(
                        image_file, model, confidence, image_size, selected_classes, show_labels
                    )
                st.session_state.image_result = {
                    "original": bgr_to_rgb(original), "annotated": bgr_to_rgb(annotated), "rows": rows
                }
            except Exception as exc:
                st.error(f"Image detection failed: {exc}")
    if "image_result" in st.session_state:
        result = st.session_state.image_result
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original image**")
            st.image(result["original"], use_container_width=True)
        with c2:
            st.markdown("**AI detection result**")
            st.image(result["annotated"], use_container_width=True)
        rows = result["rows"]
        m1, m2 = st.columns(2)
        m1.metric("Objects detected", len(rows))
        m2.metric("Unique classes", len({r["class"] for r in rows}))
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No objects detected above the selected confidence threshold.")
        buf = io.BytesIO()
        Image.fromarray(result["annotated"]).save(buf, format="PNG")
        st.download_button("⬇️ Download annotated image", buf.getvalue(),
                           "visionpro_detection.png", "image/png")

with tab_video:
    st.subheader("Video object detection & tracking")
    st.caption("Upload a video. The app processes it on the server and creates an annotated MP4 plus a CSV detection report.")
    video_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv", "webm"], key="video_upload")
    max_frames = st.number_input("Maximum frames to process", min_value=30, max_value=3000,
                                 value=600, step=30,
                                 help="Limit processing to keep free hosting responsive.")
    if video_file:
        st.caption(f"Selected: {video_file.name} · {video_file.size / (1024*1024):.1f} MB")
        if st.button("▶️ Process video", type="primary", key="run_video"):
            progress = st.progress(0)
            status = st.empty()
            try:
                with st.spinner("Processing video with YOLO..."):
                    output_path, processed, fps, counts, detections, unique_ids, total = process_video(
                        video_file, model, confidence, image_size, selected_classes,
                        show_labels, tracking, int(max_frames), progress, status
                    )
                st.session_state.video_result = {
                    "path": output_path, "processed": processed, "fps": fps,
                    "counts": dict(counts), "detections": detections,
                    "unique_ids": unique_ids, "total": total,
                }
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
    if "video_result" in st.session_state:
        vr = st.session_state.video_result
        a, b, c, d = st.columns(4)
        a.metric("Frames processed", f"{vr['processed']:,}")
        b.metric("Detection records", f"{len(vr['detections']):,}")
        c.metric("Unique tracked IDs", f"{vr['unique_ids']:,}")
        d.metric("Input FPS", f"{vr['fps']:.1f}")
        if vr["total"] > vr["processed"]:
            st.warning(f"Processed {vr['processed']} of {vr['total']} frames due to the frame limit.")
        st.markdown("#### Annotated output")
        if os.path.exists(vr["path"]):
            with open(vr["path"], "rb") as f:
                video_data = f.read()
            st.video(video_data)
            st.download_button("⬇️ Download annotated MP4", video_data,
                               "visionpro_annotated.mp4", "video/mp4")
        if vr["counts"]:
            count_df = pd.DataFrame([{"Class": k, "Detection count across frames": v}
                                     for k, v in sorted(vr["counts"].items(), key=lambda x: -x[1])])
            st.markdown("#### Class frequency")
            st.bar_chart(count_df.set_index("Class"))
        if vr["detections"]:
            df = pd.DataFrame(vr["detections"])
            st.markdown("#### Detection report")
            st.dataframe(df.head(500), use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download detection CSV", df.to_csv(index=False).encode("utf-8"),
                               "visionpro_detections.csv", "text/csv")

with tab_camera:
    st.subheader("Camera snapshot detection")
    st.caption("Take a still photo with your browser camera, then run object detection. This is not continuous live video.")
    camera_image = st.camera_input("Take a photo")
    if camera_image and st.button("🔎 Detect in camera photo", type="primary"):
        try:
            with st.spinner("Analyzing camera snapshot..."):
                original, annotated, rows = process_image(
                    camera_image, model, confidence, image_size, selected_classes, show_labels
                )
            st.image(bgr_to_rgb(annotated), caption=f"{len(rows)} objects detected", use_container_width=True)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            out = io.BytesIO()
            Image.fromarray(bgr_to_rgb(annotated)).save(out, format="PNG")
            st.download_button("⬇️ Download camera result", out.getvalue(),
                               "visionpro_camera_detection.png", "image/png")
        except Exception as exc:
            st.error(f"Camera detection failed: {exc}")

with tab_about:
    st.subheader("About VisionPro")
    st.write("VisionPro is a practical computer-vision demo for image inference, uploaded-video processing, multi-object tracking and downloadable reports.")
    st.markdown("""
    **Included**
    - YOLO11 object detection for images and videos
    - Confidence threshold and class filtering
    - ByteTrack tracking for uploaded videos
    - Annotated MP4 output and CSV detection report
    - Browser camera snapshot analysis
    - Downloadable annotated images
    """)
    st.warning("Cloud hosting does not provide direct access to your laptop's webcam or a server-side camera index. Use the Camera snapshot tab for browser camera permission, or upload a video.")
    st.caption("Owner: Ritesh Kumar Singh · VisionPro")
