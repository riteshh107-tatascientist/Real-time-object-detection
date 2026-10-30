# VisionPro — Streamlit AI Detection Studio

A Streamlit implementation of VisionPro for practical, testable computer-vision workflows.

## Features
- YOLO11 object detection on JPG/JPEG/PNG/WEBP images
- Uploaded video processing with YOLO11
- Optional ByteTrack multi-object tracking
- Confidence threshold and COCO class filters
- Annotated image and MP4 downloads
- CSV detection report and class-frequency chart
- Browser camera **snapshot** detection using `st.camera_input`

## Run locally (Windows PowerShell)

```powershell
cd VisionPro_Streamlit
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

The first model load downloads YOLO11 weights (`yolo11n.pt` by default), so internet access is required the first time.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository, e.g. `visionpro-streamlit`.
2. Upload the contents of this folder to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Click **Create app**, select the repository and branch `main`.
5. Set **Main file path** to `app.py`.
6. Deploy.

Streamlit Cloud installs dependencies from `requirements.txt` and system packages from `packages.txt`.

## Resource guidance
- Start with `yolo11n.pt` (Nano) and 320/416/512 image size on free CPU hosting.
- Use short videos and set a reasonable maximum frame count.
- Larger models and long videos may exceed free CPU/RAM/time limits.
- The app processes uploaded videos synchronously during the request; very long videos are not suitable for free hosting.
- Uploaded files and generated outputs are temporary on hosted instances. Download outputs you need to keep.

## Camera limitation
`st.camera_input` captures a browser camera still image after the visitor grants permission. It is not continuous live webcam streaming. A cloud server's OpenCV camera index `0` refers to a camera attached to the server, not the visitor's laptop.

## Troubleshooting
- If model download fails, check that the host has outbound internet access.
- If the app runs out of memory, choose `yolo11n.pt`, reduce image size, and process fewer frames.
- If a video cannot be decoded, convert it to MP4 (H.264) and upload again.
- If the app is slow on Streamlit Cloud free CPU, use a smaller video and lower image size.
