# VisionPro - AI Vision Studio (Streamlit)
Real-time object detection, tracking and counting with **YOLO11 + ByteTrack**.
Built by **Ritesh Kumar Singh**, Founder & CEO.

## Features
- Live browser webcam detection (WebRTC), snapshot and recording
- Video studio: upload a video, get an annotated H.264 video, IN/OUT line counts and a class summary
- Photo detect: camera photo or image upload
- ByteTrack persistent IDs, IN/OUT counting line, motion trails, heatmap, class filter, confidence slider
- Secure login (PBKDF2-hashed password, no default credentials, login lockout)
- SQLite history (auto-created) with charts and CSV export
- Dark UI with animated branding

## Project layout
```
streamlit_app.py   UI + pages            engine.py   YOLO11 + ByteTrack, counting, recording
db.py              SQLite + auth         config.py   all settings from environment variables
coco.py            class names           tests/smoke_test.py   real-model tests
requirements.txt   pinned dependencies   packages.txt          system libs (Streamlit Cloud)
Dockerfile, render.yaml                  .streamlit/config.toml  theme + upload limit
```

## Environment variables
| Name | Required | Meaning |
|---|---|---|
| `ADMIN_USERNAME` | yes | Login username |
| `ADMIN_PASSWORD` | yes | Login password, 8+ characters. Changing it and restarting updates the account |
| `PROCESS_WIDTH` / `YOLO_IMGSZ` | no | Frame width (default 960) / model input size (640). Use 640 / 480 on small servers |
| `ALLOWED_MODELS` | no | Default `yolo11n.pt,yolo11s.pt,yolo11m.pt` |
| `MAX_VIDEO_FRAMES` | no | Frames processed per uploaded video (900) |
| `DATA_DIR`, `DB_PATH` | no | Where the SQLite file, snapshots, recordings live (`./data`) |
| `PERSISTENT_STORAGE` | no | `true` only if `DATA_DIR` is on a persistent disk (hides the "history is temporary" warning) |
| `RETENTION_DAYS` | no | History older than this is deleted (30) |
| `TURN_URL`, `TURN_USERNAME`, `TURN_CREDENTIAL` | no | Optional TURN relay so live webcam works on strict networks |

Upload size limit is set in `.streamlit/config.toml` (`maxUploadSize`, default 50 MB).

## Run locally on Windows
Use **Python 3.12** (python.org). In PowerShell inside the project folder:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env          # set ADMIN_USERNAME and ADMIN_PASSWORD
streamlit run streamlit_app.py
```
Open http://localhost:8501. The model downloads on first use. The webcam works on `localhost`
because browsers treat it as secure. If `pip` cannot find `torch==2.14.1`, run
`pip install torch torchvision` first, then the rest.
Run the tests: `python tests/smoke_test.py`

## Upload to GitHub
1. Create an empty repository on github.com (do not tick "add README").
2. In the project folder:
```
git init
git add .
git commit -m "VisionPro"
git branch -M main
git remote add origin https://github.com/YOUR_USER/visionpro.git
git push -u origin main
```
3. Check on GitHub that `.env` and `.streamlit/secrets.toml` are **not** there (they are git-ignored).
   Passwords go into the hosting dashboard only.

## Deploy option A: Streamlit Community Cloud (free, easiest)
1. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
2. Pick your repo, branch `main`, main file `streamlit_app.py`.
3. **Advanced settings**: Python version **3.12**; paste the contents of `.streamlit/secrets.toml.example`
   into **Secrets** with your own username and password.
4. Deploy. The first build takes several minutes (PyTorch). Open the app URL and sign in.

## Deploy option B: Render with Docker
1. Push the repo to GitHub. In Render: **New > Web Service**, connect the repo, runtime **Docker**
   (or **New > Blueprint** to use `render.yaml`).
2. Add environment variables `ADMIN_USERNAME` and `ADMIN_PASSWORD` (plus the optional ones from `render.yaml`).
3. Health check path: `/_stcore/health`. Deploy and open the URL.
4. If uploads fail with a 403 error behind Render's proxy, add `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false`.

Hugging Face Spaces (Docker SDK) also works with the same Dockerfile and gives more free RAM than Render's free plan.

## Known limitations of free hosting
- **Memory:** YOLO + PyTorch needs roughly 0.5-1 GB. Render free (512 MB) will likely run out of memory; use a paid
  instance or Streamlit Cloud / Hugging Face. Each browser session loads its own model copy, so keep demos to 1-2 viewers.
- **Speed:** free hosts have 1-2 slow CPU cores, so expect only a few FPS. Use `yolo11n.pt`, `PROCESS_WIDTH=640`, `YOLO_IMGSZ=480`.
- **Storage is temporary:** history, snapshots and recordings are lost on restart or redeploy unless you attach a persistent disk
  (paid on Render) and set `DATA_DIR` there. Download the CSV before you leave.
- **Sleeping:** free apps sleep when idle and need a cold start (model load) on the next visit.
- **Live webcam** uses WebRTC. It needs HTTPS (hosts provide it) and may fail on strict corporate/mobile networks
  without a TURN server. Photo detect and Video studio always work.
- **Line counting** can double-count if the tracker loses an object and assigns a new ID (heavy video compression, occlusion).
- Streamlit reruns the script on every click, so changing a setting while a video is processing stops that run.
- Single shared admin account; no user management.

## Tested vs not tested
See the summary delivered with this project. Tests that need a real browser, Docker or the cloud hosts are listed there honestly.
