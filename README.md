# OCI Vision Video Analysis

An interactive **Streamlit app** to explore [OCI Vision Video Analysis](https://www.oracle.com/in/artificial-intelligence/vision/) results.  
It visualizes **detections, bounding boxes, OCR text, and labels** at the **frame level** with an easy UI.

---

## Features
- Upload a **video** and its **OCI Vision Video Job JSON**
- Frame-by-frame annotated viewer
- **Stepper**: jump to next/previous detection, label, or specific class
- **Context Peek**: preview upcoming detection frames
- Export **detections as CSV** and **annotated frames as PNG**
- Side-by-side layout:
  - Raw video
  - Annotated frame
  - All labels & detections
- Filter by object classes or OCR tokens
- Confidence threshold slider
- Timeline chart for labels (Altair)

---

## Requirements

Make sure you have **Python 3.9+** installed.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
.
├── app.py             # Main Streamlit app
├── requirements.txt   # Dependencies
└── README.md          # Project info
```

---

## Usage

1. Start the app:
   ```bash
   streamlit run app.py
   ```
2. Open the browser link Streamlit provides (default: `http://localhost:8501`).
3. Upload:
   - A video file (`.mp4`, `.mov`, `.mkv`, `.avi`)  
   - The corresponding **OCI Vision video job JSON**
4. Use the **frame slider** or **stepper buttons** to navigate detections.
5. Export annotated frames (PNG) or detections (CSV).

---

## Example Workflow
- Upload `video.mp4` and its `video_job.json`
- Use the **confidence slider** to adjust detection threshold
- Filter by object class (`Human face`, `Cake`, etc.)
- Click **Next Detection** to jump to the next frame containing bounding boxes
- Use **Context Peek** thumbnails to preview upcoming detections
- Save annotated frames or download detection tables

---

## Notes
- OpenCV (`opencv-python`) is optional. Without it, frames will be blank canvases with overlays only.
- Altair is optional for the timeline chart.
- Tested on **Python 3.9–3.11** with the pinned dependencies in `requirements.txt`.

---
