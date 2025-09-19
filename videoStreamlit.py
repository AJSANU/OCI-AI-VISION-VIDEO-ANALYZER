import io
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

st.set_page_config(page_title="OCI Vision Video Analysis", layout="wide", page_icon="🎥")

st.markdown("""
<div style="
    background: linear-gradient(90deg, #ff9a9e, #ff6a00);
    padding: 1.2rem 1.5rem;
    border-radius: 14px;
    margin-bottom: 1rem;
">
  <h1 style="color:white;margin:0;">OCI Vision Video Analysis</h1>
  <p style="color:#fff8f0;margin:0;">Frame-level detections • OCR • Labels • Export • Stepper</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
:root { --pill: 9999px; }
.stTabs [data-baseweb="tab-list"] { gap: .5rem; }
.stTabs [data-baseweb="tab"] { background: #111; color: #bbb; border-radius: 12px; padding: .35rem .8rem; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg,#6EE7F9,#A7F3D0); color:#0b1220; }
.badge{display:inline-flex;align-items:center;font-weight:600;border-radius:12px;padding:.15rem .55rem;margin:.12rem .12rem 0 0;font-size:.78rem}
.badge.obj{background:#1f6b3a20;border:1px solid #29c77150;color:#a5f3c7}
.badge.txt{background:#1f3f6b20;border:1px solid #5ab7ff50;color:#cde8ff}
.badge.lab{background:#6b1f6220;border:1px solid #ff7bf150;color:#ffd3f8}
.card{background:#0b1220;border:1px solid #22304a;border-radius:16px;padding:14px}
.header{position:sticky;top:0;z-index:50;background:linear-gradient(180deg,#0b1220 70%,transparent);padding:.3rem .2rem .6rem .2rem;border-bottom:1px solid #1d2940}
.kbd{background:#0f172a;border:1px solid #334155;color:#e2e8f0;border-radius:6px;padding:.05rem .35rem;font-size:.8rem}
hr{border:none;border-top:1px solid #1d2940;margin:.6rem 0;}
</style>
""", unsafe_allow_html=True)

@dataclass
class Meta:
    frame_count: int
    fps: float
    w: int
    h: int

@dataclass
class Box:
    cls: str
    conf: float
    xyxy: Tuple[int, int, int, int]
    source: str

def safe_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def norm2pix(vertices: List[Dict[str, float]], W: int, H: int) -> Tuple[int, int, int, int]:
    xs = [max(0.0, min(1.0, v.get("x", 0.0))) for v in vertices]
    ys = [max(0.0, min(1.0, v.get("y", 0.0))) for v in vertices]
    return int(min(xs) * W), int(min(ys) * H), int(max(xs) * W), int(max(ys) * H)

def ms_to_frame(ms: int, fps: float) -> int:
    return max(0, int(round((ms / 1000.0) * fps)))

def frame_to_ms(frame_idx: int, fps: float) -> int:
    return int(round((frame_idx / max(1e-9, fps)) * 1000))

def parse_video_job_json(blob: dict):
    vm = blob.get("videoMetadata", {})
    meta = Meta(
        int(vm.get("frameCount", 0)),
        float(vm.get("frameRate", 25.0)),
        int(vm.get("frameWidth", 1280)),
        int(vm.get("frameHeight", 720)),
    )
    objects_by_frame: Dict[int, List[Box]] = {}
    for obj in blob.get("videoObjects", []):
        name = obj.get("name", "Object")
        for seg in obj.get("segments", []):
            seg_conf = float(seg.get("confidence", seg.get("segmentScore", 0.0)))
            for fr in seg.get("frames", []):
                fid = ms_to_frame(int(fr.get("timeOffsetMs", 0)), meta.fps)
                conf = float(fr.get("confidence", seg_conf))
                verts = fr.get("boundingPolygon", {}).get("normalizedVertices", [])
                xyxy = norm2pix(verts, meta.w, meta.h) if verts else (0, 0, 0, 0)
                objects_by_frame.setdefault(fid, []).append(Box(name, conf, xyxy, "object"))
    texts_by_frame: Dict[int, List[Box]] = {}
    for tx in blob.get("videoText", []):
        text_str = tx.get("text", "").strip()
        for seg in tx.get("segments", []):
            seg_conf = float(seg.get("confidence", 0.0))
            for fr in seg.get("frames", []):
                fid = ms_to_frame(int(fr.get("timeOffsetMs", 0)), meta.fps)
                conf = float(fr.get("confidence", seg_conf))
                verts = fr.get("boundingPolygon", {}).get("normalizedVertices", [])
                xyxy = norm2pix(verts, meta.w, meta.h) if verts else (0, 0, 0, 0)
                texts_by_frame.setdefault(fid, []).append(Box(text_str, conf, xyxy, "text"))
    labels_by_frame: Dict[int, List[Tuple[str, float]]] = {}
    for lab in blob.get("videoLabels", []):
        lname = lab.get("name", "Label")
        for seg in lab.get("segments", []):
            conf = float(seg.get("confidence", 0.0))
            vs = seg.get("videoSegment", {})
            for tkey in ("startTimeOffsetMs", "endTimeOffsetMs"):
                if tkey in vs:
                    fid = ms_to_frame(int(vs[tkey]), meta.fps)
                    labels_by_frame.setdefault(fid, []).append((lname, conf))
    return meta, objects_by_frame, texts_by_frame, labels_by_frame

PALETTE = {"object": (41, 199, 113), "text": (90, 183, 255)}

def draw_boxes(img: Image.Image, boxes: List[Box], min_conf: float, show_tags: bool, allowed: Optional[set]):
    out = img.copy()
    drw = ImageDraw.Draw(out, "RGBA")
    font = safe_font(14)
    for b in boxes:
        if b.conf < min_conf:
            continue
        if allowed and b.cls not in allowed:
            continue
        x1, y1, x2, y2 = b.xyxy
        col = PALETTE.get(b.source, (255, 215, 0))
        drw.rectangle([x1, y1, x2, y2], outline=col + (255,), width=3)
        if show_tags:
            tag = f"{b.cls} {b.conf:.2f}"
            tw, th = drw.textbbox((0, 0), tag, font=font)[2:]
            pad = 4
            drw.rectangle([x1, max(0, y1 - (th + 2 * pad)), x1 + tw + 2 * pad, y1], fill=col + (150,))
            drw.text((x1 + pad, y1 - th - pad), tag, fill=(0, 0, 0, 255), font=font)
    return out

@st.cache_resource(show_spinner=False)
def open_video(path: str):
    if not HAS_CV2:
        return None
    return cv2.VideoCapture(path)

def read_frame(cap, frame_idx: int, fallback_size: Tuple[int, int]) -> Image.Image:
    if cap is None:
        return Image.new("RGB", fallback_size, (8, 10, 18))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or fallback_size[0]
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or fallback_size[1]
        return Image.new("RGB", (W, H), (8, 10, 18))
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

st.markdown('<div class="header">', unsafe_allow_html=True)
cA, cB, cC, cD = st.columns([2.2, 2.2, 2.2, 1.4])
with cA:
    up_video = st.file_uploader("Video (optional)", type=["mp4", "mov", "mkv", "avi"], label_visibility="collapsed")
    if up_video:
        vid_path = f"/tmp/{up_video.name}"
        with open(vid_path, "wb") as f:
            f.write(up_video.getbuffer())
    else:
        vid_path = None
with cB:
    up_json = st.file_uploader("Video Job JSON", type=["json"], label_visibility="collapsed")
with cC:
    sample = st.selectbox("Samples", ["None", "Sample A (face)", "Sample B (cake)"])
with cD:
    min_conf = st.slider("Min conf", 0.0, 1.0, 0.45, 0.01)
st.markdown('</div>', unsafe_allow_html=True)

json_data = None
if up_json is not None:
    json_data = json.load(io.StringIO(up_json.getvalue().decode("utf-8")))
elif sample != "None":
    try:
        path = {
            "Sample A (face)": "/mnt/data/ocid1.aivisionvideojob.oc1.phx.amaaaaaarof4xcqa4kmlcvmspj2etiyoa2b6zfwynju4fz4cv5niuxtqm72q.json",
            "Sample B (cake)": "/mnt/data/ocid1.aivisionvideojob.oc1.phx.amaaaaaarof4xcqahuz76nq6hwc7upodfss5yaoiliaeoyfeq4grb6req52q.json",
        }[sample]
        with open(path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        st.error(f"Could not load sample: {e}")

if not json_data:
    st.info("Upload a **Video Job JSON** or pick a sample to start.")
    st.stop()

meta, obj_by_frame, txt_by_frame, lab_by_frame = parse_video_job_json(json_data)

det_frames = sorted(set(obj_by_frame.keys()) | set(txt_by_frame.keys()))
lab_frames = sorted(set(lab_by_frame.keys()))

def next_in_list(sorted_list, cur, *, wrap=False):
    if not sorted_list:
        return cur
    for f in sorted_list:
        if f > cur:
            return f
    return sorted_list[0] if wrap else sorted_list[-1]

def prev_in_list(sorted_list, cur, *, wrap=False):
    if not sorted_list:
        return cur
    prev = sorted_list[0]
    for f in sorted_list:
        if f >= cur:
            return prev
        prev = f
    return sorted_list[-1] if wrap else prev

from collections import defaultdict
per_class_frames = defaultdict(list)
for f, boxes in obj_by_frame.items():
    for b in boxes:
        per_class_frames[f"[obj] {b.cls}"].append(f)
for f, boxes in txt_by_frame.items():
    for b in boxes:
        per_class_frames[f"[txt] {b.cls}"].append(f)
for k in per_class_frames:
    per_class_frames[k] = sorted(set(per_class_frames[k]))

all_obj = sorted({b.cls for L in obj_by_frame.values() for b in L})
all_txt = sorted({b.cls for L in txt_by_frame.values() for b in L})

st.markdown("#### Filters")
f1, f2, f3 = st.columns([2.2, 2.2, 6])
with f1:
    filt_obj = st.multiselect("Object classes", all_obj, default=all_obj)
with f2:
    filt_txt = st.multiselect("Text tokens", all_txt, default=all_txt)
with f3:
    st.markdown("<span class='badge obj'>Object</span><span class='badge txt'>Text</span><span class='badge lab'>Label</span>", unsafe_allow_html=True)

if "frame_slider" not in st.session_state:
    st.session_state.frame_slider = 0

step_left, step_mid, step_right = st.columns([2.4, 3.2, 2.4], gap="small")

with step_left:
    st.markdown("#### Frame")
    st.session_state.frame_slider = st.slider(" ", 0, max(0, meta.frame_count - 1), st.session_state.frame_slider, 1, label_visibility="collapsed")

with step_mid:
    st.markdown("#### Stepper (Detections)")
    c1, c2, c3 = st.columns(3)
    if c1.button("⏮ Prev detection", use_container_width=True):
        st.session_state.frame_slider = prev_in_list(det_frames, st.session_state.frame_slider)
    if c2.button("⏭ Next detection", use_container_width=True):
        st.session_state.frame_slider = next_in_list(det_frames, st.session_state.frame_slider)
    if c3.button("⏹ Reset", use_container_width=True):
        st.session_state.frame_slider = 0

with step_right:
    st.markdown("#### Stepper (Labels)")
    c4, c5 = st.columns(2)
    if c4.button("⏮ Prev label", use_container_width=True):
        st.session_state.frame_slider = prev_in_list(lab_frames, st.session_state.frame_slider)
    if c5.button("⏭ Next label", use_container_width=True):
        st.session_state.frame_slider = next_in_list(lab_frames, st.session_state.frame_slider)

classes_sorted = sorted(per_class_frames.keys())
pc1, pc2 = st.columns([3, 1.3])
with pc1:
    chosen_pc = st.selectbox("Per-class step:", options=["— Select class/text —"] + classes_sorted)
with pc2:
    colA, colB = st.columns(2)
    if chosen_pc != "— Select class/text —":
        frames_pc = per_class_frames[chosen_pc]
        if colA.button("⏮ Prev", use_container_width=True):
            st.session_state.frame_slider = prev_in_list(frames_pc, st.session_state.frame_slider)
        if colB.button("⏭ Next", use_container_width=True):
            st.session_state.frame_slider = next_in_list(frames_pc, st.session_state.frame_slider)

fidx = int(st.session_state.frame_slider)
st.caption(f"Frame **{fidx}** / {meta.frame_count-1} • ~{frame_to_ms(fidx, meta.fps)} ms • {meta.fps:.2f} fps • {meta.w}×{meta.h}")

cap = open_video(vid_path) if vid_path else None

obj_now = [b for b in obj_by_frame.get(fidx, []) if b.cls in filt_obj]
txt_now = [b for b in txt_by_frame.get(fidx, []) if b.cls in filt_txt]
boxes_now = obj_now + txt_now

col_left, col_mid, col_right = st.columns([3.2, 3.2, 2.6], gap="large")

with col_left:
    st.markdown("### ▶️ Video")
    if vid_path:
        st.video(vid_path, format="video/mp4")
    else:
        st.info("No video uploaded — showing synthetic canvas in middle column.")
    with st.expander("Quick actions"):
        if st.button("Export detections (CSV)"):
            import pandas as pd
            rows = []
            for b in boxes_now:
                x1, y1, x2, y2 = b.xyxy
                rows.append({"type": b.source, "class/text": b.cls, "conf": b.conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1})
            df = pd.DataFrame(rows)
            st.download_button("Download CSV", df.to_csv(index=False).encode(), file_name=f"detections_f{fidx}.csv", mime="text/csv")
        st.caption("Keyboard: <span class='kbd'>←</span> / <span class='kbd'>→</span> to scrub", unsafe_allow_html=True)

with col_mid:
    st.markdown("### 🖼️ Annotated Frame")
    base = read_frame(cap, fidx, (meta.w, meta.h))
    vis = draw_boxes(base, boxes_now, min_conf=min_conf, show_tags=True, allowed=None)
    st.image(vis, use_column_width=True, caption="Overlays: objects + OCR")
    c1, c2, c3 = st.columns(3)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    with c1:
        st.metric("Objects", len([b for b in obj_now if b.conf >= min_conf]))
    with c2:
        st.metric("Text boxes", len([b for b in txt_now if b.conf >= min_conf]))
    with c3:
        st.metric("Labels (this frame)", len(lab_by_frame.get(fidx, [])))
    st.markdown("</div>", unsafe_allow_html=True)
    buf = io.BytesIO()
    vis.save(buf, format="PNG")
    st.download_button("💾 Download annotated PNG", data=buf.getvalue(), file_name=f"frame_{fidx}_annotated.png", mime="image/png")

with col_right:
    st.markdown("### 🏷️ All Labels & Detections")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("**Detections (current frame)**")
    if boxes_now:
        boxes_now_sorted = sorted([b for b in boxes_now if b.conf >= min_conf], key=lambda x: (-x.conf, x.cls))
        for b in boxes_now_sorted:
            typ = "obj" if b.source == "object" else "txt"
            st.markdown(f"<span class='badge {typ}'>{b.source}</span> **{b.cls}** &nbsp;·&nbsp; {b.conf:.2f}", unsafe_allow_html=True)
    else:
        st.caption("No detections at this frame with current filters.")
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("**Labels (time-indexed, pinned to frames)**")
    labs = lab_by_frame.get(fidx, [])
    if labs:
        for name, conf in sorted(labs, key=lambda x: (-x[1], x[0])):
            st.markdown(f"<span class='badge lab'>label</span> **{name}** · {conf:.2f}", unsafe_allow_html=True)
    else:
        st.caption("No labels at this frame.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("#### 🔎 Context Peek (upcoming detections)")
if det_frames:
    upcoming = [f for f in det_frames if f > fidx][:4]
    if not upcoming:
        st.caption("No upcoming detection frames.")
    else:
        cols = st.columns(len(upcoming))
        for i, fnext in enumerate(upcoming):
            with cols[i]:
                thumb_base = read_frame(cap, fnext, (meta.w, meta.h))
                mini_boxes = (obj_by_frame.get(fnext, []) + txt_by_frame.get(fnext, []))[:6]
                thumb = draw_boxes(thumb_base, mini_boxes, min_conf=min_conf, show_tags=False, allowed=None)
                st.image(thumb, use_container_width=True, caption=f"f={fnext}")
                if st.button(f"Jump to {fnext}", key=f"jump_{i}", use_container_width=True):
                    st.session_state.frame_slider = fnext

st.markdown("#### Details")
t1, t2, t3 = st.tabs(["Objects table", "Text table", "Timeline (labels)"])

with t1:
    import pandas as pd
    rows = []
    for b in obj_by_frame.get(fidx, []):
        if b.cls not in filt_obj or b.conf < min_conf:
            continue
        x1, y1, x2, y2 = b.xyxy
        rows.append({"class": b.cls, "conf": round(b.conf, 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t2:
    import pandas as pd
    rows = []
    for b in txt_by_frame.get(fidx, []):
        if b.cls not in filt_txt or b.conf < min_conf:
            continue
        x1, y1, x2, y2 = b.xyxy
        rows.append({"text": b.cls, "conf": round(b.conf, 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with t3:
    try:
        import pandas as pd
        import altair as alt
        data = []
        for f, items in lab_by_frame.items():
            for (lab, conf) in items:
                data.append({"frame": f, "ms": frame_to_ms(f, meta.fps), "label": lab, "conf": conf})
        df = pd.DataFrame(data)
        if df.empty:
            st.caption("No labels found.")
        else:
            chosen = st.multiselect("Filter labels", sorted(df["label"].unique()), default=sorted(df["label"].unique()))
            dfv = df[df["label"].isin(chosen)]
            chart = alt.Chart(dfv).mark_circle().encode(
                x="frame:Q",
                y="label:N",
                size=alt.Size("conf:Q", legend=None),
                tooltip=["frame", "ms", "label", "conf"],
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.info("Install `altair` for the label timeline plot (optional).")
