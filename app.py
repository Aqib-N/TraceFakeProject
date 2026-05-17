import streamlit as st
import tempfile
import os
import sys
import time
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS

# Path setup 
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

Path("reports").mkdir(exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

from src.inference.predict_system import final_predict

# Config 
try:
    from config import MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, ALLOWED_PIL_FORMATS
except ImportError:
    MAX_FILE_SIZE_MB    = 10
    ALLOWED_EXTENSIONS  = {"jpg", "jpeg", "png"}
    ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}


# File Validation 

def _safe_filename(name: str) -> str:
    return Path(name).name


def validate_upload(uploaded_file):
    """Three-layer: size → extension → PIL format verification."""
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File too large. Max: {MAX_FILE_SIZE_MB} MB"

    suffix = Path(uploaded_file.name).suffix.lstrip(".").lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported extension '.{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"

    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        if img.format not in ALLOWED_PIL_FORMATS:
            return False, f"Image format '{img.format}' not supported. Use JPG or PNG."
    except Exception:
        return False, "Invalid or corrupt image file."

    uploaded_file.seek(0)
    return True, "Valid"


# Page Config 
st.set_page_config(
    page_title="TraceFake AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Font smoothing (fixes blurry text) ── */
*, *::before, *::after {
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #E2E8F0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Card columns 
   Use :has() to style only columns that contain actual content.
   Empty spacer columns (in [1,2,1] layouts) have no child divs → no style.
   This replaces the broken split <div class='card'> / </div> pattern.
 */

/* Middle column of 3-column layouts (steps 1, 2, 4) */
[data-testid="stHorizontalBlock"]:has(
    > [data-testid="stColumn"]:nth-child(3)
) > [data-testid="stColumn"]:nth-child(2) {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    border: 1px solid rgba(0,212,232,0.1);
    padding: 20px 24px;
}

/* ── Explicit .card class for step-3 pure-HTML cards ── */
.card {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(0,212,232,0.1);
    margin-bottom: 4px;
}

/* ── Header ── */
.tf-header { text-align: center; padding: 10px 0 18px; }
.tf-logo {
    font-family: 'Space Mono', monospace;
    font-size: 30px; color: #00D4E8;
    letter-spacing: 3px; font-weight: 700;
}
.tf-logo span { color: #FF4D6D; }
.tf-tagline {
    font-size: 12px; color: #3A6070;
    letter-spacing: 2px; text-transform: uppercase; margin-top: 4px;
}

/* ── Stepper ── */
.stepper-wrap {
    display: flex; justify-content: center;
    align-items: center; gap: 0; margin-bottom: 28px;
}
.step {
    display: flex; align-items: center; gap: 7px;
    font-family: 'Space Mono', monospace; font-size: 11px; color: #2A4050;
}
.step.active { color: #00D4E8; }
.step.done   { color: #00C98A; }
.step-num { width: 24px; height: 24px; border-radius: 50%; border: 1.5px solid #1A3040; display: flex; align-items: center; justify-content: center; font-size: 10px; background: #080C14; }
.step.active .step-num { border-color: #00D4E8; color: #00D4E8; }
.step.done   .step-num { border-color: #00C98A; background: rgba(0,201,138,0.1); color: #00C98A; }
.step-line { width: 36px; height: 1px; background: #1A3040; margin: 0 6px; }
.step-line.done { background: rgba(0,201,138,0.4); }

/* ── Scan items (Step 2) ── */
.scan-item {
    display: flex; align-items: center; gap: 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px; padding: 10px 14px;
    margin-bottom: 8px; font-size: 12px;
}
.scan-label { flex: 1; color: #6A8A9A; }
.scan-done  { color: #00C98A; font-family: 'Space Mono', monospace; font-size: 11px; }
.scan-wait  { color: #FFAA32; font-family: 'Space Mono', monospace; font-size: 11px; }

/* ── Verdicts ── */
.verdict-fake      { font-family: 'Space Mono', monospace; font-size: 26px; font-weight: 700; color: #FF4D6D; text-align: center; padding: 18px 0 10px; letter-spacing: 1px; }
.verdict-real      { font-family: 'Space Mono', monospace; font-size: 26px; font-weight: 700; color: #00C98A; text-align: center; padding: 18px 0 10px; letter-spacing: 1px; }
.verdict-uncertain { font-family: 'Space Mono', monospace; font-size: 26px; font-weight: 700; color: #FFAA32; text-align: center; padding: 18px 0 10px; letter-spacing: 1px; }
.verdict-sub { text-align: center; font-size: 12px; color: #4A7080; margin-bottom: 18px; letter-spacing: 1px; }

/* ── Score cards ── */
.score-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 14px; text-align: center;
}
.score-label { font-size: 11px; color: #4A7080; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.score-val   { font-family: 'Space Mono', monospace; font-size: 22px; color: #00D4E8; }

/* ── EXIF rows ── */
.exif-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 12px;
}
.exif-row:last-child { border-bottom: none; }
.exif-key  { color: #4A7080; }
.exif-val  { color: #9ABAC8; font-family: 'Space Mono', monospace; font-size: 11px; }
.exif-miss { color: rgba(255,77,109,0.4); font-family: 'Space Mono', monospace; font-size: 11px; }

/* ── Section title ── */
.sec-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #00D4E8;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(0,212,232,0.1);
    padding-bottom: 6px;
}

/* ── Confidence bar ── */
.conf-bar-bg   { background: rgba(255,255,255,0.05); border-radius: 6px; height: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s ease; }

/* ── History rows ── */
.hist-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; border-radius: 8px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 6px; font-size: 12px;
}
.hist-date      { color: #4A7080; }
.hist-real      { color: #00C98A; font-family: 'Space Mono', monospace; font-size: 11px; }
.hist-fake      { color: #FF4D6D; font-family: 'Space Mono', monospace; font-size: 11px; }
.hist-uncertain { color: #FFAA32; font-family: 'Space Mono', monospace; font-size: 11px; }
.hist-score     { color: #6A8A9A; font-family: 'Space Mono', monospace; font-size: 11px; }

/* ── Buttons ── */
div.stButton > button {
    background: linear-gradient(135deg, #00A8BE, #00D4E8) !important;
    color: #080C14 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important; font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important; border-radius: 8px !important;
    padding: 10px 20px !important; width: 100% !important;
}
div.stButton > button:hover { opacity: 0.85 !important; }
div.stDownloadButton > button {
    background: rgba(0,212,232,0.08) !important;
    color: #00D4E8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important; font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: 1px solid rgba(0,212,232,0.3) !important;
    border-radius: 8px !important; width: 100% !important;
}
div[data-testid="stProgress"] > div > div > div { background-color: #00D4E8 !important; }

/* ── File uploader — style Streamlit's native component ── */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(0,212,232,0.3) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
    color: #4A7080 !important;
    font-size: 13px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
    color: #3A6070 !important;
}
[data-testid="stFileUploader"] section > button {
    background: rgba(0,212,232,0.1) !important;
    border: 1px solid rgba(0,212,232,0.3) !important;
    color: #00D4E8 !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}
</style>
""", unsafe_allow_html=True)

# Session State 
for key, default in [
    ("step", 1), ("result", None), ("exif_data", {}),
    ("img_info", {}), ("history", []), ("tmp_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Session State 
for key, default in [
    ("step", 1), ("result", None), ("exif_data", {}),
    ("img_info", {}), ("history", []), ("tmp_path", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# Helpers 

def extract_exif(image_path):
    exif_info = {}
    try:
        img = Image.open(image_path)
        try:
            raw = img.getexif()
        except AttributeError:
            raw = img._getexif() or {}

        if raw:
            for tag_id, value in raw.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in {"Make", "Model", "DateTime", "Software", "GPSInfo",
                           "Flash", "ExifVersion", "Orientation",
                           "DateTimeOriginal", "DateTimeDigitized"}:
                    exif_info[tag] = str(value)[:60]

        st.session_state.img_info = {
            "format": img.format or "JPEG",
            "size":   f"{img.width} × {img.height} px",
            "mode":   img.mode,
        }
    except Exception:
        pass
    return exif_info


def build_report(result, exif_data, img_info):
    lines = [
        "",
        "         TRACEFAKE AI — FORENSIC REPORT     ",
        "",
        "",
        f"VERDICT       : {result['result']}",
        f"CONFIDENCE    : {result['confidence']:.2%}",
        f"FINAL SCORE   : {result['final_score']:.4f}",
        f"CNN SCORE     : {result['cnn_score']:.4f}",
        f"EXIF SCORE    : {result['exif_score']:.4f}",
        f"FORENSIC SCORE: {result['forensic_score']:.4f}",
        f"EXIF PRESENT  : {'Yes' if result.get('exif_present') else 'No (social media stripped)'}",
        "",
        "---- IMAGE INFO ----",
    ]
    for k, v in img_info.items():
        lines.append(f"{k.upper():<14}: {v}")
    lines += ["", "---- EXIF METADATA ----"]
    if exif_data:
        for k, v in exif_data.items():
            lines.append(f"{k:<14}: {v}")
    else:
        lines.append("No EXIF metadata found.")
    lines += [
        "",
        "",
        "  Generated by TraceFake AI Forensic System ",
        "",
    ]
    return "\n".join(lines)


# Header 
st.markdown("""
<div class="tf-header">
    <div class="tf-logo">TRACE<span>FAKE</span> AI</div>
    <div class="tf-tagline">Image Authenticity Forensic System</div>
</div>
""", unsafe_allow_html=True)

# Stepper 
step = st.session_state.step

def step_class(n):
    if n < step:  return "step done"
    if n == step: return "step active"
    return "step"

def line_class(n):
    return "step-line done" if n < step else "step-line"

st.markdown(f"""
<div class="stepper-wrap">
  <div class="{step_class(1)}"><div class="step-num">{'✓' if step > 1 else '1'}</div>UPLOAD</div>
  <div class="{line_class(1)}"></div>
  <div class="{step_class(2)}"><div class="step-num">{'✓' if step > 2 else '2'}</div>ANALYZE</div>
  <div class="{line_class(2)}"></div>
  <div class="{step_class(3)}"><div class="step-num">{'✓' if step > 3 else '3'}</div>RESULTS</div>
  <div class="{line_class(3)}"></div>
  <div class="{step_class(4)}"><div class="step-num">4</div>HISTORY</div>
</div>
""", unsafe_allow_html=True)


# Step 1 — Upload

if step == 1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:

        st.markdown("<div class='sec-title'>◈ Upload Image</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag & drop or click — Max 10 MB · JPG / PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            is_valid, msg = validate_upload(uploaded_file)
            if not is_valid:
                st.error(f"⚠️ {msg}")
            else:
                uploaded_file.seek(0)
                img = Image.open(uploaded_file)
                st.image(img, use_container_width=True, caption="Preview")
                st.markdown(
                    f"<div style='font-size:12px; color:#4A7080; margin:8px 0;'>"
                    f"📄 <b style='color:#6A9AAA'>{_safe_filename(uploaded_file.name)}</b>"
                    f" &nbsp;|&nbsp; {uploaded_file.size // 1024} KB</div>",
                    unsafe_allow_html=True,
                )
                if st.button("▶  START ANALYSIS"):
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        uploaded_file.seek(0)
                        tmp.write(uploaded_file.read())
                        st.session_state.tmp_path = tmp.name
                    st.session_state.exif_data = extract_exif(st.session_state.tmp_path)
                    st.session_state.step = 2
                    st.rerun()


# Step 2 — Analyzing
elif step == 2:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='sec-title'>◈ Running Forensic Analysis</div>", unsafe_allow_html=True)

        for icon, label, detail in [
            ("🧠", "CNN Deep Feature Extraction",  "Analyzing visual patterns..."),
            ("📷", "EXIF Metadata Analysis",        "Checking camera signatures..."),
            ("🔬", "ELA Forensic Check",            "Detecting compression artifacts..."),
            ("🎨", "Noise & FFT Pattern Analysis",  "Identifying GAN/diffusion artifacts..."),
        ]:
            st.markdown(
                f"<div class='scan-item'>"
                f"<span style='font-size:16px'>{icon}</span>"
                f"<span class='scan-label'>{label}<br>"
                f"<small style='color:#3A6070'>{detail}</small></span>"
                f"<span class='scan-wait'>PROCESSING...</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        bar = st.progress(0)
        for pct in range(0, 101, 5):
            time.sleep(0.05)
            bar.progress(pct)

        result = final_predict(st.session_state.tmp_path)
        st.session_state.result = result
        st.session_state.history.append({
            "verdict":    result["result"],
            "score":      result["final_score"],
            "confidence": result["confidence"],
            "file":       Path(st.session_state.tmp_path).name,
        })

        st.session_state.step = 3
        st.rerun()


# Step 3 — Results
elif step == 3:
    result    = st.session_state.result
    exif_data = st.session_state.exif_data
    img_info  = st.session_state.img_info
    verdict   = result["result"]
    conf      = result["confidence"]

    if verdict == "FAKE":
        bar_color = "#FF4D6D"
        st.markdown("<div class='verdict-fake'>✗ FAKE IMAGE DETECTED</div>", unsafe_allow_html=True)
    elif verdict == "REAL":
        bar_color = "#00C98A"
        st.markdown("<div class='verdict-real'>✓ REAL IMAGE VERIFIED</div>", unsafe_allow_html=True)
    else:  # UNCERTAIN
        bar_color = "#FFAA32"
        st.markdown("<div class='verdict-uncertain'>? RESULT UNCERTAIN</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center; font-size:12px; color:#FFAA32; margin-bottom:8px;'>"
            "Signals are ambiguous — consider manual review or additional context.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div class='verdict-sub'>Confidence: {conf:.2%} | Final Score: {result['final_score']:.4f}</div>",
        unsafe_allow_html=True,
    )

    # Score cards (pure HTML, self-contained per column) 
    c1, c2, c3, c4 = st.columns(4)
    for col, (label, val) in zip(
        [c1, c2, c3, c4],
        [("CNN Score",      result["cnn_score"]),
         ("EXIF Score",     result["exif_score"]),
         ("Forensic Score", result["forensic_score"]),
         ("Final Score",    result["final_score"])],
    ):
        with col:
            st.markdown(
                f"<div class='score-card'>"
                f"<div class='score-label'>{label}</div>"
                f"<div class='score-val'>{val:.3f}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)

    # Left card: EXIF Metadata 

    with left:
        exif_fields = {
            "Camera Make":   exif_data.get("Make"),
            "Camera Model":  exif_data.get("Model"),
            "Date Taken":    exif_data.get("DateTime"),
            "Date Original": exif_data.get("DateTimeOriginal"),
            "Software":      exif_data.get("Software"),
            "Orientation":   exif_data.get("Orientation"),
            "Flash":         exif_data.get("Flash"),
        }
        exif_count = sum(1 for v in exif_fields.values() if v)
        suspicious = exif_count < 2

        rows_html = "".join(
            f"<div class='exif-row'>"
            f"<span class='exif-key'>{k}</span>"
            f"<span class='{'exif-val' if v else 'exif-miss'}'>{v if v else '— missing —'}</span>"
            f"</div>"
            for k, v in exif_fields.items()
        )
        status_badge = (
            '<span style="color:#FF4D6D">⚠ Suspicious</span>'
            if suspicious else
            '<span style="color:#00C98A">✓ Normal</span>'
        )

        st.markdown(f"""
        <div class='card'>
            <div class='sec-title'>◈ EXIF Metadata</div>
            {rows_html}
            <div style='margin-top:12px; font-size:11px; color:#3A6070;'>
                {exif_count} of {len(exif_fields)} fields present
                &nbsp;&nbsp;{status_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Right card: Image Info 
    with right:
        pct = int(conf * 100)
        info_rows_html = "".join(
            f"<div class='exif-row'>"
            f"<span class='exif-key'>{k}</span>"
            f"<span class='exif-val'>{v}</span>"
            f"</div>"
            for k, v in [
                ("Format",       img_info.get("format", "—")),
                ("Dimensions",   img_info.get("size",   "—")),
                ("Color Mode",   img_info.get("mode",   "—")),
                ("EXIF Present", "Yes" if exif_data else "No"),
                ("EXIF Fields",  f"{len(exif_data)} fields"),
            ]
        )

        st.markdown(f"""
        <div class='card'>
            <div class='sec-title'>◈ Image Info</div>
            {info_rows_html}
            <div style='margin-top:14px;'>
                <div style='font-size:11px; color:#4A7080; margin-bottom:5px;'>
                    CONFIDENCE — {pct}%
                </div>
                <div class='conf-bar-bg'>
                    <div class='conf-bar-fill'
                         style='width:{pct}%; background:{bar_color};'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    btn1, btn2, btn3 = st.columns(3)
    report_bytes = build_report(result, exif_data, img_info).encode("utf-8")

    with btn1:
        st.download_button(
            "⬇  DOWNLOAD REPORT (.txt)",
            data=report_bytes,
            file_name=f"tracefake_report_{int(time.time())}.txt",
            mime="text/plain",
        )
    with btn2:
        if st.button("📋  VIEW HISTORY"):
            st.session_state.step = 4
            st.rerun()
    with btn3:
        if st.button("↺  ANALYZE ANOTHER"):
            if st.session_state.tmp_path and os.path.exists(st.session_state.tmp_path):
                os.unlink(st.session_state.tmp_path)
            for key, default in [
                ("step", 1), ("result", None), ("exif_data", {}),
                ("img_info", {}), ("tmp_path", None),
            ]:
                st.session_state[key] = default
            st.rerun()


# Step 4 — History
elif step == 4:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='sec-title'>◈ Analysis History</div>", unsafe_allow_html=True)

        history = st.session_state.history
        if not history:
            st.markdown(
                "<p style='color:#3A6070; font-size:13px;'>No analyses yet.</p>",
                unsafe_allow_html=True,
            )
        else:
            for i, entry in enumerate(reversed(history), 1):
                v      = entry["verdict"]
                vclass = ("hist-fake" if v == "FAKE"
                          else "hist-uncertain" if v == "UNCERTAIN"
                          else "hist-real")
                st.markdown(
                    f"<div class='hist-row'>"
                    f"<span class='hist-date'>#{len(history) - i + 1} — {entry.get('file','')[:20]}</span>"
                    f"<span class='{vclass}'>{v}</span>"
                    f"<span class='hist-score'>Conf: {entry['confidence']:.1%}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("◀  BACK TO RESULTS"):
            st.session_state.step = 3
            st.rerun()