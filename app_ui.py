import streamlit as st
import tempfile
import os
import io
import time
from PIL import Image
from PIL.ExifTags import TAGS
from src.inference.predict_system import final_predict

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="TraceFake AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Cyber UI Styling
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #E2E8F0;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ---- HEADER ---- */
.tf-header {
    text-align: center;
    padding: 10px 0 18px;
}
.tf-logo {
    font-family: 'Space Mono', monospace;
    font-size: 30px;
    color: #00D4E8;
    letter-spacing: 3px;
}
.tf-logo span { color: #FF4D6D; }
.tf-tagline {
    font-size: 12px;
    color: #3A6070;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ---- STEPPER ---- */
.stepper-wrap {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0;
    margin-bottom: 28px;
}
.step {
    display: flex; align-items: center; gap: 7px;
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #2A4050;
}
.step.active { color: #00D4E8; }
.step.done   { color: #00C98A; }
.step-num {
    width: 24px; height: 24px; border-radius: 50%;
    border: 1.5px solid #1A3040;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; background: #080C14;
}
.step.active .step-num { border-color: #00D4E8; color: #00D4E8; }
.step.done   .step-num { border-color: #00C98A; background: rgba(0,201,138,0.1); color: #00C98A; }
.step-line { width: 36px; height: 1px; background: #1A3040; margin: 0 6px; }
.step-line.done { background: rgba(0,201,138,0.4); }

/* ---- GLASS CARD ---- */
.card {
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(0,212,232,0.1);
    margin-bottom: 20px;
}

/* ---- UPLOAD ZONE ---- */
.upload-hint {
    border: 1.5px dashed rgba(0,212,232,0.3);
    border-radius: 12px;
    padding: 28px;
    text-align: center;
    color: #4A7080;
    font-size: 13px;
    margin-bottom: 6px;
}

/* ---- SCAN ITEMS ---- */
.scan-item {
    display: flex; align-items: center; gap: 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 12px;
}
.scan-label { flex: 1; color: #6A8A9A; }
.scan-done  { color: #00C98A; font-family: 'Space Mono', monospace; font-size: 11px; }
.scan-wait  { color: #FFAA32; font-family: 'Space Mono', monospace; font-size: 11px; }

/* ---- VERDICT ---- */
.verdict-fake {
    font-family: 'Space Mono', monospace;
    font-size: 26px; font-weight: 700;
    color: #FF4D6D;
    text-align: center; padding: 18px 0 10px;
    letter-spacing: 1px;
}
.verdict-real {
    font-family: 'Space Mono', monospace;
    font-size: 26px; font-weight: 700;
    color: #00C98A;
    text-align: center; padding: 18px 0 10px;
    letter-spacing: 1px;
}
.verdict-sub {
    text-align: center; font-size: 12px;
    color: #4A7080; margin-bottom: 18px;
    letter-spacing: 1px;
}

/* ---- SCORE GRID ---- */
.score-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
}
.score-label {
    font-size: 11px; color: #4A7080;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 6px;
}
.score-val {
    font-family: 'Space Mono', monospace;
    font-size: 22px; color: #00D4E8;
}

/* ---- EXIF TABLE ---- */
.exif-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 12px;
}
.exif-row:last-child { border-bottom: none; }
.exif-key { color: #4A7080; }
.exif-val { color: #9ABAC8; font-family: 'Space Mono', monospace; font-size: 11px; }
.exif-miss { color: #FF4D6D55; font-family: 'Space Mono', monospace; font-size: 11px; }

/* ---- SECTION TITLE ---- */
.sec-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px; color: #00D4E8;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 12px; border-bottom: 1px solid rgba(0,212,232,0.1);
    padding-bottom: 6px;
}

/* ---- CONFIDENCE BAR ---- */
.conf-bar-wrap { margin: 10px 0 16px; }
.conf-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 6px; height: 8px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 6px;
    transition: width 0.6s ease;
}

/* ---- HISTORY TABLE ---- */
.hist-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 6px;
    font-size: 12px;
}
.hist-date { color: #4A7080; }
.hist-real { color: #00C98A; font-family: 'Space Mono', monospace; font-size: 11px; }
.hist-fake { color: #FF4D6D; font-family: 'Space Mono', monospace; font-size: 11px; }
.hist-score { color: #6A8A9A; font-family: 'Space Mono', monospace; font-size: 11px; }

/* Override Streamlit button */
div.stButton > button {
    background: linear-gradient(135deg, #00A8BE, #00D4E8) !important;
    color: #080C14 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    width: 100% !important;
}
div.stButton > button:hover { opacity: 0.85 !important; }

div.stDownloadButton > button {
    background: rgba(0,212,232,0.08) !important;
    color: #00D4E8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: 1px solid rgba(0,212,232,0.3) !important;
    border-radius: 8px !important;
    width: 100% !important;
}

/* Progress bar color */
div[data-testid="stProgress"] > div > div > div {
    background-color: #00D4E8 !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Session State Init
# -------------------------
if "step" not in st.session_state:
    st.session_state.step = 1          # 1=upload 2=analyzing 3=results 4=history
if "result" not in st.session_state:
    st.session_state.result = None
if "exif_data" not in st.session_state:
    st.session_state.exif_data = {}
if "img_info" not in st.session_state:
    st.session_state.img_info = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "tmp_path" not in st.session_state:
    st.session_state.tmp_path = None


# -------------------------
# Helper: Extract EXIF
# -------------------------
def extract_exif(image_path):
    exif_info = {}
    try:
        img = Image.open(image_path)
        raw = img._getexif()
        if raw:
            for tag_id, value in raw.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag in ["Make", "Model", "DateTime", "Software",
                            "GPSInfo", "Flash", "ExifVersion", "Orientation"]:
                    exif_info[tag] = str(value)[:60]
        st.session_state.img_info = {
            "format": img.format or "JPEG",
            "size": f"{img.width} × {img.height} px",
            "mode": img.mode,
        }
    except Exception:
        pass
    return exif_info


# -------------------------
# Helper: Build Report Text
# -------------------------
def build_report(result, exif_data, img_info):
    lines = [
        "============================================",
        "         TRACEFAKE AI — FORENSIC REPORT     ",
        "============================================",
        "",
        f"VERDICT       : {result['result']}",
        f"FINAL SCORE   : {result['final_score']:.4f}",
        f"CNN SCORE     : {result['cnn_score']:.4f}",
        f"EXIF SCORE    : {result['exif_score']:.4f}",
        f"FORENSIC SCORE: {result['forensic_score']:.4f}",
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
        lines.append("No EXIF metadata found in image.")
    lines += [
        "",
        "============================================",
        "  Generated by TraceFake AI Forensic System ",
        "============================================",
    ]
    return "\n".join(lines)


# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class="tf-header">
    <div class="tf-logo">TRACE<span>FAKE</span> AI</div>
    <div class="tf-tagline">Image Authenticity Forensic System</div>
</div>
""", unsafe_allow_html=True)


# -------------------------
# STEPPER
# -------------------------
step = st.session_state.step

def step_class(n):
    if n < step: return "step done"
    if n == step: return "step active"
    return "step"

def line_class(n):
    return "step-line done" if n < step else "step-line"

st.markdown(f"""
<div class="stepper-wrap">
  <div class="{step_class(1)}">
    <div class="step-num">{'✓' if step > 1 else '1'}</div>UPLOAD
  </div>
  <div class="{line_class(1)}"></div>
  <div class="{step_class(2)}">
    <div class="step-num">{'✓' if step > 2 else '2'}</div>ANALYZE
  </div>
  <div class="{line_class(2)}"></div>
  <div class="{step_class(3)}">
    <div class="step-num">{'✓' if step > 3 else '3'}</div>RESULTS
  </div>
  <div class="{line_class(3)}"></div>
  <div class="{step_class(4)}">
    <div class="step-num">4</div>HISTORY
  </div>
</div>
""", unsafe_allow_html=True)


# ============================
# STEP 1 — UPLOAD
# ============================
if step == 1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>◈ Upload Image</div>", unsafe_allow_html=True)
        st.markdown("<div class='upload-hint'>Drag & drop or click below to upload a JPG / PNG image</div>",
                    unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True, caption="Preview")

            st.markdown(f"""
            <div style='font-size:12px; color:#4A7080; margin:8px 0;'>
                📄 <b style='color:#6A9AAA'>{uploaded_file.name}</b> &nbsp;|&nbsp;
                {uploaded_file.size // 1024} KB
            </div>
            """, unsafe_allow_html=True)

            if st.button("▶  START ANALYSIS"):
                # Save to temp file
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                uploaded_file.seek(0)
                tmp.write(uploaded_file.read())
                tmp.flush()
                st.session_state.tmp_path = tmp.name
                st.session_state.exif_data = extract_exif(tmp.name)
                st.session_state.step = 2
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ============================
# STEP 2 — ANALYZING
# ============================
elif step == 2:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>◈ Running Forensic Analysis</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='scan-item'>
            <span style='font-size:16px'>🧠</span>
            <span class='scan-label'>CNN Deep Feature Extraction</span>
            <span class='scan-wait'>PROCESSING...</span>
        </div>
        <div class='scan-item'>
            <span style='font-size:16px'>📷</span>
            <span class='scan-label'>EXIF Metadata Analysis</span>
            <span class='scan-wait'>PROCESSING...</span>
        </div>
        <div class='scan-item'>
            <span style='font-size:16px'>🔬</span>
            <span class='scan-label'>ELA Forensic Check</span>
            <span class='scan-wait'>PROCESSING...</span>
        </div>
        """, unsafe_allow_html=True)

        bar = st.progress(0)
        for pct in range(0, 61, 5):
            time.sleep(0.04)
            bar.progress(pct)

        # Run actual prediction
        has_exif = 1 if st.session_state.exif_data else 0
        missing   = 3 - has_exif
        features  = {"missing": missing, "has_exif": has_exif, "suspicious": 0}

        result = final_predict(st.session_state.tmp_path, features)
        st.session_state.result = result

        # Add to history
        st.session_state.history.append({
            "verdict": result["result"],
            "score":   result["final_score"],
        })

        for pct in range(60, 101, 5):
            time.sleep(0.03)
            bar.progress(pct)

        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state.step = 3
        st.rerun()


# ============================
# STEP 3 — RESULTS
# ============================
elif step == 3:
    result   = st.session_state.result
    exif_data = st.session_state.exif_data
    img_info  = st.session_state.img_info

    is_fake = result["result"] == "FAKE"
    conf    = result["final_score"]
    bar_color = "#FF4D6D" if is_fake else "#00C98A"

    # --- Verdict banner ---
    if is_fake:
        st.markdown("<div class='verdict-fake'>✗ FAKE IMAGE DETECTED</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-real'>✓ REAL IMAGE VERIFIED</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='verdict-sub'>Confidence Score: {conf:.2%}</div>", unsafe_allow_html=True)

    # --- Score grid ---
    c1, c2, c3, c4 = st.columns(4)
    scores = [
        ("CNN Score",     result["cnn_score"]),
        ("EXIF Score",    result["exif_score"]),
        ("Forensic Score",result["forensic_score"]),
        ("Final Score",   result["final_score"]),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], scores):
        with col:
            st.markdown(f"""
            <div class='score-card'>
                <div class='score-label'>{label}</div>
                <div class='score-val'>{val:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Two columns: EXIF + image info ---
    left, right = st.columns([1, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>◈ EXIF Metadata</div>", unsafe_allow_html=True)

        exif_fields = {
            "Camera Make":  exif_data.get("Make"),
            "Camera Model": exif_data.get("Model"),
            "Date Taken":   exif_data.get("DateTime"),
            "Software":     exif_data.get("Software"),
            "Orientation":  exif_data.get("Orientation"),
            "Flash":        exif_data.get("Flash"),
        }

        for key, val in exif_fields.items():
            if val:
                st.markdown(f"""
                <div class='exif-row'>
                    <span class='exif-key'>{key}</span>
                    <span class='exif-val'>{val}</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='exif-row'>
                    <span class='exif-key'>{key}</span>
                    <span class='exif-miss'>— missing —</span>
                </div>""", unsafe_allow_html=True)

        exif_count = sum(1 for v in exif_fields.values() if v)
        st.markdown(f"""
        <div style='margin-top:12px; font-size:11px; color:#3A6070;'>
            {exif_count} of {len(exif_fields)} fields present
            {'&nbsp;&nbsp;<span style="color:#FF4D6D">⚠ Suspicious</span>' if exif_count < 2 else ''}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>◈ Image Info</div>", unsafe_allow_html=True)

        info_rows = [
            ("Format",     img_info.get("format", "—")),
            ("Dimensions", img_info.get("size",   "—")),
            ("Color Mode", img_info.get("mode",   "—")),
            ("EXIF Present", "Yes" if exif_data else "No"),
            ("Metadata Count", str(len(exif_data)) + " fields"),
        ]
        for key, val in info_rows:
            st.markdown(f"""
            <div class='exif-row'>
                <span class='exif-key'>{key}</span>
                <span class='exif-val'>{val}</span>
            </div>""", unsafe_allow_html=True)

        # Confidence bar
        pct = int(conf * 100)
        st.markdown(f"""
        <div style='margin-top:14px;'>
            <div style='font-size:11px; color:#4A7080; margin-bottom:5px;'>
                CONFIDENCE LEVEL — {pct}%
            </div>
            <div class='conf-bar-bg'>
                <div class='conf-bar-fill'
                     style='width:{pct}%; background:{bar_color};'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Actions ---
    btn1, btn2, btn3 = st.columns(3)

    report_text = build_report(result, exif_data, img_info)
    report_bytes = report_text.encode("utf-8")

    with btn1:
        st.download_button(
            "⬇  DOWNLOAD REPORT (.txt)",
            data=report_bytes,
            file_name="tracefake_report.txt",
            mime="text/plain",
        )
    with btn2:
        if st.button("📋  VIEW HISTORY"):
            st.session_state.step = 4
            st.rerun()
    with btn3:
        if st.button("↺  ANALYZE ANOTHER"):
            st.session_state.step = 1
            st.session_state.result = None
            st.session_state.exif_data = {}
            st.session_state.img_info = {}
            if st.session_state.tmp_path and os.path.exists(st.session_state.tmp_path):
                os.unlink(st.session_state.tmp_path)
            st.session_state.tmp_path = None
            st.rerun()


# ============================
# STEP 4 — HISTORY
# ============================
elif step == 4:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='sec-title'>◈ Analysis Report History</div>", unsafe_allow_html=True)

        history = st.session_state.history
        if not history:
            st.markdown("<p style='color:#3A6070; font-size:13px;'>No analyses yet.</p>",
                        unsafe_allow_html=True)
        else:
            for i, entry in enumerate(reversed(history), 1):
                verdict_class = "hist-fake" if entry["verdict"] == "FAKE" else "hist-real"
                st.markdown(f"""
                <div class='hist-row'>
                    <span class='hist-date'>Analysis #{len(history) - i + 1}</span>
                    <span class='{verdict_class}'>{entry['verdict']}</span>
                    <span class='hist-score'>Score: {entry['score']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("◀  BACK TO RESULTS"):
            st.session_state.step = 3
            st.rerun()
