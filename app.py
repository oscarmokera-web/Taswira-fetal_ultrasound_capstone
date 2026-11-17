import datetime
import io
import zipfile
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# --- Files exported by the notebook (section 6 + the "Export for the app" cell) ---
HERE = Path(__file__).parent
MODEL_PATH = HERE / "outputs" / "taswira_plane_finder.tflite"
CONFIG_PATH = HERE / "outputs" / "app_config.json"
BRAIN_PATH = HERE / "outputs" / "taswira_brain_check.tflite"
BRAIN_CONFIG_PATH = HERE / "outputs" / "brain_config.json"
BRAIN_REF_PATH = HERE / "outputs" / "brain_reference.npy"

# Same settings as the notebook's preprocessing
IMG_SIZE, CONTRAST_PCT = 224, (1, 99.5)
CLASS_NAMES = ["biparietal", "abdominal", "heart", "spine", "femur", "no_plane"]
PLAIN = {"biparietal": "Head", "abdominal": "Tummy (abdomen)", "heart": "Heart",
         "spine": "Spine", "femur": "Thigh bone (femur)", "no_plane": "No key view"}
NO_PLANE = CLASS_NAMES.index("no_plane")


def stretch_pad(a):
    # Contrast stretch, pad to a square, resize: as stretch_pad() in the notebook
    lo, hi = np.percentile(a, CONTRAST_PCT)
    a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    h, w = a.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return np.asarray(Image.fromarray(sq).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))


def preprocess(img):
    # Crop to the scan area, then stretch and pad: as preprocess_external() in the notebook
    a = np.asarray(img.convert("L"), dtype=np.float32)
    m = a > 8
    r, c = np.where(m.mean(1) > 0.02)[0], np.where(m.mean(0) > 0.02)[0]
    if len(r) > 10 and len(c) > 10:
        a = a[r[0]:r[-1] + 1, c[0]:c[-1] + 1]
    return stretch_pad(a)


def make_interpreter(path):
    try:
        from tensorflow import lite as tflite          # full TensorFlow (as in the notebook)
        interp = tflite.Interpreter(model_path=str(path))
    except ImportError:
        from ai_edge_litert.interpreter import Interpreter  # lighter option: pip install ai-edge-litert
        interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


@st.cache_resource
def load_model():
    return make_interpreter(MODEL_PATH)


@st.cache_resource
def load_brain():
    # The brain check, its thresholds and the guard's reference set (all from notebook section 8.2a)
    if not (BRAIN_PATH.exists() and BRAIN_CONFIG_PATH.exists() and BRAIN_REF_PATH.exists()):
        return None
    cfg = json.loads(BRAIN_CONFIG_PATH.read_text())
    ref = np.load(BRAIN_REF_PATH).astype(np.float32)
    return make_interpreter(BRAIN_PATH), cfg, ref


def run(interp, frame):
    x = np.repeat(frame[..., None], 3, -1).astype(np.float32)[None]
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    return [interp.get_tensor(o["index"])[0] for o in interp.get_output_details()]


def predict(interp, frame):
    return run(interp, frame)[0]


def brain_check(brain, frame):
    # Same rules as brain_check() in the notebook: guard first, then a tiered, non-diagnostic message
    interp, cfg, ref = brain
    outs = run(interp, frame)
    n = len(cfg["classes"])
    probs = next(o for o in outs if o.shape[-1] == n)
    unit = next(o for o in outs if o.shape[-1] != n)
    if 1 - float((ref @ unit).max()) > cfg["ood_limit"]:
        return "not_assessed", None, ("Could not assess: this picture looks unlike the pictures the brain check "
                                      "learned from. Follow routine care and the usual anomaly scan.")
    ni = cfg["classes"].index(cfg["normal"])
    pa = 1 - float(probs[ni])
    top = [cfg["classes"][i] for i in np.argsort(-probs) if i != ni][:2]
    name = lambda c: cfg["names"].get(c, c)
    if pa >= cfg["tau_refer"]:
        alt = f" (or {name(top[1])})" if len(top) > 1 else ""
        return "refer", pa, f"Possible signs of {name(top[0])}{alt}. Refer to a specialist for a detailed scan."
    if pa >= cfg["tau_flag"]:
        return "consult", pa, (f"Possible brain abnormality (most like {name(top[0])}). "
                               "Book the client for a consultation and detailed scan.")
    return "no_flag", pa, ("The brain check did not flag a concern. This does not rule out a defect; "
                           "routine anomaly scanning still applies.")



# ==============================================================================================
# Interface: step-by-step screens with Next / Back, dark "night console" look.
# Works with older Streamlit versions. PLACEHOLDER clinical thresholds: to be confirmed by a
# clinician against Kenyan MOH guidelines.
# ==============================================================================================
import base64
import html
import os
import random
import re
import time

DANGER_SIGNS = ["Vaginal bleeding", "Fits or convulsions", "Severe headache or blurred vision",
                "Severe abdominal pain", "Difficulty breathing", "Leaking fluid (waters broken)",
                "Reduced or no fetal movements", "Swelling of face or hands"]
ACTIONS = ["Emergency referral now", "Refer today", "Refer to a specialist (urgent anomaly scan)",
           "Book a consultation and detailed scan", "Rescan before the client leaves", "Routine care"]
REFERRAL_ACTIONS = ACTIONS[:4]
VIEWS = ["biparietal", "abdominal", "heart", "spine", "femur"]
SHORT = {"biparietal": "Head", "abdominal": "Tummy", "heart": "Heart", "spine": "Spine",
         "femur": "Thigh bone", "no_plane": "No key view"}
TIER_RANK = {"refer": 3, "consult": 2, "no_flag": 1, "not_assessed": 0}
TIER_KIND = {"refer": "red", "consult": "amber", "no_flag": "green", "not_assessed": "blue"}
TIER_TITLE = {"refer": "Refer to a specialist", "consult": "Book a consultation", "no_flag": "No concern flagged",
              "not_assessed": "Could not assess"}
STEPS = ["Examination", "Scan", "Decision", "Summary"]
FRAMES_PER_SWEEP = 24                     # pictures shown during the simulated sweep
DEMO_DIR = HERE / "demo_sweep"            # optional: put your own demo pictures here (played in name order)

LOGO = ('<svg width="46" height="46" viewBox="0 0 160 160" aria-label="Taswira logo"><circle cx="80" cy="80" r="78" fill="#F2A541"/>'
        '<path d="M64 34 Q80 27 96 34" fill="none" stroke="#13212E" stroke-width="6" stroke-linecap="round"/>'
        '<path d="M53 49 Q80 37 107 49" fill="none" stroke="#13212E" stroke-width="6" stroke-linecap="round"/>'
        '<path d="M42 70 Q80 54 118 70 L124 86 H36 Z" fill="#13212E"/><rect x="66" y="84" width="28" height="50" rx="12" fill="#13212E"/>'
        '<path d="M80 132 C80 144 94 148 106 143" fill="none" stroke="#13212E" stroke-width="5" stroke-linecap="round"/></svg>')

CSS = """
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: radial-gradient(1200px 700px at 75% -10%, #123a5c 0%, #071321 55%, #040a12 100%) fixed !important; color:#E6F1FF;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {background:#06101c !important; border-right:1px solid #ffffff14;}
.block-container {padding-top:1.2rem; max-width:1300px;}
label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, .stCheckbox label, .stCheckbox label p,
.stRadio label p, [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"], .stSelectbox label p {color:#E6F1FF !important; opacity:1 !important;}
.stCheckbox label p {font-size:1.05rem !important;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {color:#9fb8d6 !important;}
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="textarea"] {background:#0b1d30 !important; color:#E6F1FF !important; border-color:#2a4a6a !important;}
[data-testid="stExpander"] details, [data-testid="stFileUploader"] section {background:#0b1d30 !important; border-color:#2a4a6a !important; color:#E6F1FF !important;}
[data-testid="stMetric"] {background:linear-gradient(180deg,#0d1f33cc,#081627cc); border:1px solid #2dd4ff26; border-radius:16px; padding:14px 16px;}
[data-testid="stMetricValue"] {color:#E6F1FF;}
.stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
  background:linear-gradient(90deg,#f2a541,#ffcf7a); color:#1a1206 !important; border:0; border-radius:12px; font-weight:700; padding:.55rem 1.4rem;
  box-shadow:0 0 18px #f2a54155;}
.stButton button p, .stDownloadButton button p, [data-testid="stFormSubmitButton"] button p {color:#1a1206 !important; font-weight:700;}
.tw-head {display:flex; align-items:center; justify-content:space-between; gap:16px;}
.tw-brand {display:flex; align-items:center; gap:14px;}
.tw-brand b {font-size:1.8rem; color:#fff;} .tw-brand small {display:block; color:#7fa3c7; letter-spacing:.16em; font-size:.75rem;}
.tw-pill {display:inline-block; padding:6px 13px; border-radius:999px; font-size:.82rem; border:1px solid #2dd4ff55; color:#bdf3ff; background:#0b2236; margin-left:8px;}
.tw-pill.amber {border-color:#f2a54177; color:#ffd79a; background:#2a1d08;}
.tw-stepper {display:flex; align-items:center; gap:0; margin:18px 0 22px;}
.tw-sdot {display:flex; align-items:center; gap:10px; color:#7fa3c7; font-weight:600; white-space:nowrap;}
.tw-sdot i {width:34px; height:34px; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-style:normal;
  border:2px solid #2a4a6a; color:#7fa3c7;}
.tw-sdot.done i {background:#14b8a6; border-color:#14b8a6; color:#04121d; box-shadow:0 0 12px #14b8a6aa;} .tw-sdot.done {color:#9fe8dc;}
.tw-sdot.on i {border-color:#2dd4ff; color:#2dd4ff; box-shadow:0 0 14px #2dd4ff88;} .tw-sdot.on {color:#fff;}
.tw-sline {flex:1; height:2px; margin:0 14px; background:#2a4a6a;} .tw-sline.done {background:#14b8a6;}
.tw-title {font-size:1.7rem; font-weight:700; color:#fff; margin:0 0 4px;} .tw-sub {color:#9fb8d6; margin-bottom:16px;}
.tw-card {background:linear-gradient(180deg,#0d1f33cc,#081627cc); border:1px solid #2dd4ff26; border-radius:18px; padding:18px 20px; margin-bottom:14px; box-shadow:0 10px 40px #0006;}
.tw-card h4, .tw-label {font-size:.78rem; letter-spacing:.16em; color:#7fd8f0; text-transform:uppercase; margin:0 0 12px;}
.tw-monitor {height:380px; border-radius:18px; background:#02070c; border:1px solid #2dd4ff40; position:relative; overflow:hidden;
  display:flex; align-items:center; justify-content:center; box-shadow:inset 0 0 60px #2dd4ff1a;}
.tw-monitor svg {opacity:.55;} .tw-monitor p {position:absolute; bottom:18px; width:100%; text-align:center; color:#7fd8f0 !important; letter-spacing:.12em; font-size:.85rem;}
.tw-rings {display:flex; gap:8px; justify-content:space-between;}
.tw-ring {flex:1; min-width:0; text-align:center; font-size:.82rem; color:#9fb8d6;}
.tw-ring svg {display:block; margin:0 auto 6px; width:100%; max-width:74px; height:auto;}
.tw-banner {display:flex; gap:16px; align-items:center; padding:16px 18px; border-radius:16px; margin:10px 0;}
.tw-banner b {font-size:1.3rem;} .tw-banner span {font-size:.95rem;}
.tw-banner.amber {background:linear-gradient(90deg,#3a2606,#1a1206); border:1px solid #f2a54188;} .tw-banner.amber b {color:#ffc76b;} .tw-banner.amber span {color:#e9d6b5;}
.tw-banner.red {background:linear-gradient(90deg,#4a0f16,#2a0b10); border:1px solid #ff5a6a88; box-shadow:0 0 30px #ff3b4f33;} .tw-banner.red b {color:#ff9aa4;} .tw-banner.red span {color:#f3c9cd;}
.tw-banner.green {background:linear-gradient(90deg,#06332c,#071a17); border:1px solid #14b8a688;} .tw-banner.green b {color:#5eead4;} .tw-banner.green span {color:#c8efe7;}
.tw-banner.blue {background:linear-gradient(90deg,#0e2a44,#08182a); border:1px solid #2dd4ff55;} .tw-banner.blue b {color:#7fe3ff;} .tw-banner.blue span {color:#cfe6f7;}
.tw-gal {display:grid; grid-template-columns:repeat(5,1fr); gap:10px;}
.tw-tile {position:relative; border-radius:12px; overflow:hidden; border:1px solid #ffffff18;}
.tw-tile img {display:block; width:100%; height:120px; object-fit:cover;}
.tw-tag {position:absolute; left:6px; bottom:6px; font-size:.74rem; padding:2px 8px; border-radius:999px; background:#04121dcc; border:1px solid #2dd4ff66; color:#bdf3ff;}
.tw-tag.red {border-color:#ff5a6a; color:#ffb3bb;}
.tw-empty {height:120px; border-radius:12px; border:1px dashed #f2a54177; display:flex; align-items:center; justify-content:center; color:#ffd79a; font-size:.85rem; text-align:center;}
.tw-tl {position:relative; padding-left:26px;}
.tw-tl:before {content:""; position:absolute; left:8px; top:6px; bottom:6px; width:2px; background:linear-gradient(#ff5a6a,#f2a541,#2a4a6a);}
.tw-ev {position:relative; margin-bottom:16px;} .tw-ev b {display:block; color:#fff;} .tw-ev span {color:#9fb8d6; font-size:.9rem;}
.tw-ev:before {content:""; position:absolute; left:-23px; top:5px; width:12px; height:12px; border-radius:50%; background:#f2a541; box-shadow:0 0 10px #f2a541;}
.tw-ev.r:before {background:#ff5a6a; box-shadow:0 0 10px #ff5a6a;} .tw-ev.g:before {background:#14b8a6; box-shadow:0 0 10px #14b8a6;}
.tw-row {display:flex; gap:16px; padding:10px 0; border-bottom:1px solid #ffffff12;} .tw-row span {width:190px; color:#7fa3c7;} .tw-row div {flex:1;}
.tw-foot {display:flex; justify-content:space-between; color:#5f7f9f; font-size:.8rem; margin-top:26px;}
</style>
"""

MONITOR_IDLE = ('<div class="tw-monitor"><svg width="280" height="230" viewBox="0 0 220 180"><path d="M110 10 L20 160 Q110 190 200 160 Z" '
                'fill="none" stroke="#2dd4ff" stroke-width="2" stroke-dasharray="6 6"/><circle cx="110" cy="10" r="6" fill="#2dd4ff"/></svg>'
                '<p>PROBE READY · PRESS “START SWEEP”</p></div>')


def md(x):
    st.markdown(x, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def banner(kind, title, text="", target=None):
    icon = {"red": "#ff5a6a", "amber": "#f2a541", "green": "#14b8a6", "blue": "#2dd4ff"}[kind]
    mark = "✓" if kind == "green" else ("i" if kind == "blue" else "!")
    h = (f'<div class="tw-banner {kind}"><svg width="38" height="38" viewBox="0 0 38 38"><circle cx="19" cy="19" r="17" fill="{icon}"/>'
         f'<text x="19" y="25" text-anchor="middle" font-size="18" font-weight="700" fill="#06101c">{mark}</text></svg>'
         f'<div><b>{esc(title)}</b><br><span>{esc(text)}</span></div></div>')
    (target or st).markdown(h, unsafe_allow_html=True)


def ring(label, pct, ok):
    c = "#2dd4ff" if ok else "#f2a541"
    txt = f"{pct:.0f}" if ok else "–"
    dash = max(pct, 6) * 1.95
    return (f'<div class="tw-ring"><svg width="74" height="74" viewBox="0 0 74 74"><circle cx="37" cy="37" r="31" fill="none" stroke="#14304a" stroke-width="7"/>'
            f'<circle cx="37" cy="37" r="31" fill="none" stroke="{c}" stroke-width="7" stroke-linecap="round" stroke-dasharray="{dash:.0f} 300" '
            f'transform="rotate(-90 37 37)" style="filter:drop-shadow(0 0 5px {c})"/>'
            f'<text x="37" y="43" text-anchor="middle" font-size="17" fill="#E6F1FF">{txt}</text></svg>{esc(label)}</div>')


def rings_html(found):
    return ('<div class="tw-card"><h4>Views captured</h4><div class="tw-rings">' +
            "".join(ring(SHORT[v], float(found[v]["conf"]) * 100 if v in found else 0, v in found) for v in VIEWS) + '</div></div>')


def thumb(img, w=320):
    im = img.convert("RGB"); im.thumbnail((w, w))
    b = io.BytesIO(); im.save(b, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def rerun():
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun"))()


def go(step):
    st.session_state["step"] = step
    rerun()


# --- Analysis ---------------------------------------------------------------------------------
def analyse(img, name):
    frame = preprocess(img)
    probs = predict(interp, frame)
    p_view = float(1 - probs[NO_PLANE])
    best = int(np.argmax(np.where(np.arange(len(probs)) == NO_PLANE, -1, probs)))
    r = {"name": name, "probs": probs, "p_view": p_view, "conf": float(probs[best]),
         "found": p_view >= tau, "view": CLASS_NAMES[best] if p_view >= tau else "no_plane"}
    if brain is not None and r["view"] == "biparietal":
        r["tier"], r["pa"], r["msg"] = brain_check(brain, frame)
    return r


def summarise(results):
    found = {}
    for r in results:
        if r["found"] and (r["view"] not in found or r["conf"] > found[r["view"]]["conf"]):
            found[r["view"]] = r
    need = [v for v in VIEWS if PLAIN[v] in st.session_state.get("required_views", [PLAIN[v] for v in VIEWS])]
    missing = [v for v in need if v not in found]
    flagged = [r for r in results if r.get("tier")]
    worst = max(flagged, key=lambda r: TIER_RANK[r["tier"]]) if flagged else None
    return found, missing, worst


@st.cache_data(show_spinner=False)
def find_demo_sweeps(root):
    # Folders next to app.py whose name starts with "demo_sweep" (e.g. demo_sweep_1_rescan): played in name order.
    exts = {".png", ".jpg", ".jpeg"}
    out = {}
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and d.name.lower().startswith("demo_sweep"):
            files = sorted(str(p) for p in d.iterdir() if p.suffix.lower() in exts)
            if files:
                label = d.name[len("demo_sweep"):].strip("_- ").replace("_", " ") or "demo sweep"
                out[label] = files
    return out


@st.cache_data(show_spinner=False)
def find_sweeps(root):
    # NatalIA frames grouped by recording (used when there are no demo_sweep folders).
    exts = {".png", ".jpg", ".jpeg"}
    groups = {}
    for d, _, fs in os.walk(root):
        if "natalia" not in d.lower():
            continue
        for f in fs:
            if os.path.splitext(f)[1].lower() in exts:
                key = (d, re.sub(r"[_\-\s]*\d+$", "", os.path.splitext(f)[0]))
                groups.setdefault(key, []).append(os.path.join(d, f))
    return [sorted(v) for v in groups.values() if len(v) >= 20]


# --- Page -------------------------------------------------------------------------------------
st.set_page_config(page_title="Taswira AI", page_icon="🩺", layout="wide")
md(CSS)
md(f'<div class="tw-head"><div class="tw-brand">{LOGO}<div><b>Taswira AI</b><small>SCAN ASSIST</small></div></div>'
   '<div><span class="tw-pill">Model 7 MB · on-device · offline</span><span class="tw-pill amber">Study prototype</span></div></div>')

if not MODEL_PATH.exists():
    banner("red", "Model file not found", f"Run the notebook (section 6) and keep its 'outputs' folder next to app.py. Looking for: {MODEL_PATH}")
    st.stop()
tau = 0.5
if CONFIG_PATH.exists():
    tau = float(json.loads(CONFIG_PATH.read_text()).get("tau", tau))
interp = load_model()
brain = load_brain()

S = st.session_state
for k, v in {"step": 1, "visit": 1, "sweeps_done": 0, "results": [], "pictures": {},
             "exam": {"signs": [], "sys": 115, "dia": 75, "fhr": 140, "temp": 36.8}}.items():
    if k not in S:
        S[k] = v
V = S["visit"]

with st.sidebar:
    md('<div class="tw-label">This visit</div>')
    client = st.text_input("Client reference (code, not name)", "DEMO-001", key=f"client_{V}")
    ga = st.number_input("Weeks of pregnancy", 4, 44, 20, key=f"ga_{V}")
    midwife = st.text_input("Midwife", "", key=f"mw_{V}")
    md('<div class="tw-label" style="margin-top:16px">Settings</div>')
    S["required_views"] = st.multiselect("Views needed for a good scan", [PLAIN[v] for v in VIEWS],
                                         default=[PLAIN[v] for v in VIEWS], key="required")
    with st.expander("About this tool"):
        st.write("Study prototype for a Moringa School capstone (Taswira Analytics, Group 11). Not a medical device: "
                 "it does not diagnose, and a trained clinician makes every decision.")
        st.write("The sweep is simulated from saved recordings (NatalIA dummy sweeps, or pictures you put in a "
                 "'demo_sweep' folder next to app.py).")
        st.write("Brain check: experimental; its data is not verified and it flagged 1 in 4 normal African head pictures.")
    st.caption("Demo: use made-up details. Nothing is saved or sent.")

# --- Examination results (kept between screens) ----------------------------------------------
E = S["exam"]
emergency, urgent = list(E["signs"]), []
if E["sys"] >= 160 or E["dia"] >= 110:
    emergency.append(f"Severe high blood pressure ({E['sys']}/{E['dia']})")
elif E["sys"] >= 140 or E["dia"] >= 90:
    urgent.append(f"High blood pressure ({E['sys']}/{E['dia']})")
if E["fhr"] == 0:
    emergency.append("Baby's heartbeat not heard")
elif E["fhr"] < 110 or E["fhr"] > 160:
    urgent.append(f"Baby's heart rate outside 110–160 ({E['fhr']}/min)")
if E["temp"] >= 38.0:
    urgent.append(f"Fever ({E['temp']:.1f} °C)")
exam_level = "emergency" if emergency else ("urgent" if urgent else "none")
exam_reasons = emergency + urgent

# --- Stepper ----------------------------------------------------------------------------------
step = S["step"]
parts = []
for i, name in enumerate(STEPS, 1):
    cls = "done" if i < step else ("on" if i == step else "")
    parts.append(f'<div class="tw-sdot {cls}"><i>{"✓" if i < step else i}</i>{name}</div>')
    if i < len(STEPS):
        parts.append(f'<div class="tw-sline {"done" if i < step else ""}"></div>')
md('<div class="tw-stepper">' + "".join(parts) + '</div>')

# ============================== Screen 1: examination =========================================
if step == 1:
    md('<div class="tw-title">Midwife examination</div><div class="tw-sub">Always first. Your examination overrides anything the scan shows.</div>')
    left, right = st.columns(2)
    with left:
        md('<div class="tw-label">Danger signs today</div>')
        signs = [d for d in DANGER_SIGNS if st.checkbox(d, value=d in E["signs"], key=f"ds_{d}_{V}")]
    with right:
        md('<div class="tw-label">Readings</div>')
        c1, c2 = st.columns(2)
        sys_bp = int(c1.number_input("Blood pressure, systolic (mmHg)", 50, 260, int(E["sys"]), step=1, key=f"sys_{V}"))
        dia_bp = int(c2.number_input("Blood pressure, diastolic (mmHg)", 20, 160, int(E["dia"]), step=1, key=f"dia_{V}"))
        c3, c4 = st.columns(2)
        fhr = int(c3.number_input("Baby's heart rate (beats/min, 0 = not heard)", 0, 240, int(E["fhr"]), step=1, key=f"fhr_{V}"))
        temp = round(float(c4.number_input("Temperature (°C)", 33.0, 43.0, float(E["temp"]), step=0.1, format="%.1f", key=f"temp_{V}")), 1)
    if (signs, sys_bp, dia_bp, fhr, temp) != (E["signs"], E["sys"], E["dia"], E["fhr"], E["temp"]):
        S["exam"] = {"signs": signs, "sys": sys_bp, "dia": dia_bp, "fhr": fhr, "temp": temp}
        rerun()
    if exam_level == "emergency":
        banner("red", "Emergency: refer now", "; ".join(exam_reasons) + ". Arrange transfer first.")
        banner("blue", "Recommended", "While transport is arranged, a quick sweep (next screen) helps the hospital prepare. Never delay transfer to scan.")
    elif exam_level == "urgent":
        banner("amber", "Refer today for review", "; ".join(exam_reasons) + ".")
        banner("blue", "Recommended", "Scan before the client leaves (next screen) and attach the pictures to the referral.")
    else:
        banner("green", "No danger signs", "Readings are normal. Continue to the scan.")

# ============================== Screen 2: scan ================================================
elif step == 2:
    md('<div class="tw-title">Scan</div><div class="tw-sub">Sweep the probe slowly across the belly. Taswira checks every picture as it arrives.</div>')
    mon_col, live_col = st.columns([1.35, 1])
    with mon_col:
        monitor = st.empty()
        status = st.empty()
        bar = st.empty()
        demos = find_demo_sweeps(str(HERE))
        b1, b2 = st.columns([1, 1.6])
        pick_demo = b2.selectbox("Recording to play", list(demos), key="demo_pick") if demos else None
        start = b1.button("▶  Start sweep" if not S["results"] else "↻  Sweep again")
    with live_col:
        live = st.empty()
        verdict = st.empty()

    sweeps = [] if pick_demo else find_sweeps(str(HERE))
    if start:
        if not (pick_demo or sweeps):
            banner("amber", "No saved sweeps found", "Put a folder of pictures called 'demo_sweep' next to app.py, "
                   "or keep the NatalIA folder there. You can also upload pictures below.", target=status)
        else:
            S["sweeps_done"] += 1
            files = demos[pick_demo] if pick_demo else random.Random(S["sweeps_done"]).choice(sweeps)
            idx = np.linspace(0, len(files) - 1, min(FRAMES_PER_SWEEP, len(files))).astype(int)
            results, pictures = [], {}
            for k, i in enumerate(idx, 1):
                img = Image.open(files[i])
                r = analyse(img, Path(files[i]).name)
                results.append(r)
                pictures[r["name"]] = img.copy()
                tag = f"{SHORT[r['view']]} · {r['conf']:.0%}" if r["found"] else "searching…"
                monitor.image(img, caption=f"Live sweep · picture {k} of {len(idx)} · {tag}", width=540)
                bar.progress(int(100 * k / len(idx)))
                found, _, _ = summarise(results)
                live.markdown(rings_html(found), unsafe_allow_html=True)
                time.sleep(0.08)
            S["results"], S["pictures"] = results, pictures
            rerun()

    results = S["results"]
    found, missing, worst = summarise(results)
    if not results:
        monitor.markdown(MONITOR_IDLE, unsafe_allow_html=True)
        live.markdown(rings_html({}), unsafe_allow_html=True)
        banner("blue", "Waiting for a sweep", "Press Start sweep. Views light up as they are found.", target=verdict)
    else:
        best_pic = max(results, key=lambda r: r["conf"] if r["found"] else -1)
        monitor.image(S["pictures"][best_pic["name"]], caption=f"Sweep complete · {len(results)} pictures checked", width=540)
        live.markdown(rings_html(found), unsafe_allow_html=True)
        if missing:
            banner("amber", "Please scan again", "Missing: " + ", ".join(SHORT[v] for v in missing) + " · try one slow vertical sweep", target=verdict)
        else:
            banner("green", "Good scan", "Every view needed for this visit was captured.", target=verdict)
        tiles = ""
        for v in VIEWS:
            if v in found:
                r = found[v]
                tag = f'<span class="tw-tag">{SHORT[v]} · {r["conf"]:.0%}</span>'
                tiles += f'<div class="tw-tile"><img src="{thumb(S["pictures"][r["name"]])}" alt="{SHORT[v]}">{tag}</div>'
            else:
                tiles += f'<div class="tw-empty">{SHORT[v]}<br>not found</div>'
        md(f'<div class="tw-card"><h4>Best picture of each view · kept for the specialist</h4><div class="tw-gal">{tiles}</div></div>')
        if worst:
            banner(TIER_KIND[worst["tier"]], "Brain check · experimental: " + TIER_TITLE[worst["tier"]], worst["msg"])

    with st.expander("Use your own pictures instead of the simulated sweep"):
        files = st.file_uploader("Pictures from a sweep", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                 key=f"up_{V}_{S['sweeps_done']}") or []
        if files and st.button("Check these pictures"):
            results, pictures = [], {}
            for f in files:
                img = Image.open(f)
                r = analyse(img, f.name)
                results.append(r); pictures[f.name] = img.copy()
            S["results"], S["pictures"] = results, pictures
            rerun()

# ============================== Screen 3: decision ============================================
found, missing, worst = summarise(S["results"])
scan = None
if S["results"]:
    scan = {"n": len(S["results"]), "found": [PLAIN[v] for v in found], "missing": [PLAIN[v] for v in missing],
            "tier": worst["tier"] if worst else None, "pa": worst.get("pa") if worst else None,
            "msg": worst["msg"] if worst else None, "brain_picture": worst["name"] if worst else None}
if exam_level == "emergency":
    action, source = "Emergency referral now", "midwife examination"
elif exam_level == "urgent":
    action, source = "Refer today", "midwife examination"
elif scan and scan["tier"] == "refer":
    action, source = "Refer to a specialist (urgent anomaly scan)", "brain check (experimental)"
elif scan and scan["tier"] == "consult":
    action, source = "Book a consultation and detailed scan", "brain check (experimental)"
elif scan and scan["missing"]:
    action, source = "Rescan before the client leaves", "missing views"
elif not scan:
    action, source = "Routine care", "examination only (no scan)"
else:
    action, source = "Routine care", "no concern found"
D = S.setdefault("decision", {"change": False, "new": None, "reason": ""})
decision_note = f"{action} (suggested by {source}; midwife agreed)"
final = action
if D["change"] and D["reason"].strip() and D["new"]:
    final = D["new"]
    decision_note = f"{final}. Midwife changed the suggested action ({action}, from {source}). Reason: {D['reason'].strip()}"
findings = {"BP": f"{E['sys']}/{E['dia']}", "Fetal heart rate": "not heard" if E["fhr"] == 0 else f"{E['fhr']}/min",
            "Temperature": f"{E['temp']:.1f} °C", "Danger signs": ", ".join(E["signs"]) or "none"}
attach = [(found[v]["name"], S["pictures"][found[v]["name"]]) for v in found]
if worst and worst["name"] not in [n for n, _ in attach]:
    attach.append((worst["name"], S["pictures"][worst["name"]]))


def build_note(title, facility, refer_to, urgency, notes):
    lines = [title, "=" * len(title), f"Date: {datetime.date.today().isoformat()}",
             f"From: {facility}" + (f", midwife {midwife}" if midwife else ""), f"To: {refer_to}",
             f"Client reference: {client}", f"Gestational age: {ga} weeks", f"Urgency: {urgency}", f"Action: {final}",
             "", "Midwife examination (decides first)"] + [f"- {k}: {v}" for k, v in findings.items()]
    if exam_reasons:
        lines.append("- Reasons: " + "; ".join(exam_reasons))
    lines += ["", "Ultrasound screening tool (study prototype)"]
    if scan:
        lines += [f"- Pictures checked: {scan['n']}", "- Views found: " + (", ".join(scan["found"]) or "none"),
                  "- Views missing: " + (", ".join(scan["missing"]) or "none")]
        if scan["msg"]:
            lines.append(f"- Brain check, experimental ({scan['brain_picture']}): {scan['msg']}")
    else:
        lines.append("- No scan done")
    lines.append("- Pictures attached: " + (", ".join(n for n, _ in attach) or "none"))
    lines += ["", "Decision: " + decision_note, "", "Clinical notes: " + (notes or "none"), "",
              "IMPORTANT: Taswira AI is a student study prototype (Moringa capstone), not a medical device.",
              "It does not diagnose. Any scan finding must be confirmed by a qualified specialist.",
              "", "Signature of referring health worker: ____________________"]
    return "\n".join(lines)


if step == 3:
    md('<div class="tw-title">Decision</div><div class="tw-sub">Taswira suggests an action. You decide.</div>')
    kind = "red" if final in ACTIONS[:2] else ("amber" if final in REFERRAL_ACTIONS or final.startswith("Rescan") else "green")
    banner(kind, final, f"Suggested from: {source}." if final == action else "Changed by the midwife.")
    why, act = st.columns([1, 1.4])
    with why:
        evs = "".join(f'<div class="tw-ev r"><b>{esc(x)}</b><span>Midwife examination</span></div>' for x in exam_reasons)
        if scan:
            evs += (f'<div class="tw-ev"><b>Sweep checked</b><span>{scan["n"]} pictures · found: '
                    f'{esc(", ".join(SHORT[v] for v in found) or "none")}</span></div>')
            if scan["tier"] in ("refer", "consult"):
                evs += f'<div class="tw-ev r"><b>Brain check flag (experimental)</b><span>{esc(TIER_TITLE[scan["tier"]])}</span></div>'
        if not evs:
            evs = '<div class="tw-ev g"><b>No concerns</b><span>Examination normal; no scan flags</span></div>'
        md(f'<div class="tw-card"><h4>Why this action</h4><div class="tw-tl">{evs}</div></div>')
    with act:
        change = st.checkbox("I want to change this action", value=D["change"], key=f"chg_{V}")
        new, reason = D["new"], D["reason"]
        if change:
            if exam_level == "emergency":
                banner("red", "Danger signs were found", "Lowering the action needs a clinician's agreement; the note will record it.")
            opts = [a for a in ACTIONS if a != action]
            new = st.selectbox("New action", opts, index=opts.index(D["new"]) if D["new"] in opts else 0, key=f"new_{V}")
            reason = st.text_input("Reason (required)", value=D["reason"], key=f"why_{V}")
            if not reason.strip():
                st.caption("Enter a reason to confirm the change.")
        if (change, new, reason) != (D["change"], D["new"], D["reason"]):
            S["decision"] = {"change": change, "new": new, "reason": reason}
            rerun()
        if final in REFERRAL_ACTIONS:
            urg = {"Emergency referral now": 0, "Refer today": 1, "Refer to a specialist (urgent anomaly scan)": 2}.get(final, 3)
            md('<div class="tw-label" style="margin-top:10px">Referral note</div>')
            with st.form(f"ref_{V}"):
                facility = st.text_input("Referring facility", "Example Health Centre")
                refer_to = st.text_input("Refer to", "County referral hospital" if urg < 2 else "County referral hospital, obstetric ultrasound")
                urgency = st.radio("Urgency", ["Now", "Today", "Within 1 week", "Within 2 weeks"], index=urg, horizontal=True)
                notes = st.text_area("Clinical notes", "")
                done = st.form_submit_button("Create referral note")
            if done:
                note = build_note("REFERRAL NOTE (STUDY DEMO)", facility, refer_to, urgency, notes)
                z = io.BytesIO()
                with zipfile.ZipFile(z, "w") as zf:
                    zf.writestr("referral_note.txt", note)
                    for k, (n, im) in enumerate(attach, 1):
                        b = io.BytesIO(); im.convert("RGB").save(b, format="JPEG", quality=92)
                        zf.writestr(f"scan_{k:02d}_{Path(n).stem}.jpg", b.getvalue())
                S["ref_note"], S["ref_zip"], S["ref_for"] = note, z.getvalue(), V
            if S.get("ref_for") == V:
                st.download_button("Download referral pack: note + pictures (.zip)", S["ref_zip"], file_name="taswira_referral_pack.zip",
                                   mime="application/zip")
                st.caption(f"{len(attach)} picture(s) attached. Any scan finding must be confirmed by a specialist.")
        elif final.startswith("Rescan"):
            banner("amber", "Go back and sweep again", "Use Back to return to the scan screen.")
        else:
            banner("green", "Routine care", "Give the guidance card and book the next antenatal visit.")

# ============================== Screen 4: summary =============================================
if step == 4:
    md('<div class="tw-title">Visit summary</div><div class="tw-sub">Check, download, then start the next visit.</div>')
    a, b, c = st.columns(3)
    a.metric("Examination", {"emergency": "Emergency", "urgent": "Refer today", "none": "No danger signs"}[exam_level])
    b.metric("Views found", f"{len(found)} of {len(VIEWS)}" if scan else "no scan")
    c.metric("Action", final)
    rows = [("Client", f"{client} · {ga} weeks" + (f" · midwife {midwife}" if midwife else "")),
            ("Examination", "; ".join(f"{k}: {v}" for k, v in findings.items())),
            ("Scan", (f'{scan["n"]} pictures; found: {", ".join(scan["found"]) or "none"}; missing: {", ".join(scan["missing"]) or "none"}')
             if scan else "no scan"),
            ("Brain check (experimental)", scan["msg"] if scan and scan["msg"] else "not run"),
            ("Decision", decision_note)]
    md('<div class="tw-card">' + "".join(f'<div class="tw-row"><span>{esc(k)}</span><div>{esc(v)}</div></div>' for k, v in rows) + '</div>')
    st.download_button("Download visit summary (.txt)", build_note("VISIT SUMMARY (STUDY DEMO)", "Example Health Centre", "n/a", "n/a", ""),
                       file_name="taswira_visit_summary.txt", mime="text/plain")

# --- Navigation -------------------------------------------------------------------------------
st.write("")
n1, n2, n3 = st.columns([1, 4, 1])
if step > 1 and n1.button("←  Back"):
    go(step - 1)
if step < 4:
    if n3.button("Next  →"):
        go(step + 1)
else:
    if n3.button("New visit"):
        for k in ("results", "pictures", "exam", "decision", "ref_note", "ref_zip", "ref_for"):
            S.pop(k, None)
        S["visit"] = V + 1
        go(1)
md('<div class="tw-foot"><span>Taswira Analytics · Group 11 · study prototype</span><span>Not a medical device · the midwife decides</span></div>')
