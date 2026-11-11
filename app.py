import datetime
import io
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



def referral_form(tier, pa, msg, view, image_name, img):
    # Referral note for a flagged case. Demo: use made-up details, never a real client's name or ID.
    st.subheader("Referral form")
    st.caption("Demo only: enter made-up details. Nothing is saved or sent; the note is created on this computer.")
    urgent = tier == "refer"
    if st.session_state.get("referral_for") != image_name:        # a new picture: clear the old note
        for k in ("referral_note", "referral_image"):
            st.session_state.pop(k, None)
        st.session_state["referral_for"] = image_name
    with st.form("referral"):
        c1, c2 = st.columns(2)
        facility = c1.text_input("Referring facility", "Example Health Centre")
        midwife = c2.text_input("Referring midwife", "")
        client = c1.text_input("Client reference (code, not name)", "DEMO-001")
        ga = c2.number_input("Gestational age (weeks)", min_value=4, max_value=44, value=20)
        refer_to = c1.text_input("Refer to", "County referral hospital, obstetric ultrasound")
        urgency = c2.selectbox("Urgency", ["Urgent: within 1 week", "Soon: within 2 weeks", "Routine"],
                               index=0 if urgent else 1)
        danger = st.multiselect("Danger signs present today (if any: refer to emergency care now)",
                                ["Vaginal bleeding", "Severe headache or blurred vision", "Fits or convulsions",
                                 "Fever", "Severe abdominal pain", "Reduced or no fetal movements", "Leaking fluid"])
        notes = st.text_area("Clinical notes", "")
        done = st.form_submit_button("Create referral note")
    if done:
        today = datetime.date.today().isoformat()
        lines = [
            "REFERRAL NOTE: FETAL ULTRASOUND SCREENING (STUDY DEMO)",
            "=" * 54,
            f"Date: {today}",
            f"From: {facility}" + (f", midwife {midwife}" if midwife else ""),
            f"To: {refer_to}",
            f"Client reference: {client}",
            f"Gestational age: {ga} weeks",
            f"Urgency: {urgency}",
            "",
            "Reason for referral",
            f"- Screening tool result: {msg}",
            f"- Score: {pa:.2f} (flag line and refer line set by the study model)" if pa is not None else "- Score: n/a",
            f"- View in the picture: {view}",
            f"- Picture file: {image_name}",
            "",
            "Danger signs today: " + (", ".join(danger) if danger else "none reported"),
        ]
        if danger:
            lines.append("  -> Danger signs present: send to emergency care now; do not wait for this referral.")
        lines += ["", "Clinical notes: " + (notes or "none"), "",
                  "Requested: specialist detailed anomaly scan and counselling.",
                  "",
                  "IMPORTANT: produced by Taswira AI, a student study prototype (Moringa capstone).",
                  "It is not a medical device and does not diagnose. The finding must be confirmed",
                  "by a qualified specialist before anything is said to the client about a defect.",
                  "", "Signature of referring health worker: ____________________"]
        st.session_state["referral_note"] = "\n".join(lines)
        buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=92)
        st.session_state["referral_image"] = buf.getvalue()
    if "referral_note" in st.session_state:
        st.text(st.session_state["referral_note"])
        d1, d2 = st.columns(2)
        d1.download_button("Download referral note (.txt)", st.session_state["referral_note"],
                           file_name="taswira_referral_note.txt", mime="text/plain")
        d2.download_button("Download the flagged picture (.jpg)", st.session_state["referral_image"],
                           file_name="taswira_flagged_picture.jpg", mime="image/jpeg")

st.header("Taswira AI")
st.write("Fetal AI ultrasound image analysis and prediction tool to support resource-limited midwives")
st.warning("Study prototype for a Moringa capstone. Not a medical device: it does not diagnose, "
           "and a trained clinician makes every decision.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}. Run the notebook (section 6) and keep its "
             "'outputs' folder next to app.py.")
    st.stop()

tau = 0.5
if CONFIG_PATH.exists():
    tau = float(json.loads(CONFIG_PATH.read_text()).get("tau", tau))
interp = load_model()

st.subheader("Review an ultrasound frame")
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = preprocess(img)
    probs = predict(interp, frame)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Uploaded image")
    col2.image(frame, caption="What the model sees (cropped and cleaned)")

    p_view = 1 - probs[NO_PLANE]
    best = int(np.argmax(np.where(np.arange(len(probs)) == NO_PLANE, -1, probs)))
    if p_view >= tau:
        st.success(f"Key view found: **{PLAIN[CLASS_NAMES[best]]}** "
                   f"(confidence {probs[best]:.0%}). Keep this picture for the specialist.")
    else:
        st.info(f"No key view found in this picture (chance of a key view {p_view:.0%}). "
                "If no picture in the sweep shows the view, scan again.")

    st.write("Model scores for each view")
    for c, p in zip(CLASS_NAMES, probs):          # simple bars that work in any Streamlit version
        st.write(f"{PLAIN[c]}: {float(p):.0%}")
        st.progress(int(round(float(p) * 100)))
    st.caption("Trained mainly on dummy-belly sweeps plus real Spanish hospital images. "
               "On real African scans it named the right view about 8 times in 10.")

    # --- Brain check (exploratory study demo) on head pictures ---
    st.subheader("Brain check (study demo)")
    brain = load_brain()
    is_head = p_view >= tau and CLASS_NAMES[best] == "biparietal"
    if brain is None:
        st.write("Brain check not available: run notebook section 8.2a and keep its files in the 'outputs' folder.")
    else:
        if not is_head:
            st.write("The brain check runs on head pictures. This picture was not recognised as a head view.")
        if is_head or st.checkbox("Run the brain check on this picture anyway (demo)", key="brain_anyway"):
            tier, pa, msg = brain_check(brain, frame)
            show = {"refer": st.error, "consult": st.warning, "no_flag": st.success, "not_assessed": st.info}[tier]
            show(msg + (f" (score {pa:.2f})" if pa is not None else ""))
            if tier in ("refer", "consult"):
                referral_form(tier, pa, msg, PLAIN[CLASS_NAMES[best]] if p_view >= tau else "not recognised",
                              uploaded_file.name, img)
        st.caption("Exploratory: trained on a public dataset whose source is not verified. In tests it raised false "
                   "alarms on 1 in 4 normal African head pictures. It never gives a diagnosis; any flag goes to a specialist.")
