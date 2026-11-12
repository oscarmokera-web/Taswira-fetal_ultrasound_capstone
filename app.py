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



# --- Midwife examination: always comes first and overrides the scan --------------------------
# PLACEHOLDER thresholds for the demo: to be confirmed by a clinician against Kenyan MOH guidelines.
DANGER_SIGNS = ["Vaginal bleeding", "Fits or convulsions", "Severe headache or blurred vision",
                "Severe abdominal pain", "Difficulty breathing", "Leaking fluid (waters broken)",
                "Reduced or no fetal movements", "Swelling of face or hands"]
ACTIONS = ["Emergency referral now", "Refer today", "Refer to a specialist (urgent anomaly scan)",
           "Book a consultation and detailed scan", "Rescan before the client leaves", "Routine care"]
REFERRAL_ACTIONS = ACTIONS[:4]


def examination():
    st.subheader("1. Midwife examination (always first)")
    st.caption("Your examination overrides anything the scan shows. Thresholds here are demo placeholders, "
               "to be confirmed by a clinician against Kenyan MOH guidelines.")
    signs = [d for d in DANGER_SIGNS if st.checkbox(d, key=f"ds_{d}")]
    c1, c2, c3, c4 = st.columns(4)
    sys_bp = c1.number_input("BP systolic", 60, 250, 115)
    dia_bp = c2.number_input("BP diastolic", 30, 160, 75)
    fhr = c3.number_input("Fetal heart rate (0 = not heard)", 0, 240, 140)
    temp = c4.number_input("Temperature (°C)", 34.0, 42.0, 36.8, step=0.1)
    emergency = list(signs)
    urgent = []
    if sys_bp >= 160 or dia_bp >= 110:
        emergency.append(f"Severe high blood pressure ({sys_bp}/{dia_bp})")
    elif sys_bp >= 140 or dia_bp >= 90:
        urgent.append(f"High blood pressure ({sys_bp}/{dia_bp})")
    if fhr == 0:
        emergency.append("Fetal heart not heard")
    elif fhr < 110 or fhr > 160:
        urgent.append(f"Fetal heart rate outside 110–160 ({fhr}/min)")
    if temp >= 38.0:
        urgent.append(f"Fever ({temp:.1f} °C)")
    findings = {"BP": f"{sys_bp}/{dia_bp}", "Fetal heart rate": "not heard" if fhr == 0 else f"{fhr}/min",
                "Temperature": f"{temp:.1f} °C", "Danger signs": ", ".join(signs) or "none"}
    if emergency:
        st.error("**Emergency: refer now.** " + "; ".join(emergency + urgent) + ". The scan result below does not change this.")
        st.info("**Recommended:** while transport is being arranged, do a quick sweep (step 2) and attach the pictures "
                "to the referral form. They help the hospital prepare (baby's heartbeat, position and key views). "
                "Never delay transfer to scan.")
        return "emergency", emergency + urgent, findings
    if urgent:
        st.warning("**Refer today for review.** " + "; ".join(urgent) + ".")
        st.info("**Recommended:** scan before the client leaves (step 2) and attach the pictures to the referral form, "
                "so the hospital can prepare for her admission.")
        return "urgent", urgent, findings
    st.success("No danger signs or abnormal readings. Continue with the scan.")
    return "none", [], findings


def suggested_action(exam_level, scan):
    if exam_level == "emergency":
        return "Emergency referral now", "midwife examination"
    if exam_level == "urgent":
        return "Refer today", "midwife examination"
    if scan and scan.get("tier") == "refer":
        return "Refer to a specialist (urgent anomaly scan)", "brain check (study demo)"
    if scan and scan.get("tier") == "consult":
        return "Book a consultation and detailed scan", "brain check (study demo)"
    if scan and not scan.get("view_found"):
        return "Rescan before the client leaves", "plane finder"
    return "Routine care", "no concern found"


def referral_form(action, exam_reasons, findings, scan, decision_note, image_name, img):
    # Referral note. Demo: use made-up details, never a real client's name or ID.
    st.subheader("Referral form")
    st.caption("Demo only: enter made-up details. Nothing is saved or sent; the note is created on this computer.")
    key = f"{image_name}|{action}"
    if st.session_state.get("referral_for") != key:          # a new case: clear the old note
        for k in ("referral_note", "referral_image"):
            st.session_state.pop(k, None)
        st.session_state["referral_for"] = key
    # Scan pictures for the hospital: the reviewed picture plus any others from the sweep
    extra = st.file_uploader("Attach more scan pictures for the hospital (optional)", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="extra_pics") or []
    pics = ([(image_name, img)] if img is not None else []) + [(f.name, Image.open(f)) for f in extra]
    if pics:
        st.write(f"{len(pics)} scan picture(s) will be attached to the referral.")
    else:
        st.warning("No scan pictures attached. Recommended: scan (step 2) and attach pictures so the hospital can "
                   "prepare, unless scanning would delay transfer.")
    urg = {"Emergency referral now": 0, "Refer today": 1, "Refer to a specialist (urgent anomaly scan)": 2}.get(action, 3)
    with st.form("referral"):
        c1, c2 = st.columns(2)
        facility = c1.text_input("Referring facility", "Example Health Centre")
        midwife = c2.text_input("Referring midwife", "")
        client = c1.text_input("Client reference (code, not name)", "DEMO-001")
        ga = c2.number_input("Gestational age (weeks)", min_value=4, max_value=44, value=20)
        refer_to = c1.text_input("Refer to", "County referral hospital" if urg < 2 else
                                 "County referral hospital, obstetric ultrasound")
        urgency = c2.selectbox("Urgency", ["Emergency: now", "Today", "Urgent: within 1 week", "Soon: within 2 weeks"],
                               index=urg)
        no_scan = "" if pics else st.text_input("Reason no scan pictures are attached", "")
        notes = st.text_area("Clinical notes", "")
        done = st.form_submit_button("Create referral note")
    if done:
        lines = ["REFERRAL NOTE: ANTENATAL VISIT WITH ULTRASOUND SCREENING (STUDY DEMO)", "=" * 68,
                 f"Date: {datetime.date.today().isoformat()}",
                 f"From: {facility}" + (f", midwife {midwife}" if midwife else ""),
                 f"To: {refer_to}", f"Client reference: {client}", f"Gestational age: {ga} weeks",
                 f"Urgency: {urgency}", f"Action: {action}", "",
                 "Midwife examination (decides first)"]
        lines += [f"- {k}: {v}" for k, v in findings.items()]
        if exam_reasons:
            lines.append("- Reasons: " + "; ".join(exam_reasons))
        lines += ["", "Ultrasound screening tool (study prototype)"]
        if scan:
            lines += [f"- View in the picture: {scan['view']}", f"- Picture file: {image_name}"]
            if scan.get("msg"):
                lines.append(f"- Brain check: {scan['msg']}" + (f" (score {scan['pa']:.2f})" if scan.get("pa") is not None else ""))
        else:
            lines.append("- No scan picture reviewed")
        lines.append("- Pictures attached for the receiving hospital: " +
                     (", ".join(n for n, _ in pics) if pics else "none" + (f" (reason: {no_scan})" if no_scan else "")))
        lines += ["", "Decision: " + decision_note, "", "Clinical notes: " + (notes or "none"), "",
                  "IMPORTANT: Taswira AI is a student study prototype (Moringa capstone), not a medical device.",
                  "It does not diagnose. Any scan finding must be confirmed by a qualified specialist",
                  "before anything is said to the client about a defect.",
                  "", "Signature of referring health worker: ____________________"]
        st.session_state["referral_note"] = "\n".join(lines)
        zbuf = io.BytesIO()                        # referral pack: the note plus every attached picture
        with zipfile.ZipFile(zbuf, "w") as z:
            z.writestr("referral_note.txt", st.session_state["referral_note"])
            for k, (n, im) in enumerate(pics, 1):
                b = io.BytesIO(); im.convert("RGB").save(b, format="JPEG", quality=92)
                z.writestr(f"scan_{k:02d}_{Path(n).stem}.jpg", b.getvalue())
        st.session_state["referral_image"] = zbuf.getvalue()
    if "referral_note" in st.session_state:
        st.text(st.session_state["referral_note"])
        d1, d2 = st.columns(2)
        d1.download_button("Download referral note (.txt)", st.session_state["referral_note"],
                           file_name="taswira_referral_note.txt", mime="text/plain")
        if "referral_image" in st.session_state:
            d2.download_button("Download referral pack: note + pictures (.zip)", st.session_state["referral_image"],
                               file_name="taswira_referral_pack.zip", mime="application/zip")


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

exam_level, exam_reasons, findings = examination()

st.subheader("2. Review an ultrasound frame")
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])
scan, img = None, None
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    frame = preprocess(img)
    probs = predict(interp, frame)

    col1, col2 = st.columns(2)
    col1.image(img, caption="Uploaded image")
    col2.image(frame, caption="What the model sees (cropped and cleaned)")

    p_view = 1 - probs[NO_PLANE]
    best = int(np.argmax(np.where(np.arange(len(probs)) == NO_PLANE, -1, probs)))
    scan = {"view_found": p_view >= tau, "view": PLAIN[CLASS_NAMES[best]] if p_view >= tau else "no key view"}
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
    st.write("**Brain check (study demo)**")
    brain = load_brain()
    is_head = p_view >= tau and CLASS_NAMES[best] == "biparietal"
    if brain is None:
        missing = [p.name for p in (BRAIN_PATH, BRAIN_CONFIG_PATH, BRAIN_REF_PATH) if not p.exists()]
        st.warning("Brain check not available. Missing from the 'outputs' folder: " + ", ".join(missing) +
                   ". Run the notebook from the top (it includes section 8.2a, 'Export the brain check for the app'), "
                   f"then refresh this page. Looking in: {BRAIN_PATH.parent}")
    else:
        if not is_head:
            st.write("The brain check runs on head pictures. This picture was not recognised as a head view.")
        if is_head or st.checkbox("Run the brain check on this picture anyway (demo)", key="brain_anyway"):
            tier, pa, msg = brain_check(brain, frame)
            show = {"refer": st.error, "consult": st.warning, "no_flag": st.success, "not_assessed": st.info}[tier]
            show(msg + (f" (score {pa:.2f})" if pa is not None else ""))
            scan.update({"tier": tier, "pa": pa, "msg": msg})
        st.caption("Exploratory: trained on a public dataset whose source is not verified. In tests it raised false "
                   "alarms on 1 in 4 normal African head pictures. It never gives a diagnosis; any flag goes to a specialist.")

# --- 3. Final decision: the midwife decides ---------------------------------------------------
st.subheader("3. Decision (the midwife decides)")
action, source = suggested_action(exam_level, scan)
st.write(f"Suggested action: **{action}** (based on: {source}).")
choice = st.radio("Your decision", ["Agree with the suggested action", "Change the action"], key="decision")
decision_note = f"{action} (suggested by {source}; midwife agreed)"
if choice == "Change the action":
    options = [a for a in ACTIONS if a != action]
    if exam_level == "emergency":
        st.error("Danger signs were found. Changing the action to anything other than emergency referral "
                 "needs a clinician's agreement, and the note will record this.")
    new_action = st.selectbox("New action", options, key="new_action")
    reason = st.text_input("Reason for changing (required)", key="override_reason")
    if not reason.strip():
        st.info("Enter a reason to confirm the change.")
    else:
        decision_note = f"{new_action}. Midwife changed the suggested action ({action}, from {source}). Reason: {reason.strip()}"
        action = new_action
        st.write(f"Final action: **{action}**")

if action in REFERRAL_ACTIONS:
    referral_form(action, exam_reasons, findings, scan, decision_note,
                  uploaded_file.name if uploaded_file is not None else "none", img)
elif action == "Rescan before the client leaves":
    st.info("Sweep again, then upload the new pictures.")
else:
    st.success("Routine care: give the guidance card for this visit and book the next antenatal visit.")
