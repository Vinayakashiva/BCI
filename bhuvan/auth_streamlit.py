"""
=============================================================================
BCI EEG AUTHENTICATION GATE — Streamlit UI
=============================================================================
A login screen that uses a person's EEG brainwave as a biometric.

Flow:
  1. User selects their claimed subject id and a trial from the chosen
     dataset split.
  2. The system extracts subject-id features and computes a match score
     against the enrolled subject model.
  3. If score >= threshold  ->  ACCESS GRANTED  ->  word prediction card
     appears (Hello / Helpme / Stop / Thankyou / Yes).
  4. If score  <  threshold  ->  ACCESS DENIED  ->  word prediction locked.

Run:
    streamlit run auth_streamlit.py

Requirements:
    pip install streamlit numpy scipy scikit-learn joblib plotly
=============================================================================
"""

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

from authentication import (
    FS,
    CLASS_NAMES,
    AUTH_DIR,
    SUBJECT_MODEL_PATH,
    WORD_MODEL_PATH,
    EER_PATH,
    discover_dataset,
    discover_mat_files,
    load_epoch,
    extract_subject_features,
    reject_artefact_trials,
)

# Colours
COL_GREEN = "#639922"
COL_RED   = "#E24B4A"
COL_BLUE  = "#378ADD"
COL_GRAY  = "#888780"


# =============================================================================
# HELPERS
# =============================================================================

def load_models():
    """Load saved authenticator + word predictor."""
    auth = None
    wp = None
    if SUBJECT_MODEL_PATH.exists():
        with open(SUBJECT_MODEL_PATH, "rb") as fh:
            auth = pickle.load(fh)["auth"]
    if WORD_MODEL_PATH.exists():
        with open(WORD_MODEL_PATH, "rb") as fh:
            wp = pickle.load(fh)
    return auth, wp


def load_eer():
    if EER_PATH.exists():
        with open(EER_PATH) as fh:
            return json.load(fh)
    return None


def result_card_html(res, word=None, true_word=None):
    if res["granted"]:
        color = COL_GREEN
        icon = "&#10003;"
        status = "ACCESS GRANTED"
    else:
        color = COL_RED
        icon = "&#10007;"
        status = "ACCESS DENIED"

    score = res["score"]
    thr = res["threshold"]
    conf = res["confidence"]

    # Score bar (map score to a 0-100 visual)
    max_abs = max(abs(score), abs(thr), 1.0) * 1.2
    bar = max(0.0, min(100.0, (score + max_abs) / (2 * max_abs) * 100))

    word_html = ""
    if res["granted"] and word is not None:
        word_html = f"""
        <div style="margin-top:16px;border-top:1px solid #ddd;padding-top:12px">
            <div style="font-size:12px;color:{COL_GRAY};text-transform:uppercase;
                        letter-spacing:1px">Predicted imagined word</div>
            <div style="font-size:34px;font-weight:500;color:{COL_BLUE};margin-top:4px">
                {word}
            </div>
            <div style="font-size:12px;color:{COL_GRAY}">
                True word: {true_word or '?'}
            </div>
        </div>
        """
    elif not res["granted"]:
        word_html = f"""
        <div style="margin-top:16px;border-top:1px solid #ddd;padding-top:12px;
                    color:{COL_GRAY};font-size:13px;font-style:italic">
            🔒 Word prediction locked — authentication failed.
        </div>
        """

    return f"""
    <div style="border:2px solid {color};border-radius:12px;padding:24px;
                text-align:center;background:{color}18">
        <div style="font-size:14px;color:{COL_GRAY};text-transform:uppercase;
                    letter-spacing:1px">Authentication result</div>
        <div style="font-size:30px;font-weight:500;color:{color};margin-top:6px">
            {icon} {status}
        </div>
        <div style="font-size:13px;color:{COL_GRAY};margin-top:4px">
            Claimed subject: <strong>{res['subject']}</strong>
        </div>
        <div style="margin:14px auto 0;max-width:260px">
            <div style="display:flex;justify-content:space-between;font-size:11px;
                        color:{COL_GRAY}">
                <span>Score {score:.3f}</span>
                <span>Threshold {thr:.3f}</span>
            </div>
            <div style="background:#eee;border-radius:4px;height:8px;margin-top:4px">
                <div style="background:{color};height:8px;border-radius:4px;
                            width:{bar:.1f}%"></div>
            </div>
        </div>
        <div style="font-size:12px;color:{COL_GRAY};margin-top:8px">
            Confidence proxy: {conf:.1f}%
        </div>
        {word_html}
    </div>
    """


def explain_html():
    return st.markdown("""
### How this works
Each person's EEG is unique — a **"brainprint"**. This system uses a single
trial to verify a claimed identity:

1. **CAR** spatial filtering removes common noise.
2. **Epoch trim** keeps the post-stimulus window.
3. **Hanning taper + 5-band bandpass** isolates theta/mu/beta/gamma rhythms.
4. **Per-channel log band-power + broadband log-variance** form the subject
   fingerprint.
5. A one-vs-rest **SVM** scores the trial against every enrolled subject.
6. If the score **≥ threshold** → **GRANTED** and the word decoder unlocks.
   Otherwise → **DENIED** and the word is locked.
""")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="BCI EEG Authentication Gate",
                       page_icon="🔐", layout="wide")

    st.markdown(
        "<h2 style='margin-bottom:2px'>🔐 BCI EEG Authentication Gate</h2>"
        "<p style='color:#73726c;font-size:13px;margin-top:0'>"
        "Biometric brainwave login · GRANTED → word prediction · DENIED → locked</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Load artifacts ─────────────────────────────────────────────────────
    auth, wp = load_models()
    eer = load_eer()

    if auth is None:
        st.warning(
            "No enrolled subject model found. Please run one of:\n\n"
            "`python authentication.py enroll --path <dataset_root>`\n\n"
            "or\n\n"
            "`python authentication.py self-test`\n\n"
            "then reload this page."
        )
        st.stop()

    # ── Dataset discovery ──────────────────────────────────────────────────
    dataset_root = discover_dataset()
    mat_map = None
    if dataset_root is not None:
        mat_map = discover_mat_files(dataset_root)

    threshold = auth.threshold if auth.threshold is not None else 0.0

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Dataset")
        if dataset_root is not None:
            st.success(f"Found: `{dataset_root}`")
        else:
            st.info("Dataset not auto-found. Use auth Streamlit with a detected path.")

        if mat_map:
            subjects = sorted(mat_map.keys())
        else:
            subjects = sorted(auth.classes_) if auth.classes_ else []

        st.markdown("### Claimed identity")
        subject = st.selectbox("Subject", subjects)
        split = st.selectbox("Split", ["val", "train", "test"])

        trial_idx = st.number_input("Trial index", min_value=0, value=0,
                                    step=1)

        st.markdown("### System")
        st.metric("Threshold", f"{threshold:.3f}")
        if eer:
            st.metric("EER (validation)", f"{eer['eer']*100:.2f}%")
        st.caption("EER = Equal-Error-Rate. Lower is better. Tuned during "
                   "enrollment on genuine vs imposter trials.")

        st.divider()
        st.markdown("""
**Classes:** Hello, Helpme, Stop, Thankyou, Yes

**Who is enrolled?** The system can only 'recognise' subjects used in
enrollment. A subject not in the set will always be **DENIED**.
""")

    # ── Main ──────────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        explain_html()

        if mat_map is None or subject not in mat_map:
            st.error("No dataset/mat files for this subject. Word decoder "
                     "unavailable; authentication only.")
            F = None
        else:
            fpaths = mat_map[subject].get(split) or mat_map[subject].get("train")
            if not fpaths:
                st.error(f"No {split} data for {subject}.")
                F = None
            else:
                fpath = fpaths[0]
                epo_key = {"train": "epo_train", "val": "epo_validation",
                           "test": "epo_test"}[split]
                try:
                    X, y = load_epoch(fpath, epo_key)
                    n_trials = X.shape[0]
                    trial_idx = int(min(trial_idx, n_trials - 1))
                    trial_X = X[trial_idx: trial_idx + 1]
                    F = extract_subject_features(trial_X)
                    st.caption(
                        f"Trial {trial_idx}/{n_trials-1} · "
                        f"true word: {CLASS_NAMES.get(int(y[trial_idx]), '?')}"
                    )
                except Exception as e:
                    st.error(f"Load error: {e}")
                    F = None

    with right:
        st.markdown("**Authentication**")
        if F is not None:
            res = auth.authenticate(F, subject)
            word = None
            true_word = None
            if res["granted"] and wp is not None:
                _, wname = wp.predict(F, subject)
                word = wname
            if mat_map is not None and subject in mat_map:
                try:
                    fpaths = mat_map[subject].get(split) or mat_map[subject].get("train")
                    fpath = fpaths[0]
                    epo_key = {"train": "epo_train", "val": "epo_validation",
                               "test": "epo_test"}[split]
                    X, y = load_epoch(fpath, epo_key)
                    true_word = CLASS_NAMES.get(int(y[trial_idx]), "?")
                except Exception:
                    pass
            st.markdown(result_card_html(res, word, true_word),
                        unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='color:#73726c;font-size:13px;padding:40px 20px;"
                "text-align:center;border:1px dashed #D3D1C7;border-radius:10px'>"
                "No trial data available for authentication.</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("""
**Note:** If the dataset is not present on this machine, run
`python authentication.py self-test` to generate synthetic enrolled subjects,
then reload this page. Any subject in the synthetic set will be granted;
an out-of-set subject will be denied.
""")


if __name__ == "__main__":
    main()
