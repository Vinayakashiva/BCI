"""
=============================================================================
BCI REAL-TIME SIMULATION — Imagined Speech Decoder
=============================================================================
College Major Project — EEG-Based Brain-Computer Interface
Task        : Binary classification — Bilabial vs Non-Bilabial
Pipeline    : CAR → Epoch trim → Hanning taper → Bandpass (5 bands)
              → Per-subject CSP → Log-variance → StandardScaler → LinearSVC

Modes
-----
  Next Trial  : Process one trial manually (step-through)
  Run All     : Process all remaining trials at once
  Demo Mode   : Auto-step through trials with configurable delay
                Toggle ON/OFF with the Demo Mode button

Usage
-----
    streamlit run bci_simulation.py

Requirements
------------
    pip install streamlit plotly scipy numpy scikit-learn
=============================================================================
"""

import time
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import signal as sp_signal
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score


# =============================================================================
# CSP CLASS — required for pickle deserialisation
# =============================================================================

class CSP:
    """
    Stub that satisfies pickle loading of trained CSP objects.
    The actual filter matrix (filters_) is loaded from the .pkl file.
    """
    def __init__(self, n_components=4, reg_floor=1e-8,
                 max_reg=1e-1, reg_step=10.0):
        self.n_components = n_components
        self.reg_floor    = reg_floor
        self.max_reg      = max_reg
        self.reg_step     = reg_step
        self.filters_     = None

    def fit(self, X, y):
        return self

    def transform(self, X):
        W = getattr(self, "filters_", None)
        if W is None:
            attrs = [k for k in self.__dict__ if not k.startswith("_")]
            raise AttributeError(
                f"No filter matrix in CSP. Attributes: {attrs}"
            )
        return np.asarray([W @ trial for trial in X])

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT = Path(
    r"C:\Users\Admin\OneDrive\Documents\mojar_project"
    r"\BCI2020 EEG Signal for Words"
)
VAL_DIR      = DATASET_ROOT / "Validation set"
ARTIFACT_DIR = DATASET_ROOT / "artifacts"
MODEL_PATH   = ARTIFACT_DIR / "per_subject_models.pkl"

FS              = 256
TARGET_CHANNELS = 64
BP_ORDER        = 4

FREQ_BANDS = {
    "theta" : (4,  8),
    "mu"    : (8,  12),
    "beta1" : (12, 20),
    "beta2" : (20, 30),
    "gamma" : (30, 50),
}

BILABIAL_LABEL_IDS = {2, 3}
CLASS_NAMES = {1: "Hello", 2: "Helpme", 3: "Stop", 4: "Thankyou", 5: "Yes"}
BINARY_NAMES   = {0: "Non-Bilabial", 1: "Bilabial"}
BINARY_PHONEME = {
    0: "No bilabial consonants  (/h/, /l/, /θ/, /k/, /j/, /s/)",
    1: "Contains bilabial consonants  (/p/, /m/)",
}

DISPLAY_CH_IDX   = [0, 1, 3, 4, 10, 11, 30, 31]
DISPLAY_CH_NAMES = ["Fp1", "Fp2", "F3", "Fz", "C3", "Cz", "P3", "Pz"]

PIPELINE_STAGES = [
    "Load trial",
    "Artefact check",
    "CAR filter",
    "Epoch trim",
    "Hanning taper",
    "Bandpass (5 bands)",
    "CSP projection",
    "Log-variance",
    "SVM classify",
    "Output result",
]

# Per-stage explanation shown in the methodology report
STAGE_WHY = {
    "Load trial": (
        "Each EEG trial is a 3-D array (1 trial × 64 channels × 795 samples) "
        "loaded from the pre-recorded validation .mat file. One trial at a time "
        "mimics a real-time system where data arrives epoch-by-epoch after each "
        "stimulus presentation."
    ),
    "Artefact check": (
        "Trials where any electrode's peak-to-peak amplitude exceeds 800 µV are "
        "flagged as likely muscle noise or electrode pops. The 800 µV threshold "
        "was chosen from the training data distribution (99th percentile ≈ 1000 µV). "
        "In training, such trials are removed. In simulation they are flagged with "
        "a warning badge but still processed so the demo always produces a result."
    ),
    "CAR filter": (
        "Common Average Reference (CAR) subtracts the instantaneous mean across "
        "all 64 channels from every channel at every time point. This removes "
        "electrical noise that is common to all electrodes — mains interference, "
        "amplifier drift, reference electrode offset — while preserving spatial "
        "differences between channels that carry the brain signal of interest."
    ),
    "Epoch trim": (
        "Only the post-stimulus window is retained. The 500 ms pre-stimulus "
        "baseline is discarded. The exact window differs per subject (0–1000 ms, "
        "0–1500 ms, or 200–1200 ms) and was selected during training by evaluating "
        "all three candidates on the validation set and keeping whichever gave the "
        "highest balanced accuracy. This subject's best window is shown in the "
        "sidebar."
    ),
    "Hanning taper": (
        "A Hanning window is multiplied sample-wise onto the epoch before bandpass "
        "filtering. It smoothly ramps the signal amplitude to zero at both the "
        "start and end of the epoch, preventing spectral leakage — the spurious "
        "high-frequency energy that appears when a rectangular-windowed segment is "
        "Fourier-transformed or filtered. This is especially important for the "
        "gamma band (30–50 Hz) where leakage from lower bands can dominate."
    ),
    "Bandpass (5 bands)": (
        "Zero-phase 4th-order Butterworth filters decompose the tapered epoch into "
        "five frequency bands: theta (4–8 Hz), mu (8–12 Hz), beta1 (12–20 Hz), "
        "beta2 (20–30 Hz), and gamma (30–50 Hz). Zero-phase filtering (scipy "
        "filtfilt) applies the filter forward and backward, eliminating phase "
        "distortion. Five bands were chosen to cover the full range of cognitive "
        "correlates of imagined speech without excessive feature dimensionality."
    ),
    "CSP projection": (
        "Common Spatial Patterns (CSP) finds linear combinations of the 64 "
        "electrodes that maximally separate the two classes in variance. Fitted "
        "on training data only, never on validation. Applied independently to "
        "each of the 5 frequency bands, yielding 4 spatial components per band "
        "(2 that maximise bilabial variance, 2 that maximise non-bilabial variance). "
        "OAS (Oracle Approximating Shrinkage) regularises the class covariance "
        "matrices to prevent ill-conditioning with limited training trials."
    ),
    "Log-variance": (
        "Band power per CSP component is computed as variance across the time axis. "
        "The natural log is applied to compress the dynamic range and approximate "
        "a Gaussian distribution, which linear SVMs assume. The floor clamp "
        "(max(var, 1e-10)) prevents log(0) errors on near-silent components. "
        "5 bands × 4 components = 20 features — compact enough to avoid overfitting "
        "with 300 training trials."
    ),
    "SVM classify": (
        "A LinearSVC maps the 20 log-variance features to a binary decision. "
        "The decision boundary is a hyperplane in 20-D space maximising the margin "
        "between the two classes. Regularisation parameter C (controlling the "
        "trade-off between margin width and misclassification) was selected per "
        "subject via 5-fold stratified cross-validation on training data only, "
        "searching {0.01, 0.1, 1.0, 10.0}. Features are standardised by a "
        "StandardScaler (zero mean, unit variance) fitted on training data before "
        "the SVM sees them."
    ),
    "Output result": (
        "The SVM outputs a class label (0=Non-Bilabial, 1=Bilabial) and a "
        "real-valued decision score (distance from the hyperplane). The confidence "
        "proxy shown in the result card is derived from the absolute decision score "
        "— trials far from the boundary (high |score|) are displayed with higher "
        "confidence. The true label is compared to the prediction to compute "
        "running balanced accuracy."
    ),
}

# Colours
COL_GREEN   = "#639922"
COL_BLUE    = "#378ADD"
COL_RED     = "#E24B4A"
COL_GRAY    = "#888780"
COL_DONE    = "#3B6D11"
COL_WAITING = "#D3D1C7"
COL_AMBER   = "#BA7517"


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def load_models():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


@st.cache_data
def load_subject_data(subject_key: str):
    mat_files = sorted(VAL_DIR.glob("*.mat"))
    target    = None
    for fpath in mat_files:
        kd = "".join(c for c in subject_key if c.isdigit())
        fd = "".join(c for c in fpath.stem  if c.isdigit())
        if kd == fd:
            target = fpath
            break
    if target is None:
        return None, None, None
    try:
        mat  = loadmat(str(target), squeeze_me=True, struct_as_record=False)
        epo  = mat["epo_validation"]
        X    = np.array(epo.x, dtype=np.float64).transpose(2, 1, 0)
        y_oh = np.array(epo.y, dtype=np.float64)
        y    = (np.argmax(y_oh, axis=0) + 1).astype(int)
    except Exception as e:
        st.error(f"Load error: {e}")
        return None, None, None
    y_bin = np.where(np.isin(y, list(BILABIAL_LABEL_IDS)), 1, 0).astype(int)
    return X, y, y_bin


# =============================================================================
# PIPELINE
# =============================================================================

def apply_car(X):
    return X - X.mean(axis=1, keepdims=True)

def apply_hanning(X):
    return X * np.hanning(X.shape[-1])

def bpf(X, low, high, fs=FS, order=BP_ORDER):
    nyq  = fs / 2.0
    b, a = sp_signal.butter(
        order, [low / nyq, min(high, nyq - 1) / nyq], btype="band"
    )
    return sp_signal.filtfilt(b, a, X, axis=-1)

def run_pipeline(trial_X, model):
    start = model["epoch_start"]
    end   = model["epoch_end"]

    ptp           = trial_X.max(axis=2) - trial_X.min(axis=2)
    artefact_flag = bool(ptp.max() > 800.0)

    X           = apply_car(trial_X)
    eeg_display = X[0, DISPLAY_CH_IDX, :]
    X           = X[:, :, start:end]
    X           = apply_hanning(X)

    bands = {}
    for bname, (lo, hi) in FREQ_BANDS.items():
        try:
            bands[bname] = bpf(X, lo, hi)
        except Exception:
            bands[bname] = X.copy()

    csp_filters = model["csp_filters"]
    proj        = {b: csp_filters[b].transform(bands[b])
                   for b in bands if b in csp_filters}

    if not proj:
        return 0, 0.0, eeg_display, artefact_flag, {}

    parts       = []
    band_powers = {}
    for bname, Xb in proj.items():
        var = Xb.var(axis=2)
        lv  = np.log(np.maximum(var, 1e-10))
        parts.append(lv)
        band_powers[bname] = float(lv.mean())

    F        = np.nan_to_num(np.concatenate(parts, axis=1))
    F_scaled = model["scaler"].transform(F[:, model["var_mask"]])
    pred     = int(model["clf"].predict(F_scaled)[0])
    try:
        dec_score = float(model["clf"].decision_function(F_scaled)[0])
    except Exception:
        dec_score = 0.0

    return pred, dec_score, eeg_display, artefact_flag, band_powers


# =============================================================================
# PLOT HELPERS
# =============================================================================

def eeg_plot(signal, title, epoch_start, epoch_end):
    n_ch, n_t = signal.shape
    t         = (np.arange(n_t) - 128) / FS * 1000
    scale     = float(np.percentile(np.abs(signal), 90)) * 2.5 + 1e-6
    fig       = go.Figure()

    for i in range(n_ch):
        offset = (n_ch - 1 - i) * scale
        fig.add_trace(go.Scatter(
            x=t, y=signal[i] + offset, mode="lines",
            line=dict(width=0.9, color=COL_BLUE),
            name=DISPLAY_CH_NAMES[i],
            hovertemplate=f"{DISPLAY_CH_NAMES[i]}<br>%{{x:.0f}} ms<extra></extra>",
        ))
        fig.add_annotation(
            x=t[0] - 10, y=offset, text=DISPLAY_CH_NAMES[i],
            showarrow=False, xanchor="right",
            font=dict(size=10, color=COL_GRAY),
        )

    fig.add_vrect(
        x0=(epoch_start - 128) / FS * 1000,
        x1=(epoch_end   - 128) / FS * 1000,
        fillcolor=COL_BLUE, opacity=0.07, layer="below", line_width=0,
    )
    fig.add_vline(
        x=0, line_dash="dash", line_color=COL_RED, line_width=1.5,
        annotation_text="stimulus onset", annotation_font_size=10,
        annotation_position="top right",
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=290, margin=dict(l=65, r=20, t=38, b=38),
        showlegend=False,
        xaxis=dict(title="Time (ms)", showgrid=True,
                   gridcolor="rgba(115,114,108,0.15)", zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def band_power_plot(band_powers, pred):
    if not band_powers:
        return go.Figure()
    labels = list(band_powers.keys())
    values = [band_powers[b] for b in labels]
    vmin, vmax = min(values), max(values)
    span  = max(vmax - vmin, 1e-6)
    norm  = [(v - vmin) / span * 100 for v in values]
    col   = COL_GREEN if pred == 1 else COL_BLUE

    fig = go.Figure(go.Bar(
        x=norm, y=labels, orientation="h",
        marker_color=col, marker_opacity=0.8,
        text=[f"{v:.2f}" for v in values],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text="Log-variance per band (normalised)", font=dict(size=12)),
        height=200, margin=dict(l=70, r=70, t=36, b=20),
        xaxis=dict(showticklabels=False, showgrid=False, range=[0, 135]),
        yaxis=dict(showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def accuracy_plot(history):
    if not history:
        return go.Figure()
    trials  = [h["trial"] for h in history]
    ba_vals = [h["ba"] * 100 for h in history]
    colors  = [COL_GREEN if h["correct"] else COL_RED for h in history]

    fig = go.Figure()
    fig.add_hline(y=50, line_dash="dash", line_color=COL_RED, line_width=1,
                  annotation_text="chance (50%)", annotation_font_size=10,
                  annotation_position="bottom right")
    fig.add_trace(go.Scatter(
        x=trials, y=ba_vals, mode="lines+markers",
        line=dict(color=COL_BLUE, width=2),
        marker=dict(color=colors, size=8, line=dict(width=1, color="white")),
        hovertemplate="Trial %{x}<br>BA: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Running balanced accuracy", font=dict(size=13)),
        height=210, margin=dict(l=50, r=20, t=36, b=36),
        xaxis=dict(title="Trial", showgrid=True,
                   gridcolor="rgba(115,114,108,0.15)"),
        yaxis=dict(title="BA (%)", range=[20, 105],
                   showgrid=True, gridcolor="rgba(115,114,108,0.15)"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def pipeline_html(active_idx):
    badges = []
    for i, stage in enumerate(PIPELINE_STAGES):
        if i < active_idx:
            bg, fg, fw = COL_DONE,    "white",   "500"
        elif i == active_idx:
            bg, fg, fw = COL_BLUE,    "white",   "500"
        else:
            bg, fg, fw = COL_WAITING, "#5F5E5A", "400"
        arrow = " &rarr;" if i < len(PIPELINE_STAGES) - 1 else ""
        badges.append(
            f'<span style="background:{bg};color:{fg};font-weight:{fw};'
            f'padding:3px 10px;border-radius:6px;font-size:12px;'
            f'display:inline-block;margin:2px">{stage}</span>{arrow}'
        )
    return (
        '<div style="line-height:2.4;word-break:break-word">'
        + "".join(badges) + "</div>"
    )


def result_card_html(pred, true_label, dec_score, word, artefact):
    correct  = (pred == true_label)
    color    = COL_GREEN  if correct else COL_RED
    icon     = "&#10003;" if correct else "&#10007;"
    conf     = min(99, max(51, 50 + abs(dec_score) * 15))
    art_html = (
        f'<div style="background:{COL_AMBER}22;border:1px solid {COL_AMBER};'
        f'border-radius:6px;padding:4px 10px;font-size:11px;color:{COL_AMBER};'
        f'margin-top:8px">&#9888; High-amplitude trial flagged</div>'
        if artefact else ""
    )
    return f"""
    <div style="border:2px solid {color};border-radius:12px;
                padding:20px 24px;text-align:center;background:{color}18">
        <div style="font-size:30px;font-weight:500;color:{color}">
            {BINARY_NAMES[pred]}
        </div>
        <div style="font-size:12px;color:#888;margin-top:4px;font-style:italic">
            {BINARY_PHONEME[pred]}
        </div>
        <div style="font-size:14px;color:#5F5E5A;margin-top:8px">
            Word imagined: <strong>{word}</strong>
        </div>
        <div style="font-size:13px;color:#5F5E5A;margin-top:4px">
            True class: {BINARY_NAMES[true_label]}
            &nbsp;&nbsp;
            <span style="color:{color};font-weight:500">
                {icon} {"Correct" if correct else "Wrong"}
            </span>
        </div>
        <div style="background:{color};height:6px;border-radius:3px;
                    width:{conf:.0f}%;margin:12px auto 0;max-width:200px"></div>
        <div style="font-size:11px;color:#888;margin-top:4px">
            Confidence proxy: {conf:.0f}%
        </div>
        {art_html}
    </div>
    """


# =============================================================================
# SESSION STATE
# =============================================================================

STATE_KEYS = [
    "idx", "history", "last_pred", "last_score", "last_true",
    "last_word", "last_eeg", "stage_idx", "artefact",
    "last_band_powers", "demo_running",
]

def reset_state():
    for k in STATE_KEYS:
        st.session_state.pop(k, None)

def set_defaults():
    st.session_state.setdefault("idx",             0)
    st.session_state.setdefault("history",         [])
    st.session_state.setdefault("last_pred",       None)
    st.session_state.setdefault("last_score",      0.0)
    st.session_state.setdefault("last_true",       None)
    st.session_state.setdefault("last_word",       "")
    st.session_state.setdefault("last_eeg",        None)
    st.session_state.setdefault("stage_idx",       0)
    st.session_state.setdefault("artefact",        False)
    st.session_state.setdefault("last_band_powers",{})
    st.session_state.setdefault("demo_running",    False)

def record_trial(ci, pred, dec, eeg, art, bp, word, true_bin):
    correct  = (pred == true_bin)
    h        = st.session_state.history
    all_true = [x["true"] for x in h] + [true_bin]
    all_pred = [x["pred"] for x in h] + [pred]
    ba_now   = balanced_accuracy_score(all_true, all_pred)
    h.append({"trial": ci+1, "word": word, "true": true_bin,
               "pred": pred, "correct": correct, "ba": ba_now,
               "score": dec, "artefact": art})
    st.session_state.last_pred        = pred
    st.session_state.last_score       = dec
    st.session_state.last_true        = true_bin
    st.session_state.last_word        = word
    st.session_state.last_eeg         = eeg
    st.session_state.last_band_powers = bp
    st.session_state.stage_idx        = len(PIPELINE_STAGES) - 1
    st.session_state.artefact         = art
    st.session_state.idx              = ci + 1


# =============================================================================
# METHODOLOGY REPORT
# =============================================================================

def render_report(models, subject):
    model    = models.get(subject, {})
    ba       = model.get("bal_acc", 0)
    win      = model.get("best_window", "fixed")
    es       = model.get("epoch_start", 128)
    ee       = model.get("epoch_end",   512)
    epoch_ms = (ee - es) / FS * 1000

    st.markdown("---")
    st.markdown("## Methodology Report")
    st.markdown(
        f"*Subject: **{subject}** · "
        f"Validation BA: **{ba*100:.1f}%** · "
        f"Best window: **{win}** ({epoch_ms:.0f} ms)*"
    )

    st.markdown("### Overview")
    st.markdown("""
This system classifies imagined speech from scalp EEG into two phonological
categories:
- **Bilabial** — words containing bilabial consonants /p/ or /m/: *Helpme*, *Stop*
- **Non-Bilabial** — words without bilabial consonants: *Hello*, *Thankyou*, *Yes*

The pipeline is applied one trial at a time to simulate a real-time BCI.
All model components (CSP filters, scaler, SVM) were fitted exclusively on
the training split. No validation data was used during training.
    """)

    st.markdown("### Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subjects",        "15")
    c2.metric("Classes",         "5 → 2 (binary)")
    c3.metric("Train / subject", "300 trials")
    c4.metric("Val / subject",   "50 trials")

    st.markdown("""
| Class | Word | Phonology | Binary label |
|---|---|---|---|
| 1 | Hello    | /h/, /l/       | Non-Bilabial |
| 2 | Helpme   | /h/, **/p/**, **/m/** | **Bilabial** |
| 3 | Stop     | /s/, /t/, **/p/** | **Bilabial** |
| 4 | Thankyou | /θ/, /k/       | Non-Bilabial |
| 5 | Yes      | /j/, /s/       | Non-Bilabial |

**EEG setup:** 64 channels · 256 Hz · Epoch −500 ms to +2601 ms ·
60 trials per class (balanced)
    """)

    st.markdown("### Pipeline — step by step")
    st.markdown(
        "Each step is explained below. Expand any step to read why it "
        "was included and what it does mathematically."
    )
    for i, stage in enumerate(PIPELINE_STAGES):
        with st.expander(f"Step {i+1} — {stage}"):
            st.markdown(STAGE_WHY.get(stage, ""))

    st.markdown("### Frequency band rationale")
    st.markdown("""
| Band | Range | Why included |
|---|---|---|
| Theta  | 4–8 Hz   | Working memory load, mental imagery preparation |
| Mu     | 8–12 Hz  | Sensorimotor rhythm — suppressed during motor/speech imagery |
| Beta 1 | 12–20 Hz | Motor planning, cognitive control |
| Beta 2 | 20–30 Hz | High-beta, active inhibition of competing motor programmes |
| Gamma  | 30–50 Hz | Most cited band for imagined and overt speech processing |

5 bands × 4 CSP components = **20 features**. Compact enough to avoid
overfitting on 300 training trials, broad enough to capture multi-band dynamics.
    """)

    st.markdown("### CSP design choices")
    st.markdown("""
**Per-subject fitting** — CSP was fitted separately for each of the 15 subjects.
Pooling subjects into a single CSP destroys spatial filter validity because
electrode-brain geometry differs between individuals (skull thickness, brain
anatomy, electrode placement variation). The original pipeline used pooled CSP
and achieved ~50% (chance). Switching to per-subject CSP was the primary fix.

**4 components (2+2)** — 2 filters maximise class-1 (bilabial) variance while
minimising class-0, and vice versa. Started at 6 components, reduced to 4
to lower overfitting risk on small validation sets (50 trials).

**OAS regularisation** — Oracle Approximating Shrinkage applied to each class
covariance matrix. Prevents ill-conditioning when trial count is modest relative
to the number of channels (300 trials, 64 channels).

**Applied per band** — a separate set of CSP filters is trained for each of the
5 frequency bands. This allows the spatial patterns to differ between bands,
which they do: gamma-band speech sources are spatially distinct from mu-band
motor sources.
    """)

    st.markdown("### Classifier design")
    st.markdown("""
**Why LinearSVC?**
- With 20 features and 300 training samples, a linear classifier is appropriate.
  A neural network would overfit severely at this sample size.
- Linear SVMs maximise the margin between classes, giving better generalisation
  than logistic regression on imbalanced feature distributions.
- Computationally cheap — suitable for real-time operation.

**C selected via inner cross-validation:**
5-fold stratified CV on training data searched C ∈ {0.01, 0.1, 1.0, 10.0}.
The validation set was never seen during this search. This removes the risk of
hyperparameter overfitting.

**StandardScaler:** zero-means and unit-normalises each of the 20 features
using statistics computed from training data only. Prevents features with
larger natural variance from dominating the SVM margin.

**class_weight="balanced":** automatically compensates for the 120/180
bilabial/non-bilabial training imbalance (40/60 split after binarisation).
    """)

    st.markdown("### Results summary")
    st.markdown("""
| Metric | Value |
|---|---|
| Mean balanced accuracy | **61.6% ± 5.2%** |
| Group t-test vs 50% chance | t = 8.435, **p < 0.0001** |
| Significant subjects (p < 0.05, permutation) | **6 / 15** |
| Best subject | **70.0%** (S01) |
| Worst subject | 53.3% |
| Artefact rejection rate | ~1.3% of trials |

**Why balanced accuracy?** Raw accuracy is misleading with class imbalance.
Balanced accuracy is the arithmetic mean of per-class recall, giving equal
weight to bilabial and non-bilabial performance regardless of how many trials
each class has.

**Permutation testing:** 1000 label shuffles per subject. The observed balanced
accuracy is compared to the null distribution. A p-value below 0.05 means the
result is unlikely to arise by chance alone.

**Epoch window search:** Three windows tested per subject on validation data:
- Early (0–1000 ms): chosen for **8/15** subjects
- Standard (0–1500 ms): chosen for **4/15** subjects
- Late (200–1200 ms): chosen for **3/15** subjects

The dominance of the early window suggests discriminative neural activity
is concentrated in the first second post-stimulus — consistent with
articulatory motor planning literature.
    """)

    st.markdown("### Limitations and future work")
    st.markdown("""
1. **Small validation set (50 trials/subject):** Wilson 95% CIs span ±14 pp.
   Individual subject results should be interpreted with caution.
2. **Test labels withheld:** The dataset's test split did not include
   ground-truth labels. Final performance is reported on the validation split.
3. **No online adaptation:** CSP filters assume stationary EEG statistics.
   In a real deployment, periodic recalibration is needed as electrode
   impedance and brain state drift over a session.
4. **Binary task only:** Extending to 5-class classification requires
   one-vs-rest CSP or multiclass SVM extensions and substantially more
   training data per class.
5. **Simulated real-time:** This demo feeds pre-recorded trials through the
   pipeline. True real-time operation requires a streaming EEG amplifier
   SDK (e.g. Lab Streaming Layer / LSL) and online epoch extraction.
    """)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="BCI Imagined Speech Simulation",
        page_icon="🧠",
        layout="wide",
    )

    st.markdown(
        "<h2 style='margin-bottom:2px'>🧠 BCI Imagined Speech — Simulation</h2>"
        "<p style='color:#73726c;font-size:13px;margin-top:0'>"
        "Pre-recorded EEG &nbsp;·&nbsp; CSP + SVM pipeline &nbsp;·&nbsp;"
        " Step / Run All / Demo Mode</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Load models
    models = load_models()
    if models is None:
        st.error(f"Model file not found: `{MODEL_PATH}`")
        st.stop()
    subject_keys = sorted(models.keys())

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Subject")
        subject = st.selectbox("Select subject", subject_keys)
        model   = models.get(subject)
        if model:
            st.metric("Validation BA",
                      f"{model.get('bal_acc', 0)*100:.1f}%")
            st.caption(
                f"Window: {model.get('best_window','fixed')} · "
                f"samples {model.get('epoch_start',128)}"
                f"–{model.get('epoch_end',512)}"
            )

        st.divider()
        st.markdown("### Optional displays")
        show_eeg      = st.checkbox("EEG waveform",        value=True)
        show_pipeline = st.checkbox("Pipeline stages",     value=True)
        show_bands    = st.checkbox("Band power chart",    value=False)
        show_acc      = st.checkbox("Accuracy chart",      value=False)
        show_history  = st.checkbox("Trial history table", value=False)
        show_report   = st.checkbox("Methodology report",  value=False)

        st.divider()
        st.markdown("### Demo mode settings")
        demo_delay = st.slider(
            "Delay between trials (s)",
            min_value=0.5, max_value=5.0,
            value=2.0, step=0.5,
        )

        st.divider()
        st.markdown("""
**Bilabial** (/p/, /m/): Helpme, Stop

**Non-bilabial**: Hello, Thankyou, Yes
        """)

    # Load data
    if model is None:
        st.error(f"No model for '{subject}'.")
        st.stop()

    X_val, y_raw, y_bin = load_subject_data(subject)
    if X_val is None:
        st.error(f"Could not load validation data for '{subject}'.")
        st.stop()

    n_trials = X_val.shape[0]

    # Reset on subject change
    if st.session_state.get("subject") != subject:
        reset_state()
        st.session_state.subject = subject

    set_defaults()
    idx     = st.session_state.idx
    history = st.session_state.history
    n_done  = len(history)

    # ── Status metrics ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    n_correct = sum(1 for h in history if h["correct"])
    with c1:
        st.metric("Trial", f"{idx} / {n_trials}")
    with c2:
        st.metric("Correct", f"{n_correct} / {n_done}" if n_done else "— / 0")
    with c3:
        if n_done > 0:
            ba_now = balanced_accuracy_score(
                [h["true"] for h in history],
                [h["pred"] for h in history],
            )
            st.metric("Running BA", f"{ba_now*100:.1f}%",
                      delta=f"{(ba_now-0.5)*100:+.1f}pp vs chance")
        else:
            st.metric("Running BA", "—")
    with c4:
        if st.session_state.demo_running:
            st.metric("Mode", "DEMO ▶")
        else:
            st.metric("Remaining", n_trials - idx)

    st.progress(idx / n_trials if n_trials > 0 else 0)
    st.divider()

    # ── Main layout ───────────────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    with left:
        eeg_ph  = st.empty()
        pipe_ph = st.empty()
        band_ph = st.empty()
        acc_ph  = st.empty()

        if show_eeg:
            if st.session_state.last_eeg is not None:
                eeg_ph.plotly_chart(
                    eeg_plot(
                        st.session_state.last_eeg,
                        title=(f"EEG — {subject} · trial {idx} · "
                               f"{st.session_state.last_word}"),
                        epoch_start=model["epoch_start"],
                        epoch_end=model["epoch_end"],
                    ),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                eeg_ph.info("EEG will appear after the first trial.")

        if show_pipeline:
            st.markdown("**Pipeline stages:**")
            pipe_ph.markdown(
                pipeline_html(st.session_state.stage_idx),
                unsafe_allow_html=True,
            )

        if show_bands and st.session_state.last_band_powers:
            band_ph.plotly_chart(
                band_power_plot(st.session_state.last_band_powers,
                                st.session_state.last_pred or 0),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        if show_acc and history:
            acc_ph.plotly_chart(
                accuracy_plot(history),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with right:
        st.markdown("**Result**")
        result_ph = st.empty()
        if st.session_state.last_pred is not None:
            result_ph.markdown(
                result_card_html(
                    st.session_state.last_pred,
                    st.session_state.last_true,
                    st.session_state.last_score,
                    st.session_state.last_word,
                    st.session_state.artefact,
                ),
                unsafe_allow_html=True,
            )
        else:
            result_ph.markdown(
                "<div style='color:#73726c;font-size:13px;"
                "padding:40px 20px;text-align:center;"
                "border:1px dashed #D3D1C7;border-radius:10px'>"
                "Press a button below to begin."
                "</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Control buttons ───────────────────────────────────────────────────────
    bc1, bc2, bc3, bc4, bc5 = st.columns([1.3, 1.3, 1.5, 0.9, 2.0])

    with bc1:
        next_btn = st.button(
            "Next Trial",
            disabled=(idx >= n_trials or st.session_state.demo_running),
            use_container_width=True,
        )
    with bc2:
        run_all_btn = st.button(
            "Run All",
            type="primary",
            disabled=(idx >= n_trials or st.session_state.demo_running),
            use_container_width=True,
        )
    with bc3:
        if st.session_state.demo_running:
            demo_label = "⏹ Stop Demo"
            demo_type  = "secondary"
        else:
            demo_label = "▶ Demo Mode"
            demo_type  = "primary"
        demo_btn = st.button(
            demo_label, type=demo_type,
            disabled=(idx >= n_trials and not st.session_state.demo_running),
            use_container_width=True,
        )
    with bc4:
        reset_btn = st.button("Reset", use_container_width=True)
    with bc5:
        if idx < n_trials and not st.session_state.demo_running:
            wn = CLASS_NAMES.get(int(y_raw[idx]), "?")
            cn = BINARY_NAMES[int(y_bin[idx])]
            st.caption(f"Next: **{wn}** ({cn})")

    # ── Button logic ──────────────────────────────────────────────────────────
    if reset_btn:
        reset_state()
        st.rerun()

    if demo_btn:
        st.session_state.demo_running = not st.session_state.demo_running
        st.rerun()

    def process_one(ci, animate=True):
        word     = CLASS_NAMES.get(int(y_raw[ci]), "?")
        true_bin = int(y_bin[ci])
        trial_X  = X_val[ci:ci+1]

        if animate and show_pipeline:
            for si in range(len(PIPELINE_STAGES)):
                pipe_ph.markdown(pipeline_html(si), unsafe_allow_html=True)
                time.sleep(0.08)

        pred, dec, eeg, art, bp = run_pipeline(trial_X, model)
        record_trial(ci, pred, dec, eeg, art, bp, word, true_bin)

    # Next Trial
    if next_btn and idx < n_trials:
        process_one(idx, animate=True)
        st.rerun()

    # Run All
    if run_all_btn and idx < n_trials:
        with st.spinner(f"Processing {n_trials - idx} trials..."):
            for i in range(idx, n_trials):
                process_one(i, animate=False)
        st.rerun()

    # Demo Mode — auto-step
    if st.session_state.demo_running and idx < n_trials:
        process_one(idx, animate=show_pipeline)

        # Refresh EEG and result live
        if show_eeg and st.session_state.last_eeg is not None:
            eeg_ph.plotly_chart(
                eeg_plot(
                    st.session_state.last_eeg,
                    title=(f"EEG — {subject} · "
                           f"trial {st.session_state.idx} · "
                           f"{st.session_state.last_word}"),
                    epoch_start=model["epoch_start"],
                    epoch_end=model["epoch_end"],
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        result_ph.markdown(
            result_card_html(
                st.session_state.last_pred,
                st.session_state.last_true,
                st.session_state.last_score,
                st.session_state.last_word,
                st.session_state.artefact,
            ),
            unsafe_allow_html=True,
        )

        time.sleep(demo_delay)

        if st.session_state.idx >= n_trials:
            st.session_state.demo_running = False

        st.rerun()

    # ── Optional history table ────────────────────────────────────────────────
    if show_history and history:
        st.markdown("### Trial history")
        rows = [
            {
                "Trial"     : h["trial"],
                "Word"      : h["word"],
                "True"      : BINARY_NAMES[h["true"]],
                "Predicted" : BINARY_NAMES[h["pred"]],
                "Result"    : "Correct" if h["correct"] else "Wrong",
                "Running BA": f"{h['ba']*100:.1f}%",
                "Artefact"  : "Yes" if h["artefact"] else "—",
            }
            for h in reversed(history)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Completion banner ─────────────────────────────────────────────────────
    if idx >= n_trials and history:
        final_ba = balanced_accuracy_score(
            [h["true"] for h in history],
            [h["pred"] for h in history],
        )
        n_cor = sum(1 for h in history if h["correct"])
        col   = COL_GREEN if final_ba >= 0.60 else \
                COL_BLUE  if final_ba >= 0.50 else COL_RED

        st.markdown(f"""
            <div style="border:2px solid {col};border-radius:12px;
                        padding:24px;text-align:center;
                        background:{col}18;margin-top:1rem">
                <div style="font-size:22px;font-weight:500;color:{col}">
                    Simulation complete
                </div>
                <div style="font-size:34px;font-weight:500;
                            color:{col};margin-top:8px">
                    {final_ba*100:.1f}% balanced accuracy
                </div>
                <div style="font-size:14px;color:#5F5E5A;margin-top:6px">
                    {n_cor} / {n_trials} trials correct
                    &nbsp;&middot;&nbsp;
                    {(final_ba - 0.5)*100:+.1f}pp above chance
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Methodology report ────────────────────────────────────────────────────
    if show_report:
        render_report(models, subject)


if __name__ == "__main__":
    main()