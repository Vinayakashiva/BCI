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

Display modes
-------------
  1. Normal (plots)        : EEG waveform + pipeline badges + charts (original)
  2. Video playback         : Plays a local .mp4 clip in a fullscreen,
                               timeline-free "cutscene" player each time a
                               trial is run. Hold ENTER for 5 seconds to
                               skip straight to the result.
  3. Animated illustration  : Professional-styled SVG/HTML neural-signal
                               animation, stepping through each pipeline stage

Usage
-----
    streamlit run bci_simulation.py

Requirements
------------
    pip install streamlit plotly scipy numpy scikit-learn

    Optional (for "Voice output" — imagined word -> sentence -> speech):
    pip install edge-tts anthropic
=============================================================================
"""

import time
import json
import base64
import pickle
import asyncio
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from scipy import signal as sp_signal
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score

# Optional deps for the "imagined word -> sentence -> voice" feature.
# The app still runs fine without them (falls back to canned sentences /
# no audio), so these are soft imports rather than hard requirements.
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# EEG authentication gate (authentication.py must sit next to this file).
# Soft import: if it's missing, the app still runs — the login gate simply
# reports that it can't find the module instead of crashing the whole app.
try:
    from authentication import (
        AUTH_DIR,
        SUBJECT_MODEL_PATH,
        WORD_MODEL_PATH,
        EER_PATH,
        discover_dataset as auth_discover_dataset,
        discover_mat_files as auth_discover_mat_files,
        load_epoch as auth_load_epoch,
        extract_subject_features,
        SubjectAuthenticator,
        WordPredictor,
    )
    AUTH_MODULE_AVAILABLE = True

    # --- pickle compatibility shim -----------------------------------------
    # If auth_artifacts/*.pkl were created by running `python authentication.py
    # ...` directly, SubjectAuthenticator/WordPredictor were pickled as
    # belonging to "__main__" (whatever script was run at the time), not to
    # "authentication". When THIS script — a different __main__ — later tries
    # to unpickle them, Python looks for those classes on its own __main__ and
    # raises AttributeError. Registering them here makes both old-style
    # (__main__-pickled) and new-style (module-pickled) artifacts load fine,
    # regardless of how enroll/self-test was invoked.
    import sys as _sys
    _main_mod = _sys.modules.get("__main__")
    if _main_mod is not None:
        if not hasattr(_main_mod, "SubjectAuthenticator"):
            _main_mod.SubjectAuthenticator = SubjectAuthenticator
        if not hasattr(_main_mod, "WordPredictor"):
            _main_mod.WordPredictor = WordPredictor
except ImportError:
    AUTH_MODULE_AVAILABLE = False


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

# Local mp4 clip that plays each time a trial is processed in "Video playback"
# mode. ONE single generic clip is used for every trial — update this path.
VIDEO_PATH = r"eeg_capture_demo.mp4"

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

# Short label + description used by the animated-illustration mode
# (kept intentionally brief so they fit on-screen during the step animation)
ANIM_STAGE_INFO = [
    ("Acquisition",   "64-ch EEG cap samples the scalp at 256 Hz"),
    ("Artefact check","Peak-to-peak amplitude screened for noise"),
    ("CAR filter",    "Common average reference removes shared noise"),
    ("Epoch trim",    "Subject-specific post-stimulus window extracted"),
    ("Hanning taper", "Edges smoothed to prevent spectral leakage"),
    ("Band filters",  "Split into theta / mu / beta1 / beta2 / gamma"),
    ("CSP projection","Spatial filters maximise class separation"),
    ("Log-variance",  "Band power per component → 20-D feature vector"),
    ("SVM classify",  "LinearSVC maps features to a decision boundary"),
    ("Prediction",    "Predicted word and confidence are produced"),
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

# Palette used only by the animated-illustration mode (kept separate so it
# can be restyled without touching the rest of the app)
ANIM_INK       = "#1B1B1F"
ANIM_MUTED     = "#6B6B76"
ANIM_LINE      = "#D8D8E0"
ANIM_PANEL     = "#F6F6F9"
ANIM_ACCENT    = "#4C6FFF"
ANIM_ACCENT_2  = "#22C3A6"
ANIM_SUCCESS   = "#1F9D63"


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


@st.cache_data
def load_video_base64(path_str: str):
    """
    Load a local mp4 file once, cache it, and return a base64 string so it
    can be embedded directly as a <video> data URI inside a custom HTML
    component (needed for the fullscreen, timeline-free "cutscene" player —
    Streamlit's built-in st.video() cannot hide the scrubber or enter
    fullscreen programmatically).
    """
    p = Path(path_str)
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


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
# ANIMATED ILLUSTRATION — professional step-by-step neural signal animation
# =============================================================================

def build_capture_animation_html(word: str, pred_label: int, true_label: int,
                                  band_powers: dict, dec_score: float = 0.0,
                                  height: int = 460) -> str:
    """
    Sankey-style "signal flow" illustration in the spirit of the transformer-
    explainer visualisation: EEG channel tokens fan out into colour-coded
    ribbons (CAR / Bandpass / CSP-variance — echoing Key/Query/Value lanes),
    converge on a CSP "attention-like" component grid, flow through an SVM
    funnel, and resolve into a probability panel on the right, exactly the
    way the reference diagram ends in a ranked token-probability list.

    The whole thing steps through the real pipeline stages (ANIM_STAGE_INFO):
    channels pulse first, then each ribbon lights up in turn, then the CSP
    grid, then the SVM funnel, then the prediction panel fades in — so it
    reads as an actual animation rather than a static picture.

    The SVG's rendered width is capped (not left at 100% of the Streamlit
    column) so its height can never grow past the fixed iframe height the
    caller allocates, regardless of how wide the page layout is.
    """
    correct       = (pred_label == true_label)
    result_color  = ANIM_SUCCESS if pred_label == 1 else ANIM_ACCENT
    conf          = min(99, max(51, 50 + abs(dec_score) * 15))
    other_pct     = 100 - conf
    pred_name     = BINARY_NAMES[pred_label]
    other_name    = BINARY_NAMES[1 - pred_label]

    KEY_COL, QUERY_COL, VALUE_COL = "#E2574C", ANIM_ACCENT, ANIM_ACCENT_2

    CH_X, CH_Y0, CH_STEP = 55, 38, 34
    KQV_X                = 250
    KEY_Y, QUERY_Y, VALUE_Y = 82, 176, 270
    GRID_X0, GRID_X1     = 380, 570
    GRID_Y0, GRID_Y1     = 92, 262
    OUT_X, OUT_Y         = 630, 176
    SVM_X0, SVM_X1       = 670, 830
    FINAL_X              = 862

    bands    = list(FREQ_BANDS.keys())
    vals     = [abs(band_powers.get(b, 0.0)) for b in bands]
    vmax     = max(vals) or 1.0
    col_norm = [v / vmax for v in vals]
    top_band_idx = col_norm.index(max(col_norm)) if col_norm else 0

    ch_ys = [CH_Y0 + i * CH_STEP for i in range(len(DISPLAY_CH_NAMES))]

    def curve(x0, y0, x1, y1):
        mx = (x0 + x1) / 2
        return f"M {x0} {y0} C {mx} {y0}, {mx} {y1}, {x1} {y1}"

    ribbons_in, ch_dots = [], []
    for name, y in zip(DISPLAY_CH_NAMES, ch_ys):
        ribbons_in.append(
            f'<text x="{CH_X-12}" y="{y+4}" text-anchor="end" font-size="10.5" '
            f'fill="{ANIM_MUTED}">{name}</text>'
        )
        ch_dots.append(
            f'<circle class="ch-dot" cx="{CH_X}" cy="{y}" r="3" fill="{ANIM_INK}" opacity="0.5"/>'
        )
        for ty, col, lane in ((KEY_Y, KEY_COL, "car"), (QUERY_Y, QUERY_COL, "bandpass"),
                               (VALUE_Y, VALUE_COL, "cspvar")):
            ribbons_in.append(
                f'<path d="{curve(CH_X, y, KQV_X, ty)}" stroke="{col}" '
                f'stroke-width="1.2" fill="none" opacity="0.15" class="ribbon lane-{lane}"/>'
            )

    grid_dots = []
    for c in range(5):
        gx = GRID_X0 + 30 + c * 34
        for r in range(4):
            gy = GRID_Y0 + 20 + r * 40
            inten = col_norm[c] * (1 - r * 0.15)
            fill  = f"rgba(76,111,255,{0.15 + inten * 0.7:.2f})"
            hi    = ' class="grid-dot grid-dot-top"' if c == top_band_idx else ' class="grid-dot"'
            grid_dots.append(f'<circle{hi} cx="{gx}" cy="{gy}" r="7" fill="{fill}"/>')

    band_labels = "".join(
        f'<text x="{GRID_X0+30+c*34}" y="{GRID_Y1+18}" text-anchor="middle" '
        f'font-size="9" fill="{ANIM_MUTED}">{bands[c]}</text>'
        for c in range(5)
    )

    key_to_grid   = curve(KQV_X, KEY_Y, GRID_X0, (GRID_Y0 + GRID_Y1) // 2)
    query_to_grid = curve(KQV_X, QUERY_Y, GRID_X0, (GRID_Y0 + GRID_Y1) // 2)
    value_to_out  = (
        f"M {KQV_X} {VALUE_Y} C {(KQV_X+OUT_X)//2} {VALUE_Y}, "
        f"{(KQV_X+OUT_X)//2} {OUT_Y+60}, {OUT_X} {OUT_Y}"
    )
    grid_to_out   = curve(GRID_X1, (GRID_Y0 + GRID_Y1) // 2, OUT_X, OUT_Y)
    out_to_svm    = curve(OUT_X, OUT_Y, SVM_X0, OUT_Y)
    svm_mid       = (SVM_X0 + SVM_X1) // 2
    svm_shape     = (
        f"M {SVM_X0} {OUT_Y-60} C {svm_mid} {OUT_Y-60}, {svm_mid} {OUT_Y}, {svm_mid} {OUT_Y} "
        f"C {svm_mid} {OUT_Y}, {svm_mid} {OUT_Y+60}, {SVM_X0} {OUT_Y+60} "
        f"M {SVM_X1} {OUT_Y-60} C {svm_mid} {OUT_Y-60}, {svm_mid} {OUT_Y}, {svm_mid} {OUT_Y} "
        f"C {svm_mid} {OUT_Y}, {svm_mid} {OUT_Y+60}, {SVM_X1} {OUT_Y+60}"
    )
    svm_to_final  = curve(SVM_X1, OUT_Y, FINAL_X, OUT_Y)

    svg = f"""
    <svg viewBox="0 0 900 340" style="width:100%;height:auto;display:block">
      <text x="{CH_X}" y="16" font-size="10.5" font-weight="700" fill="{ANIM_MUTED}"
            style="text-transform:uppercase;letter-spacing:0.06em">Channels</text>
      <text x="{KQV_X-32}" y="{KEY_Y-12}" font-size="11.5" font-weight="700" fill="{KEY_COL}"
            class="lbl lane-car-lbl">CAR</text>
      <text x="{KQV_X-52}" y="{QUERY_Y-12}" font-size="11.5" font-weight="700" fill="{QUERY_COL}"
            class="lbl lane-bandpass-lbl">Bandpass</text>
      <text x="{KQV_X-40}" y="{VALUE_Y+24}" font-size="11.5" font-weight="700" fill="{VALUE_COL}"
            class="lbl lane-cspvar-lbl">CSP var</text>

      {''.join(ribbons_in)}
      {''.join(ch_dots)}

      <path id="path-car" d="{key_to_grid}" stroke="{KEY_COL}" stroke-width="2.2" fill="none"
            opacity="0.45" class="ribbon"/>
      <path id="path-bandpass" d="{query_to_grid}" stroke="{QUERY_COL}" stroke-width="2.2" fill="none"
            opacity="0.45" class="ribbon"/>
      <path id="path-cspvar" d="{value_to_out}" stroke="{VALUE_COL}" stroke-width="2.2" fill="none"
            opacity="0.45" class="ribbon"/>

      <g id="csp-grid">
        <rect x="{GRID_X0}" y="{GRID_Y0}" width="{GRID_X1-GRID_X0}" height="{GRID_Y1-GRID_Y0}"
              rx="12" fill="{ANIM_PANEL}" stroke="{ANIM_LINE}"/>
        <text x="{(GRID_X0+GRID_X1)//2}" y="{GRID_Y0-8}" text-anchor="middle" font-size="11.5"
              font-weight="700" fill="{ANIM_INK}">CSP projection</text>
        {''.join(grid_dots)}
        {band_labels}
        <text x="{(GRID_X0+GRID_X1)//2}" y="{GRID_Y1+32}" text-anchor="middle" font-size="9.5"
              fill="{ANIM_MUTED}">5 bands &times; 4 components</text>
      </g>

      <path id="path-logvar" d="{grid_to_out}" stroke="{ANIM_ACCENT}" stroke-width="2.2" fill="none"
            opacity="0.45" class="ribbon"/>
      <path id="logvar-arrow" d="M {OUT_X-13} {OUT_Y-22} L {OUT_X+13} {OUT_Y} L {OUT_X-13} {OUT_Y+22} Z"
            fill="{ANIM_ACCENT}" opacity="0.25"/>
      <text x="{OUT_X}" y="{OUT_Y-30}" text-anchor="middle" font-size="10.5" fill="{ANIM_MUTED}">Log-var</text>

      <path id="path-svm-in" d="{out_to_svm}" stroke="{ANIM_ACCENT}" stroke-width="2.2" fill="none"
            opacity="0.4" class="ribbon"/>
      <path id="svm-shape" d="{svm_shape}" stroke="{ANIM_ACCENT}" stroke-width="2" fill="none" opacity="0.35"/>
      <text x="{svm_mid}" y="{OUT_Y-68}" text-anchor="middle" font-size="11.5" font-weight="700"
            fill="{ANIM_INK}">SVM</text>
      <path id="path-final" d="{svm_to_final}" stroke="{result_color}" stroke-width="3" fill="none"
            opacity="0.6" class="ribbon"/>
      <circle id="final-dot" cx="{FINAL_X}" cy="{OUT_Y}" r="5" fill="{result_color}"/>
    </svg>
    """

    prob_rows = ""
    for name, val, is_top in ((pred_name, conf, True), (other_name, other_pct, False)):
        col    = result_color if is_top else ANIM_MUTED
        weight = 700 if is_top else 400
        prob_rows += f"""
        <div class="prob-row">
          <span style="color:{col};font-weight:{weight}">{name}</span>
          <div class="prob-bar-track"><div class="prob-bar-fill" data-w="{val}" style="width:0%;background:{col}"></div></div>
          <span style="color:{col};font-weight:{weight}">{val:.0f}%</span>
        </div>"""

    return f"""
    <style>
      .neuro-wrap2 {{
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        color: {ANIM_INK};
        display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap;
      }}
      .svg-col {{ max-width: 1180px; flex: 1 1 700px; min-width: 280px; }}
      .neuro-head {{
        display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:4px;
        font-size:13px; color:{ANIM_MUTED}; margin-bottom:4px;
      }}
      .flow-caption {{
        font-size:13px; color:{ANIM_ACCENT}; min-height:18px; margin-bottom:4px; font-weight:600;
      }}
      .ribbon {{ stroke-dasharray: 6 5; animation: flow 1.6s linear infinite; }}
      @keyframes flow {{ to {{ stroke-dashoffset: -22; }} }}
      .ribbon.hi {{ opacity: 0.95 !important; stroke-width: 3.4; filter: drop-shadow(0 0 3px currentColor); }}
      .lbl {{ opacity: 0.55; transition: opacity 0.25s ease; }}
      .lbl.hi {{ opacity: 1; }}
      #csp-grid {{ opacity: 0.5; transition: opacity 0.3s ease; }}
      #csp-grid.hi {{ opacity: 1; }}
      .grid-dot {{ transition: r 0.25s ease; }}
      #csp-grid.hi .grid-dot-top {{ r: 8.5; }}
      #svm-shape {{ transition: opacity 0.3s ease, stroke-width 0.3s ease; }}
      #svm-shape.hi {{ opacity: 0.9; stroke-width: 3; }}
      .ch-dot {{ transition: fill 0.2s ease, opacity 0.2s ease; }}
      .prob-panel {{
        background:{ANIM_PANEL}; border:1px solid {ANIM_LINE}; border-radius:16px; padding:20px 22px;
        flex: 0 0 280px; opacity: 0; transform: translateY(6px);
        transition: opacity 0.5s ease, transform 0.5s ease;
      }}
      .prob-panel.show {{ opacity: 1; transform: translateY(0); }}
      .prob-panel-title {{
        font-size:12px; font-weight:700; text-transform:uppercase;
        letter-spacing:0.06em; color:{ANIM_MUTED}; margin-bottom:14px;
      }}
      .prob-row {{ display:flex; align-items:center; gap:10px; font-size:14px; margin:13px 0; }}
      .prob-row span:first-child {{ width:118px; }}
      .prob-row span:last-child  {{ width:42px; text-align:right; }}
      .prob-bar-track {{ flex:1; height:8px; background:{ANIM_LINE}; border-radius:4px; overflow:hidden; }}
      .prob-bar-fill  {{ height:100%; border-radius:4px; transition:width 0.9s ease; }}
      .word-card {{
        margin-top:18px; text-align:center; padding:14px; border-radius:12px;
        background:{result_color}14; border:1px solid {result_color}55;
      }}
      .word-card .w {{ font-size:24px; font-weight:700; color:{result_color}; }}
      .word-card .t {{ font-size:12.5px; color:{ANIM_MUTED}; margin-top:4px; }}
    </style>
    <div class="neuro-head">
      <span>Trial pipeline &middot; imagined word &ldquo;{word}&rdquo;</span>
      <span>5 frequency bands &middot; per-subject CSP</span>
    </div>
    <div class="flow-caption" id="flow-caption">Positioning electrodes&hellip;</div>
    <div class="neuro-wrap2">
      <div class="svg-col">{svg}</div>
      <div class="prob-panel" id="prob-panel">
        <div class="prob-panel-title">Prediction</div>
        {prob_rows}
        <div class="word-card">
          <div class="w">{pred_name}</div>
          <div class="t">{"&#10003; matches true label" if correct else "&#10007; differs from true label"}</div>
        </div>
      </div>
    </div>

    <script>
    (function() {{
      const stages = {json.dumps([{"t": t, "d": d} for t, d in ANIM_STAGE_INFO])};
      const STEP_MS = 460;
      const chDots = Array.from(document.querySelectorAll('.ch-dot'));
      const accent = "{ANIM_ACCENT}";
      let i = 0;

      function clearHi() {{
        document.querySelectorAll('.hi').forEach(function(el) {{ el.classList.remove('hi'); }});
      }}

      function setStage(idx) {{
        const capt = document.getElementById('flow-caption');
        if (stages[idx]) capt.textContent = stages[idx].d;
        clearHi();

        if (idx === 0 || idx === 1) {{
          chDots.forEach(function(el, k) {{
            setTimeout(function() {{
              el.setAttribute('fill', accent);
              el.setAttribute('opacity', '1');
              setTimeout(function() {{
                el.setAttribute('fill', '{ANIM_INK}');
                el.setAttribute('opacity', '0.5');
              }}, 220);
            }}, k * 30);
          }});
        }}
        if (idx === 2) {{
          document.querySelectorAll('.lane-car, .lane-car-lbl, #path-car')
            .forEach(function(el) {{ el.classList.add('hi'); }});
        }}
        if (idx === 4 || idx === 5) {{
          document.querySelectorAll('.lane-bandpass, .lane-bandpass-lbl, #path-bandpass')
            .forEach(function(el) {{ el.classList.add('hi'); }});
          if (idx === 5) document.getElementById('csp-grid').classList.add('hi');
        }}
        if (idx === 6) {{
          document.querySelectorAll('.lane-cspvar, .lane-cspvar-lbl, #path-cspvar')
            .forEach(function(el) {{ el.classList.add('hi'); }});
          document.getElementById('csp-grid').classList.add('hi');
        }}
        if (idx === 7) {{
          document.querySelectorAll('#path-logvar, #path-svm-in').forEach(function(el) {{
            el.classList.add('hi');
          }});
        }}
        if (idx === 8) {{
          document.getElementById('svm-shape').classList.add('hi');
          document.getElementById('path-final').classList.add('hi');
        }}
        if (idx === 9) {{
          document.getElementById('path-final').classList.add('hi');
          const panel = document.getElementById('prob-panel');
          panel.classList.add('show');
          panel.querySelectorAll('.prob-bar-fill').forEach(function(el) {{
            el.style.width = el.getAttribute('data-w') + '%';
          }});
        }}
      }}

      function tick() {{
        if (i >= stages.length) return;
        setStage(i);
        i += 1;
        setTimeout(tick, STEP_MS);
      }}
      tick();
    }})();
    </script>
    """


# =============================================================================
# VIDEO PLAYBACK -- fullscreen, timeline-free "cutscene" player
# =============================================================================

def build_video_capture_html(video_b64: str, mime: str = "video/mp4",
                              nonce: str = "0") -> str:
    """
    Custom HTML5 video player (replaces st.video()) that:
      - Has NO native controls, so no seek bar / timeline is ever visible
      - Auto-plays muted (browser autoplay policy requires this without a
        prior user gesture)
      - Fills the ENTIRE browser viewport as a "cutscene" -- this does NOT
        use the real Fullscreen API, because Streamlit's components.html
        iframe is not marked allow="fullscreen" and the browser silently
        blocks requestFullscreen() inside it (a known Streamlit platform
        limitation). Instead, this script reaches into the parent page
        (window.parent.document -- same-origin, so this is allowed) and
        injects a fixed, full-viewport black overlay containing the video,
        covering the sidebar/buttons/everything -- visually identical to a
        game cutscene without depending on a browser permission that
        Streamlit blocks.
      - Lets the user hold ENTER for 5 seconds to skip: a thin progress bar
        fills at the bottom while held, and on completion the overlay is
        removed, instantly revealing the result already rendered on the
        underlying Streamlit page.
      - Also auto-removes the overlay when the clip finishes naturally.

    `nonce` should change on every trial (e.g. the trial index) so the
    component re-mounts and restarts playback instead of Streamlit re-using
    a cached iframe.
    """
    if not video_b64:
        return (
            '<div style="padding:2rem;text-align:center;'
            'font-family:sans-serif;color:#B3261E;'
            'border:1px dashed #E5B3AE;border-radius:10px">'
            'Video file not found. Update VIDEO_PATH at the top of the script.'
            '</div>'
        )

    return f"""
    <div id="cutscene-status-{nonce}" style="
        padding:10px 14px; font-family:-apple-system,'Segoe UI',Roboto,sans-serif;
        font-size:12px; color:#6B6B76; background:#F6F6F9; border-radius:8px;">
      Playing capture video in fullscreen&hellip; hold <strong>ENTER</strong> for 5s to skip.
    </div>

    <script>
    (function() {{
      const nonce = "{nonce}";
      const videoSrc = "data:{mime};base64,{video_b64}";

      let pdoc, pwin;
      try {{
        pdoc = window.parent.document;
        pwin = window.parent;
      }} catch (e) {{
        pdoc = document;
        pwin = window;
      }}

      // Clean up any previous cutscene overlay + listeners from an earlier
      // trial before creating a new one, so they don't stack up.
      const prevOverlay = pdoc.getElementById('bci-cutscene-overlay');
      if (prevOverlay) prevOverlay.remove();
      if (pwin.__bciCutsceneCleanup) {{
        try {{ pwin.__bciCutsceneCleanup(); }} catch (e) {{}}
      }}

      // ---- Build the full-viewport overlay in the PARENT document ----
      const overlay = pdoc.createElement('div');
      overlay.id = 'bci-cutscene-overlay';
      overlay.tabIndex = -1;
      overlay.style.cssText = [
        'position:fixed', 'top:0', 'left:0', 'width:100vw', 'height:100vh',
        'background:#000', 'z-index:2147483647', 'display:flex',
        'align-items:center', 'justify-content:center', 'flex-direction:column',
        'outline:none',
      ].join(';');

      const video = pdoc.createElement('video');
      video.src = videoSrc;
      video.muted = true;
      video.autoplay = true;
      video.playsInline = true;
      video.disablePictureInPicture = true;
      video.style.cssText = 'width:100%;height:100%;object-fit:contain;background:#000;';
      video.oncontextmenu = function() {{ return false; }};

      const hint = pdoc.createElement('div');
      hint.style.cssText = [
        'position:absolute', 'left:0', 'right:0', 'bottom:0', 'padding:16px 24px',
        'background:linear-gradient(transparent, rgba(0,0,0,0.7))',
        "font-family:-apple-system,'Segoe UI',Roboto,sans-serif",
        'color:#fff', 'font-size:13px', 'display:flex', 'flex-direction:column', 'gap:8px',
      ].join(';');
      hint.innerHTML =
        '<span>Hold <strong>ENTER</strong> for 5 seconds to skip</span>' +
        '<div style="height:5px;width:100%;background:rgba(255,255,255,0.25);' +
        'border-radius:3px;overflow:hidden;">' +
        '<div id="bci-skip-bar" style="height:100%;width:0%;background:#4C6FFF;' +
        'border-radius:3px;"></div></div>';

      overlay.appendChild(video);
      overlay.appendChild(hint);
      pdoc.body.appendChild(overlay);

      // The "Next Trial" button (or whatever was clicked to get here) still
      // holds keyboard focus at this point. On most browsers, pressing
      // ENTER while a <button> is focused re-clicks that button instead of
      // reaching our listener -- which is exactly why skip silently did
      // nothing. Move focus onto the overlay itself so ENTER has nothing
      // else to activate.
      if (pdoc.activeElement && typeof pdoc.activeElement.blur === 'function') {{
        pdoc.activeElement.blur();
      }}
      overlay.focus();

      const skipBar = hint.querySelector('#bci-skip-bar');

      function removeOverlay() {{
        video.pause();
        if (overlay.parentNode) overlay.remove();
        cleanup();
      }}

      // ---- Hold-ENTER-for-5s-to-skip ----
      let holdStart = null;
      let rafId = null;

      function tickHold() {{
        if (holdStart === null) return;
        const elapsed = Date.now() - holdStart;
        const pct = Math.min(100, (elapsed / 5000) * 100);
        if (skipBar) skipBar.style.width = pct + '%';
        if (elapsed >= 5000) {{
          holdStart = null;
          removeOverlay();
          return;
        }}
        rafId = pwin.requestAnimationFrame(tickHold);
      }}

      function onKeyDown(e) {{
        if (e.key !== 'Enter') return;
        // Stop the browser from treating this as "activate the focused
        // button" and stop it from reaching any other page-level handler.
        e.preventDefault();
        e.stopPropagation();
        if (holdStart === null) {{
          holdStart = Date.now();
          tickHold();
        }}
      }}
      function onKeyUp(e) {{
        if (e.key !== 'Enter') return;
        e.preventDefault();
        e.stopPropagation();
        holdStart = null;
        if (skipBar) skipBar.style.width = '0%';
        if (rafId) pwin.cancelAnimationFrame(rafId);
      }}
      function onEnded() {{ removeOverlay(); }}

      // Capture phase + listen on both document AND window so the ENTER
      // press is intercepted before it can reach the focused button or any
      // other page-level shortcut handler.
      pdoc.addEventListener('keydown', onKeyDown, true);
      pdoc.addEventListener('keyup', onKeyUp, true);
      pwin.addEventListener('keydown', onKeyDown, true);
      pwin.addEventListener('keyup', onKeyUp, true);
      video.addEventListener('ended', onEnded);

      function cleanup() {{
        pdoc.removeEventListener('keydown', onKeyDown, true);
        pdoc.removeEventListener('keyup', onKeyUp, true);
        pwin.removeEventListener('keydown', onKeyDown, true);
        pwin.removeEventListener('keyup', onKeyUp, true);
        pwin.__bciCutsceneCleanup = null;
      }}
      pwin.__bciCutsceneCleanup = cleanup;

      video.play().catch(function() {{
        // Autoplay blocked even muted (rare) -- let the user click the video.
        video.addEventListener('click', function() {{ video.play(); }}, {{ once: true }});
      }});
    }})();
    </script>
    """




# =============================================================================
# VOICE OUTPUT — decoded word -> sentence (API) -> speech (edge-tts)
# =============================================================================
#
#   decoded word "Hello"
#        -> Anthropic API writes a short natural sentence using that word
#           e.g. "Hello, how are you today?"
#        -> edge-tts synthesises that sentence to speech (MP3 bytes)
#        -> st.audio plays it in the UI
#
# Only 5 distinct words exist in this dataset, so results are cached per
# word — repeats (very common across trials / Demo Mode) cost no extra
# API calls or TTS synthesis.
# =============================================================================

EDGE_TTS_VOICE = "en-US-AriaNeural"   # any voice from `edge-tts --list-voices`

# Used when no API key is supplied, the anthropic SDK isn't installed, or
# the API call fails for any reason — the demo should never break because
# of this feature.
FALLBACK_SENTENCES = {
    "Hello":    "Hello, how are you today?",
    "Helpme":   "Please, can you help me right now?",
    "Stop":     "Stop, please wait just a moment.",
    "Thankyou": "Thank you so much for your help.",
    "Yes":      "Yes, that sounds good to me.",
}


def generate_sentence_from_word(word: str, api_key: str) -> str:
    """
    Calls the Anthropic API to turn a single decoded word into one short,
    natural spoken sentence that uses that word, e.g. "Hello" -> "Hello,
    how are you?". Falls back to a canned sentence if no key / SDK /
    network is available.
    """
    if not api_key or not ANTHROPIC_AVAILABLE:
        return FALLBACK_SENTENCES.get(word, word)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=40,
            messages=[{
                "role": "user",
                "content": (
                    f'The word "{word}" was just decoded from imagined '
                    f'speech on a BCI. Write ONE short, natural spoken '
                    f'sentence (under 12 words) that a person might say, '
                    f'using the word "{word}" naturally in it. Reply with '
                    f'ONLY the sentence — no quotes, no preamble.'
                ),
            }],
        )
        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", "") == "text"
        ).strip()
        return text or FALLBACK_SENTENCES.get(word, word)
    except Exception:
        return FALLBACK_SENTENCES.get(word, word)


async def _edge_tts_bytes(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def synthesize_speech(text: str, voice: str = EDGE_TTS_VOICE) -> Optional[bytes]:
    """Runs edge-tts synchronously and returns MP3 bytes, or None on failure."""
    if not EDGE_TTS_AVAILABLE or not text:
        return None
    try:
        return asyncio.run(_edge_tts_bytes(text, voice))
    except Exception:
        return None


def get_sentence_and_audio(word: str, api_key: str, voice: str):
    """
    Cached word -> (sentence, audio_bytes) lookup, stored in session state
    so reruns / repeated words (Demo Mode, Run All) don't re-call the API
    or re-synthesise audio unnecessarily.
    """
    cache = st.session_state.setdefault("voice_cache", {})
    key = (word, voice)
    if key in cache:
        return cache[key]
    sentence = generate_sentence_from_word(word, api_key)
    audio    = synthesize_speech(sentence, voice)
    cache[key] = (sentence, audio)
    return sentence, audio


# =============================================================================
# SESSION STATE
# =============================================================================

STATE_KEYS = [
    "idx", "history", "last_pred", "last_score", "last_true",
    "last_word", "last_eeg", "stage_idx", "artefact",
    "last_band_powers", "demo_running", "voice_cache",
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
# EEG AUTHENTICATION GATE — ported from auth_streamlit.py
# =============================================================================
# A login screen that uses a person's EEG brainwave as a biometric, gating
# access to the simulation dashboard below. Uses authentication.py's own
# enrolled subject model (auth_artifacts/subject_model.pkl), which is a
# separate model from the per-subject CSP+SVM models used by the simulation
# itself — run `python authentication.py self-test` (or `enroll`) once to
# produce it.

def load_auth_models():
    """Load the saved authenticator + word predictor for the login gate."""
    auth = None
    wp = None
    if SUBJECT_MODEL_PATH.exists():
        with open(SUBJECT_MODEL_PATH, "rb") as fh:
            auth = pickle.load(fh)["auth"]
    if WORD_MODEL_PATH.exists():
        with open(WORD_MODEL_PATH, "rb") as fh:
            wp = pickle.load(fh)
    return auth, wp


def load_auth_eer():
    if EER_PATH.exists():
        with open(EER_PATH) as fh:
            return json.load(fh)
    return None


def auth_result_card_html(res, word=None, true_word=None):
    if res["granted"]:
        color, icon, status = COL_GREEN, "&#10003;", "ACCESS GRANTED"
    else:
        color, icon, status = COL_RED, "&#10007;", "ACCESS DENIED"

    score = res["score"]
    thr   = res["threshold"]
    conf  = res["confidence"]

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
            &#128274; Word prediction locked — authentication failed.
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


def render_auth_gate():
    """
    Full-page EEG login gate. Returns True once the person has been granted
    access in this session (and the dashboard below should render); False
    while the gate itself is still on screen.
    """
    st.markdown(
        "<h2 style='margin-bottom:2px'>&#128274; BCI EEG Authentication Gate</h2>"
        "<p style='color:#73726c;font-size:13px;margin-top:0'>"
        "Biometric brainwave login &middot; GRANTED &rarr; enter simulation "
        "&middot; DENIED &rarr; locked</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if not AUTH_MODULE_AVAILABLE:
        st.error(
            "`authentication.py` wasn't found next to this script. Place it "
            "in the same folder and reload."
        )
        st.stop()

    auth, wp = load_auth_models()
    eer = load_auth_eer()

    if auth is None:
        st.warning(
            "No enrolled subject model found. Run one of:\n\n"
            "`python authentication.py enroll --path <dataset_root>`\n\n"
            "or\n\n"
            "`python authentication.py self-test`\n\n"
            "then reload this page."
        )
        st.stop()

    dataset_root = auth_discover_dataset()
    mat_map = auth_discover_mat_files(dataset_root) if dataset_root is not None else None
    threshold = auth.threshold if auth.threshold is not None else 0.0

    with st.sidebar:
        st.markdown("### Dataset")
        if dataset_root is not None:
            st.success(f"Found: `{dataset_root}`")
        else:
            st.info("Dataset not auto-found.")

        subjects = sorted(mat_map.keys()) if mat_map else (
            sorted(auth.classes_) if auth.classes_ else []
        )

        st.markdown("### Claimed identity")
        subject   = st.selectbox("Subject", subjects, key="auth_subject_select")
        split     = st.selectbox("Split", ["val", "train", "test"], key="auth_split_select")
        trial_idx = st.number_input("Trial index", min_value=0, value=0, step=1,
                                     key="auth_trial_select")

        st.markdown("### System")
        st.metric("Threshold", f"{threshold:.3f}")
        if eer:
            st.metric("EER (validation)", f"{eer['eer']*100:.2f}%")
        st.caption("EER = Equal-Error-Rate. Lower is better.")

        st.divider()
        st.markdown("""
**Classes:** Hello, Helpme, Stop, Thankyou, Yes

**Who is enrolled?** Only subjects used in enrollment can be recognised.
A subject not in the set will always be **DENIED**.
""")

    left, right = st.columns([3, 2], gap="large")

    F, true_word = None, None
    with left:
        st.markdown("""
### How this works
Each person's EEG is unique — a **"brainprint"**. This gate uses a single
trial to verify a claimed identity:

1. **CAR** spatial filtering removes common noise.
2. **Epoch trim** keeps the post-stimulus window.
3. **Hanning taper + 5-band bandpass** isolates theta/mu/beta/gamma rhythms.
4. **Per-channel log band-power + broadband log-variance** form the subject
   fingerprint.
5. A one-vs-rest **SVM** scores the trial against every enrolled subject.
6. If the score **≥ threshold** → **GRANTED** and the simulation unlocks.
   Otherwise → **DENIED**.
""")

        if mat_map is None or subject not in mat_map:
            st.error("No dataset/mat files for this subject.")
        else:
            fpaths = mat_map[subject].get(split) or mat_map[subject].get("train")
            if not fpaths:
                st.error(f"No {split} data for {subject}.")
            else:
                fpath = fpaths[0]
                epo_key = {"train": "epo_train", "val": "epo_validation",
                           "test": "epo_test"}[split]
                try:
                    X, y = auth_load_epoch(fpath, epo_key)
                    n_trials = X.shape[0]
                    trial_idx = int(min(trial_idx, n_trials - 1))
                    trial_X = X[trial_idx: trial_idx + 1]
                    F = extract_subject_features(trial_X)
                    true_word = CLASS_NAMES.get(int(y[trial_idx]), "?")
                    st.caption(f"Trial {trial_idx}/{n_trials-1} · true word: {true_word}")
                except Exception as e:
                    st.error(f"Load error: {e}")
                    F = None

    granted_now = False
    with right:
        st.markdown("**Authentication**")
        if F is not None:
            res = auth.authenticate(F, subject)
            word = None
            if res["granted"] and wp is not None:
                _, wname = wp.predict(F, subject)
                word = wname
            st.markdown(auth_result_card_html(res, word, true_word),
                        unsafe_allow_html=True)

            if res["granted"]:
                granted_now = True
                if st.button("Enter simulation \u2192", type="primary",
                             use_container_width=True):
                    st.session_state.authenticated  = True
                    st.session_state.auth_subject    = subject
                    st.rerun()
            else:
                st.button("Enter simulation \u2192", disabled=True,
                          use_container_width=True,
                          help="Authentication must succeed first.")
        else:
            st.markdown(
                "<div style='color:#73726c;font-size:13px;padding:40px 20px;"
                "text-align:center;border:1px dashed #D3D1C7;border-radius:10px'>"
                "No trial data available for authentication.</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("""
**Note:** If the dataset isn't present on this machine, run
`python authentication.py self-test` to generate synthetic enrolled subjects,
then reload. Any subject in the synthetic set will be granted; an
out-of-set subject will be denied.
""")
    return granted_now


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(
        page_title="BCI Imagined Speech Simulation",
        page_icon="🧠",
        layout="wide",
    )

    # ── EEG authentication gate ──────────────────────────────────────────
    # The dashboard below (Normal / Video / Animated modes, voice output,
    # everything) is unchanged and only unlocks once the person's EEG trial
    # is verified against an enrolled subject model.
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("auth_subject", None)

    if not st.session_state.authenticated:
        render_auth_gate()
        return

    with st.sidebar:
        st.success(f"🔓 Authenticated as **{st.session_state.auth_subject}**")
        if st.button("Log out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.auth_subject  = None
            st.rerun()
        st.divider()

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
        st.markdown("### Display mode")
        display_mode = st.radio(
            "Choose how a trial is visualised",
            [
                "1. Normal (plots)",
                "2. Video playback",
                "3. Animated illustration",
            ],
            index=0,
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
        st.markdown("### Voice output")
        enable_voice = st.checkbox(
            "Speak decoded word as a sentence",
            value=False,
            help="Word -> API generates a sentence -> edge-tts speaks it.",
        )
        tts_api_key = st.text_input(
            "Anthropic API key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            help="Leave blank to use built-in fallback sentences.",
        )
        if enable_voice and not EDGE_TTS_AVAILABLE:
            st.warning("`edge-tts` not installed — run `pip install edge-tts`.")
        if enable_voice and not ANTHROPIC_AVAILABLE:
            st.caption(
                "`anthropic` not installed — using fallback sentences. "
                "Run `pip install anthropic` for real API sentences."
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
    is_normal = display_mode.startswith("1")
    is_video  = display_mode.startswith("2")
    is_anim   = display_mode.startswith("3")

    eeg_ph  = pipe_ph = band_ph = acc_ph = None
    video_ph = anim_ph = None

    if is_normal:
        left, right = st.columns([3, 2], gap="large")
    elif is_video:
        left, right = st.columns([3, 2], gap="large")
    else:
        left, right = st.columns([1, 1], gap="large")  # animation needs more room

    with left:
        if is_normal:
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

        elif is_video:
            video_ph = st.empty()
            if st.session_state.last_pred is not None:
                video_b64 = load_video_base64(str(VIDEO_PATH))
                if video_b64 is None:
                    video_ph.error(
                        f"Video file not found at `{VIDEO_PATH}`. "
                        "Update VIDEO_PATH at the top of this script to "
                        "point at your .mp4 file."
                    )
                else:
                    with video_ph:
                        components.html(
                            build_video_capture_html(
                                video_b64, nonce=str(idx)
                            ),
                            height=60,
                            scrolling=False,
                        )
                    st.caption(
                        f"Trial {idx} · predicted word: "
                        f"**{st.session_state.last_word}** · "
                        "hold ENTER 5s to skip to the result"
                    )
            else:
                video_ph.info("Click Next Trial to play the capture video.")

        else:  # animated illustration — spans full width below
            pass

    if is_anim:
        anim_ph = st.empty()
        if st.session_state.last_pred is not None:
            html = build_capture_animation_html(
                word=st.session_state.last_word,
                pred_label=st.session_state.last_pred,
                true_label=st.session_state.last_true,
                band_powers=st.session_state.last_band_powers,
                dec_score=st.session_state.last_score,
            )
            with anim_ph:
                components.html(html, height=560, scrolling=False)
        else:
            anim_ph.info("Click Next Trial to run the animation.")

    with right:
        if not is_anim:
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
        else:
            # The Animated illustration widget renders its own prediction
            # panel inline, so nothing is duplicated here. result_ph still
            # needs to exist (Demo Mode refreshes it below) — make it a
            # harmless no-op placeholder.
            result_ph = st.empty()

        # ── Voice output: word -> sentence (API) -> speech (edge-tts) ──────
        if enable_voice and st.session_state.last_word:
            sentence, audio = get_sentence_and_audio(
                st.session_state.last_word, tts_api_key, EDGE_TTS_VOICE,
            )
            st.caption(f'🗣️ "{sentence}"')
            if audio:
                st.audio(audio, format="audio/mp3")
            elif EDGE_TTS_AVAILABLE:
                st.caption("Voice synthesis failed for this trial.")

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

        if animate and is_normal and show_pipeline and pipe_ph is not None:
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
        process_one(idx, animate=(is_normal and show_pipeline))

        # Refresh live displays for the current display mode
        if is_normal:
            if show_eeg and st.session_state.last_eeg is not None and eeg_ph is not None:
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
        elif is_video and video_ph is not None:
            video_b64 = load_video_base64(str(VIDEO_PATH))
            if video_b64 is not None:
                with video_ph:
                    components.html(
                        build_video_capture_html(
                            video_b64, nonce=str(st.session_state.idx)
                        ),
                        height=60,
                        scrolling=False,
                    )
        elif is_anim and anim_ph is not None:
            html = build_capture_animation_html(
                word=st.session_state.last_word,
                pred_label=st.session_state.last_pred,
                true_label=st.session_state.last_true,
                band_powers=st.session_state.last_band_powers,
                dec_score=st.session_state.last_score,
            )
            with anim_ph:
                components.html(html, height=560, scrolling=False)

        if not is_anim:
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
