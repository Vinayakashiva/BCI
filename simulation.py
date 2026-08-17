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
=============================================================================
"""

import time
import json
import base64
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from scipy import signal as sp_signal
from scipy.io import loadmat
from sklearn.metrics import balanced_accuracy_score
from ai_engine import process_prediction, text_to_wav
from speech_imagery import generate_speech_imagery
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

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_ROOT = BASE_DIR
VAL_DIR = DATASET_ROOT / "Validation set"
ARTIFACT_DIR = DATASET_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "per_subject_models.pkl"

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
                                  band_powers: dict, height: int = 460) -> str:
    """
    A single self-contained HTML/CSS/SVG/JS animation that steps through the
    real pipeline stages (ANIM_STAGE_INFO) with a professional, minimal
    design language: a head silhouette with electrodes, a flowing signal
    line, a stage tracker with a moving progress rail, and a final result
    reveal card. Auto-plays once on mount (Streamlit re-renders this
    component fresh every time a trial is processed).
    """
    result_color = ANIM_SUCCESS if pred_label == 1 else ANIM_ACCENT
    correct      = (pred_label == true_label)
    stages_json  = ANIM_STAGE_INFO

    stage_dots = "".join(
        f'<div class="stage" id="stage-{i}">'
        f'<div class="dot"></div>'
        f'<div class="stage-text"><div class="stage-title">{title}</div>'
        f'<div class="stage-desc">{desc}</div></div>'
        f'</div>'
        for i, (title, desc) in enumerate(stages_json)
    )

    band_bars = ""
    if band_powers:
        vals = list(band_powers.items())
        vmax = max(abs(v) for _, v in vals) or 1.0
        for name, v in vals:
            pct = min(100, max(6, int(abs(v) / vmax * 100)))
            band_bars += (
                f'<div class="band-row">'
                f'<span class="band-label">{name}</span>'
                f'<div class="band-track"><div class="band-fill" '
                f'style="width:{pct}%"></div></div>'
                f'</div>'
            )

    return f"""
    <style>
      .neuro-wrap {{
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        color: {ANIM_INK};
        display: grid;
        grid-template-columns: 230px 1fr 220px;
        gap: 18px;
        align-items: start;
      }}
      .neuro-card {{
        background: {ANIM_PANEL};
        border: 1px solid {ANIM_LINE};
        border-radius: 14px;
        padding: 14px 16px;
      }}
      .stage {{
        display: flex; align-items: flex-start; gap: 10px;
        padding: 7px 4px; border-radius: 8px; opacity: 0.35;
        transition: opacity 0.35s ease, background 0.35s ease;
      }}
      .stage.active {{ opacity: 1; background: rgba(76,111,255,0.08); }}
      .stage.done   {{ opacity: 0.75; }}
      .dot {{
        width: 9px; height: 9px; border-radius: 50%;
        background: {ANIM_LINE}; margin-top: 5px; flex: none;
        transition: background 0.3s ease, box-shadow 0.3s ease;
      }}
      .stage.active .dot {{
        background: {ANIM_ACCENT};
        box-shadow: 0 0 0 4px rgba(76,111,255,0.18);
      }}
      .stage.done .dot {{ background: {ANIM_ACCENT_2}; }}
      .stage-title {{ font-size: 12.5px; font-weight: 600; line-height: 1.3; }}
      .stage-desc  {{ font-size: 10.5px; color: {ANIM_MUTED}; line-height: 1.35; margin-top: 1px; }}

      .stage-panel-title {{
        font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
        text-transform: uppercase; color: {ANIM_MUTED}; margin-bottom: 8px;
      }}

      .center-panel {{ display:flex; flex-direction:column; align-items:center; gap:10px; }}
      .flow-caption {{
        font-size: 12px; color: {ANIM_MUTED}; text-align:center; min-height: 16px;
      }}

      .band-row {{ display:flex; align-items:center; gap:8px; margin:6px 0; }}
      .band-label {{ font-size: 10.5px; width: 42px; color:{ANIM_MUTED}; text-transform:capitalize; }}
      .band-track {{ flex:1; height:7px; background:{ANIM_LINE}; border-radius:4px; overflow:hidden; }}
      .band-fill {{ height:100%; background:{ANIM_ACCENT}; border-radius:4px;
                    transition: width 1s ease; }}

      .result-card {{
        margin-top: 12px; border-radius: 12px; padding: 14px;
        background: {result_color}14; border: 1px solid {result_color}55;
        text-align: center; opacity: 0; transform: translateY(6px);
        transition: opacity 0.5s ease, transform 0.5s ease;
      }}
      .result-card.show {{ opacity: 1; transform: translateY(0); }}
      .result-word {{ font-size: 22px; font-weight: 700; color: {result_color}; }}
      .result-sub  {{ font-size: 11px; color: {ANIM_MUTED}; margin-top: 2px; }}
      .result-tag  {{
        display:inline-block; margin-top:8px; font-size:10.5px; font-weight:600;
        padding:2px 8px; border-radius:20px;
        color: {ANIM_SUCCESS if correct else "#B3261E"};
        background: {ANIM_SUCCESS + "1A" if correct else "#B3261E1A"};
      }}
    </style>

    <div class="neuro-wrap">

      <div class="neuro-card">
        <div class="stage-panel-title">Pipeline</div>
        {stage_dots}
      </div>

      <div class="neuro-card center-panel">
        <div class="stage-panel-title" style="align-self:flex-start">Signal capture</div>
        <svg viewBox="0 0 320 230" style="width:100%;max-width:320px">
          <ellipse cx="160" cy="118" rx="86" ry="100" fill="none" stroke="{ANIM_LINE}" stroke-width="2"/>
          <path d="M 78 82 A 86 100 0 0 1 242 82" fill="none" stroke="{ANIM_LINE}" stroke-width="2"/>
          <g id="electrodes"></g>
          <g id="signal-line" opacity="0">
            <path id="wave-path" d="M 40 200 q 10 -18 20 0 q 10 18 20 0 q 10 -18 20 0 q 10 18 20 0"
                  fill="none" stroke="{ANIM_ACCENT}" stroke-width="2"/>
          </g>
        </svg>
        <div class="flow-caption" id="flow-caption">Positioning electrodes…</div>
      </div>

      <div class="neuro-card">
        <div class="stage-panel-title">Band power</div>
        <div id="band-container">{band_bars if band_bars else '<div style="font-size:11px;color:' + ANIM_MUTED + '">Computed after CSP…</div>'}</div>
        <div class="result-card" id="result-card">
          <div class="result-word" id="result-word">{word}</div>
          <div class="result-sub">{BINARY_NAMES[pred_label]}</div>
          <div class="result-tag">{"&#10003; Matches true label" if correct else "&#10007; Differs from true label"}</div>
        </div>
      </div>

    </div>

    <script>
    (function() {{
      const stages = {json.dumps([{"t": t, "d": d} for t, d in stages_json])};
      const positions = [[160,44],[196,50],[124,50],[220,68],[100,68],
                          [232,96],[88,96],[220,128],[100,128],[160,150]];
      const eg = document.getElementById('electrodes');
      positions.forEach((p,i)=>{{
        const c = document.createElementNS("http://www.w3.org/2000/svg","circle");
        c.setAttribute("cx",p[0]); c.setAttribute("cy",p[1]); c.setAttribute("r",5.5);
        c.setAttribute("fill","#FFFFFF"); c.setAttribute("stroke","{ANIM_LINE}");
        c.setAttribute("stroke-width","2"); c.setAttribute("id","e"+i);
        eg.appendChild(c);
      }});

      const STEP_MS = 480;
      let i = 0;

      function setStage(idx) {{
        for (let k=0; k<10; k++) {{
          const el = document.getElementById('stage-'+k);
          if (!el) continue;
          el.classList.remove('active');
          if (k < idx) el.classList.add('done'); else el.classList.remove('done');
          if (k === idx) el.classList.add('active');
        }}
        const capt = document.getElementById('flow-caption');
        if (stages[idx]) capt.textContent = stages[idx].d;
      }}

      function tick() {{
        if (i >= 10) return;
        setStage(i);

        if (i === 0) {{
          positions.forEach((p, k) => {{
            setTimeout(() => {{
              const el = document.getElementById('e'+k);
              el.setAttribute("fill", "{ANIM_ACCENT}");
              setTimeout(() => el.setAttribute("fill", "#FFFFFF"), 200);
            }}, k * 35);
          }});
        }}
        if (i === 5) {{
          document.getElementById('signal-line').setAttribute('opacity', '1');
        }}
        if (i === 9) {{
          setTimeout(() => {{
            document.getElementById('result-card').classList.add('show');
          }}, 150);
        }}

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
# SESSION STATE
# =============================================================================

STATE_KEYS = [
    "idx", "history", "last_pred", "last_score", "last_true",
    "last_word", "last_eeg", "stage_idx", "artefact",
    "last_band_powers", "demo_running",
    "generated_sentence", "generated_audio", "ai_confidence",
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
    st.session_state.setdefault("generated_sentence", "")
    st.session_state.setdefault("generated_audio",    None)
    st.session_state.setdefault("ai_confidence",      0.0)

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
            )
            with anim_ph:
                components.html(html, height=430, scrolling=False)
        else:
            anim_ph.info("Click Next Trial to run the animation.")

    with right:
        st.markdown("**Result**")
        result_ph = st.empty()
        if st.session_state.last_pred is not None and not is_anim:
            # Animated illustration mode already shows its own result card,
            # so avoid duplicating it here.
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

            if (
                    "generated_sentence" in st.session_state
                    and "generated_audio" in st.session_state
            ):
                st.markdown("### 🧠 AI Engine Output")

                st.success(st.session_state.generated_sentence)

                st.audio(
                    st.session_state.generated_audio,
                    format="audio/wav"
                )
        elif is_anim and st.session_state.last_pred is not None:
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

    def process_one(ci, animate=True, generate_audio=True):
        """Process exactly one EEG trial without changing the global index."""
        word = CLASS_NAMES.get(int(y_raw[ci]), "?")
        true_bin = int(y_bin[ci])
        trial_X = X_val[ci:ci + 1]

        # Pipeline animation
        if animate and is_normal and show_pipeline and pipe_ph is not None:
            for si in range(len(PIPELINE_STAGES)):
                pipe_ph.markdown(pipeline_html(si), unsafe_allow_html=True)
                time.sleep(0.08)

        # Run EEG pipeline
        pred, dec, eeg, art, bp = run_pipeline(trial_X, model)

        # Store/display trial results
        record_trial(ci, pred, dec, eeg, art, bp, word, true_bin)

        # ==========================
        # AI ENGINE
        # ==========================
        confidence = min(99, max(51, 50 + abs(dec) * 15)) / 100.0

        imagined_word = generate_speech_imagery(
            word,
            verbose=False
        )

        sentence = process_prediction(
            anchor=imagined_word,
            emotion="neutral",
            confidence=confidence
        )

        st.session_state.generated_sentence = sentence
        st.session_state.ai_confidence = confidence

        # Generate a unique WAV for this trial
        if generate_audio:
            audio_file = BASE_DIR / f"output_trial_{ci + 1}.wav"
            text_to_wav(sentence, str(audio_file))
            st.session_state.generated_audio = str(audio_file)

        return sentence


    # Next Trial — process exactly one sample
    if next_btn and idx < n_trials:
        process_one(idx, animate=True, generate_audio=True)
        st.session_state.idx = idx + 1
        st.rerun()

    # Run All — process every remaining sample exactly once
    if run_all_btn and idx < n_trials:
        start_idx = int(st.session_state.idx)
        remaining = n_trials - start_idx

        with st.spinner(
            f"Processing all {remaining} remaining trials "
            f"({start_idx + 1}–{n_trials})..."
        ):
            for i in range(start_idx, n_trials):
                process_one(
                    i,
                    animate=False,
                    generate_audio=True
                )

                # Advance only after that trial has completed
                st.session_state.idx = i + 1

        # Force the UI to show 50/50 and prevent the Run All button
        # from starting the same run again. This is one controlled rerun.
        st.session_state.idx = n_trials
        st.session_state.demo_running = False
        st.rerun()

    # Demo Mode — auto-step
    if st.session_state.demo_running and idx < n_trials:
        process_one(idx, animate=(is_normal and show_pipeline),generate_audio=False)

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
            )
            with anim_ph:
                components.html(html, height=430, scrolling=False)

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
        if (
                "generated_sentence" in st.session_state
                and st.session_state.get("generated_audio")
        ):
            st.markdown("### 🧠 AI Engine Output")

            st.success(st.session_state.generated_sentence)

            st.audio(
                st.session_state.generated_audio,
                format="audio/wav"
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
