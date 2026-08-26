# BCI Imagined Speech — Decoder & EEG Authentication Gate

A college major project that turns pre-recorded EEG into two things:

1. **A biometric login** — your EEG trial *is* your password. A one-vs-rest SVM
   trained on your brain's spectral fingerprint decides `GRANTED` or `DENIED`.
2. **A binary imagined-speech decoder** — once inside, a per-subject CSP + SVM
   pipeline classifies each trial as **Bilabial** (words containing /p/ or /m/:
   *Helpme*, *Stop*) or **Non-Bilabial** (*Hello*, *Thankyou*, *Yes*), replayed
   trial-by-trial through a real-time-style Streamlit simulator.

Built on the [BCI2020 EEG Signal for Words](https://www.kaggle.com/datasets/kylesnowbc/2020-international-bci-competition-track-3)
imagined-speech dataset (15 subjects, 5 words, 64-channel EEG, 256 Hz).

---

## How it works

Both the authenticator and the word decoder share the same signal-processing
front end, then diverge:

Raw EEG Trial (64 Channels × Time)
            │
            ▼
 Common Average Reference (CAR)
            │
            ▼
 Epoch Trim (Post-Stimulus Window)
            │
            ▼
 Hanning Taper
            │
            ▼
 Bandpass Filtering × 5
 (Theta, Mu, Beta1, Beta2, Gamma)
            │
     ┌──────┴──────┐
     ▼             ▼

AUTHENTICATION   WORD DECODING
     │             │
Per-Channel      CSP
Log Band-Power   (4 Components/Band)
+ Broadband      │
Log-Variance     ▼
     │        Log-Variance
     ▼        (20 Features)
One-vs-Rest       │
LinearSVC         ▼
     │       StandardScaler
     ▼             │
Subject Score      ▼
     │         LinearSVC
     ▼             │
EER Threshold      ▼
     │      Bilabial /
     ▼      Non-Bilabial
 Authentication
     │
 ┌───┴───┐
 ▼       ▼

GRANTED  DENIED
   │        │
Unlock   Word Prediction
Simulator   Locked

The authentication threshold is tuned automatically during enrollment by
computing genuine-vs-imposter scores on the validation split and finding the
**Equal Error Rate (EER)** — the point where false-accept rate equals
false-reject rate.

---

## Project structure

| File | What it does |
|---|---|
| `bci_simulation.py` | Main Streamlit app. Opens on the **EEG authentication gate**; once `GRANTED`, unlocks the full simulation dashboard (see [Display modes](#display-modes) below). |
| `authentication.py` | Core auth module — dataset discovery, signal processing, feature extraction, `SubjectAuthenticator`, `WordPredictor`, EER/threshold tuning, and a CLI (`enroll`, `stats`, `login`, `simulate`, `self-test`, `tune`). |
| `auth_streamlit.py` | Standalone version of the login gate (same logic as the gate baked into `bci_simulation.py`, useful if you want authentication as its own page). |
| `TODO.md` | Build checklist / progress notes for the authentication module. |

> `bci_simulation.py` does `from authentication import ...`, so keep both
> files in the same folder.

---

## Installation

```bash
pip install streamlit numpy scipy scikit-learn plotly

# optional — only needed for the "voice output" feature
# (imagined word → spoken sentence)
pip install edge-tts anthropic
```

Python 3.9+ recommended.

---

## Dataset setup

Point the app at your local copy of the **BCI2020 EEG Signal for Words**
dataset. It expects `Training set` / `Validation set` / `Test set`
subfolders, each containing one `.mat` file per subject
(`Data_Sample01.mat` … `Data_Sample15.mat`, or `S01_Training.mat` style —
both naming conventions are supported).

Either drop the dataset into one of the hardcoded `COMMON_PATHS` in
`authentication.py`, or pass `--path` explicitly on the CLI (see below).

---

## Usage

### 1. Train the authentication model

`self-test` fabricates synthetic EEG for 5 fake subjects, just to prove the
pipeline runs end-to-end — it will **never recognise your real dataset**:

```bash
python authentication.py self-test
```

To actually enroll on your real subjects:

```bash
# sanity-check what subjects/files get discovered first
python authentication.py stats --path "/path/to/BCI2020 EEG Signal for Words"

# train for real — extracts features, fits the subject SVM,
# tunes the EER threshold, and trains the per-subject word decoder
python authentication.py enroll --path "/path/to/BCI2020 EEG Signal for Words"
```

This writes to `auth_artifacts/`:

| File | Contents |
|---|---|
| `subject_model.pkl` | The enrolled `SubjectAuthenticator` |
| `word_models.pkl` | Per-subject `WordPredictor` (5-class: Hello/Helpme/Stop/Thankyou/Yes) |
| `eer.json` | Equal Error Rate + tuned accept/reject threshold |
| `subject_templates.json` | Per-subject trial counts / metadata |

Other CLI modes:

```bash
python authentication.py login --subject S01 --trial 3   # test one login attempt
python authentication.py simulate --path <dataset_root>  # interactive genuine/imposter demo
```

### 2. Run the app

```bash
streamlit run bci_simulation.py
```

- **Login gate** — pick a claimed subject + trial in the sidebar. `GRANTED`
  reveals a confidence bar and an **"Enter simulation →"** button. `DENIED`
  locks the word prediction and keeps that button disabled.
- Once inside, the sidebar shows `🔓 Authenticated as S01` with a **Log out**
  button that drops you back to the gate.

---

## Display modes

Inside the simulator, pick how each trial is visualised:

1. **Normal (plots)** — live EEG waveform, pipeline-stage badges, band-power
   bar chart, running balanced-accuracy chart.
2. **Video playback** — plays a local `.mp4` capture clip fullscreen as a
   timeline-free "cutscene" each time a trial runs (hold **Enter** 5s to
   skip). Set `VIDEO_PATH` at the top of `bci_simulation.py`.
3. **Animated illustration** — a Sankey-style flow diagram: EEG channels fan
   into color-coded CAR / Bandpass / CSP-variance ribbons, converge on a CSP
   projection grid, flow through an SVM funnel, and resolve into a live
   prediction panel — animated step-by-step through the real pipeline stages.

Other controls: **Next Trial** (step through one at a time), **Run All**,
**Demo Mode** (auto-advances with a configurable delay), and a **Methodology
report** with the full write-up of design choices, per-subject results, and
limitations.

---

## Voice output (optional)

Toggle **"Speak decoded word as a sentence"** in the sidebar to have each
decoded word turned into a short natural sentence (via the Anthropic API,
with canned fallbacks if no key is supplied) and spoken aloud with
`edge-tts`. Results are cached per word so repeats in Demo Mode don't re-call
the API.

---

## Results (validation set)

| Metric | Value |
|---|---|
| Mean balanced accuracy (word decoding) | 61.6% ± 5.2% |
| Best / worst subject | 70.0% / 53.3% |
| Significant subjects (p < 0.05, permutation test) | 6 / 15 |
| Authentication EER (self-test, synthetic) | 0.00% |

See the in-app **Methodology report** for the full breakdown: CSP design
choices, classifier design, frequency-band rationale, and limitations
(small validation set, no online adaptation, simulated rather than
true real-time operation).

---

## Known limitations

- Both models assume **stationary EEG statistics** — no online recalibration
  as electrode impedance or brain state drifts over a session.
- The authentication gate can only recognise subjects it was enrolled on;
  anyone outside the enrolled set is always `DENIED` by design.
- This is a simulation over pre-recorded trials, not a live streaming BCI —
  real-time operation would need an amplifier SDK (e.g. Lab Streaming Layer)
  and online epoch extraction.

---

## Credits

Dataset: *BCI2020 — International BCI Competition, Track 3 (Imagined Speech
Classification)*.
