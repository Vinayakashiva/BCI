"""
=============================================================================
BCI EEG AUTHENTICATION GATE — Imagined Speech Access Control
=============================================================================
A separate, standalone module that uses a person's EEG brainwave as a
biometric "password".

Flow
----
1. A user presents a single EEG trial (idle / imagined speech).
2. The system extracts subject-id features (per-band log power + broadband
   log-variance) and asks: "Does this wave match an enrolled person?"
3. If the match score >= threshold  ->  GRANT ACCESS  ->  run the word
   predictor (Hello / Helpme / Stop / Thankyou / Yes).
4. If the match score <  threshold  ->  DENY ACCESS  ->  word prediction is
   locked (the person is not who they claim to be / not enrolled).

Pipeline (feature extraction, mirrors train.py)
-----------------------------------------------
CAR -> epoch trim -> Hanning taper -> 5-band bandpass -> per-channel
log band-power + broadband log-variance -> StandardScaler -> subject model.

Subject model: One-vs-Rest LinearSVC over enrolled subjects. The decision
score on the claimed subject's class is used as the accept/reject score.
Threshold may be tuned via Equal-Error-Rate (EER) on genuine vs imposter
trials.

Dataset layout (supported)
--------------------------
The BCI2020 imagined-speech dataset (Kaggle) uses per-subject .mat files
with 64 channels @ 256 Hz and keys epo_train / epo_validation / epo_test.
Auto-discovery searches common locations for a folder containing
"Training set" / "Validation set" / "Test set", or a flat set of *_Training*
/ *_Validation* / *_Test* .mat files.

If the dataset is not present, run `--self-test` to generate synthetic data
that exercises the whole machinery end-to-end.

Usage
-----
    python authentication.py enroll --path <dataset_root>
    python authentication.py stats --path <dataset_root>
    python authentication.py login --subject S01 --trial 3 --path <dataset_root>
    python authentication.py simulate --path <dataset_root>
    python authentication.py self-test            # no dataset needed
    python authentication.py tune --path <dataset_root>

Requirements
------------
    pip install numpy scipy scikit-learn joblib
=============================================================================
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

FS              = 256          # Hz
TARGET_CHANNELS = 64           # BCI2020 uses 64 channels
BP_ORDER        = 4

# Default epoch window (sample indices at 256 Hz; sample 128 = 0 ms onset)
EPOCH_START = 128
EPOCH_END   = 512

FREQ_BANDS = {
    "theta" : (4,  8),
    "mu"    : (8,  12),
    "beta1" : (12, 20),
    "beta2" : (20, 30),
    "gamma" : (30, 50),
}

ARTIFACT_THRESHOLD_UV = 800.0

# Subject labels for words (from the BCI2020 dataset)
CLASS_NAMES = {1: "Hello", 2: "Helpme", 3: "Stop", 4: "Thankyou", 5: "Yes"}

# Artifact directory for this module's outputs
AUTH_DIR = Path(__file__).resolve().parent / "auth_artifacts"
SUBJECT_MODEL_PATH = AUTH_DIR / "subject_model.pkl"
SUBJECT_TEMPLATES_PATH = AUTH_DIR / "subject_templates.json"
WORD_MODEL_PATH = AUTH_DIR / "word_models.pkl"
EER_PATH = AUTH_DIR / "eer.json"

RANDOM_STATE = 42

# Common dataset locations to auto-discover (in priority order)
COMMON_PATHS = [
    Path(r"C:\Users\Admin\OneDrive\Documents\mojar_project\BCI2020 EEG Signal for Words"),
    Path("C:/Users/Bhuva/Downloads/BCI2020 EEG Signal for Words"),
    Path("C:/Users/Bhuva/Downloads/BCI2020"),
    Path("C:/Users/Bhuva/Downloads/imagined-speech-eeg-signal-bci2020"),
    Path("C:/Users/Bhuva/Downloads"),
    Path("BCI2020 EEG Signal for Words"),
    Path("dataset"),
]


# =============================================================================
# DATASET DISCOVERY
# =============================================================================

def _slashes(path):
    return str(path).replace("\\", "/")


def discover_dataset(explicit: str = None) -> Path:
    """Find the dataset root. Returns the folder containing the .mat files."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates += COMMON_PATHS

    seen = set()
    for cand in candidates:
        try:
            if not cand.exists():
                continue
        except Exception:
            continue
        key = _slashes(cand.resolve())
        if key in seen:
            continue
        seen.add(key)

        root = cand
        # If the candidate itself is a file, use its parent
        if cand.is_file():
            root = cand.parent
        # If it contains a Training set / Validation set / Test set folder, use it
        for sub in ("Training set", "Validation set", "Test set"):
            if (root / sub).exists():
                return root
        # If it contains .mat files directly, use it
        if list(root.glob("*.mat")):
            return root

    # Not found -> return None (caller decides)
    return None


def _load_mat(fpath: Path) -> dict:
    """Load a .mat file (scipy v5, or h5py v7.3)."""
    from scipy.io import loadmat
    try:
        return loadmat(str(fpath), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py
        out = {}
        with h5py.File(str(fpath), "r") as f:
            for key in f.keys():
                if key.startswith("#"):
                    continue
                out[key] = _H5Struct(f, f[key])
        return out


class _Struct:
    """Simple namespace to mimic scipy mat_struct for saving synthetic .mat."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _H5Struct:
    """Minimal h5py wrapper for field access (obj.x, obj.y)."""
    def __init__(self, h5file, grp):
        import h5py
        self._fields = {}
        for name in grp.keys():
            item = grp[name]
            if isinstance(item, h5py.Dataset):
                raw = item[()]
                if raw.dtype == object:
                    try:
                        raw = np.array([
                            "".join(chr(c) for c in h5file[ref][()])
                            for ref in raw.flatten()
                        ])
                    except Exception:
                        pass
                self._fields[name] = raw
            elif isinstance(item, h5py.Group):
                self._fields[name] = _H5Struct(h5file, item)

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name)


import re


def _subject_id_from_filename(stem: str) -> str:
    """
    Extract a subject id from a .mat filename stem. Two naming schemes are
    in the wild for this dataset:
      - "S01_Training", "S01_Validation"      -> "S01"  (leading-token style)
      - "Data_Sample01", "Data_Sample12_test" -> "S01", "S12"  (Kaggle's
        actual BCI2020 release names every file "Data_SampleNN.mat" — the
        subject only shows up as a number after "Sample", not as the
        leading token, which is the same literal word "Data" for every
        subject. Blindly taking the first token collapses every subject
        into one bucket called "Data".)
    Falls back to the leading token if neither pattern is recognised.
    """
    m = re.search(r'(?:sample|subject|subj)[\s_-]*?(\d+)', stem, re.IGNORECASE)
    if m:
        return f"S{int(m.group(1)):02d}"

    tokens = stem.replace("_", " ").split()
    if tokens:
        tok = tokens[0]
        # Generic leading words carry no subject identity by themselves —
        # if there's a number anywhere else in the filename, that's the
        # real subject id (e.g. "Data01", "File_01_train").
        if tok.lower() in {"data", "file", "sample", "subject", "eeg", "trial"}:
            digits = re.findall(r'\d+', stem)
            if digits:
                return f"S{int(digits[-1]):02d}"
        return tok
    return stem


def discover_mat_files(dataset_root: Path):
    """
    Return a dict: {subject_id: {"train": [paths], "val": [paths], "test": [paths]}}
    Searches Training set / Validation set / Test set subfolders, or a flat
    folder of *_Training* / *_Validation* / *_Test* files.
    """
    result = {}

    def add(fpath, split):
        sid = _subject_id_from_filename(fpath.stem)
        result.setdefault(sid, {"train": [], "val": [], "test": []})
        result[sid][split].append(fpath)

    # Case 1: structured subfolders
    for sub, split in (("Training set", "train"),
                       ("Validation set", "val"),
                       ("Test set", "test")):
        folder = dataset_root / sub
        if folder.exists():
            for f in sorted(folder.glob("*.mat")):
                add(f, split)

    # Case 2: flat folder with split keywords in filename
    if not any(result[sid]["train"] for sid in result):
        for f in sorted(dataset_root.glob("*.mat")):
            low = f.stem.lower()
            if "train" in low or "training" in low:
                add(f, "train")
            elif "val" in low or "validation" in low:
                add(f, "val")
            elif "test" in low or "eval" in low:
                add(f, "test")

    return result


def load_epoch(fpath: Path, epo_key: str):
    """Return (X, y) where X:(trials,channels,samples), y:(trials,) labels 1..5."""
    mat = _load_mat(fpath)
    if epo_key not in mat:
        # try to auto-detect the only struct with .x and .y
        for k, v in mat.items():
            if hasattr(v, "x") and hasattr(v, "y"):
                epo_key = k
                break
        else:
            raise RuntimeError(
                f"No epoch struct found in {fpath.name}. Keys: {list(mat.keys())}"
            )
    epo = mat[epo_key]
    X_raw = np.array(epo.x, dtype=np.float64)          # (samples, channels, trials)
    y_oh = np.array(epo.y, dtype=np.float64)
    # y_oh may be (classes, trials) or (trials, classes)
    if y_oh.ndim == 2:
        if y_oh.shape[0] == X_raw.shape[2] and y_oh.shape[1] > 1:
            y = (np.argmax(y_oh, axis=1) + 1).astype(int)
        elif y_oh.shape[1] == X_raw.shape[2] and y_oh.shape[0] > 1:
            y = (np.argmax(y_oh, axis=0) + 1).astype(int)
        else:
            y = y_oh.flatten().astype(int)
    else:
        y = y_oh.flatten().astype(int)
    X = X_raw.transpose(2, 1, 0)                        # (trials, channels, samples)
    if X.shape[0] != y.shape[0]:
        y = y[: X.shape[0]]
    return X, y


# =============================================================================
# SIGNAL PROCESSING (mirrors train.py)
# =============================================================================

def apply_car(X):
    return X - X.mean(axis=1, keepdims=True)


def apply_hanning(X):
    return X * np.hanning(X.shape[-1])


def bpf(X, low, high, fs=FS, order=BP_ORDER):
    nyq = fs / 2.0
    lo = low / nyq
    hi = min(high, nyq - 1.0) / nyq
    b, a = sp_signal.butter(order, [lo, hi], btype="band")
    return sp_signal.filtfilt(b, a, X, axis=-1)


def preprocess_bands(X, start=EPOCH_START, end=EPOCH_END):
    """
    X : (trials, channels, samples) -> dict band -> (trials, channels, epoch_samples)
    """
    X = apply_car(X)
    X = X[:, :, start:end]
    X = apply_hanning(X)
    bands = {}
    for name, (lo, hi) in FREQ_BANDS.items():
        try:
            bands[name] = bpf(X, lo, hi)
        except Exception:
            bands[name] = X.copy()
    return bands


def reject_artefact_trials(X, y, threshold_uv=ARTIFACT_THRESHOLD_UV):
    ptp = X.max(axis=2) - X.min(axis=2)
    mask = ptp.max(axis=1) < threshold_uv
    return X[mask], y[mask]


# =============================================================================
# SUBJECT-ID FEATURES ("brainprint")
# =============================================================================

def extract_subject_features(X, start=EPOCH_START, end=EPOCH_END):
    """
    X : (trials, channels, samples)
    Returns (trials, features) where features = per-band per-channel log-power
    + broadband log-variance per channel.
    """
    bands = preprocess_bands(X, start, end)
    parts = []
    for name, Xb in bands.items():
        # log band power per channel: (trials, channels)
        var = Xb.var(axis=2)
        lp = np.log(np.maximum(var, 1e-10))
        parts.append(lp)
    # broadband log-variance on the CAR'd, trimmed epoch
    Xc = apply_car(X[:, :, start:end])
    bv = np.log(np.maximum(Xc.var(axis=2), 1e-10))
    parts.append(bv)
    F = np.concatenate(parts, axis=1)      # (trials, n_bands*n_ch + n_ch)
    return np.nan_to_num(F)


# =============================================================================
# SUBJECT CLASSIFIER (enrollment + verification)
# =============================================================================

class SubjectAuthenticator:
    """Enrolls subjects and verifies a claimed identity from a trial."""

    def __init__(self, threshold=None):
        self.scaler = None
        self.model = None
        self.classes_ = None
        self.threshold = threshold
        self.feature_dim = None

    def fit(self, X_all, y_subj):
        """X_all: (trials, features); y_subj: subject labels (strings)."""
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X_all)

        self.model = LinearSVC(
            C=1.0, max_iter=10000, random_state=RANDOM_STATE,
            class_weight="balanced",
        )
        self.model.fit(Xs, y_subj)
        self.classes_ = list(self.model.classes_)
        self.feature_dim = X_all.shape[1]
        return self

    def _score(self, X, subject):
        """Signed decision score for a claimed subject (positive = match)."""
        if subject not in self.model.classes_:
            return -1e9
        Xs = self.scaler.transform(X)
        dec = self.model.decision_function(Xs)     # (n_trials, n_classes)
        idx = list(self.model.classes_).index(subject)
        return float(dec[:, idx][0])

    def authenticate(self, X, subject):
        """
        Returns dict: granted, score, threshold, subject, confidence.
        """
        score = self._score(X, subject)
        if self.threshold is None:
            granted = score >= 0.0
            conf = min(99.0, max(1.0, 50.0 + abs(score) * 5.0))
        else:
            granted = score >= self.threshold
            conf = min(99.0, max(1.0, 50.0 + abs(score) * 5.0))
        return {
            "granted": bool(granted),
            "score": score,
            "threshold": self.threshold if self.threshold is not None else 0.0,
            "subject": subject,
            "confidence": conf,
        }

    def esperanza(self, X, y_subj):
        """Verification experiment: genuine vs imposter, returns scores."""
        scores = []
        labels = []
        n = len(y_subj)
        for i in range(n):
            # genuine
            s = self._score(X[i:i+1], y_subj[i])
            scores.append(s)
            labels.append(1)
            # imposter: pick a different subject
            others = [c for c in self.classes_ if c != y_subj[i]]
            if others:
                imp = others[np.random.RandomState(i).choice(len(others))]
                s2 = self._score(X[i:i+1], imp)
                scores.append(s2)
                labels.append(0)
        return np.array(scores), np.array(labels)


def compute_eer(scores, labels):
    """
    Returns (eer, threshold_at_eer). Scores are signed decision values;
    larger = more likely genuine.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    thresholds = np.sort(np.unique(scores))
    best = None
    for thr in thresholds:
        gen = scores[labels == 1]
        imp = scores[labels == 0]
        frr = (gen < thr).mean() if len(gen) else 1.0
        far = (imp >= thr).mean() if len(imp) else 1.0
        if best is None or abs(far - frr) < abs(best[0] - best[1]):
            best = (far, frr, thr)
    far, frr, thr = best
    eer = (far + frr) / 2.0
    return float(eer), float(thr)


# =============================================================================
# WORD PREDICTION GATE (per-subject 5-class word decoder)
# =============================================================================

class WordPredictor:
    """Per-subject 5-class word classifier (Hello/Helpme/Stop/Thankyou/Yes)."""

    def __init__(self):
        self.models = {}

    def fit(self, X_all, y_words, subj_ids):
        """X_all: (trials, features); y_words: labels 1..5; subj_ids: subject strings."""
        for sid in np.unique(subj_ids):
            mask = subj_ids == sid
            Xs = X_all[mask]
            ys = y_words[mask]
            classes = np.unique(ys)
            if len(classes) < 2:
                continue
            scaler = StandardScaler()
            Xss = scaler.fit_transform(Xs)
            clf = LinearSVC(
                C=1.0, max_iter=10000, random_state=RANDOM_STATE,
                class_weight="balanced",
            )
            clf.fit(Xss, ys)
            self.models[sid] = {"scaler": scaler, "clf": clf, "classes": classes}
        return self

    def predict(self, X, subj_id):
        if subj_id not in self.models:
            return None, None
        m = self.models[subj_id]
        Xs = m["scaler"].transform(X)
        p = int(m["clf"].predict(Xs)[0])
        return p, CLASS_NAMES.get(p, "?")


# =============================================================================
# ENROLLMENT PIPELINE
# =============================================================================

def build_features_from_dataset(dataset_root: Path, split="train"):
    """
    Load all trials for the given split across subjects, compute subject
    features, and return (F, subj_ids, raw_y, subject_meta).
    Uses the best available split (prefer train; fall back to val/test).
    """
    mat_map = discover_mat_files(dataset_root)
    if not mat_map:
        raise RuntimeError("No subject .mat files discovered.")

    Fs, subj_ids, y_words = [], [], []
    meta = {}

    for sid, files in mat_map.items():
        fpaths = files.get(split) or files.get("train") or files.get("val") or files.get("test")
        if not fpaths:
            continue
        fpath = fpaths[0]
        epo_key = {"train": "epo_train", "val": "epo_validation", "test": "epo_test"}.get(split, "epo_train")
        X, y = load_epoch(fpath, epo_key)
        X, y = reject_artefact_trials(X, y)
        if X.shape[0] == 0:
            continue
        F = extract_subject_features(X)
        Fs.append(F)
        subj_ids.extend([sid] * X.shape[0])
        y_words.extend(list(y))
        meta[sid] = {"file": str(fpath), "n_trials": X.shape[0]}

    if not Fs:
        raise RuntimeError(f"No data loaded for split '{split}'.")
    F = np.concatenate(Fs, axis=0)
    return F, np.array(subj_ids), np.array(y_words), meta


def enroll(dataset_root: Path, verbose=True):
    """Train the subject authenticator + word predictor, save artifacts."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer training split; fall back to validation if training empty.
    try:
        F, subj_ids, y_words, meta = build_features_from_dataset(dataset_root, "train")
    except Exception:
        F, subj_ids, y_words, meta = build_features_from_dataset(dataset_root, "val")

    if verbose:
        print(f"Enrolled subjects: {len(meta)}")
        print(f"Total trials      : {F.shape[0]}")
        print(f"Feature dim       : {F.shape[1]}")
        print(f"Subjects          : {sorted(meta.keys())}")

    auth = SubjectAuthenticator()
    auth.fit(F, subj_ids)

    # Tune threshold using validation split (genuine vs imposter)
    try:
        Fv, v_subj, v_words, _ = build_features_from_dataset(dataset_root, "val")
        scores, labels = auth.esperanza(Fv, v_subj)
        eer, thr = compute_eer(scores, labels)
        auth.threshold = thr
        if verbose:
            print(f"\nThreshold tuning (validation):")
            print(f"  EER = {eer*100:.2f}%  threshold = {thr:.3f}")
        with open(EER_PATH, "w") as fh:
            json.dump({"eer": eer, "threshold": thr}, fh, indent=2)
    except Exception as e:
        if verbose:
            print(f"\n[!] Threshold tuning skipped: {e}")
        auth.threshold = 0.0

    # Word predictor (per-subject 5-class)
    wp = WordPredictor()
    try:
        wp.fit(F, y_words, subj_ids)
        if verbose:
            print(f"  Word models trained for {len(wp.models)} subject(s).")
    except Exception as e:
        if verbose:
            print(f"[!] Word model training failed: {e}")

    with open(SUBJECT_MODEL_PATH, "wb") as fh:
        pickle.dump({"auth": auth, "meta": meta}, fh)
    with open(WORD_MODEL_PATH, "wb") as fh:
        pickle.dump(wp, fh)
    with open(SUBJECT_TEMPLATES_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    if verbose:
        print(f"\nArtifacts saved to {AUTH_DIR}")
    return auth, wp, meta


def load_artifacts():
    auth = None
    wp = None
    meta = {}
    if SUBJECT_MODEL_PATH.exists():
        with open(SUBJECT_MODEL_PATH, "rb") as fh:
            auth = pickle.load(fh)["auth"]
    if WORD_MODEL_PATH.exists():
        with open(WORD_MODEL_PATH, "rb") as fh:
            wp = pickle.load(fh)
    if SUBJECT_TEMPLATES_PATH.exists():
        with open(SUBJECT_TEMPLATES_PATH) as fh:
            meta = json.load(fh)
    return auth, wp, meta


# =============================================================================
# LOGIN / VERIFICATION
# =============================================================================

def login(dataset_root, subject, trial_idx, split="val"):
    auth, wp, meta = load_artifacts()
    if auth is None:
        print("No enrolled model found. Run `enroll` first.")
        return None

    mat_map = discover_mat_files(dataset_root)
    if subject not in mat_map:
        print(f"Subject '{subject}' not found in dataset.")
        return None
    fpaths = mat_map[subject].get(split) or mat_map[subject].get("train")
    if not fpaths:
        print(f"No {split} data for '{subject}'.")
        return None
    fpath = fpaths[0]

    # Try loading with the correct epoch key
    epo_key = {"train": "epo_train", "val": "epo_validation", "test": "epo_test"}[split]
    X, y = load_epoch(fpath, epo_key)
    if trial_idx >= X.shape[0]:
        print(f"Trial {trial_idx} out of range (0..{X.shape[0]-1}).")
        return None

    trial_X = X[trial_idx: trial_idx + 1]
    F = extract_subject_features(trial_X)

    result = auth.authenticate(F, subject)
    print("\n" + "=" * 55)
    print(f"  AUTHENTICATION — claimed subject: {subject}")
    print(f"  Trial: {trial_idx}   Score: {result['score']:.3f}   "
          f"Threshold: {result['threshold']:.3f}")
    print(f"  Confidence: {result['confidence']:.1f}%")
    if result["granted"]:
        print("  >>> ACCESS GRANTED <<<")
        if wp is not None:
            word, word_name = wp.predict(F, subject)
            print(f"  Predicted word: {word_name} (class {word})")
            print(f"  True word     : {CLASS_NAMES.get(int(y[trial_idx]), '?')}")
        else:
            print("  (No word model available.)")
    else:
        print("  >>> ACCESS DENIED <<<")
        print("  Word prediction locked.")
    print("=" * 55)
    return result


# =============================================================================
# SYNTHETIC SELF-TEST (no dataset needed)
# =============================================================================

def self_test():
    """Generate synthetic per-subject EEG-like data and verify machinery."""
    print("\n=== SELF-TEST (synthetic data) ===")
    rng = np.random.RandomState(RANDOM_STATE)
    n_subjects = 5
    n_trials = 40
    n_samples = 795
    n_ch = 64

    Fs, subj_ids, y_words = [], [], []
    for s in range(n_subjects):
        sid = f"S{s+1:02d}"
        # each subject has a distinct "brainprint" spectral profile
        base = rng.randn(n_ch, 1) * (s + 1)
        for t in range(n_trials):
            trial = np.zeros((n_ch, n_samples))
            freqs = np.linspace(0.5, 50, 8)
            for f in freqs:
                trial += (0.5 + s * 0.1) * np.sin(2 * np.pi * f * np.arange(n_samples) / FS)[None, :]
                trial += rng.randn(n_ch, 1) * 0.1
            trial += base
            trial += 0.2 * np.sin(2 * np.pi * (6 + s) * np.arange(n_samples) / FS)[None, :]
            Fs.append(extract_subject_features(trial[None, :, :]))
            subj_ids.append(sid)
            y_words.append((t % 5) + 1)

    F = np.concatenate(Fs, axis=0)
    subj_ids = np.array(subj_ids)
    y_words = np.array(y_words)

    auth = SubjectAuthenticator()
    auth.fit(F, subj_ids)
    scores, labels = auth.esperanza(F, subj_ids)
    eer, thr = compute_eer(scores, labels)
    auth.threshold = thr

    print(f"Subjects: {n_subjects}, Trials: {F.shape[0]}, Features: {F.shape[1]}")
    print(f"EER = {eer*100:.2f}%  (lower is better)")

    # Genuine verification accuracy
    np.random.seed(0)
    correct = 0
    total = 0
    idx = np.random.choice(F.shape[0], min(200, F.shape[0]), replace=False)
    for i in idx:
        got = auth.authenticate(F[i:i+1], subj_ids[i])["granted"]
        correct += got
        total += 1
        # an imposter attempt
        imp = subj_ids[np.random.choice(F.shape[0])]
        while imp == subj_ids[i]:
            imp = subj_ids[np.random.choice(F.shape[0])]
        got2 = auth.authenticate(F[i:i+1], imp)["granted"]
        correct += (not got2)
        total += 1
    print(f"Verification accuracy (genuine+imposter): {correct/total*100:.1f}%")

    # Word predictor
    wp = WordPredictor()
    wp.fit(F, y_words, subj_ids)
    p, name = wp.predict(F[0:1], subj_ids[0])
    print(f"Word model: {len(wp.models)} subject(s), sample pred '{name}' (true {CLASS_NAMES.get(int(y_words[0]))})")

# Save synthetic artifacts for demo
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUBJECT_MODEL_PATH, "wb") as fh:
        pickle.dump({"auth": auth, "meta": {}}, fh)
        # Persist a synthetic dataset (same folder layout as Kaggle) so that
        # `login` / `simulate` / Streamlit all work without the real dataset.
        from scipy.io import savemat

        synth_root = Path(__file__).resolve().parent / "synthetic_dataset"
        (synth_root / "Training set").mkdir(parents=True, exist_ok=True)
        (synth_root / "Validation set").mkdir(parents=True, exist_ok=True)
        (synth_root / "Test set").mkdir(parents=True, exist_ok=True)

        rng2 = np.random.RandomState(RANDOM_STATE + 1)
        for s in range(n_subjects):
            sid = f"S{s+1:02d}"
            base = rng2.randn(n_ch, 1) * (s + 1)
            counts = {"train": 300, "val": 50, "test": 50}
            for split, n_tr in counts.items():
                Xs = np.zeros((n_tr, n_ch, n_samples))
                ys = np.zeros(n_tr, dtype=int)
                for t in range(n_tr):
                    trial = np.zeros((n_ch, n_samples))
                    for f in np.linspace(0.5, 50, 8):
                        trial += (0.5 + s * 0.1) * np.sin(
                            2 * np.pi * f * np.arange(n_samples) / FS)[None, :]
                    trial += rng2.randn(n_ch, 1) * 0.1
                    trial += base
                    trial += 0.2 * np.sin(
                        2 * np.pi * (6 + s) * np.arange(n_samples) / FS)[None, :]
                    Xs[t] = trial
                    ys[t] = (t % 5) + 1
                y_oh = np.zeros((5, n_tr), dtype=int)
                y_oh[ys - 1, np.arange(n_tr)] = 1
                epo = _Struct(
                    x=Xs.transpose(2, 1, 0),   # (samples, channels, trials)
                    y=y_oh,
                    fs=FS,
                    t=np.arange(n_samples) / FS,
                    className=np.array([f"word{i}" for i in range(1, 6)]),
                )
                folder = {"train": "Training set", "val": "Validation set",
                          "test": "Test set"}[split]
                savemat(str(synth_root / folder / f"{sid}_{split}.mat"),
                        {f"epo_{split}": epo})

        print("Synthetic dataset saved to:", synth_root)
    with open(WORD_MODEL_PATH, "wb") as fh:
        pickle.dump(wp, fh)
    with open(EER_PATH, "w") as fh:
        json.dump({"eer": eer, "threshold": thr}, fh, indent=2)

    print("\nSelf-test artifacts saved. Try:")
    print("  python authentication.py login --subject S01 --trial 3")
    print("  (uses synthetic enrollment; any subject will be 'granted',")
    print("   a subject not in the set will be 'denied')")
    return auth, wp, subj_ids, F


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="BCI EEG Authentication Gate")
    parser.add_argument("mode", choices=["enroll", "stats", "login",
                                         "simulate", "self-test", "tune"])
    parser.add_argument("--path", default=None, help="Dataset root path")
    parser.add_argument("--subject", default=None, help="Claimed subject id")
    parser.add_argument("--trial", type=int, default=0, help="Trial index")
    parser.add_argument("--split", default="val",
                        choices=["train", "val", "test"])
    args = parser.parse_args(argv)

    if args.mode == "self-test":
        self_test()
        return

    dataset_root = discover_dataset(args.path)
    if dataset_root is None:
        print(f"[!] Dataset not found. Provide --path or run `self-test`.")
        print(f"    {COMMON_PATHS}")
        return

    print(f"Dataset root: {dataset_root}")

    if args.mode == "enroll":
        enroll(dataset_root)
    elif args.mode == "tune":
        enroll(dataset_root)  # enroll includes tuning
    elif args.mode == "stats":
        mat_map = discover_mat_files(dataset_root)
        print(f"Subjects: {len(mat_map)}")
        for sid, files in sorted(mat_map.items()):
            tr = len(files["train"]); va = len(files["val"]); te = len(files["test"])
            print(f"  {sid:<8s} train_files={tr}  val_files={va}  test_files={te}")
    elif args.mode == "login":
        if not args.subject:
            parser.error("login requires --subject")
        login(dataset_root, args.subject, args.trial, args.split)
    elif args.mode == "simulate":
        simulate(dataset_root)


def simulate(dataset_root):
    """Interactive genuine/imposter demo over validation trials."""
    auth, wp, meta = load_artifacts()
    if auth is None:
        print("No enrolled model. Run `enroll` or `self-test` first.")
        return
    mat_map = discover_mat_files(dataset_root)
    subjects = sorted(mat_map.keys())
    if not subjects:
        print("No subjects found.")
        return

    print("\n=== SIMULATE: authentication demo ===")
    print("Subjects:", subjects)
    chosen = input("Claimed subject (e.g. S01): ").strip()
    if chosen not in subjects:
        print(f"'{chosen}' is not an enrolled subject -> ACCESS DENIED (imposter).")
        return
    fpaths = mat_map[chosen].get("val") or mat_map[chosen].get("train")
    fpath = fpaths[0]
    epo_key = "epo_validation" if (mat_map[chosen].get("val")) else "epo_train"
    X, y = load_epoch(fpath, epo_key)
    for t in range(X.shape[0]):
        trial_X = X[t:t+1]
        F = extract_subject_features(trial_X)
        res = auth.authenticate(F, chosen)
        status = "GRANTED" if res["granted"] else "DENIED"
        word = ""
        if res["granted"] and wp is not None:
            _, wname = wp.predict(F, chosen)
            word = wname
        print(f"  Trial {t:>3d} subject={chosen:<5s} score={res['score']:+.3f} "
              f"-> {status:<8s} word={word}")
        if t >= 19:
            break


if __name__ == "__main__":
    main()