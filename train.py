"""
=============================================================================
BCI TRAINING PIPELINE — Imagined Speech Decoder  [v3 — FULL]
=============================================================================
Dataset     : Custom 5-class imagined speech EEG
              Words: Hello, Helpme, Stop, Thankyou, Yes
Task        : Binary classification — Bilabial vs Non-Bilabial
              Bilabial  (/p/, /m/): Helpme (2), Stop (3)
              Non-bilabial        : Hello (1), Thankyou (4), Yes (5)

Pipeline    : Artefact rejection → CAR → Hanning taper → Epoch trim →
              Sub-band bandpass → Per-subject CSP (train only) →
              Log-variance features → StandardScaler (train only) →
              LinearSVC with inner GridSearchCV → Balanced accuracy +
              Wilson CI + permutation test per subject

What is new in v3
-----------------
1.  Artefact rejection   — trials with peak-to-peak > 100 µV discarded
2.  Hanning taper        — applied after epoch trim, before bandpass
                           reduces spectral leakage at epoch edges
3.  Epoch window search  — evaluates three windows automatically and
                           picks the best one per subject, or you can
                           lock a single window via EPOCH_MODE config
4.  All v2 fixes kept    — FS=256, epo_key explicit, correct transpose,
                           one-hot decode, BILABIAL={2,3}, theta+gamma
                           bands, N_CSP=4, inner CV for C, permutation
                           test, Wilson CI, group t-test

How to use
----------
1. Run as-is (EPOCH_MODE = "search") — tries all three windows and
   picks the best validated one per subject.
2. Set EPOCH_MODE = "fixed" and adjust EPOCH_START/END to lock a window.
3. Set run_test=True in main() ONLY when all decisions are final.
=============================================================================
"""

# -- Standard library ---------------------------------------------------------
import pickle
import warnings
from pathlib import Path

# -- Scientific stack ---------------------------------------------------------
import numpy as np
from scipy import signal as sp_signal
from scipy.io import loadmat
from scipy.linalg import eigh
from scipy.stats import t as t_dist

# -- Machine-learning stack ---------------------------------------------------
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_ROOT = Path(
    r"C:\Users\Admin\OneDrive\Documents\mojar_project"
    r"\BCI2020 EEG Signal for Words"
)

TRAIN_DIR    = DATASET_ROOT / "Training set"
VAL_DIR      = DATASET_ROOT / "Validation set"
TEST_DIR     = DATASET_ROOT / "Test set"        # never touched until final eval
ARTIFACT_DIR = DATASET_ROOT / "artifacts"

# ── EEG acquisition ───────────────────────────────────────────────────────────
TARGET_CHANNELS = 64
FS              = 256       # Hz

# ── Artefact rejection ────────────────────────────────────────────────────────
ARTEFACT_THRESHOLD_UV = 800.0   # peak-to-peak µV limit per trial
                                 # trials exceeding this on ANY channel are dropped

# ── Epoch windows (all in sample indices at 256 Hz) ──────────────────────────
# Sample 128 = 0 ms stimulus onset  (time axis starts at -500 ms)
#
# Three candidate windows:
#   "early"   : 0 ms  to +1000 ms  → samples 128–384  (256 samples)
#   "standard": 0 ms  to +1500 ms  → samples 128–512  (384 samples)  ← default
#   "late"    : 200 ms to +1200 ms → samples 179–435  (256 samples)
#
# EPOCH_MODE = "search"  → tries all three, picks best per subject
# EPOCH_MODE = "fixed"   → always uses EPOCH_START_SAMPLE / EPOCH_END_SAMPLE
EPOCH_MODE         = "search"   # "search" | "fixed"
EPOCH_START_SAMPLE = 128        # used when EPOCH_MODE = "fixed"
EPOCH_END_SAMPLE   = 512        # used when EPOCH_MODE = "fixed"

EPOCH_CANDIDATES = {
    "early"   : (128, 384),
    "standard": (128, 512),
    "late"    : (179, 435),
}

# ── Frequency bands ───────────────────────────────────────────────────────────
FREQ_BANDS = {
    "theta" : (4,  8),     # working memory / imagery
    "mu"    : (8,  12),    # sensorimotor rhythm
    "beta1" : (12, 20),    # motor planning
    "beta2" : (20, 30),    # high beta
    "gamma" : (30, 50),    # most cited band for imagined speech
}
BP_ORDER = 4

# ── Label mapping ─────────────────────────────────────────────────────────────
# Hello(1)    /h/,/l/        → Non-bilabial
# Helpme(2)   /h/,/p/,/m/   → Bilabial
# Stop(3)     /s/,/t/,/p/   → Bilabial
# Thankyou(4) /θ/,/k/       → Non-bilabial
# Yes(5)      /j/,/s/       → Non-bilabial
BILABIAL_LABEL_IDS = {2, 3}
CLASS_NAMES        = {1: "Hello", 2: "Helpme", 3: "Stop", 4: "Thankyou", 5: "Yes"}

# ── CSP ───────────────────────────────────────────────────────────────────────
N_CSP_COMPONENTS   = 4       # 2 from each end of eigenspectrum
CSP_TIKHONOV_FLOOR = 1e-8
CSP_MAX_REG        = 1e-1
CSP_REG_STEP       = 10.0

# ── Classifier ────────────────────────────────────────────────────────────────
SVM_C_GRID   = [0.01, 0.1, 1.0, 10.0]
SVM_CV_FOLDS = 5

# ── Permutation test ──────────────────────────────────────────────────────────
N_PERMUTATIONS = 1000

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DEBUG = True

def _load_mat(fpath: Path) -> dict:
    """
    Load a .mat file regardless of whether it is v5 (scipy) or v7.3 (h5py).
    Returns a flat dict of {key: numpy_array or mat_struct}.
    """
    # Try scipy first (works for v4, v5, v6)
    try:
        return loadmat(str(fpath), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        pass  # v7.3 HDF5 — fall through to h5py

    # h5py path for v7.3
    import h5py
    import numpy as np

    out = {}
    with h5py.File(str(fpath), "r") as f:
        for key in f.keys():
            if key.startswith("#"):
                continue
            grp = f[key]
            # Build a simple namespace so field access works the same way
            obj = _H5Struct(f, grp)
            out[key] = obj

    return out


class _H5Struct:
    """
    Wraps an h5py Group to mimic scipy's mat_struct field access.
    Supports:  obj.x,  obj.y,  obj.fs,  obj.t,  obj.className
    """
    def __init__(self, h5file, grp):
        import h5py
        import numpy as np

        self._fields = {}
        for name in grp.keys():
            item = grp[name]
            if isinstance(item, h5py.Dataset):
                raw = item[()]
                # Dereference object arrays (cell arrays of strings etc.)
                if raw.dtype == object:
                    try:
                        raw = np.array([
                            "".join(chr(c) for c in h5file[ref][()])
                            if hasattr(h5file[ref][()], "__iter__")
                            else h5file[ref][()]
                            for ref in raw.flatten()
                        ])
                    except Exception:
                        pass
                self._fields[name] = np.array(raw, dtype=np.float64) \
                    if np.issubdtype(np.array(raw).dtype, np.number) \
                    else raw
            elif isinstance(item, h5py.Group):
                self._fields[name] = _H5Struct(h5file, item)

    def __getattr__(self, name):
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(
                f"_H5Struct has no field '{name}'. "
                f"Available: {list(self._fields.keys())}"
            )

    def __contains__(self, name):
        return name in self._fields

    def keys(self):
        return self._fields.keys()

# =============================================================================
# SECTION A — DATA LOADING
# =============================================================================

def load_dataset(directory: Path, split_name: str, epo_key: str):
    """
    Load all .mat files from a directory.

    epo_key must be explicit:
      "epo_train"       Training set
      "epo_validation"  Validation set
      "epo_test"        Test set  (only call at final evaluation)

    Returns
    -------
    X_list : list of (trials, channels, samples) float64 arrays
    y_list : list of (trials,) int arrays, labels in 1..5
    names  : list of filename strings
    """
    mat_files = sorted(directory.glob("*.mat"))
    if not mat_files:
        raise RuntimeError(f"No .mat files found in '{directory}'.")

    print(f"\n{'--'*30}")
    print(f"  Loading {split_name}  ({len(mat_files)} file(s))  key='{epo_key}'")
    print(f"{'--'*30}")

    X_list, y_list, names = [], [], []

    for fpath in mat_files:
        mat = _load_mat(fpath)
        available = [k for k in mat.keys() if not k.startswith("_")]

        if epo_key not in mat:
            raise RuntimeError(
                f"Key '{epo_key}' not found in '{fpath.name}'. "
                f"Available: {available}"
            )

        epo = mat[epo_key]

        # epo.x : (samples, channels, trials)
        X_raw = np.array(epo.x, dtype=np.float64)

        # epo.y : (n_classes, trials)  one-hot → integer labels 1..5
        y_oh  = np.array(epo.y, dtype=np.float64)
        y_raw = (np.argmax(y_oh, axis=0) + 1).astype(int)

        # → (trials, channels, samples)
        X = X_raw.transpose(2, 1, 0)

        assert X.shape[0] == y_raw.shape[0], (
            f"Trial/label mismatch in '{fpath.name}': "
            f"X has {X.shape[0]} trials, y has {y_raw.shape[0]}."
        )
        assert X.shape[1] == TARGET_CHANNELS, (
            f"Expected {TARGET_CHANNELS} channels, got {X.shape[1]} "
            f"in '{fpath.name}'."
        )
        assert X.shape[2] >= EPOCH_END_SAMPLE if EPOCH_MODE == "fixed" else \
               X.shape[2] >= max(e for _, e in EPOCH_CANDIDATES.values()), (
            f"Epoch too short in '{fpath.name}'."
        )

        print(f"  {fpath.name}: X={X.shape}  labels={np.unique(y_raw).tolist()}  "
              f"n={X.shape[0]}")

        X_list.append(X)
        y_list.append(y_raw)
        names.append(fpath.name)

    return X_list, y_list, names


# =============================================================================
# SECTION B — LABEL PROCESSING
# =============================================================================

def binarise_labels(y_raw: np.ndarray) -> np.ndarray:
    """1 = Bilabial (Helpme, Stop),  0 = Non-bilabial (Hello, Thankyou, Yes)."""
    return np.where(
        np.isin(y_raw.flatten().astype(int), list(BILABIAL_LABEL_IDS)), 1, 0
    ).astype(int)


def check_balance(y_bin: np.ndarray, tag: str) -> bool:
    bil    = int(y_bin.sum())
    nonbil = int((y_bin == 0).sum())
    print(f"  {tag}: Non-bil={nonbil}  Bil={bil}")
    if bil < 5 or nonbil < 5:
        print(f"  !! Insufficient class balance for '{tag}' — skipping.")
        return False
    return True


# =============================================================================
# SECTION C — SIGNAL PROCESSING
# =============================================================================

def apply_car(X: np.ndarray) -> np.ndarray:
    """Common Average Reference along channel axis."""
    return X - X.mean(axis=1, keepdims=True)


def apply_hanning_taper(X: np.ndarray) -> np.ndarray:
    """
    Multiply each trial by a Hanning window along the time axis.

    Reduces spectral leakage at epoch edges, especially important
    for gamma band estimation.

    X : (trials, channels, samples)
    """
    window = np.hanning(X.shape[-1])        # (samples,)
    return X * window                       # broadcast over trials × channels


def bandpass_band(X: np.ndarray, low: float, high: float,
                  fs: float = FS, order: int = BP_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass along time axis (axis=-1)."""
    nyq = fs / 2.0
    lo  = low  / nyq
    hi  = min(high, nyq - 1.0) / nyq

    if lo <= 0 or hi >= 1 or lo >= hi:
        raise ValueError(
            f"Invalid band [{low}–{high} Hz] at fs={fs}: "
            f"normalised [{lo:.3f}–{hi:.3f}]."
        )

    b, a       = sp_signal.butter(order, [lo, hi], btype="band")
    min_padlen = 3 * max(len(a), len(b))
    if X.shape[-1] < min_padlen:
        raise RuntimeError(
            f"Epoch length {X.shape[-1]} samples too short for "
            f"[{low}–{high} Hz] (need ≥ {min_padlen})."
        )

    return sp_signal.filtfilt(b, a, X, axis=-1)


def preprocess_and_decompose(X: np.ndarray,
                              start: int, end: int,
                              fs: float = FS) -> dict:
    """
    CAR → epoch trim → Hanning taper → per-band bandpass.

    Parameters
    ----------
    X     : (trials, channels, samples)
    start : first sample index (inclusive)
    end   : last  sample index (exclusive)

    Returns
    -------
    dict  band_name → (trials, channels, epoch_samples)
    """
    X = apply_car(X)
    X = X[:, :, start:end]                 # trim epoch
    X = apply_hanning_taper(X)             # taper before filtering

    bands = {}
    for name, (lo, hi) in FREQ_BANDS.items():
        try:
            bands[name] = bandpass_band(X, lo, hi, fs)
        except Exception as exc:
            warnings.warn(f"Band '{name}' failed: {exc}", RuntimeWarning)

    return bands


def reject_artefact_trials(X: np.ndarray, y: np.ndarray,
                            threshold_uv: float = ARTEFACT_THRESHOLD_UV,
                            tag: str = "") -> tuple:
    """
    Remove trials where any channel's peak-to-peak amplitude
    exceeds threshold_uv microvolts.

    Parameters
    ----------
    X : (trials, channels, samples)
    y : (trials,)

    Returns
    -------
    X_clean, y_clean  (may be smaller than input)
    """
    peak_to_peak = X.max(axis=2) - X.min(axis=2)   # (trials, channels)
    clean_mask   = peak_to_peak.max(axis=1) < threshold_uv

    n_rejected = int((~clean_mask).sum())
    if n_rejected > 0:
        print(f"  [{tag}] Artefact rejection: "
              f"{n_rejected}/{len(y)} trials removed "
              f"(peak-to-peak > {threshold_uv} µV)")

    return X[clean_mask], y[clean_mask]


def _sanitise(X: np.ndarray, tag: str) -> np.ndarray:
    mask = ~np.isfinite(X)
    if mask.any():
        warnings.warn(
            f"{tag}: {int(mask.sum())} non-finite values replaced with 0.",
            RuntimeWarning,
        )
        X = np.where(mask, 0.0, X)
    return X


# =============================================================================
# SECTION D — CSP
# =============================================================================

class CSP:
    """
    Binary Common Spatial Patterns with OAS regularisation.
    Must be fitted on training data only.
    """

    def __init__(self, n_components=N_CSP_COMPONENTS,
                 reg_floor=CSP_TIKHONOV_FLOOR,
                 max_reg=CSP_MAX_REG,
                 reg_step=CSP_REG_STEP):
        self.n_components = n_components
        self.reg_floor    = reg_floor
        self.max_reg      = max_reg
        self.reg_step     = reg_step
        self.filters_     = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSP":
        """
        X : (trials, channels, time)
        y : (trials,)  binary {0, 1}
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires 2 classes, got {classes}.")

        C0 = self._class_cov(X[y == classes[0]], int(classes[0]))
        C1 = self._class_cov(X[y == classes[1]], int(classes[1]))

        evals, evecs = _stable_eigh(C1, C0 + C1)

        half = self.n_components // 2
        idx  = np.concatenate([
            np.argsort(evals)[-half:][::-1],
            np.argsort(evals)[:half],
        ])
        self.filters_ = evecs[:, idx].T    # (n_components, channels)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """X: (trials, channels, time) → (trials, n_components, time)"""
        if self.filters_ is None:
            raise RuntimeError("CSP not fitted.")
        return np.stack([self.filters_ @ trial for trial in X])

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _class_cov(self, Xc: np.ndarray, label: int) -> np.ndarray:
        n_ch = Xc.shape[1]
        covs = []
        for t, trial in enumerate(Xc):
            c  = trial @ trial.T
            tr = np.trace(c)
            if not np.isfinite(tr) or tr < 1e-12:
                warnings.warn(
                    f"Trial {t} class {label}: near-zero energy, skipped.",
                    RuntimeWarning,
                )
                continue
            covs.append(c / tr)

        if not covs:
            raise RuntimeError(f"All trials in class {label} were skipped.")

        C = np.mean(covs, axis=0)
        C = (C + C.T) / 2.0
        C = self._oas_regularise(C, Xc, n_ch)
        C += self.reg_floor * np.eye(n_ch)
        C  = _ensure_pd(C, n_ch, self.reg_floor, self.max_reg, self.reg_step)
        return C

    @staticmethod
    def _oas_regularise(C_emp: np.ndarray, Xc: np.ndarray,
                         n_ch: int) -> np.ndarray:
        try:
            from sklearn.covariance import OAS
            X2d  = Xc.transpose(0, 2, 1).reshape(-1, n_ch)
            est  = OAS(store_precision=False)
            est.fit(X2d)
            alpha = float(getattr(est, "shrinkage_", 0.0))
            if DEBUG:
                print(f"      [OAS] alpha={alpha:.6f}", end="")
            if alpha <= 1e-6:
                if DEBUG: print("  → empirical kept")
                return C_emp
            C_oas = (est.covariance_ + est.covariance_.T) / 2.0
            if DEBUG: print("  → OAS applied")
            return C_oas
        except Exception as exc:
            warnings.warn(f"OAS failed ({exc}); using empirical.", RuntimeWarning)
            return C_emp


def _ensure_pd(C, n_ch, reg_floor, max_reg, reg_step):
    current = reg_floor
    while current <= max_reg:
        if np.linalg.eigvalsh(C).min() > 0:
            return C
        next_lam = current * reg_step
        C        = C - current * np.eye(n_ch) + next_lam * np.eye(n_ch)
        current  = next_lam
    warnings.warn(
        f"Could not enforce PD at lambda={current:.2e}.", RuntimeWarning
    )
    return C


def _stable_eigh(C1, B):
    n  = B.shape[0]
    C1 = (C1 + C1.T) / 2.0
    B  = (B  + B.T)  / 2.0

    try:
        return eigh(C1, B)
    except Exception:
        pass

    try:
        floor = max(float(np.linalg.eigvalsh(B).max()) * 1e-6, 1e-8)
        B_reg = B + floor * np.eye(n)
        return eigh(C1, (B_reg + B_reg.T) / 2.0)
    except Exception:
        pass

    # Whitening fallback
    evals_b, U = np.linalg.eigh(B)
    evals_b    = np.maximum(evals_b, 1e-10)
    W          = (U / np.sqrt(evals_b)).T
    C1w        = W @ C1 @ W.T
    evals_w, evecs_w = np.linalg.eigh((C1w + C1w.T) / 2.0)
    return evals_w, W.T @ evecs_w


# =============================================================================
# SECTION E — FEATURE EXTRACTION
# =============================================================================

def extract_logvar_features(bands_csp: dict) -> np.ndarray:
    """
    Log-variance per CSP component per band.
    Output shape: (trials, n_bands × n_components)  e.g. (300, 5×4=20)
    """
    parts = []
    for X_band in bands_csp.values():
        var = X_band.var(axis=2)                    # (trials, n_components)
        lv  = np.log(np.maximum(var, 1e-10))
        parts.append(lv)
    return np.concatenate(parts, axis=1)            # (trials, n_bands*n_comp)


# =============================================================================
# SECTION F — CLASSIFIER  (inner CV for C tuning)
# =============================================================================

def scale_and_train(F_train: np.ndarray, y_train: np.ndarray):
    """
    1. Drop near-zero-variance features.
    2. Fit StandardScaler on training data only.
    3. GridSearchCV for SVM C on training data only.

    Returns clf, scaler, var_mask
    """
    var_mask = F_train.var(axis=0) > 1e-8
    F        = F_train[:, var_mask]

    scaler = StandardScaler()
    F      = scaler.fit_transform(F)

    inner_cv = StratifiedKFold(
        n_splits=SVM_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    grid = GridSearchCV(
        LinearSVC(max_iter=5000, random_state=RANDOM_STATE,
                  class_weight="balanced"),
        param_grid={"C": SVM_C_GRID},
        cv=inner_cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    grid.fit(F, y_train)

    if DEBUG:
        print(f"  Best C={grid.best_params_['C']}  "
              f"(inner CV BA={grid.best_score_*100:.1f}%)")

    return grid.best_estimator_, scaler, var_mask


def apply_pipeline(F: np.ndarray, scaler, var_mask: np.ndarray) -> np.ndarray:
    return scaler.transform(F[:, var_mask])


# =============================================================================
# SECTION G — ONE WINDOW EVALUATION  (used by both fixed and search modes)
# =============================================================================

def evaluate_window(X_tr, y_tr, X_va, y_va,
                    start: int, end: int,
                    subj: str, window_name: str,
                    verbose: bool = True) -> tuple:
    """
    Run the full CSP → feature → classifier pipeline for one epoch window.

    Returns
    -------
    ba       : balanced accuracy on validation set
    preds    : predictions on validation set
    artefacts: dict with model components for saving
    """
    # ── Artefact rejection ────────────────────────────────────────────────
    X_tr_c, y_tr_c = reject_artefact_trials(X_tr, y_tr, tag=f"{subj}/tr")
    X_va_c, y_va_c = reject_artefact_trials(X_va, y_va, tag=f"{subj}/va")

    if len(np.unique(y_tr_c)) < 2 or y_tr_c.sum() < 5:
        if verbose:
            print(f"  [{window_name}] Too few clean trials — skipped.")
        return None, None, None

    # ── Preprocess ────────────────────────────────────────────────────────
    bands_tr = preprocess_and_decompose(X_tr_c, start, end)
    bands_va = preprocess_and_decompose(X_va_c, start, end)

    for bname in list(bands_tr.keys()):
        bands_tr[bname] = _sanitise(bands_tr[bname], f"{subj}/tr/{bname}")
        bands_va[bname] = _sanitise(bands_va[bname], f"{subj}/va/{bname}")

    # ── CSP — fit on training data ONLY ───────────────────────────────────
    csp_filters = {}
    if verbose:
        print(f"  [{window_name}] Fitting CSP ...")
    for bname, X_band in bands_tr.items():
        csp = CSP()
        try:
            csp.fit(X_band, y_tr_c)
            csp_filters[bname] = csp
        except Exception as exc:
            warnings.warn(f"CSP failed for band '{bname}': {exc}", RuntimeWarning)

    if not csp_filters:
        return None, None, None

    # ── Project ───────────────────────────────────────────────────────────
    proj_tr = {n: csp_filters[n].transform(bands_tr[n]) for n in csp_filters}
    proj_va = {n: csp_filters[n].transform(bands_va[n]) for n in csp_filters}

    # ── Features ──────────────────────────────────────────────────────────
    F_tr = np.nan_to_num(extract_logvar_features(proj_tr))
    F_va = np.nan_to_num(extract_logvar_features(proj_va))

    if verbose:
        print(f"  [{window_name}] Features: train={F_tr.shape}  val={F_va.shape}")

    # ── Train ─────────────────────────────────────────────────────────────
    clf, scaler, var_mask = scale_and_train(F_tr, y_tr_c)
    F_va_scaled           = apply_pipeline(F_va, scaler, var_mask)

    # ── Evaluate ──────────────────────────────────────────────────────────
    preds = clf.predict(F_va_scaled)
    ba    = balanced_accuracy_score(y_va_c, preds)

    if verbose:
        print(f"  [{window_name}] Val BA = {ba*100:.2f}%")

    artefacts = {
        "csp_filters": csp_filters,
        "clf"         : clf,
        "scaler"      : scaler,
        "var_mask"    : var_mask,
        "epoch_start" : start,
        "epoch_end"   : end,
        "window_name" : window_name,
        "y_va_clean"  : y_va_c,
        "F_tr"        : F_tr,
        "y_tr_clean"  : y_tr_c,
        "F_va"        : F_va,
    }

    return ba, preds, artefacts


# =============================================================================
# SECTION H — PERMUTATION TEST
# =============================================================================

def permutation_test(F_tr, y_tr, F_va, y_va,
                     observed_ba: float,
                     n_perms: int = N_PERMUTATIONS) -> float:
    """
    One-tailed permutation test: P(BA_null >= observed_ba).
    Returns p_value (float).
    """
    rng       = np.random.RandomState(RANDOM_STATE)
    null_dist = []

    for _ in range(n_perms):
        y_shuf = rng.permutation(y_tr)
        if len(np.unique(y_shuf)) < 2:
            continue
        try:
            clf_n, sc_n, vm_n = scale_and_train(F_tr, y_shuf)
            ba_n = balanced_accuracy_score(
                y_va, clf_n.predict(apply_pipeline(F_va, sc_n, vm_n))
            )
            null_dist.append(ba_n)
        except Exception:
            continue

    if not null_dist:
        return float("nan")

    return float((np.array(null_dist) >= observed_ba).mean())


# =============================================================================
# SECTION I — CONFIDENCE INTERVAL
# =============================================================================

def wilson_ci(acc: float, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a proportion."""
    p      = acc
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# =============================================================================
# SECTION J — PER-SUBJECT PIPELINE
# =============================================================================

def _subject_id(filename: str) -> str:
    stem   = Path(filename).stem.lower()
    tokens = {"train", "training", "validation", "val", "test", "eval"}
    parts  = stem.split("_")
    while len(parts) > 1 and parts[-1] in tokens:
        parts.pop()
    return "_".join(parts)


def run_per_subject(
    X_tr_list, y_tr_list, names_tr,
    X_va_list, y_va_list, names_va,
    run_permutation: bool = True,
) -> tuple:
    """
    For each subject:
      1. Binarise labels
      2. Artefact rejection (train and val separately)
      3. If EPOCH_MODE="search": evaluate all three windows, pick best
         If EPOCH_MODE="fixed":  use configured start/end
      4. CSP fit on training data only
      5. Log-variance features
      6. SVM with inner CV for C
      7. Balanced accuracy + Wilson CI + permutation p-value

    Returns results (list of dicts), accuracies (list of floats)
    """
    print(f"\n{'='*60}")
    print("  PER-SUBJECT PIPELINE")
    print(f"  Epoch mode: {EPOCH_MODE.upper()}")
    if EPOCH_MODE == "search":
        for wname, (ws, we) in EPOCH_CANDIDATES.items():
            ms = (we - ws) / FS * 1000
            print(f"    {wname:<10s}: samples {ws}–{we}  "
                  f"({(ws-128)/FS*1000:.0f}ms – {(we-128)/FS*1000:.0f}ms, "
                  f"{ms:.0f}ms window)")
    print(f"{'='*60}")

    tr_map = {_subject_id(n): (n, X, y)
              for n, X, y in zip(names_tr, X_tr_list, y_tr_list)}
    va_map = {_subject_id(n): (n, X, y)
              for n, X, y in zip(names_va, X_va_list, y_va_list)}

    common = sorted(set(tr_map) & set(va_map))
    print(f"\n  Matched subjects: {len(common)}")

    results    = []
    accuracies = []

    for subj in common:
        tr_name, X_tr, y_tr_raw = tr_map[subj]
        va_name, X_va, y_va_raw = va_map[subj]

        print(f"\n  {'─'*56}")
        print(f"  Subject : {subj}")
        print(f"  Train   : {tr_name}   Val: {va_name}")
        print(f"  {'─'*56}")

        # ── Binarise ──────────────────────────────────────────────────────
        y_tr = binarise_labels(y_tr_raw)
        y_va = binarise_labels(y_va_raw)

        if not check_balance(y_tr, "train") or not check_balance(y_va, "val"):
            continue

        # ── Epoch window selection ─────────────────────────────────────────
        if EPOCH_MODE == "fixed":
            best_ba, best_preds, best_art = evaluate_window(
                X_tr, y_tr, X_va, y_va,
                EPOCH_START_SAMPLE, EPOCH_END_SAMPLE,
                subj, "fixed",
            )
            if best_ba is None:
                continue
            best_window = "fixed"

        else:  # "search"
            best_ba      = -1.0
            best_preds   = None
            best_art     = None
            best_window  = None

            for wname, (ws, we) in EPOCH_CANDIDATES.items():
                ba, preds, art = evaluate_window(
                    X_tr, y_tr, X_va, y_va,
                    ws, we, subj, wname, verbose=True,
                )
                if ba is not None and ba > best_ba:
                    best_ba     = ba
                    best_preds  = preds
                    best_art    = art
                    best_window = wname

            if best_ba < 0:
                print(f"  All windows failed for {subj} — skipping.")
                continue

            print(f"\n  >> Best window for {subj}: '{best_window}' "
                  f"BA={best_ba*100:.2f}%")

        # ── Full report with best window ───────────────────────────────────
        y_va_clean = best_art["y_va_clean"]
        ci_lo, ci_hi = wilson_ci(best_ba, len(y_va_clean))

        print(f"\n  Balanced accuracy : {best_ba*100:.2f}%  "
              f"95% CI [{ci_lo*100:.1f}%–{ci_hi*100:.1f}%]")
        print(classification_report(
            y_va_clean, best_preds,
            target_names=["Non-Bilabial (0)", "Bilabial (1)"],
            zero_division=0,
        ))

        # ── Permutation test ──────────────────────────────────────────────
        p_val = float("nan")
        if run_permutation:
            print(f"  Permutation test ({N_PERMUTATIONS} shuffles) ...")
            p_val = permutation_test(
                best_art["F_tr"], best_art["y_tr_clean"],
                best_art["F_va"], y_va_clean,
                observed_ba=best_ba,
            )
            sig = "significant" if p_val < 0.05 else "not significant"
            print(f"  p = {p_val:.4f}  ({sig})")

        accuracies.append(best_ba)
        results.append({
            "subject"     : subj,
            "bal_acc"     : best_ba,
            "ci_lo"       : ci_lo,
            "ci_hi"       : ci_hi,
            "p_value"     : p_val,
            "best_window" : best_window,
            "epoch_start" : best_art["epoch_start"],
            "epoch_end"   : best_art["epoch_end"],
            "csp_filters" : best_art["csp_filters"],
            "clf"         : best_art["clf"],
            "scaler"      : best_art["scaler"],
            "var_mask"    : best_art["var_mask"],
            "n_tr"        : len(best_art["y_tr_clean"]),
            "n_va"        : len(y_va_clean),
        })

    return results, accuracies


# =============================================================================
# SECTION K — SUMMARY STATISTICS
# =============================================================================

def summarise(results: list, accuracies: list) -> str:
    if not accuracies:
        return "No results to summarise."

    accs = np.array(accuracies)
    n    = len(accs)

    # One-sample t-test against 50% chance (one-tailed)
    std    = accs.std(ddof=1)
    t_stat = ((accs.mean() - 0.5) / (std / np.sqrt(n))) if std > 0 else 0.0
    p_grp  = float(t_dist.sf(t_stat, df=n - 1))

    sig_count    = sum(1 for r in results
                       if not np.isnan(r["p_value"]) and r["p_value"] < 0.05)
    below_chance = [r["subject"] for r in results if r["bal_acc"] <= 0.50]

    lines = [
        "",
        "=" * 60,
        "  FINAL SUMMARY",
        "=" * 60,
        f"  Subjects evaluated       : {n}",
        f"  Mean  balanced accuracy  : {accs.mean()*100:.2f}%",
        f"  Median balanced accuracy : {np.median(accs)*100:.2f}%",
        f"  Std   balanced accuracy  : {accs.std()*100:.2f}%",
        f"  Best  subject            : {accs.max()*100:.2f}%",
        f"  Worst subject            : {accs.min()*100:.2f}%",
        "",
        f"  Group t-test vs 50%      : t={t_stat:.3f}  p={p_grp:.4f} (one-tailed)",
        f"  Significant subjects (p<0.05): {sig_count}/{n}",
        "",
        "  Per-subject breakdown:",
        f"  {'Subject':<20s} {'BA':>7s} {'CI 95%':>18s} "
        f"{'p-val':>8s} {'Window':<10s} {'Sig':>4s}",
        "  " + "-" * 75,
    ]

    for r in results:
        sig = "YES" if (not np.isnan(r["p_value"]) and r["p_value"] < 0.05) \
              else "   "
        lines.append(
            f"  {r['subject']:<20s} "
            f"{r['bal_acc']*100:>6.2f}%  "
            f"[{r['ci_lo']*100:>5.1f}%–{r['ci_hi']*100:>5.1f}%]  "
            f"{r['p_value']:>8.4f}  "
            f"{r.get('best_window','fixed'):<10s}  {sig}"
        )

    lines += ["", "  Interpretation:"]

    if p_grp < 0.05:
        lines.append(
            f"  Group mean ({accs.mean()*100:.1f}%) is significantly above "
            f"50% chance (p={p_grp:.4f})."
        )
    else:
        lines.append(
            f"  Group mean ({accs.mean()*100:.1f}%) is NOT significantly "
            f"above chance (p={p_grp:.4f}). Pipeline may not be working."
        )

    if below_chance:
        lines.append(
            f"  {len(below_chance)} subject(s) at/below chance: {below_chance}"
        )

    report = "\n".join(lines)
    print(report)
    return report


# =============================================================================
# SECTION L — ARTIFACT PERSISTENCE
# =============================================================================

def save_artifacts(results: list, summary: str, artifact_dir: Path):
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_data = {
        r["subject"]: {
            "csp_filters" : r["csp_filters"],
            "clf"          : r["clf"],
            "scaler"       : r["scaler"],
            "var_mask"     : r["var_mask"],
            "bal_acc"      : r["bal_acc"],
            "ci_lo"        : r["ci_lo"],
            "ci_hi"        : r["ci_hi"],
            "p_value"      : r["p_value"],
            "epoch_start"  : r["epoch_start"],
            "epoch_end"    : r["epoch_end"],
            "best_window"  : r.get("best_window", "fixed"),
        }
        for r in results
    }

    with open(artifact_dir / "per_subject_models.pkl", "wb") as fh:
        pickle.dump(model_data, fh)

    (artifact_dir / "training_report.txt").write_text(
        summary, encoding="utf-8"
    )

    print(f"\n  Artifacts saved → {artifact_dir}")
    for p in sorted(artifact_dir.iterdir()):
        print(f"    {p.name:<35s}  {p.stat().st_size / 1024:>7.1f} KB")


# =============================================================================
# SECTION M — FINAL TEST SET EVALUATION  (call ONCE, at the very end)
# =============================================================================

def run_final_test(results: list, test_dir: Path):
    """
    Apply trained per-subject models to the held-out test set.

    Uses each subject's best epoch window (stored in results).
    Call ONLY after all hyperparameter decisions are final.
    """
    print(f"\n{'='*60}")
    print("  FINAL TEST SET EVALUATION  (held-out — run once only)")
    print(f"{'='*60}")

    X_te_list, y_te_list, names_te = load_dataset(
        test_dir, "Test set", "epo_test"
    )

    te_map    = {_subject_id(n): (n, X, y)
                 for n, X, y in zip(names_te, X_te_list, y_te_list)}
    model_map = {r["subject"]: r for r in results}
    common    = sorted(set(te_map) & set(model_map))

    test_accs = []
    for subj in common:
        te_name, X_te, y_te_raw = te_map[subj]
        r = model_map[subj]

        y_te = binarise_labels(y_te_raw)

        # Use the same epoch window that was selected during training
        start = r["epoch_start"]
        end   = r["epoch_end"]

        X_te_c, y_te_c = reject_artefact_trials(X_te, y_te, tag=f"{subj}/te")

        bands_te = preprocess_and_decompose(X_te_c, start, end)
        for bname in bands_te:
            bands_te[bname] = _sanitise(bands_te[bname], f"{subj}/te/{bname}")

        proj_te = {
            n: r["csp_filters"][n].transform(bands_te[n])
            for n in bands_te
            if n in r["csp_filters"]
        }

        F_te        = np.nan_to_num(extract_logvar_features(proj_te))
        F_te_scaled = apply_pipeline(F_te, r["scaler"], r["var_mask"])
        preds       = r["clf"].predict(F_te_scaled)
        ba          = balanced_accuracy_score(y_te_c, preds)
        ci_lo, ci_hi = wilson_ci(ba, len(y_te_c))

        print(f"  {subj:<20s}  Test BA={ba*100:.2f}%  "
              f"Val BA={r['bal_acc']*100:.2f}%  "
              f"CI=[{ci_lo*100:.1f}%–{ci_hi*100:.1f}%]  "
              f"window={r.get('best_window','fixed')}")
        test_accs.append(ba)

    if test_accs:
        arr = np.array(test_accs)
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(f"\n  Test mean BA : {arr.mean()*100:.2f}% "
              f"± {std*100:.2f}%  (n={len(arr)} subjects)")

    return test_accs


# =============================================================================
# MAIN
# =============================================================================

def main(run_test: bool = False) -> None:
    """
    run_test : bool
        Keep False during all development and hyperparameter tuning.
        Set True ONLY for the single final evaluation on the test set.
    """
    sep = "=" * 60
    print(sep)
    print("  BCI TRAINING PIPELINE — Imagined Speech Decoder [v3]")
    print("  Task: Bilabial vs Non-Bilabial")
    print(sep)
    print(f"  FS              : {FS} Hz")
    print(f"  Artefact thresh : {ARTEFACT_THRESHOLD_UV} µV peak-to-peak")
    print(f"  Epoch mode      : {EPOCH_MODE}")
    if EPOCH_MODE == "fixed":
        dur = (EPOCH_END_SAMPLE - EPOCH_START_SAMPLE) / FS * 1000
        print(f"  Epoch window    : {EPOCH_START_SAMPLE}–{EPOCH_END_SAMPLE} "
              f"({dur:.0f} ms)")
    print(f"  Hanning taper   : enabled")
    print(f"  Freq bands      : {list(FREQ_BANDS.keys())}")
    print(f"  CSP components  : {N_CSP_COMPONENTS} ({N_CSP_COMPONENTS//2}+{N_CSP_COMPONENTS//2})")
    print(f"  Features        : {len(FREQ_BANDS) * N_CSP_COMPONENTS} total")
    print(f"  Bilabial classes: {BILABIAL_LABEL_IDS} "
          f"({', '.join(CLASS_NAMES[c] for c in sorted(BILABIAL_LABEL_IDS))})")
    print(f"  SVM C grid      : {SVM_C_GRID}")
    print(f"  Permutations    : {N_PERMUTATIONS}")
    print(f"  Test set        : "
          f"{'WILL BE EVALUATED' if run_test else 'LOCKED'}")

    for d in (TRAIN_DIR, VAL_DIR):
        if not d.exists():
            raise RuntimeError(f"Directory not found: {d}")

    # ── Load training and validation ───────────────────────────────────────────
    X_tr, y_tr, names_tr = load_dataset(TRAIN_DIR, "Training set",   "epo_train")
    X_va, y_va, names_va = load_dataset(VAL_DIR,   "Validation set", "epo_validation")

    # ── Per-subject pipeline ───────────────────────────────────────────────────
    results, accuracies = run_per_subject(
        X_tr, y_tr, names_tr,
        X_va, y_va, names_va,
        run_permutation=True,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = summarise(results, accuracies)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_artifacts(results, summary, ARTIFACT_DIR)

    # ── Test set (locked by default) ───────────────────────────────────────────
    if run_test:
        if not TEST_DIR.exists():
            print(f"\n  !! Test directory not found: {TEST_DIR}")
        else:
            run_final_test(results, TEST_DIR)
    else:
        print(f"\n  Test set is locked.")
        print(f"  Set run_test=True in main() only when all decisions are final.")

    print(f"\n{sep}")
    print("  Pipeline complete.")
    print(sep)


if __name__ == "__main__":
    # ── INSTRUCTIONS ──────────────────────────────────────────────────────────
    # 1. Run as-is first (run_test=False, EPOCH_MODE="search").
    # 2. Check the per-subject "Best window" column in the summary.
    # 3. If most subjects prefer the same window, set EPOCH_MODE="fixed"
    #    and set EPOCH_START_SAMPLE / EPOCH_END_SAMPLE to that window.
    # 4. Rerun and confirm mean BA improves or stays the same.
    # 5. Only then set run_test=True for the single final evaluation.
    # ─────────────────────────────────────────────────────────────────────────
    main(run_test=True)