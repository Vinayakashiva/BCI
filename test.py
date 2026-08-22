# ============================================
# EEG IMAGINED SPEECH PIPELINE (ZHAO & RUDZICZ 2015 METHODOLOGY)
# ============================================

import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import scipy.stats as stats
import gc  # Added for RAM garbage collection
import joblib  # NEW: Added for saving the model

# Machine Learning Modules
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

# =========================
# CONFIGURATION
# =========================
BASE_PATH = r"C:\Users\Admin\OneDrive\Documents\mojar_project\BCI2020 EEG Signal for Words"

TRAIN_DIR = os.path.join(BASE_PATH, "Training set")
VAL_DIR = os.path.join(BASE_PATH, "Validation set")
TEST_DIR = os.path.join(BASE_PATH, "Test set")

FS = 256

# Original labels for reference
label_map = {0: "hello", 1: "help me", 2: "stop", 3: "thank you", 4: "yes"}


# =========================
# SIGNAL PROCESSING (SECTION 2.2)
# =========================
def bandpass_filter(data, low=1, high=50, fs=256):
    """Paper used 1Hz to 50Hz bandpass filter"""
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def apply_laplacian_filter(X):
    """
    Simulates the small Laplacian filter mentioned in Section 2.2.
    Subtracts the mean of all channels from each individual channel
    to localize activity and remove common background noise.
    """
    common_average = np.mean(X, axis=1, keepdims=True)
    return X - common_average


# =========================
# FEATURE EXTRACTION (SECTION 2.3)
# =========================
def calculate_zhao_stats(window):
    """Calculates the specific statistical suite from Section 2.3"""
    mean = np.mean(window, axis=-1)
    median = np.median(window, axis=-1)
    std = np.std(window, axis=-1)
    var = np.var(window, axis=-1)
    maximum = np.max(window, axis=-1)
    minimum = np.min(window, axis=-1)
    sum_val = np.sum(window, axis=-1)

    # Non-Gaussianity features
    skew = stats.skew(window, axis=-1)
    kurt = stats.kurtosis(window, axis=-1)

    # Absolute value features
    abs_win = np.abs(window)
    abs_mean = np.mean(abs_win, axis=-1)
    abs_max = np.max(abs_win, axis=-1)

    return [mean, median, std, var, maximum, minimum, sum_val, skew, kurt, abs_mean, abs_max]


def extract_features(X):
    """
    Implements the exact 10% windowing with 50% overlap and derivative features
    described by Zhao & Rudzicz (2015).
    X shape: (Trials, Channels, Time)
    """
    trials, channels, time_steps = X.shape

    # Windowing parameters (10% length, 50% overlap)
    window_size = max(int(0.1 * time_steps), 1)
    step_size = max(int(0.5 * window_size), 1)

    # Calculate starting indices for overlapping windows
    window_starts = np.arange(0, time_steps - window_size + 1, step_size)

    all_trial_features = []

    for trial_idx in range(trials):
        trial_data = X[trial_idx]  # Shape: (Channels, Time)

        # Calculate 1st and 2nd derivatives of the signal (Velocity and Acceleration)
        # Using np.gradient instead of np.diff to maintain array length
        d1_data = np.gradient(trial_data, axis=-1)
        d2_data = np.gradient(d1_data, axis=-1)

        trial_features = []

        for start in window_starts:
            end = start + window_size

            # Extract the raw, velocity, and acceleration windows
            w_raw = trial_data[:, start:end]
            w_d1 = d1_data[:, start:end]
            w_d2 = d2_data[:, start:end]

            # Compute stats for all three representations
            stats_raw = calculate_zhao_stats(w_raw)
            stats_d1 = calculate_zhao_stats(w_d1)
            stats_d2 = calculate_zhao_stats(w_d2)

            # Flatten channel stats for this window
            window_features = np.concatenate(stats_raw + stats_d1 + stats_d2, axis=-1)
            trial_features.append(window_features)

        # Flatten all windows into one massive feature vector for this trial
        all_trial_features.append(np.concatenate(trial_features, axis=-1))

    features = np.array(all_trial_features)
    return np.nan_to_num(features)


# =========================
# DATA LOADER
# =========================
def load_subject(file_path):
    X, y = None, None
    try:
        mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        for key in mat.keys():
            if not key.startswith('__'):
                val = mat[key]
                if hasattr(val, 'x'):
                    X = np.array(val.x)
                    if hasattr(val, 'y'):
                        y = np.array(val.y)
                        if y.size == 0: y = None
                    break
    except NotImplementedError:
        import h5py
        with h5py.File(file_path, 'r') as f:
            datasets = []

            def find_all(group):
                for k in group.keys():
                    item = group[k]
                    if isinstance(item, h5py.Dataset):
                        datasets.append(item)
                    elif isinstance(item, h5py.Group):
                        find_all(item)

            find_all(f)
            datasets.sort(key=lambda d: d.size, reverse=True)

            if len(datasets) > 0: X = np.array(datasets[0])
            if len(datasets) > 1:
                y = np.array(datasets[-1])
                if y.size == 0: y = None

    # SMART SHAPE ALIGNER
    if X is not None and X.ndim == 3:
        dims = list(X.shape)
        c_idx = dims.index(64) if 64 in dims else dims.index(min(dims))
        other_dims = [i for i in range(3) if i != c_idx]
        if dims[other_dims[0]] > dims[other_dims[1]]:
            t_idx, tr_idx = other_dims[0], other_dims[1]
        else:
            t_idx, tr_idx = other_dims[1], other_dims[0]
        X = np.transpose(X, (tr_idx, c_idx, t_idx))

    # ROBUST ONE-HOT DECODER
    if y is not None and X is not None:
        y = np.squeeze(y)
        if y.ndim == 2:
            if y.shape[0] == X.shape[0] and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            elif y.shape[1] == X.shape[0] and y.shape[0] > 1:
                y = np.argmax(y, axis=0)
            else:
                y = y.flatten()
        else:
            y = y.flatten()

        if y.size != X.shape[0]:
            y = None

    return X, y


def load_from_folder(folder):
    X_list, y_list = [], []
    if not os.path.exists(folder): return np.array([]), np.array([])
    files = sorted([f for f in os.listdir(folder) if f.endswith(".mat")])

    for file in files:
        path = os.path.join(folder, file)
        X, y = load_subject(path)
        if X is None: continue

        # Apply paper's preprocessing techniques
        X = apply_laplacian_filter(X)
        X = bandpass_filter(X, low=1, high=50, fs=FS)
        X = extract_features(X)

        X_list.append(X)
        if y is not None: y_list.append(y)

    return (np.concatenate(X_list) if X_list else np.array([])), (np.concatenate(y_list) if y_list else None)


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    print("\n=== PHASE 1: LOADING DATA (Zhao & Rudzicz Method) ===")
    print("Extracting 10% overlapping windows, Laplacian spatial filters, and derivatives...")
    X_train, y_train = load_from_folder(TRAIN_DIR)
    X_val, y_val = load_from_folder(VAL_DIR)
    X_test, y_test = load_from_folder(TEST_DIR)

    if X_train.size == 0: exit()

    print("\n=== PHASE 2: PHONOLOGICAL MAPPING (The Secret to >80% Accuracy) ===")

    # Standardize to 0-4
    if np.min(y_train) == 1:
        y_train -= 1
        if y_val is not None: y_val -= 1
        if y_test is not None: y_test -= 1

    print(f"Original Words Found: {[label_map[i] for i in np.unique(y_train)]}")


    # --- BINARY PHONOLOGICAL MAPPING ---
    # Class 1 (Bilabial Sounds - Lips Touching): "help me" (1), "stop" (2)
    # Class 0 (Non-Bilabial Sounds): "hello" (0), "thank you" (3), "yes" (4)
    def map_to_binary(y_array):
        if y_array is None: return None
        return np.where((y_array == 1) | (y_array == 2), 1, 0)


    y_train = map_to_binary(y_train)
    y_val = map_to_binary(y_val)
    y_test = map_to_binary(y_test)

    binary_label_map = {0: "Non-Bilabial (hello, thanks, yes)", 1: "Bilabial (help me, stop)"}
    print("Successfully mapped 5 words to Binary Motor-Planning Intentions!")

    print("\n=== PHASE 3: SCALING ===")
    # Using StandardScaler as it is critical for massive feature spaces (65,000+ features in the paper)
    scaler = StandardScaler()

    # CRITICAL RAM FIX: Cast data to float32. This cuts memory usage by exactly 50%
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    if X_val is not None and X_val.size > 0:
        X_val = scaler.transform(X_val).astype(np.float32)
    if X_test is not None and X_test.size > 0:
        X_test = scaler.transform(X_test).astype(np.float32)

    # Force Python to clear any unused memory before starting the heavy AI training
    gc.collect()

    # =========================
    # HIGH-ACCURACY TRAINING
    # =========================
    print(f"\n=== PHASE 4: TRAINING AI ({len(y_train)} Trials) ===")

    pipeline = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('hgb', HistGradientBoostingClassifier(random_state=42, class_weight='balanced'))
    ])

    param_grid = {
        # Increased feature selection caps since Gradient Boosters handle high dimensions well
        'feature_selection__k': [100, 300, 500],
        'hgb__learning_rate': [0.05, 0.1],
        'hgb__max_iter': [100, 200]
    }

    # CRITICAL RAM FIX: Changed n_jobs=-1 to n_jobs=1 to prevent Python from duplicating the
    # massive dataset in memory across multiple CPU cores. It will be slightly slower, but will not crash.
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)

    print("\n✅ Optimization Complete!")
    print(f"🏆 Best AI Settings Found: {grid_search.best_params_}")
    print(f"📈 Best Internal Training Accuracy: {grid_search.best_score_ * 100:.2f}%")

    best_model = grid_search.best_estimator_

    if X_val is not None and y_val is not None:
        print("\n=== PHASE 5: VALIDATION ===")
        val_pred = best_model.predict(X_val)
        print(f"Validation Accuracy: {accuracy_score(y_val, val_pred) * 100:.2f}%")

    print("\n=== PHASE 6: TEST (OFFICIAL COMPETITION TEST SET) ===")
    if X_test is not None and X_test.size > 0:
        test_pred = best_model.predict(X_test)

        if y_test is not None:
            acc = accuracy_score(y_test, test_pred)
            print(f"\n🎯 FINAL TEST SET ACCURACY: {acc * 100:.2f}%")
            target_names = [binary_label_map[i] for i in range(2) if i in np.unique(y_test)]
            print("\nClassification Report:\n", classification_report(y_test, test_pred, target_names=target_names))
        else:
            print("Test set loaded and predicted! (Labels are hidden by dataset creators to prevent cheating).")
    else:
        print("❌ No test data found.")

    # =========================
    # SAVE MODEL
    # =========================
    print("\n=== PHASE 7: SAVING MODEL ===")
    model_filename = "best_eeg_hgb_model.joblib"
    scaler_filename = "eeg_scaler.joblib"

    # Save the optimized AI model
    joblib.dump(best_model, model_filename)

    # Save the scaler (Critical: You MUST scale future brainwaves exactly the same way before predicting)
    joblib.dump(scaler, scaler_filename)

    print(f"✅ AI Model successfully saved to: {model_filename}")
    print(f"✅ Data Scaler successfully saved to: {scaler_filename}")
    print("\nTo use this model in the future without retraining, run:")
    print("  import joblib")
    print("  model = joblib.load('best_eeg_hgb_model.joblib')")
    print("  scaler = joblib.load('eeg_scaler.joblib')")