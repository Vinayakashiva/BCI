# TODO — EEG Authentication + Word Prediction Gate

## Goal

Build a separate authentication module that uses a person's EEG brainwave as a
biometric. If a presented wave matches an enrolled person from the dataset,
grant access and allow word prediction; otherwise deny access.

## Steps

- [x] 1. Create `authentication.py` — core module (config, dataset discovery, signal processing, subject-ID features, enrollment, verification, EER/FAR/FRR threshold tuning, word-prediction gate, CLI).
- [x] 2. Create `auth_streamlit.py` — Streamlit UI (login gate -> GRANTED -> word prediction; DENIED -> locked).
- [x] 3. Add dependency install instructions / requirements.txt.
- [x] 4. Test with `--self-test` synthetic data to verify machinery works end-to-end.
- [x] 5. Validate code runs without errors (imports, CLI entrypoints).

## Verification status

- `python authentication.py self-test` → PASSED (EER = 0.00%, verification accuracy 100.0%, word model trained for 5 subjects)
- Artifacts saved to `auth_artifacts/` (subject_model.pkl, word_models.pkl, eer.json)
