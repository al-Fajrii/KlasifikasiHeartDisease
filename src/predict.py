"""
predict.py
==========
Shared inference pipeline untuk prediksi penyakit jantung.
Digunakan oleh app_streamlit.py dan app_gradio.py.

Pipeline (identik dengan notebook):
  1. Imputasi nilai 0 tidak wajar → NaN → median (dari data training)
  2. Capping outlier IQR (batas dari data training)
  3. One-Hot Encoding (drop_first=True)
  4. Alignment kolom (urutan harus sama persis dengan X_train)
  5. StandardScaler (transform saja — load dari models/scaler.joblib)
  6. Prediksi dengan model KNN Tuned

PENTING: Jalankan sel "Simpan Model Terbaik" di notebook terlebih dahulu
agar file models/scaler.joblib dan models/pipeline_meta.json tersedia.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd

# ── Path default ──────────────────────────────────────────────────────────────
_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH  = os.path.join(_BASE_DIR, "models", "best_heart_disease_model.joblib")
_SCALER_PATH = os.path.join(_BASE_DIR, "models", "scaler.joblib")
_META_PATH   = os.path.join(_BASE_DIR, "models", "pipeline_meta.json")

# ── Cache (load sekali, pakai berkali-kali) ───────────────────────────────────
_cache: dict = {}


def _load_artifacts():
    """Muat model, scaler, dan metadata pipeline. Di-cache setelah load pertama."""
    if _cache:
        return _cache["model"], _cache["scaler"], _cache["meta"]

    # Cek keberadaan file
    for path, label in [
        (_MODEL_PATH,  "models/best_heart_disease_model.joblib"),
        (_SCALER_PATH, "models/scaler.joblib"),
        (_META_PATH,   "models/pipeline_meta.json"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}\n"
                "Pastikan sudah menjalankan sel 'Simpan Model Terbaik' di notebook."
            )

    model  = joblib.load(_MODEL_PATH)
    scaler = joblib.load(_SCALER_PATH)
    with open(_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _cache["model"]  = model
    _cache["scaler"] = scaler
    _cache["meta"]   = meta
    return model, scaler, meta


def preprocess(raw_input: dict) -> pd.DataFrame:
    """
    Terima dict nilai mentah (raw) dari user dan kembalikan DataFrame
    yang sudah siap diprediksi oleh model.

    Parameters
    ----------
    raw_input : dict
        Kunci sesuai nama fitur asli dataset:
        age, sex, chest pain type, resting bp s, cholesterol,
        fasting blood sugar, resting ecg, max heart rate,
        exercise angina, oldpeak, ST slope

    Returns
    -------
    pd.DataFrame dengan kolom sesuai X_train, sudah di-scale.
    """
    _, scaler, meta = _load_artifacts()

    df = pd.DataFrame([raw_input])

    # Step 1 — Imputasi nilai 0 tidak wajar → median dari training
    for col, median_val in meta["train_median"].items():
        df[col] = df[col].replace(0, np.nan).fillna(median_val)

    # Step 2 — Capping outlier IQR (batas dari training)
    for col, (lo, hi) in meta["iqr_bounds"].items():
        df[col] = df[col].clip(lower=lo, upper=hi)

    # Step 3 — One-Hot Encoding
    df_enc = pd.get_dummies(df, columns=meta["cat_cols"], drop_first=True, dtype=int)

    # Step 4 — Alignment kolom
    for col in meta["expected_cols"]:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[meta["expected_cols"]]

    # Step 5 — StandardScaler (gunakan scaler yang di-fit pada X_train)
    df_enc[meta["scale_cols"]] = scaler.transform(df_enc[meta["scale_cols"]])

    return df_enc


def predict(raw_input: dict) -> dict:
    """
    Prediksi risiko penyakit jantung dari nilai mentah.

    Parameters
    ----------
    raw_input : dict
        Nilai fitur mentah (lihat preprocess()).

    Returns
    -------
    dict dengan kunci:
        - label      : int   (0=Sehat, 1=Sakit)
        - prob_sehat : float (probabilitas kelas 0)
        - prob_sakit : float (probabilitas kelas 1)
        - keterangan : str   (teks ringkas untuk UI)
    """
    model, _, _ = _load_artifacts()
    X = preprocess(raw_input)

    proba = model.predict_proba(X)[0]

    # Gunakan threshold 0.35 agar lebih sensitif mendeteksi risiko
    # (lebih baik false positive daripada melewatkan pasien sakit)
    THRESHOLD = 0.35
    label = 1 if proba[1] >= THRESHOLD else 0

    if label == 1:
        keterangan = (
            "⚠️ Pasien diprediksi BERISIKO penyakit jantung.\n"
            f"Keyakinan model: {proba[1]*100:.1f}%.\n"
            "→ Disarankan pemeriksaan lanjutan (EKG, stress test, dll.)."
        )
    else:
        keterangan = (
            "✅ Pasien diprediksi TIDAK berisiko penyakit jantung.\n"
            f"Keyakinan model: {proba[0]*100:.1f}%.\n"
            "→ Tetap lakukan pemantauan rutin tekanan darah & kolesterol."
        )

    return {
        "label"      : label,
        "prob_sehat" : float(proba[0]),
        "prob_sakit" : float(proba[1]),
        "keterangan" : keterangan,
    }
