"""
app_streamlit.py
================
Aplikasi Streamlit untuk prediksi risiko penyakit jantung.
Jalankan dengan:
    streamlit run app_streamlit.py
"""

import sys
import os

# Tambahkan direktori src ke path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from predict import predict

# ── Konfigurasi halaman ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.markdown(
    """
    Aplikasi ini menggunakan model **K-Nearest Neighbors (KNN Tuned)** yang dilatih
    pada dataset Heart Disease untuk memprediksi risiko penyakit jantung pasien.

    > ⚠️ **Disclaimer:** Prediksi ini bersifat pendukung keputusan, **bukan** diagnosis medis.
    > Selalu konsultasikan hasil dengan dokter / tenaga medis profesional.
    """
)

st.divider()

# ── Form input ────────────────────────────────────────────────────────────────
st.subheader("📋 Data Pasien")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Usia (tahun)",
        min_value=1, max_value=120, value=55,
        help="Usia pasien dalam tahun",
    )
    sex = st.selectbox(
        "Jenis Kelamin",
        options=[1, 0],
        format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan",
    )
    chest_pain = st.selectbox(
        "Tipe Nyeri Dada",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 — Typical Angina",
            2: "2 — Atypical Angina",
            3: "3 — Non-Anginal Pain",
            4: "4 — Asymptomatic",
        }[x],
        index=3,
    )
    resting_bp = st.number_input(
        "Tekanan Darah Sistolik Istirahat (mmHg)",
        min_value=0, max_value=300, value=140,
        help="Nilai 0 akan diimputasi dengan median data training",
    )
    cholesterol = st.number_input(
        "Kolesterol Total (mg/dL)",
        min_value=0, max_value=600, value=250,
        help="Nilai 0 akan diimputasi dengan median data training",
    )
    fasting_bs = st.selectbox(
        "Gula Darah Puasa",
        options=[0, 1],
        format_func=lambda x: "≤ 120 mg/dL (Normal)" if x == 0 else "> 120 mg/dL (Tinggi)",
    )

with col2:
    resting_ecg = st.selectbox(
        "Hasil EKG Istirahat",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 — Normal",
            1: "1 — ST-T Wave Abnormality",
            2: "2 — Left Ventricular Hypertrophy",
        }[x],
    )
    max_hr = st.number_input(
        "Detak Jantung Maksimum (bpm)",
        min_value=1, max_value=300, value=130,
    )
    exercise_angina = st.selectbox(
        "Angina Saat Olahraga",
        options=[0, 1],
        format_func=lambda x: "Tidak Ada" if x == 0 else "Ada",
    )
    oldpeak = st.number_input(
        "Oldpeak — Depresi Segmen ST (mm)",
        min_value=-10.0, max_value=10.0, value=2.0, step=0.1,
        format="%.1f",
    )
    st_slope = st.selectbox(
        "Slope Segmen ST",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — Upsloping",
            2: "2 — Flat",
            3: "3 — Downsloping",
        }[x],
        index=1,
    )

st.divider()

# ── Tombol prediksi ───────────────────────────────────────────────────────────
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    raw_input = {
        "age"                : age,
        "sex"                : sex,
        "chest pain type"    : chest_pain,
        "resting bp s"       : resting_bp,
        "cholesterol"        : cholesterol,
        "fasting blood sugar": fasting_bs,
        "resting ecg"        : resting_ecg,
        "max heart rate"     : max_hr,
        "exercise angina"    : exercise_angina,
        "oldpeak"            : oldpeak,
        "ST slope"           : st_slope,
    }

    try:
        result = predict(raw_input)
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")
        st.stop()

    # Tampilkan hasil
    st.subheader("📊 Hasil Prediksi")

    if result["label"] == 1:
        st.error("🔴 **BERISIKO PENYAKIT JANTUNG**")
    else:
        st.success("🟢 **TIDAK BERISIKO PENYAKIT JANTUNG**")

    # Gauge probabilitas
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric(
            label="Probabilitas Sehat",
            value=f"{result['prob_sehat']*100:.1f}%",
        )
    with prob_col2:
        st.metric(
            label="Probabilitas Sakit",
            value=f"{result['prob_sakit']*100:.1f}%",
        )

    # Progress bar
    st.progress(result["prob_sakit"], text="Risiko penyakit jantung")

    # Keterangan
    st.info(result["keterangan"])

    st.caption(
        "Model: KNN Tuned (n_neighbors=9, metric=manhattan, weights=uniform) "
        "| Dataset: Heart Disease — Mendeley (CC BY 4.0)"
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "© 2025 Muhammad Syafiq Al Fajri — UAS Machine Learning "
    "A11.2024.15698 | Universitas Dian Nuswantoro"
)
