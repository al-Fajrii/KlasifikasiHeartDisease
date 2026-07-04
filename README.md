# Prediksi Penyakit Jantung — UAS Machine Learning

**Nama:** Muhammad Syafiq Al Fajri
**NIM :** A11.2024.15698
**Mata Kuliah:** Machine Learning
**Universitas:** Universitas Dian Nuswantoro
**LINK PRESENTASI** : https://youtu.be/y-PM2wRXrmc

---

## 📌 Deskripsi Proyek

Proyek ini membangun sistem klasifikasi untuk memprediksi **risiko penyakit jantung** pada pasien berdasarkan data rekam medis klinis. Model terbaik yang dipilih adalah **K-Nearest Neighbors (KNN Tuned)** berdasarkan kombinasi metrik Recall (meminimalkan False Negative) dan Accuracy tertinggi.

> ⚠️ **Disclaimer:** Prediksi yang dihasilkan bersifat pendukung keputusan, **bukan** diagnosis medis. Selalu konsultasikan hasil dengan dokter / tenaga medis profesional.

---

## 🗂️ Struktur Repositori

```
.
├── data/
│   ├── Heart_Disease.csv          # Dataset mentah (sumber asli)
│   ├── Heart_Disease_clean.csv    # Dataset setelah cleaning (pre-OHE & scaling)
│   ├── data_dictionary.md         # Kamus fitur dataset
│   └── source_dataset.md          # Informasi sumber dataset
│
├── dataset/
│   └── Heart_Disease_clean.csv    # Dataset bersih (copy untuk training)
│
├── models/
│   └── best_heart_disease_model.joblib   # Model KNN Tuned tersimpan
│
├── notebooks/
│   └── uas_ml_heartdisease_knn_nb_svm_optimization.ipynb  # Notebook utama
│
├── reports/
│   ├── all_experiment_results.csv     # Tabel hasil semua eksperimen
│   ├── audit_dataset.json             # Laporan audit dataset
│   └── classification_reports.json   # Classification report tiap model
│
├── src/
│   └── predict.py      # Pipeline inference (shared oleh kedua app)
│
├── app_streamlit.py    # Aplikasi web Streamlit
├── requirements.txt    # Dependensi Python
└── README.md           # Dokumentasi ini
```

---

## 📊 Dataset

| Atribut         | Nilai                                               |
|-----------------|-----------------------------------------------------|
| **Sumber**      | [Mendeley Data](https://data.mendeley.com/datasets/jtwbww4z9k/1) |
| **Kontributor** | Sherko MURAD                                        |
| **DOI**         | 10.17632/jtwbww4z9k.1                              |
| **Lisensi**     | CC BY 4.0                                           |
| **Jumlah data** | 1.190 baris, 12 kolom (11 fitur + 1 target)         |
| **Target**      | `0` = Sehat, `1` = Penyakit Jantung Terdeteksi      |

### Fitur Input

| Fitur                | Tipe    | Keterangan                                          |
|----------------------|---------|-----------------------------------------------------|
| `age`                | int     | Usia pasien (tahun)                                 |
| `sex`                | binary  | 1 = Laki-laki, 0 = Perempuan                        |
| `chest pain type`    | cat 1–4 | Tipe nyeri dada                                     |
| `resting bp s`       | int     | Tekanan darah sistolik istirahat (mmHg)             |
| `cholesterol`        | int     | Kolesterol total (mg/dL)                            |
| `fasting blood sugar`| binary  | 1 = >120 mg/dL, 0 = ≤120 mg/dL                     |
| `resting ecg`        | cat 0–2 | Hasil EKG istirahat                                 |
| `max heart rate`     | int     | Detak jantung maksimum (bpm)                        |
| `exercise angina`    | binary  | 1 = Ada angina saat olahraga, 0 = Tidak ada         |
| `oldpeak`            | float   | Depresi segmen ST (mm)                              |
| `ST slope`           | cat 1–3 | Slope segmen ST: 1=Upsloping, 2=Flat, 3=Downsloping |

---

## 🔬 Pipeline Machine Learning

```
Dataset Mentah (1190 baris)
    ↓
Konversi tipe oldpeak: str → float
    ↓
Audit: Missing Value, Duplikat (272), Outlier (boxplot)
    ↓
Hapus duplikat → 918 baris bersih
    ↓
Train-Test Split (80:20, stratified, random_state=42)
    ├── X_train: 734 sampel
    └── X_test : 184 sampel
    ↓
Preprocessing (hanya dari X_train — anti data leakage)
    ├── 1. Imputasi 0 → median (cholesterol, resting bp s)
    ├── 2. Capping outlier IQR (5 fitur numerik)
    ├── 3. One-Hot Encoding (drop_first=True)
    └── 4. StandardScaler (fit hanya pada X_train)
    ↓
Model Baseline: KNN, GaussianNB, SVM, Decision Tree
    ↓
Hyperparameter Tuning (GridSearchCV, scoring=recall, 5-Fold CV)
    ├── KNN Tuned
    ├── NB Tuned
    ├── SVM Tuned
    └── DT Tuned
    ↓
Evaluasi: Accuracy, Balanced Accuracy, Recall, Precision, F1, AUC-ROC
    ↓
Model Terbaik: KNN Tuned
    (n_neighbors=9, metric=manhattan, weights=uniform)
```

---

## 🏆 Hasil Eksperimen

| Model        | Acc Test | Bal Acc | Recall | Precision | F1    | AUC-ROC |
|--------------|----------|---------|--------|-----------|-------|---------|
| **KNN Tuned**| **0.8696**| –      | 0.8725 | 0.8980    | **0.8945** | **0.9173** |
| SVM Tuned    | 0.8587   | –       | 0.9608 | 0.8167    | 0.8829| 0.9429  |
| KNN Baseline | 0.8478   | –       | 0.8922 | 0.8426    | 0.8667| 0.9047  |
| GaussianNB   | 0.8587   | –       | 0.9118 | 0.8455    | 0.8774| 0.8957  |
| DT Tuned     | 0.7717   | –       | 0.7745 | 0.7379    | 0.8272| –       |
| Dummy Baseline| 0.5543  | –       | 1.0000 | 0.5543    | 0.7133| 0.5000  |

**Alasan pemilihan KNN Tuned:**
- Recall tertinggi di antara model non-SVM dengan gap overfitting minimal
- F1-Score dan AUC-ROC tertinggi secara keseluruhan
- Generalisasi baik (gap Acc Train – Acc Test ≈ 0)

---

## 🚀 Cara Menjalankan

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi Streamlit

```bash
streamlit run app_streamlit.py
```

Buka browser di `http://localhost:8501`

### 3. Jalankan aplikasi Gradio

---

## 📁 File Penting

| File                         | Deskripsi                                          |
|------------------------------|----------------------------------------------------|
| `src/predict.py`             | Pipeline inference yang digunakan kedua app        |
| `models/best_heart_disease_model.joblib` | Model KNN Tuned tersimpan             |
| `notebooks/uas_ml_*.ipynb`   | Notebook eksperimen lengkap                        |
| `reports/all_experiment_results.csv` | Hasil semua model                          |

---

## 📚 Referensi

- Murad, S. (n.d.). *Heart Disease Dataset*. Mendeley Data. DOI: 10.17632/jtwbww4z9k.1
- Fix, E., & Hodges, J. (1951). Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties.
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.

---

© 2025 Muhammad Syafiq Al Fajri | Universitas Dian Nuswantoro
