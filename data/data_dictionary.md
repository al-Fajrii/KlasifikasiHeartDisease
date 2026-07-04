# Data Dictionary — Heart_Disease.csv

Deskripsi singkat: dataset berisi rekam medis ringkas pasien dan indikator klinis untuk prediksi penyakit jantung (target). Kolom-kolom berikut adalah nama variabel seperti ada di file CSV.

- **age**: integer — Usia pasien dalam tahun.
- **sex**: binary (0,1) — Jenis kelamin (1 = laki-laki, 0 = perempuan).
- **chest pain type**: categorical (1–4) — Tipe nyeri dada; nilai numerik sesuai coding sumber dataset.
- **resting bp s**: integer — Tekanan darah sistolik saat istirahat (mm Hg). Nilai 0 dianggap implisit/missing pada dataset dan harus diubah menjadi NaN sebelum imputasi.
- **cholesterol**: integer — Kadar kolesterol total (mg/dL). Nilai 0 dianggap implisit/missing; ganti dengan NaN lalu imputasi.
- **fasting blood sugar**: binary (0,1) — Hasil gula darah puasa (1 = >120 mg/dL, 0 = ≤120).
- **resting ecg**: categorical (0–2) — Hasil elektrokardiogram saat istirahat (kode kategorik).
- **max heart rate**: integer — Denyut jantung maksimum yang tercatat (bpm).
- **exercise angina**: binary (0,1) — Ada angina saat olahraga (1 = ya, 0 = tidak).
- **oldpeak**: numeric — Depresi segmen ST (perubahan ST) pada latihan relative ke istirahat. Perhatikan desimal koma menggunakan koma dalam CSV (`,`) — pastikan parsing `decimal=','` jika diperlukan atau lakukan replace.
- **ST slope**: categorical (1–3) — Slope segmen ST saat latihan.
- **target**: binary (0,1) — Label target (1 = penyakit jantung terdeteksi / Sakit, 0 = tidak terdeteksi / Sehat).

Catatan preprocessing dan penggunaan kolom:
- Nilai `0` pada kolom numerik klinis (mis. `resting bp s`, `cholesterol`, kadang `max heart rate`) tampaknya dipakai sebagai placeholder untuk missing; ganti dengan `NaN` lalu imputasi menggunakan median yang dihitung hanya dari data TRAIN (untuk mencegah data leakage).
- Kolom kategoris (`chest pain type`, `resting ecg`, `ST slope`) sebaiknya di-encode (One-Hot) setelah split; pastikan alignment kolom antara train dan test.
- `oldpeak` pada CSV menggunakan tanda koma sebagai desimal (`1,5`) — saat membaca CSV, set `decimal=','` atau lakukan normalisasi string ke format titik sebelum konversi numeric.
- Cek dan hapus/pertimbangkan duplikat jika ada — laporkan jumlah duplikat dalam audit.

Saran dokumentasi lisensi/atribusi dataset:
- Sumber dataset: (cantumkan sumber asli yang Anda gunakan, mis. Mendeley / UCI / publikasi terkait)
- Lisensi rekomendasi untuk laporan: "Dataset under CC BY 4.0 — attribution required" — simpan teks lengkap lisensi di `data/LICENSE` jika Anda mendistribusikan dataset bersama repositori.

Jika Anda ingin, saya bisa menghasilkan `data/LICENSE` dan menambahkan contoh kalimat atribusi di `README.md` atau notebook.
