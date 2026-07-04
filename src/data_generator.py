    """
data_generator.py
=================
Utilitas untuk memuat dan memvalidasi dataset Heart Disease.

Tanggung jawab:
  - Memuat CSV mentah dan menangani quirk format (koma desimal pada oldpeak)
  - Melaporkan audit awal: tipe data, missing value, duplikat, nilai 0 tidak wajar
  - Menghasilkan DataFrame bersih (pre-OHE, pre-scaling) untuk digunakan ml_core.py

Cara pakai:
    from src.data_generator import load_raw, audit, load_clean
"""

import os
import numpy as np
import pandas as pd

# ── Konstanta path ─────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV    = os.path.join(_BASE_DIR, "data", "Heart_Disease.csv")
CLEAN_CSV  = os.path.join(_BASE_DIR, "data", "Heart_Disease_clean.csv")

# Kolom yang nilai 0-nya tidak masuk akal secara klinis
ZERO_IMPUTE_COLS = ["cholesterol", "resting bp s"]

# Kolom numerik murni (bukan kode kategori)
NUMERIC_COLS = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]

# Kolom kategoris (meskipun tipenya int64)
CATEGORICAL_COLS = [
    "sex", "chest pain type", "fasting blood sugar",
    "resting ecg", "exercise angina", "ST slope",
]


# ── Fungsi utama ───────────────────────────────────────────────────────────────

def load_raw(csv_path: str = RAW_CSV) -> pd.DataFrame:
    """
    Muat CSV mentah dan perbaiki tipe data.

    Kolom `oldpeak` pada CSV menggunakan koma sebagai separator desimal
    (mis. '1,5'). Fungsi ini mengganti koma → titik lalu konversi ke float.

    Parameters
    ----------
    csv_path : str
        Path ke file CSV. Default: data/Heart_Disease.csv

    Returns
    -------
    pd.DataFrame dengan 12 kolom dan tipe data sudah benar.
    """
    df = pd.read_csv(csv_path, encoding="latin1", sep=";")

    # Perbaiki tipe oldpeak: str → float
    if df["oldpeak"].dtype == object:
        df["oldpeak"] = (
            df["oldpeak"]
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    return df


def audit(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Laporkan audit kualitas data.

    Parameters
    ----------
    df      : pd.DataFrame — dataset yang akan diaudit
    verbose : bool — jika True, cetak ringkasan ke stdout

    Returns
    -------
    dict dengan kunci:
        n_rows, n_cols, dtypes, missing_total, missing_per_col,
        n_duplicates, zero_not_valid (per kolom), nunique_target
    """
    report = {
        "n_rows"         : int(df.shape[0]),
        "n_cols"         : int(df.shape[1]),
        "dtypes"         : df.dtypes.astype(str).to_dict(),
        "missing_total"  : int(df.isnull().sum().sum()),
        "missing_per_col": df.isnull().sum().to_dict(),
        "n_duplicates"   : int(df.duplicated().sum()),
        "zero_not_valid" : {
            col: int((df[col] == 0).sum())
            for col in ZERO_IMPUTE_COLS
            if col in df.columns
        },
        "nunique_target" : int(df["target"].nunique()) if "target" in df.columns else None,
    }

    if verbose:
        print("=" * 55)
        print("  AUDIT DATASET")
        print("=" * 55)
        print(f"  Jumlah baris        : {report['n_rows']}")
        print(f"  Jumlah kolom        : {report['n_cols']}")
        print(f"  Total missing value : {report['missing_total']}")
        print(f"  Jumlah duplikat     : {report['n_duplicates']}")
        for col, cnt in report["zero_not_valid"].items():
            if cnt > 0:
                print(f"  ⚠️  Nilai 0 tidak wajar di '{col}': {cnt} baris")
        print("=" * 55)

    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan dataset mentah (sebelum split):
      1. Hapus duplikat
      2. Ganti nilai 0 tidak wajar dengan NaN pada kolom klinis
      3. Imputasi NaN dengan median SELURUH dataset
         (catatan: untuk training yang benar, gunakan ml_core.preprocess_split
          yang menghitung median hanya dari X_train)

    Fungsi ini menghasilkan snapshot "data bersih" yang disimpan di
    data/Heart_Disease_clean.csv.

    Returns
    -------
    pd.DataFrame bersih (belum OHE, belum scaling)
    """
    df = df.copy()

    # 1) Hapus duplikat
    before = len(df)
    df = df.drop_duplicates()
    after  = len(df)
    print(f"  Duplikat dihapus: {before - after} baris ({before} → {after})")

    # 2) Tandai nilai 0 tidak wajar sebagai NaN
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            n_zero = (df[col] == 0).sum()
            if n_zero > 0:
                df[col] = df[col].replace(0, np.nan)
                print(f"  Nilai 0 diganti NaN di '{col}': {n_zero} baris")

    # 3) Imputasi NaN dengan median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputasi median '{col}': {median_val:.2f}")

    return df


def save_clean(df_clean: pd.DataFrame, path: str = CLEAN_CSV) -> None:
    """Simpan DataFrame bersih ke CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_clean.to_csv(path, index=False)
    print(f"  ✅ Data bersih disimpan ke: {path}")


def load_clean(path: str = CLEAN_CSV) -> pd.DataFrame:
    """
    Muat dataset yang sudah dibersihkan (dari data/Heart_Disease_clean.csv).
    Jika file belum ada, jalankan pipeline load_raw → clean → save_clean terlebih dahulu.
    """
    if not os.path.exists(path):
        print(f"  ℹ️  File clean tidak ditemukan, membuat dari raw...")
        df_raw   = load_raw()
        df_clean = clean(df_raw)
        save_clean(df_clean, path)
        return df_clean

    return pd.read_csv(path)


# ── Entry point (opsional, untuk testing cepat) ───────────────────────────────
if __name__ == "__main__":
    print("\n[ data_generator.py ] Memuat dan mengaudit dataset...\n")
    df_raw = load_raw()
    report = audit(df_raw)

    print("\nMembersihkan dataset...")
    df_clean = clean(df_raw)
    save_clean(df_clean)

    print(f"\nDataset bersih: {df_clean.shape[0]} baris × {df_clean.shape[1]} kolom")
    print(df_clean.head(3).to_string())
