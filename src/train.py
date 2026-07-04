"""
train.py
========
Script pelatihan model end-to-end.

Alur:
  1. Muat & bersihkan data      (data_generator)
  2. Preprocessing + split      (ml_core.preprocess_split)
  3. Training baseline          (ml_core.train_baseline)
  4. Hyperparameter tuning      (ml_core.tune_*)
  5. Pilih & simpan model terbaik (ml_core.select_best, save_model)
  6. Simpan laporan              (reports/)

Cara pakai:
    python src/train.py
    python src/train.py --data data/Heart_Disease.csv --metric Recall
"""

import os
import sys
import argparse
import json
import time

import pandas as pd

# Pastikan direktori src ada di path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_generator import load_raw, audit, clean, save_clean, CLEAN_CSV
from ml_core import (
    preprocess_split,
    train_baseline,
    tune_knn, tune_nb, tune_svm, tune_dt,
    select_best,
    save_model,
    summary_table,
    REPORTS_DIR,
)


# ── Argumen CLI ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Latih model prediksi penyakit jantung"
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path ke CSV mentah (default: data/Heart_Disease.csv)",
    )
    parser.add_argument(
        "--metric",
        default="Recall",
        choices=["Recall", "F1-Score", "AUC-ROC", "Acc Test", "Bal Acc"],
        help="Metrik pemilihan model terbaik (default: Recall)",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Lewati hyperparameter tuning (hanya baseline)",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Gunakan SMOTE pada data training sebelum tuning",
    )
    return parser.parse_args()


# ── Pipeline utama ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0   = time.time()

    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE — Prediksi Penyakit Jantung")
    print("=" * 60)

    # 1) Muat & bersihkan data
    print("\n[ Step 1 ] Memuat dan membersihkan data...")
    csv_path = args.data if args.data else None
    if csv_path and os.path.exists(csv_path):
        df_raw = load_raw(csv_path)
    else:
        df_raw = load_raw()

    _ = audit(df_raw)
    df_clean = clean(df_raw)
    save_clean(df_clean, CLEAN_CSV)

    # 2) Preprocessing + split
    print("\n[ Step 2 ] Preprocessing & train-test split...")
    X_train, X_test, y_train, y_test, scaler, iqr_bounds, train_medians = \
        preprocess_split(df_clean)

    # SMOTE (opsional)
    X_fit, y_fit = X_train, y_train
    if args.smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_fit, y_fit = sm.fit_resample(X_train, y_train)
            print(f"  SMOTE aktif | X_fit shape: {X_fit.shape}")
        except ImportError:
            print("  ⚠️  imbalanced-learn tidak ditemukan, lewati SMOTE")

    # 3) Baseline
    print("\n[ Step 3 ] Training model baseline...")
    hasil = train_baseline(X_train, X_test, y_train, y_test)

    # 4) Hyperparameter tuning
    if not args.no_tuning:
        print("\n[ Step 4 ] Hyperparameter tuning...")

        print("\n  -- KNN Tuning --")
        _, hasil["KNN_Tuned"] = tune_knn(
            X_fit, y_fit, X_train, X_test, y_train, y_test,
        )

        print("\n  -- NB Tuning --")
        _, hasil["NB_Tuned"] = tune_nb(
            X_fit, y_fit, X_train, X_test, y_train, y_test,
        )

        print("\n  -- SVM Tuning --")
        _, hasil["SVM_Tuned"] = tune_svm(
            X_fit, y_fit, X_train, X_test, y_train, y_test,
        )

        print("\n  -- Decision Tree Tuning --")
        _, hasil["DT_Tuned"] = tune_dt(
            X_fit, y_fit, X_train, X_test, y_train, y_test,
        )

    # 5) Pilih & simpan model terbaik
    print(f"\n[ Step 5 ] Memilih model terbaik berdasarkan '{args.metric}'...")
    best_name, best_model, best_result = select_best(hasil, metric=args.metric)
    save_model(best_model)

    # 6) Laporan
    print("\n[ Step 6 ] Menyimpan laporan...")
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Tabel ringkasan semua model
    df_summary = summary_table(hasil)
    summary_path = os.path.join(REPORTS_DIR, "all_experiment_results.csv")
    df_summary.to_csv(summary_path)
    print(f"  ✅ Tabel hasil disimpan ke: {summary_path}")

    # Audit dataset (JSON)
    from data_generator import audit as _audit
    audit_report = _audit(df_raw, verbose=False)
    audit_path = os.path.join(REPORTS_DIR, "audit_dataset.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✅ Audit dataset disimpan ke: {audit_path}")

    # ── Ringkasan akhir ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  SELESAI dalam {elapsed:.1f} detik")
    print(f"  Model terbaik : {best_name}")
    print(f"  {args.metric:12s} : {best_result[args.metric]:.4f}")
    print(f"  Acc Test      : {best_result['Acc Test']:.4f}")
    print(f"  AUC-ROC       : {best_result['AUC-ROC']:.4f}")
    print(f"{'=' * 60}\n")

    print("\nRingkasan semua model:")
    print(df_summary[["Acc Test", "Bal Acc", "Recall",
                       "Precision", "F1-Score", "AUC-ROC"]].to_string())


if __name__ == "__main__":
    main()
