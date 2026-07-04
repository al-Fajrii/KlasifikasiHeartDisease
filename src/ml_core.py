"""
ml_core.py
==========
Inti pipeline machine learning:
  - Preprocessing yang anti data leakage (fit hanya dari X_train)
  - Pelatihan model baseline (KNN, GaussianNB, SVM, DecisionTree)
  - Hyperparameter tuning dengan GridSearchCV
  - Evaluasi model (accuracy, balanced accuracy, recall, precision, F1, AUC-ROC)
  - Penyimpanan model terbaik

Cara pakai:
    from src.ml_core import preprocess_split, train_baseline, tune_knn, evaluate
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, Tuple, Any

from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── Konstanta ─────────────────────────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(_BASE_DIR, "models")
REPORTS_DIR = os.path.join(_BASE_DIR, "reports")

NUMERIC_COLS     = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
CATEGORICAL_COLS = [
    "sex", "chest pain type", "fasting blood sugar",
    "resting ecg", "exercise angina", "ST slope",
]
ZERO_IMPUTE_COLS = ["cholesterol", "resting bp s"]
SCALE_COLS       = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# ── 1. Preprocessing ──────────────────────────────────────────────────────────

def preprocess_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
    """
    Split data lalu preprocessing dengan anti data leakage.

    Urutan langkah:
      1. Train-test split (stratified)
      2. Imputasi nilai 0 tidak wajar → NaN → median (dari X_train saja)
      3. Capping outlier IQR (batas dari X_train saja)
      4. One-Hot Encoding + alignment kolom
      5. StandardScaler (fit hanya pada X_train)

    Parameters
    ----------
    df           : pd.DataFrame bersih (output dari data_generator.clean)
    test_size    : float proporsi test set
    random_state : int seed reproduksi

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    scaler       : StandardScaler yang sudah di-fit pada X_train
    iqr_bounds   : dict {col: (lo, hi)} batas capping per kolom
    train_medians: pd.Series median per kolom numerik dari X_train
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    # 1) Split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    X_train = X_train_raw.copy()
    X_test  = X_test_raw.copy()

    # 2) Imputasi 0 → NaN → median (dari X_train)
    for col in ZERO_IMPUTE_COLS:
        X_train[col] = X_train[col].replace(0, np.nan)
        X_test[col]  = X_test[col].replace(0, np.nan)

    train_medians = X_train[NUMERIC_COLS].median()
    X_train[NUMERIC_COLS] = X_train[NUMERIC_COLS].fillna(train_medians)
    X_test[NUMERIC_COLS]  = X_test[NUMERIC_COLS].fillna(train_medians)

    # 3) Capping IQR (batas dari X_train)
    iqr_bounds: Dict[str, Tuple[float, float]] = {}
    for col in NUMERIC_COLS:
        q1  = X_train[col].quantile(0.25)
        q3  = X_train[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        iqr_bounds[col] = (lo, hi)
        X_train[col] = X_train[col].clip(lower=lo, upper=hi)
        X_test[col]  = X_test[col].clip(lower=lo, upper=hi)

    # 4) One-Hot Encoding + alignment
    X_train_enc = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    X_test_enc  = pd.get_dummies(X_test,  columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    X_train_enc, X_test_enc = X_train_enc.align(
        X_test_enc, join="outer", axis=1, fill_value=0,
    )
    X_train = X_train_enc.astype(float)
    X_test  = X_test_enc.astype(float)

    # 5) StandardScaler (fit pada X_train saja)
    scaler = StandardScaler()
    scaler.fit(X_train[SCALE_COLS])
    X_train[SCALE_COLS] = scaler.transform(X_train[SCALE_COLS])
    X_test[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

    print(f"  ✅ Preprocessing selesai | Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler, iqr_bounds, train_medians


# ── 2. Evaluasi model ─────────────────────────────────────────────────────────

def evaluate(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    verbose: bool = True,
) -> dict:
    """
    Latih model dan evaluasi pada test set.

    Parameters
    ----------
    name    : str — nama model (untuk laporan)
    model   : estimator scikit-learn
    verbose : bool — jika True, cetak metrik ke stdout

    Returns
    -------
    dict metrik lengkap termasuk model_obj, y_pred, y_prob
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    y_prob_test  = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba") else None
    )

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test  = accuracy_score(y_test,  y_pred_test)
    bal_acc   = balanced_accuracy_score(y_test, y_pred_test)
    gap       = acc_train - acc_test
    status    = "OVERFIT" if gap > 0.05 else "OK"

    prec   = precision_score(y_test, y_pred_test, zero_division=0)
    rec    = recall_score(y_test,    y_pred_test, zero_division=0)
    f1     = f1_score(y_test,        y_pred_test, zero_division=0)
    f1_mac = f1_score(y_test,        y_pred_test, average="macro",    zero_division=0)
    auc    = roc_auc_score(y_test, y_prob_test) if y_prob_test is not None else 0.0

    cm           = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MODEL : {name}")
        print(f"{'='*60}")
        print(f"  Acc Train        : {acc_train:.4f}")
        print(f"  Acc Test         : {acc_test:.4f}   [{status}] (gap={gap:.4f})")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  Recall           : {rec:.4f}   ← prioritas medis")
        print(f"  Precision        : {prec:.4f}")
        print(f"  F1-Score         : {f1:.4f}  (macro: {f1_mac:.4f})")
        print(f"  AUC-ROC          : {auc:.4f}")
        print(f"  CM  → TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"  ⚠️  FN={fn}: pasien sakit diprediksi sehat")
        print()
        print(classification_report(y_test, y_pred_test,
                                    target_names=["Sehat (0)", "Sakit (1)"]))

    return {
        "Nama"        : name,
        "Acc Train"   : round(acc_train, 4),
        "Acc Test"    : round(acc_test,  4),
        "Bal Acc"     : round(bal_acc,   4),
        "Gap"         : round(gap,       4),
        "Recall"      : round(rec,       4),
        "Precision"   : round(prec,      4),
        "F1-Score"    : round(f1,        4),
        "F1 Macro"    : round(f1_mac,    4),
        "AUC-ROC"     : round(auc,       4),
        "Overfitting" : "Ya" if gap > 0.05 else "Tidak",
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "model_obj"   : model,
        "y_pred"      : y_pred_test,
        "y_prob"      : y_prob_test,
    }


# ── 3. Baseline models ────────────────────────────────────────────────────────

def train_baseline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Latih semua model baseline dengan parameter default.

    Models: Dummy, KNN, GaussianNB, SVM, DecisionTree

    Returns
    -------
    dict {nama_model: dict_metrik}
    """
    models = {
        "Baseline"    : DummyClassifier(strategy="most_frequent", random_state=42),
        "KNN"         : KNeighborsClassifier(),
        "GaussianNB"  : GaussianNB(),
        "SVM"         : SVC(probability=True, random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    hasil = {}
    print("\n[ Melatih model baseline ]")
    for name, model in models.items():
        try:
            hasil[name] = evaluate(name, model, X_train, X_test, y_train, y_test)
        except Exception as exc:
            print(f"  ⚠️  Error saat melatih {name}: {exc}")

    return hasil


# ── 4. Hyperparameter tuning ──────────────────────────────────────────────────

def tune_knn(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Any, dict]:
    """
    GridSearchCV untuk KNN. Scoring: recall.

    Parameters
    ----------
    X_fit, y_fit : data untuk CV (bisa X_train atau resampled SMOTE)

    Returns
    -------
    (best_estimator, dict_metrik)
    """
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "weights"    : ["uniform", "distance"],
        "metric"     : ["euclidean", "manhattan"],
    }
    gs = GridSearchCV(
        KNeighborsClassifier(), param_grid,
        cv=CV, scoring="recall", n_jobs=-1, verbose=0,
    )
    gs.fit(X_fit, y_fit)
    print(f"  KNN Tuned — best params : {gs.best_params_}")
    print(f"  KNN Tuned — recall CV   : {gs.best_score_:.4f}")
    result = evaluate(
        f"KNN Tuned {gs.best_params_}", gs.best_estimator_,
        X_train, X_test, y_train, y_test,
    )
    return gs.best_estimator_, result


def tune_nb(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Any, dict]:
    """GridSearchCV untuk GaussianNB. Scoring: recall."""
    param_grid = {"var_smoothing": np.logspace(-12, 0, 25)}
    gs = GridSearchCV(
        GaussianNB(), param_grid,
        cv=CV, scoring="recall", n_jobs=-1, verbose=0,
    )
    gs.fit(X_fit, y_fit)
    print(f"  NB Tuned — best params : {gs.best_params_}")
    print(f"  NB Tuned — recall CV   : {gs.best_score_:.4f}")
    result = evaluate(
        f"NB Tuned {gs.best_params_}", gs.best_estimator_,
        X_train, X_test, y_train, y_test,
    )
    return gs.best_estimator_, result


def tune_svm(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Any, dict]:
    """GridSearchCV untuk SVM. Scoring: recall."""
    param_grid = {
        "C"     : [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma" : ["scale", "auto"],
    }
    gs = GridSearchCV(
        SVC(probability=True, random_state=42), param_grid,
        cv=CV, scoring="recall", n_jobs=-1, verbose=0,
    )
    gs.fit(X_fit, y_fit)
    print(f"  SVM Tuned — best params : {gs.best_params_}")
    print(f"  SVM Tuned — recall CV   : {gs.best_score_:.4f}")
    result = evaluate(
        f"SVM Tuned {gs.best_params_}", gs.best_estimator_,
        X_train, X_test, y_train, y_test,
    )
    return gs.best_estimator_, result


def tune_dt(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Any, dict]:
    """GridSearchCV untuk DecisionTree. Scoring: recall."""
    param_grid = {
        "criterion"        : ["gini", "entropy"],
        "max_depth"        : [3, 4, 5, 6, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf" : [1, 2, 4, 8],
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=42), param_grid,
        cv=CV, scoring="recall", n_jobs=-1, verbose=0,
    )
    gs.fit(X_fit, y_fit)
    print(f"  DT Tuned — best params : {gs.best_params_}")
    print(f"  DT Tuned — recall CV   : {gs.best_score_:.4f}")
    result = evaluate(
        f"DT Tuned {gs.best_params_}", gs.best_estimator_,
        X_train, X_test, y_train, y_test,
    )
    return gs.best_estimator_, result


# ── 5. Pilih & simpan model terbaik ──────────────────────────────────────────

def select_best(hasil: dict, metric: str = "Recall") -> Tuple[str, Any, dict]:
    """
    Pilih model terbaik berdasarkan metrik tertentu.

    Parameters
    ----------
    hasil  : dict {nama: dict_metrik} — output dari train_baseline / tuning
    metric : str kolom yang dijadikan kriteria pemilihan

    Returns
    -------
    (nama_terbaik, model_obj_terbaik, dict_metrik_terbaik)
    """
    best_name   = max(hasil, key=lambda k: hasil[k].get(metric, 0))
    best_result = hasil[best_name]
    best_model  = best_result["model_obj"]

    print(f"\n  🏆 Model terbaik berdasarkan {metric}: {best_name}")
    print(f"     {metric}: {best_result[metric]:.4f}")
    return best_name, best_model, best_result


def save_model(
    model: Any,
    filename: str = "best_heart_disease_model.joblib",
    model_dir: str = MODEL_DIR,
) -> str:
    """Simpan model ke file joblib."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)
    joblib.dump(model, path)
    print(f"  ✅ Model disimpan ke: {path}")
    return path


def load_model(
    filename: str = "best_heart_disease_model.joblib",
    model_dir: str = MODEL_DIR,
) -> Any:
    """Muat model dari file joblib."""
    path = os.path.join(model_dir, filename)
    return joblib.load(path)


# ── 6. Ringkasan hasil ────────────────────────────────────────────────────────

def summary_table(hasil: dict) -> pd.DataFrame:
    """
    Buat tabel ringkasan semua model.

    Returns
    -------
    pd.DataFrame diurutkan berdasarkan Acc Test (descending)
    """
    cols = ["Acc Train", "Acc Test", "Bal Acc", "Recall",
            "Precision", "F1-Score", "AUC-ROC", "Overfitting"]
    rows = {
        k: {c: v[c] for c in cols if c in v}
        for k, v in hasil.items()
        if isinstance(v, dict)
    }
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.sort_values("Acc Test", ascending=False)
