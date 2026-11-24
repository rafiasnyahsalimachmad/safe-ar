
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold


# ==========================================
# 1. PATH DAN KONFIGURASI
# ==========================================

AI_DIR = Path(__file__).resolve().parent
DATA_PATH = AI_DIR / "dataset" / "processed" / "landmarks_normalized.csv"
SCHEMA_PATH = AI_DIR / "dataset" / "processed" / "feature_schema.json"

MODEL_DIR = AI_DIR / "models"
MODEL_PATH = MODEL_DIR / "gesture_classifier.joblib"
EVALUATION_PATH = MODEL_DIR / "evaluation.json"
CONFUSION_PATH = MODEL_DIR / "confusion_matrix.csv"

RANDOM_STATE = 42

CLASS_NAMES = [
    "DANGER_STOP",
    "NEED_MEDICAL",
    "EVACUATION",
    "SAFE",
    "NEED_LOGISTICS",
]


# ==========================================
# 2. LOAD DATASET DAN FEATURE SCHEMA
# ==========================================

print("Membaca dataset...")

with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
    schema = json.load(file)

FEATURES = schema["feature_order"]

if len(FEATURES) != 63:
    raise ValueError(
        f"Jumlah fitur seharusnya 63, ditemukan {len(FEATURES)}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [
    "safe_ar_label",
    "user_id",
    "status",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Kolom tidak ditemukan: {missing_columns}")

# Gunakan hanya sampel yang berhasil diekstraksi.
df = df[df["status"] == "success"].copy()

# Pastikan fitur berupa angka.
df[FEATURES] = df[FEATURES].apply(
    pd.to_numeric, errors="coerce"
)

# Buang data yang tidak lengkap.
before = len(df)

df = df.dropna(
    subset=FEATURES + ["safe_ar_label", "user_id"]
).copy()

print(f"Data sebelum pembersihan: {before}")
print(f"Data setelah pembersihan: {len(df)}")
print(f"Data yang dibuang: {before - len(df)}")

X = df[FEATURES].astype(np.float32)
y = df["safe_ar_label"].astype(str)
groups = df["user_id"].astype(str)

if set(y.unique()) != set(CLASS_NAMES):
    raise ValueError(
        f"Label tidak sesuai. Ditemukan: {sorted(y.unique())}"
    )


# ==========================================
# 3. SPLIT DATA BERDASARKAN USER_ID
# ==========================================

print("\nMembagi dataset berdasarkan user_id...")

# Split pertama: 80% train+validation, 20% test.
outer_split = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_val_idx, test_idx = next(
    outer_split.split(X, y, groups)
)

# Split kedua: 80% train, 20% validation
# dari bagian train+validation.
X_train_val = X.iloc[train_val_idx]
y_train_val = y.iloc[train_val_idx]
groups_train_val = groups.iloc[train_val_idx]

inner_split = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

train_rel_idx, val_rel_idx = next(
    inner_split.split(
        X_train_val,
        y_train_val,
        groups_train_val,
    )
)

train_idx = train_val_idx[train_rel_idx]
val_idx = train_val_idx[val_rel_idx]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

# Pastikan tidak ada user yang bocor antar kelompok.
train_users = set(groups.iloc[train_idx])
val_users = set(groups.iloc[val_idx])
test_users = set(groups.iloc[test_idx])

assert train_users.isdisjoint(val_users)
assert train_users.isdisjoint(test_users)
assert val_users.isdisjoint(test_users)

print("\nDistribusi dataset:")
for name, indices in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx),
]:
    print(
        f"{name}: {len(indices)} sampel, "
        f"{groups.iloc[indices].nunique()} pengguna"
    )

print("\nDistribusi kelas:")
print("Train:\n", y_train.value_counts())
print("Validation:\n", y_val.value_counts())
print("Test:\n", y_test.value_counts())


# ==========================================
# 4. TRAIN RANDOM FOREST
# ==========================================

print("\nMemulai training Random Forest...")

model = RandomForestClassifier(
    n_estimators=400,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training selesai.")


# ==========================================
# 5. EVALUASI VALIDATION
# ==========================================

val_predictions = model.predict(X_val)

print("\n===== VALIDATION RESULTS =====")
print("Accuracy:", accuracy_score(y_val, val_predictions))
print(
    classification_report(
        y_val,
        val_predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)


# ==========================================
# 6. FINAL TRAINING DAN TEST
# ==========================================

print("\nMelatih ulang model menggunakan train + validation...")

X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

final_model = RandomForestClassifier(
    n_estimators=400,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

final_model.fit(X_train_final, y_train_final)

test_predictions = final_model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_predictions)
test_balanced_accuracy = balanced_accuracy_score(
    y_test, test_predictions
)

print("\n===== FINAL TEST RESULTS =====")
print("Accuracy:", test_accuracy)
print("Balanced Accuracy:", test_balanced_accuracy)

report = classification_report(
    y_test,
    test_predictions,
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

print(
    classification_report(
        y_test,
        test_predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=CLASS_NAMES,
)

print("Confusion Matrix:")
print(cm)


# ==========================================
# 7. SIMPAN MODEL DAN HASIL EVALUASI
# ==========================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_bundle = {
    "model": final_model,
    "feature_order": FEATURES,
    "class_names": CLASS_NAMES,
    "label_column": "safe_ar_label",
    "schema_version": schema["schema_version"],
}

joblib.dump(model_bundle, MODEL_PATH)

pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
).to_csv(CONFUSION_PATH)

evaluation = {
    "model": "RandomForestClassifier",
    "random_state": RANDOM_STATE,
    "n_estimators": 400,
    "feature_count": len(FEATURES),
    "class_names": CLASS_NAMES,
    "split": {
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "train_users": len(train_users),
        "validation_users": len(val_users),
        "test_users": len(test_users),
    },
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
}

with open(EVALUATION_PATH, "w", encoding="utf-8") as file:
    json.dump(evaluation, file, indent=4)

print("\nSemua hasil berhasil disimpan.")
print("Model:", MODEL_PATH)
print("Evaluasi:", EVALUATION_PATH)
print("Confusion matrix:", CONFUSION_PATH)
