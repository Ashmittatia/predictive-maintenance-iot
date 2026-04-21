"""
models/fault_classifier.py
Random Forest multi-class fault type classifier.
Input:  engineered feature vector (tabular, per timestep)
Output: fault class label (0=normal, 1-5=fault types)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path

MODEL_PATH      = Path(__file__).parent / "fault_classifier.pkl"
LABEL_MAP_PATH  = Path(__file__).parent / "fault_label_map.pkl"

INDUSTRIAL_LABELS = {
    0: "normal",
    1: "bearing_wear",
    2: "imbalance",
    3: "overheating",
    4: "cavitation",
    5: "valve_leak",
}

SMARTHOME_LABELS = {
    0: "normal",
    1: "refrigerant_leak",
    2: "compressor_fail",
    3: "drum_imbalance",
    4: "pump_fail",
    5: "heating_element_fail",
}

COMBINED_LABELS = {
    0:  "normal",
    1:  "bearing_wear",
    2:  "imbalance",
    3:  "overheating",
    4:  "cavitation",
    5:  "valve_leak",
    6:  "refrigerant_leak",
    7:  "compressor_fail",
    8:  "drum_imbalance",
    9:  "pump_fail",
    10: "heating_element_fail",
}


def prepare_classifier_data(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    For combined dataset: re-maps smarthome fault labels to unique IDs
    so industrial and smarthome faults don't collide.
    Returns (X, y, label_map).
    """
    df = df.copy()
    df["clf_label"] = df["fault_label"]
    # Shift smarthome non-normal labels up by 5
    mask_sh = (df["domain"] == "smarthome") & (df["fault_label"] > 0)
    df.loc[mask_sh, "clf_label"] = df.loc[mask_sh, "fault_label"] + 5

    label_map = COMBINED_LABELS
    X = df[feature_cols].fillna(0).values
    y = df["clf_label"].values
    return X, y, label_map


def train_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    n_estimators: int = 200,
    test_size: float = 0.2
) -> RandomForestClassifier:
    X, y, label_map = prepare_classifier_data(df, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    label_names = [label_map.get(k, str(k)) for k in sorted(set(y))]
    print("\n[Fault Classifier] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf,       MODEL_PATH)
    joblib.dump(label_map, LABEL_MAP_PATH)
    print(f"[✓] Fault classifier saved → {MODEL_PATH}")
    return clf


def load_classifier() -> tuple:
    """Returns (classifier, label_map)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Classifier not found at {MODEL_PATH}. Run train_all.py first.")
    clf       = joblib.load(MODEL_PATH)
    label_map = joblib.load(LABEL_MAP_PATH) if LABEL_MAP_PATH.exists() else COMBINED_LABELS
    return clf, label_map


def predict_fault(clf, label_map: dict, features: np.ndarray) -> dict:
    """
    features: shape (1, n_features) — a single feature vector
    Returns fault label, name, and class probabilities.
    """
    pred_label = int(clf.predict(features)[0])
    proba      = clf.predict_proba(features)[0]
    class_ids  = clf.classes_

    return {
        "fault_label":    pred_label,
        "fault_name":     label_map.get(pred_label, "unknown"),
        "confidence":     round(float(np.max(proba)), 4),
        "probabilities":  {
            label_map.get(int(c), str(c)): round(float(p), 4)
            for c, p in zip(class_ids, proba)
        },
    }


def get_feature_importance(clf: RandomForestClassifier, feature_cols: list, top_n: int = 15) -> pd.DataFrame:
    importances = clf.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    return df_imp.sort_values("importance", ascending=False).head(top_n)


if __name__ == "__main__":
    print("Fault classifier module loaded.")
