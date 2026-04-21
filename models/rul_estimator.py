"""
models/rul_estimator.py
Gradient Boosting regressor for Remaining Useful Life (RUL) prediction.
RUL = estimated timesteps until failure.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "rul_estimator.pkl"


def train_rul_model(
    df: pd.DataFrame,
    feature_cols: list,
    test_size: float = 0.2
) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting regressor to predict RUL.
    Only samples where rul < max_rul (not brand-new) are included to help the model.
    """
    df = df.copy()
    # Cap RUL to avoid extreme outliers driving the loss
    max_rul = df["rul"].quantile(0.95)
    df["rul_capped"] = df["rul"].clip(upper=max_rul)

    X = df[feature_cols].fillna(0).values
    y = df["rul_capped"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred)   # RUL can't be negative

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n[RUL Estimator] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    # Save
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[✓] RUL estimator saved → {MODEL_PATH}")
    return model


def load_rul_model() -> GradientBoostingRegressor:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"RUL model not found at {MODEL_PATH}. Run train_all.py first.")
    return joblib.load(MODEL_PATH)


def predict_rul(model: GradientBoostingRegressor, features: np.ndarray) -> dict:
    """
    features: shape (1, n_features)
    Returns estimated RUL with health percentage.
    """
    raw_rul = float(model.predict(features)[0])
    rul     = max(0.0, raw_rul)

    # Health score: scale to 0-100 assuming max meaningful RUL = 3500 cycles
    max_rul      = 3500.0
    health_score = round(min(100.0, (rul / max_rul) * 100), 1)

    # Urgency tier
    if rul < 100:
        urgency = "critical"
    elif rul < 500:
        urgency = "warning"
    elif rul < 1500:
        urgency = "moderate"
    else:
        urgency = "healthy"

    return {
        "rul_cycles":    round(rul, 1),
        "health_score":  health_score,
        "urgency":       urgency,
    }


def get_rul_feature_importance(model: GradientBoostingRegressor, feature_cols: list, top_n: int = 15) -> pd.DataFrame:
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    return df_imp.sort_values("importance", ascending=False).head(top_n)


if __name__ == "__main__":
    print("RUL estimator module loaded.")
