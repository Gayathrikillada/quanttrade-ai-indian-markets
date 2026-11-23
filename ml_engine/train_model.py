import joblib
import shap
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from ml_engine.features import fetch_and_engineer_features, FEATURE_COLS

def train_and_save_model(save_path="ml_engine/model_bundle.pkl"):
    print("Fetching 6 years of multi-asset market data (2020–2026)...")
    df = fetch_and_engineer_features(is_live=False)
    df = df.dropna(subset=['Target'])

    X = df[FEATURE_COLS]
    y = df['Target'].astype(int)

    # Strict chronological split (No random shuffling!)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    base_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )
    base_xgb.fit(X_train, y_train)

    # Probability Calibration so confidence % matches true empirical accuracy
    calibrated_model = CalibratedClassifierCV(base_xgb, method="sigmoid", cv=3)
    calibrated_model.fit(X_train, y_train)

    preds = calibrated_model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    print("\n--- Out-of-Sample Test Performance (2025–2026) ---")
    print(classification_report(y_test, preds, zero_division=0))

    # SHAP Explainer for transparent feature attribution
    explainer = shap.TreeExplainer(base_xgb)

    bundle = {
        "model": calibrated_model,
        "base_xgb": base_xgb,
        "explainer": explainer,
        "feature_cols": FEATURE_COLS,
        "test_accuracy": round(report["accuracy"] * 100, 2),
        "trained_rows": len(df)
    }
    joblib.dump(bundle, save_path)
    print(f"Model saved successfully to {save_path}!")
    return bundle

if __name__ == "__main__":
    train_and_save_model()