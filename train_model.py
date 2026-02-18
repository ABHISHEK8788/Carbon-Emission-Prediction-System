"""
Machine Learning pipeline for Carbon Footprint prediction.
Trains a model to predict total_daily_co2_kg from role, program, transport, distance, etc.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_PATH = Path(__file__).parent / "eco_optimizer_individual_daily_carbon1.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_and_preprocess():
    """Load CSV and prepare features/target."""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "total_daily_co2_kg"])

    # Extract date features (optional, helps if there's seasonality)
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    categorical = ["role", "program", "transport_mode"]
    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Keep these features; in the UI they will be used mainly for role == Trainer.
    feature_cols = [
        "role_encoded", "program_encoded", "transport_mode_encoded",
        "distance_km", "travel_energy_kwh", "travel_co2_kg",
        "personal_computer_kwh", "personal_computer_co2_kg",
        "students_trained", "lab_energy_kwh",
        "month", "day_of_week"
    ]
    X = df[feature_cols]
    y = df["total_daily_co2_kg"]

    return X, y, encoders, feature_cols, df, categorical


def train_and_save():
    """Train Random Forest regressor and save model + encoders."""
    X, y, encoders, feature_cols, df, categorical = load_and_preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model metrics (test set):")
    print(f"  MAE:  {mae:.4f} kg CO2")
    print(f"  RMSE: {rmse:.4f} kg CO2")
    print(f"  R²:   {r2:.4f}")

    joblib.dump(model, MODEL_DIR / "carbon_model.joblib")
    joblib.dump(encoders, MODEL_DIR / "encoders.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.joblib")
    joblib.dump({"role": list(encoders["role"].classes_),
                 "program": list(encoders["program"].classes_),
                 "transport_mode": list(encoders["transport_mode"].classes_)}, MODEL_DIR / "categories.joblib")

    # Persist metrics so the Streamlit app can display the model details properly
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    joblib.dump(metrics, MODEL_DIR / "metrics.joblib")

    return model, encoders, feature_cols, metrics


if __name__ == "__main__":
    train_and_save()
    print("Model and encoders saved to ./models/")
