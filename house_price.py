"""Project 1: House Price Predictor (Regression)

Trains Linear Regression, Ridge and Random Forest on the California Housing
dataset and compares them with MAE, RMSE and R^2.

Run:  python house_price.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data():
    try:
        data = fetch_california_housing(as_frame=True)
        return data.frame  # target column: MedHouseVal (in $100,000s)
    except Exception:
        # Offline fallback so the script still runs (synthetic data, same columns)
        print("[!] Could not download dataset - using synthetic demo data.")
        rng = np.random.default_rng(42)
        n = 2000
        df = pd.DataFrame({
            "MedInc": rng.gamma(4, 1, n),
            "HouseAge": rng.integers(1, 52, n),
            "AveRooms": rng.normal(5.5, 1.2, n),
            "AveBedrms": rng.normal(1.1, 0.2, n),
            "Population": rng.integers(100, 5000, n),
            "AveOccup": rng.normal(3, 0.6, n),
            "Latitude": rng.uniform(32.5, 42, n),
            "Longitude": rng.uniform(-124, -114, n),
        })
        df["MedHouseVal"] = (0.45 * df.MedInc + 0.01 * df.HouseAge
                             + 0.1 * df.AveRooms + rng.normal(0, 0.4, n))
        return df


df = load_data()
print(df.head(), "\n")
print(df.describe().T[["mean", "min", "max"]], "\n")

X, y = df.drop(columns="MedHouseVal"), df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

results, preds = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, p),
        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
        "R2": r2_score(y_test, p),
    })

print(pd.DataFrame(results).round(3).to_string(index=False))

# Plot: actual vs predicted for the best model (by R^2)
best = max(results, key=lambda r: r["R2"])["Model"]
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds[best], alpha=0.3, s=10)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual price ($100k)")
plt.ylabel("Predicted price ($100k)")
plt.title(f"Actual vs Predicted - {best}")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)
print(f"\nBest model: {best}. Plot saved to actual_vs_predicted.png")
