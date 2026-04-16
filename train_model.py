"""
train_model.py  —  Walmart Sales Forecast
==========================================
Dataset : Walmart.csv  (Kaggle: yasserh/walmart-dataset)
          6435 rows × 8 columns | 45 stores | Feb 2010 – Oct 2012
Model   : Random Forest Regressor (single best model)

Run:
    python train_model.py
"""

import os, json, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

warnings.filterwarnings("ignore")

BASE   = os.path.dirname(os.path.abspath(__file__))
CSV    = os.path.join(BASE, "Walmart.csv")
OUTDIR = os.path.join(BASE, "models")
os.makedirs(OUTDIR, exist_ok=True)

print("\n" + "="*56)
print("  WALMART SALES FORECAST  —  TRAINING PIPELINE")
print("="*56)

# ── 1. LOAD ────────────────────────────────────────────────────
print("\n[1/5]  Loading Walmart.csv ...")
df = pd.read_csv(CSV)
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df.sort_values(["Store", "Date"], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"       Rows : {len(df):,}   Columns : {list(df.columns)}")
print(f"       Stores : {df['Store'].nunique()}  |  "
      f"Date range : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"       Missing values : {df.isnull().sum().sum()}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────
print("\n[2/5]  Feature Engineering ...")

df["Year"]        = df["Date"].dt.year
df["Month"]       = df["Date"].dt.month
df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
df["Quarter"]     = df["Date"].dt.quarter
df["DayOfYear"]   = df["Date"].dt.dayofyear
df["Is_Q4"]       = (df["Quarter"] == 4).astype(int)
df["Is_December"] = (df["Month"]   == 12).astype(int)
df["Is_November"] = (df["Month"]   == 11).astype(int)

# Lag features per store (time-series momentum)
df["Lag_1"]    = df.groupby("Store")["Weekly_Sales"].shift(1)
df["Lag_4"]    = df.groupby("Store")["Weekly_Sales"].shift(4)
df["Rolling_4"]= df.groupby("Store")["Weekly_Sales"]\
                   .transform(lambda x: x.shift(1).rolling(4).mean())

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

FEATURES = [
    "Store", "Year", "Month", "Week", "Quarter", "DayOfYear",
    "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "Is_Q4", "Is_December", "Is_November",
    "Lag_1", "Lag_4", "Rolling_4",
]

X = df[FEATURES]
y = df["Weekly_Sales"]
print(f"       Features : {len(FEATURES)}  |  Training samples after lag drop : {len(X):,}")

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────
print("\n[3/5]  Train / Test Split (80/20) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True)
print(f"       Train : {len(X_train):,}   Test : {len(X_test):,}")

# ── 4. TRAIN RANDOM FOREST ────────────────────────────────────
print("\n[4/5]  Training Random Forest Regressor ...")
model = RandomForestRegressor(
    n_estimators = 200,
    max_depth    = 20,
    min_samples_leaf = 2,
    max_features = "sqrt",
    n_jobs       = -1,
    random_state = 42,
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
mae  = float(mean_absolute_error(y_test, preds))
r2   = float(r2_score(y_test, preds))
mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)

print(f"\n       ✓  R²   = {r2:.4f}  ({r2*100:.2f}%)")
print(f"       ✓  RMSE = ${rmse:,.0f}")
print(f"       ✓  MAE  = ${mae:,.0f}")
print(f"       ✓  MAPE = {mape:.2f}%")

# ── 5. SAVE ARTIFACTS ─────────────────────────────────────────
print("\n[5/5]  Saving model artifacts ...")
joblib.dump(model, os.path.join(OUTDIR, "rf_model.pkl"))

# Feature importance for the API / UI
importance = dict(zip(FEATURES,
    [round(float(v), 5) for v in model.feature_importances_]))
importance_sorted = dict(sorted(importance.items(),
    key=lambda x: x[1], reverse=True))

# Store-level stats for dashboard
store_stats = df.groupby("Store").agg(
    avg_sales    = ("Weekly_Sales", "mean"),
    total_sales  = ("Weekly_Sales", "sum"),
    max_sales    = ("Weekly_Sales", "max"),
    min_sales    = ("Weekly_Sales", "min"),
).round(2).to_dict("index")

# Monthly averages for chart
monthly_avg = df.groupby("Month")["Weekly_Sales"].mean().round(2).to_dict()

metadata = {
    "model"         : "RandomForestRegressor",
    "n_estimators"  : 200,
    "max_depth"     : 20,
    "features"      : FEATURES,
    "metrics"       : {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape},
    "train_size"    : len(X_train),
    "test_size"     : len(X_test),
    "total_rows"    : len(df),
    "stores"        : df["Store"].nunique(),
    "date_min"      : str(df["Date"].min().date()),
    "date_max"      : str(df["Date"].max().date()),
    "feature_importance": importance_sorted,
    "store_stats"   : {str(k): v for k, v in store_stats.items()},
    "monthly_avg"   : {str(k): v for k, v in monthly_avg.items()},
    "sales_min"     : float(df["Weekly_Sales"].min()),
    "sales_max"     : float(df["Weekly_Sales"].max()),
    "sales_mean"    : float(df["Weekly_Sales"].mean()),
    "holiday_avg"   : float(df[df["Holiday_Flag"]==1]["Weekly_Sales"].mean()),
    "regular_avg"   : float(df[df["Holiday_Flag"]==0]["Weekly_Sales"].mean()),
}

with open(os.path.join(OUTDIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"       ✓  rf_model.pkl   saved to models/")
print(f"       ✓  metadata.json  saved to models/")

print("\n" + "="*56)
print(f"  DONE — Random Forest  R²={r2*100:.2f}%  MAPE={mape:.2f}%")
print("="*56 + "\n")
