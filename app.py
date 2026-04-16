"""
app.py  —  Walmart Sales Predictor
Simple prediction-only Flask app.
"""

import os, json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

app  = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE, "models", "rf_model.pkl"))
with open(os.path.join(BASE, "models", "metadata.json")) as f:
    META = json.load(f)

FEATURES = META["features"]

df_raw = pd.read_csv(os.path.join(BASE,'data',"Walmart.csv"))
df_raw["Date"] = pd.to_datetime(df_raw["Date"], dayfirst=True)
df_raw.sort_values(["Store", "Date"], inplace=True)

STORE_HIST = {sid: grp.reset_index(drop=True)
              for sid, grp in df_raw.groupby("Store")}

HOLIDAY_DATES = [
    datetime(2010,2,12), datetime(2011,2,11), datetime(2012,2,10),
    datetime(2010,9,10), datetime(2011,9,9),  datetime(2012,9,7),
    datetime(2010,11,26),datetime(2011,11,25),datetime(2012,11,23),
    datetime(2010,12,31),datetime(2011,12,30),datetime(2012,12,28),
    datetime(2013,2,10), datetime(2013,9,6),
    datetime(2013,11,29),datetime(2013,12,27),
    datetime(2014,2,9),  datetime(2014,9,5),
    datetime(2014,11,28),datetime(2014,12,26),
]

def _is_holiday(dt):
    return int(any(abs((dt - h).days) <= 3 for h in HOLIDAY_DATES))

def _econ(dt):
    ref  = datetime(2010, 2, 5)
    days = max(0, (dt - ref).days)
    temp = round(50 + 30 * np.sin(2 * np.pi * (dt.month - 3) / 12), 2)
    fuel = round(min(4.5, 2.572 + days / 365 * 0.38), 3)
    cpi  = round(211.096 + days / 365 * 3.4, 3)
    unemp= round(max(3.8, 8.106 - days / 365 * 0.46), 3)
    return temp, fuel, cpi, unemp

def _lags(store_id, target_date):
    hist = STORE_HIST.get(store_id, pd.DataFrame())
    if hist.empty:
        m = META["sales_mean"]
        return m, m, m
    past = hist[hist["Date"] < target_date]["Weekly_Sales"].values
    lag1  = float(past[-1])            if len(past) >= 1 else META["sales_mean"]
    lag4  = float(past[-4])            if len(past) >= 4 else lag1
    roll4 = float(np.mean(past[-4:]))  if len(past) >= 4 else lag1
    return lag1, lag4, roll4

def build_row(store_id, dt):
    lag1, lag4, roll4 = _lags(store_id, dt)
    temp, fuel, cpi, unemp = _econ(dt)
    m = dt.month; q = (m - 1) // 3 + 1
    return {
        "Store": store_id, "Year": dt.year, "Month": m,
        "Week": dt.isocalendar()[1], "Quarter": q,
        "DayOfYear": dt.timetuple().tm_yday,
        "Holiday_Flag": _is_holiday(dt),
        "Temperature": temp, "Fuel_Price": fuel,
        "CPI": cpi, "Unemployment": unemp,
        "Is_Q4": int(q == 4), "Is_December": int(m == 12),
        "Is_November": int(m == 11),
        "Lag_1": lag1, "Lag_4": lag4, "Rolling_4": roll4,
    }

@app.route("/")
def index():
    stores = sorted(df_raw["Store"].unique().tolist())
    return render_template("index.html", stores=stores)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body     = request.json
        store_id = int(body["store"])
        dt       = datetime.strptime(body["date"], "%Y-%m-%d")
        row      = build_row(store_id, dt)
        X        = pd.DataFrame([row])[FEATURES]
        pred     = float(model.predict(X)[0])
        return jsonify({
            "ok":         True,
            "prediction": round(pred, 2),
            "is_holiday": bool(row["Holiday_Flag"]),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
