# Walmart Sales Forecasting System

**Dataset:** [Kaggle — yasserh/walmart-dataset](https://www.kaggle.com/datasets/yasserh/walmart-dataset)  
**Model:** Random Forest (R² = 98.16%, MAPE = 4.81%, RMSE = $78,604)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model  (run once — saves models/rf_model.pkl)
python train_model.py

# 3. Start the dashboard
python app.py

# 4. Open browser
#    http://127.0.0.1:5000
```

## Project Structure

```
walmart_project/
├── app.py               ← Flask web application
├── data/
│   ├── rWalmart.csv          ← Kaggle dataset (6,435 rows × 8 cols)     
│   └── train_model.py       ← Feature engineering + RF training 
├── templates/
│   └── index.html       ← Full dashboard UI
├── models/
│   ├── rf_model.pkl     ← Trained Random Forest
│   └── metadata.json    ← Metrics + feature importance + store stats
├── requirements.txt
└── README.md
```

## Dataset Columns

| Column | Type | Description |
|---|---|---|
| Store | int | Store number (1–45) |
| Date | str | Week of sales (dd-mm-yyyy) |
| Weekly_Sales | float | Target — weekly revenue |
| Holiday_Flag | int | 1 = holiday week |
| Temperature | float | Regional temperature (°F) |
| Fuel_Price | float | Fuel price per gallon |
| CPI | float | Consumer Price Index |
| Unemployment | float | Unemployment rate (%) |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Dashboard |
| POST | `/predict` | Predict sales for store + date |
| GET | `/api/trend?store=<id\|all>` | Weekly sales time series |
| GET | `/api/monthly?store=<id\|all>` | Monthly seasonality data |
| GET | `/api/stores` | All store performance stats |
| GET | `/api/kpis` | Model performance metrics |

## Model Details

- **Algorithm:** Random Forest Regressor
- **Estimators:** 200 trees, max_depth=20
- **Features:** 17 (temporal + economic + lag features)
- **Split:** 80% train / 20% test (random_state=42)
- **R²:** 98.16% | **MAPE:** 4.81% | **RMSE:** $78,604
