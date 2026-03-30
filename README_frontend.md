# Demand Forecast Frontend

## Run locally

1. Open terminal in this folder.
2. Install packages:
   pip install -r requirements.txt
3. Start app:
   python app.py
4. Open browser:
   http://127.0.0.1:5000

## Notes

- Uses `demand_forecast_rf_model.pkl` for predictions.
- Recreates encoding maps from `Historical Product Demand.csv` using the same cleaning logic as notebook.
- Date is converted to Unix timestamp exactly like your notebook.
