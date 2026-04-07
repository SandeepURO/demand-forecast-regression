import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "demand_forecast_rf_model.pkl"
DATA_PATH = BASE_DIR / "Historical Product Demand.csv"
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror notebook cleaning so feature encoding stays consistent."""
    cleaned = df.drop_duplicates().copy()
    cleaned["Order_Demand"] = cleaned["Order_Demand"].astype(str)
    cleaned["Order_Demand"] = cleaned["Order_Demand"].str.replace("(", "", regex=False)
    cleaned["Order_Demand"] = cleaned["Order_Demand"].str.replace(")", "", regex=False)
    cleaned["Order_Demand"] = pd.to_numeric(cleaned["Order_Demand"], errors="coerce")
    cleaned = cleaned[cleaned["Order_Demand"] > 0]
    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def build_label_maps(df: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """LabelEncoder sorts classes; sorted uniques reproduce the same mapping."""
    warehouse_values = sorted(df["Warehouse"].astype(str).unique())
    category_values = sorted(df["Product_Category"].astype(str).unique())
    product_values = sorted(df["Product_Code"].astype(str).unique())

    warehouse_map = {value: idx for idx, value in enumerate(warehouse_values)}
    category_map = {value: idx for idx, value in enumerate(category_values)}
    product_map = {value: idx for idx, value in enumerate(product_values)}

    return warehouse_map, category_map, product_map


def load_assets() -> tuple[object, dict[str, int], dict[str, int], dict[str, int]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    raw_df = pd.read_csv(DATA_PATH)
    cleaned_df = clean_training_data(raw_df)
    warehouse_map, category_map, product_map = build_label_maps(cleaned_df)

    return model, warehouse_map, category_map, product_map


rf_model, WAREHOUSE_MAP, CATEGORY_MAP, PRODUCT_MAP = load_assets()


def build_options(mapping: dict[str, int]) -> list[dict[str, object]]:
    items = sorted(mapping.items(), key=lambda x: x[1])
    return [{"id": idx, "name": name} for name, idx in items]


def predict_from_encoded(
    warehouse_encoded: int, category_encoded: int, product_encoded: int, date_str: str
) -> tuple[float, float]:
    date_encoded = int(pd.Timestamp(date_str).timestamp())
    new_input = np.array(
        [[warehouse_encoded, category_encoded, product_encoded, date_encoded]]
    )

    start = perf_counter()
    log_prediction = rf_model.predict(new_input)[0]
    latency_ms = (perf_counter() - start) * 1000
    prediction = max(0.0, float(np.expm1(log_prediction)))
    return prediction, latency_ms


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Demand Forecast API is running",
            "endpoints": ["/api/options", "/api/predict"],
        }
    )


@app.route("/api/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True) or {}

    try:
        warehouse = int(payload.get("warehouse"))
        category = int(payload.get("category"))
        product = int(payload.get("product"))
        date_str = str(payload.get("date", "")).strip()

        if not date_str:
            raise ValueError("date is required in YYYY-MM-DD format")

        prediction, latency_ms = predict_from_encoded(
            warehouse, category, product, date_str
        )

        return jsonify(
            {
                "predicted_demand": int(round(prediction)),
                "predicted_demand_raw": prediction,
                "latency_ms": round(latency_ms, 2),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/options", methods=["GET"])
def options_api():
    return jsonify(
        {
            "warehouse": build_options(WAREHOUSE_MAP),
            "category": build_options(CATEGORY_MAP),
            "product": build_options(PRODUCT_MAP),
        }
    )


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=debug_mode)
