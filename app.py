import os
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "demand_forecast_rf_model.pkl"
DATA_PATH = BASE_DIR / "Historical Product Demand.csv"

app = Flask(__name__)


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
WAREHOUSE_OPTIONS = list(WAREHOUSE_MAP.keys())
CATEGORY_OPTIONS = list(CATEGORY_MAP.keys())
PRODUCT_OPTIONS = list(PRODUCT_MAP.keys())


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


@app.route("/", methods=["GET", "POST"])
def predict_page():
    prediction = None
    latency_ms = None
    error = None

    selected_warehouse = WAREHOUSE_OPTIONS[0] if WAREHOUSE_OPTIONS else ""
    selected_category = CATEGORY_OPTIONS[0] if CATEGORY_OPTIONS else ""
    selected_product = PRODUCT_OPTIONS[0] if PRODUCT_OPTIONS else ""
    date_str = ""

    if request.method == "POST":
        selected_warehouse = request.form.get("warehouse", "")
        selected_category = request.form.get("category", "")
        selected_product = request.form.get("product", "")
        date_str = request.form.get("date", "")

        try:
            if selected_warehouse not in WAREHOUSE_MAP:
                raise ValueError("Invalid warehouse selected.")
            if selected_category not in CATEGORY_MAP:
                raise ValueError("Invalid category selected.")
            if selected_product not in PRODUCT_MAP:
                raise ValueError("Invalid product selected.")
            if not date_str:
                raise ValueError("Please choose a date.")

            warehouse_encoded = WAREHOUSE_MAP[selected_warehouse]
            category_encoded = CATEGORY_MAP[selected_category]
            product_encoded = PRODUCT_MAP[selected_product]
            prediction, latency_ms = predict_from_encoded(
                warehouse_encoded, category_encoded, product_encoded, date_str
            )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        latency_ms=latency_ms,
        error=error,
        warehouse_options=WAREHOUSE_OPTIONS,
        category_options=CATEGORY_OPTIONS,
        product_options=PRODUCT_OPTIONS,
        selected_warehouse=selected_warehouse,
        selected_category=selected_category,
        selected_product=selected_product,
        date_str=date_str,
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


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    app.run(host=host, port=port, debug=debug_mode)
