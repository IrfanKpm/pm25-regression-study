import requests
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
api_key = os.getenv("OpenAQ_API_KEY")

model = joblib.load(BASE_DIR / "artifacts/model.pkl")

BASE_URL = "https://api.openaq.org/v3"
LOCATION_ID = 6358
LIMIT = 10
HEADERS = {"X-API-Key": api_key}

FEATURES = ["co", "no2", "nox", "pm10", "relativehumidity", "temperature"]
TARGET = "pm25"


print("Getting sensors...")
sensors_url = f"{BASE_URL}/locations/{LOCATION_ID}/sensors"
response = requests.get(sensors_url, headers=HEADERS)
response.raise_for_status()
sensors = response.json()["results"]
print(f"Found {len(sensors)} sensors")

all_rows = []
for sensor in sensors:
    sensor_id = sensor["id"]
    parameter_name = sensor["parameter"]["name"]

    url = (
        f"{BASE_URL}/sensors/{sensor_id}/measurements"
        f"?limit={LIMIT}&page=1"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            print(f"Skipping sensor {sensor_id} (HTTP {resp.status_code})")
            continue

        results = resp.json().get("results", [])
        for row in results:
            utc_time = None
            if "datetime" in row:
                utc_time = row["datetime"].get("utc")
            elif "period" in row:
                utc_time = row["period"].get("datetimeFrom", {}).get("utc")

            all_rows.append({
                "datetime_utc": utc_time,
                "parameter": parameter_name,
                "value": row.get("value")
            })
    except Exception as e:
        print(f"Error sensor {sensor_id}: {e}")


df = pd.DataFrame(all_rows)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
df = df.dropna(subset=["datetime_utc"])

ml_df = df.pivot_table(
    index="datetime_utc",
    columns="parameter",
    values="value",
    aggfunc="mean"
)
ml_df = ml_df.sort_index()
ml_df = ml_df.interpolate(method="time").ffill()
ml_df.columns = [c.lower() for c in ml_df.columns] 

print("\nAvailable columns:", ml_df.columns.tolist())
missing = [f for f in FEATURES + [TARGET] if f not in ml_df.columns]
if missing:
    print(f"\nMissing columns in live data: {missing}")
    print("Cannot proceed. Check if sensors are reporting these parameters.")
    exit(1)

df_final = ml_df[FEATURES + [TARGET]].dropna().tail(10)

if df_final.empty:
    print("No complete rows after dropping NaNs.")
    exit(1)

X_live = df_final[FEATURES]
y_true = df_final[TARGET]


y_pred = model.predict(X_live)
print("\n===== PM2.5 ACTUAL vs PREDICTED =====\n")
for i in range(len(X_live)):
    print(f"Sample {i+1}")
    print(f"  Actual PM2.5    : {y_true.iloc[i]:.2f}")
    print(f"  Predicted PM2.5 : {y_pred[i]:.2f}")
    print(f"  Error           : {abs(y_true.iloc[i] - y_pred[i]):.2f}")
    print("-" * 45)

print("\nDone.")