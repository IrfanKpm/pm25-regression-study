import requests
import pandas as pd
import time


from dotenv import load_dotenv
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
DATA_DIR =  BASE_DIR / "data/raw"


load_dotenv(ENV_PATH )
api_key = os.getenv("OpenAQ_API_KEY")

LOCATION_ID = 6358 # Mandir Marg, New Delhi - DPCC
BASE_URL = "https://api.openaq.org/v3"

# REQUIRED PERIOD
START_DATE = "2026-01-01T00:00:00Z"
END_DATE   = "2026-05-01T23:59:59Z"
LIMIT = 1000

print("Getting sensors...")
sensors_url = f"{BASE_URL}/locations/{LOCATION_ID}/sensors"

HEADERS = {"X-API-Key": api_key }
response = requests.get(sensors_url,headers=HEADERS)
response.raise_for_status()
sensors = response.json()["results"]
print(f"Found {len(sensors)} sensors")

all_rows = []
for sensor in sensors:
    sensor_id = sensor["id"]
    parameter_name = sensor["parameter"]["name"]
    parameter_unit = sensor["parameter"]["units"]
    print(f"\nDownloading: {parameter_name} | Sensor ID: {sensor_id}")
    page = 1
    while True:
        # use datetime_from / datetime_to
        url = (
            f"{BASE_URL}/sensors/{sensor_id}/measurements"
            f"?datetime_from={START_DATE}"
            f"&datetime_to={END_DATE}"
            f"&limit={LIMIT}"
            f"&page={page}"
        )
        try:
            response = requests.get(url,headers=HEADERS,timeout=30)
            if response.status_code != 200:
                print(
                    f"Stopped sensor {sensor_id} "
                    f"(HTTP {response.status_code})"
                )
                break
            data = response.json()
            results = data.get("results", [])
            if len(results) == 0:
                print("No more rows")
                break
            print(f"Page {page}: {len(results)} rows")

            for row in results:
                utc_time = None
                local_time = None
                if "datetime" in row:
                    utc_time = row["datetime"].get("utc")
                    local_time = row["datetime"].get("local")
                elif "period" in row:
                    dt_from = row["period"].get("datetimeFrom",{})
                    utc_time = dt_from.get("utc")
                    local_time = dt_from.get("local")
                all_rows.append({
                    "datetime_utc": utc_time,
                    "datetime_local": local_time,
                    "sensor_id": sensor_id,
                    "parameter": parameter_name,
                    "unit": parameter_unit,
                    "value": row.get("value")
                })
            if len(results) < LIMIT:
                break
            page += 1
            time.sleep(0.2) # avoid API rate limits
        except Exception as e:
            print(f"Error sensor {sensor_id}: {e}")
            break


print("\nCreating DataFrame...")
df = pd.DataFrame(all_rows)

print(f"Total rows: {len(df)}")

df["datetime_utc"] = pd.to_datetime(df["datetime_utc"],errors="coerce")
df["datetime_local"] = pd.to_datetime(df["datetime_local"],errors="coerce")
df = df.dropna(subset=["datetime_utc"])
df = df.sort_values("datetime_utc")

print("\nLong format preview:")
print(df.head())
print("\nLong format shape:")
print(df.shape)

print("\nCreating ML dataset...")
ml_df = df.pivot_table(index="datetime_utc",columns="parameter",values="value",aggfunc="mean")

ml_df = ml_df.sort_index()
ml_df = ml_df.interpolate(method="time")
ml_df = ml_df.ffill()
ml_df = ml_df.reset_index()

print("\nML DataFrame preview:")
print(ml_df.head())

print("\nML DataFrame shape:")
print(ml_df.shape)

print("\nColumns:")
print(ml_df.columns.tolist())

df.to_csv(DATA_DIR / "openaq_long_format.csv", index=False)
ml_df.to_csv(DATA_DIR / "openaq_ml_format.csv", index=False)

print("\nSaved files:")
print("1. openaq_long_format.csv")
print("2. openaq_ml_format.csv")