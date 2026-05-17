import joblib
import pandas as pd

# Load model
model = joblib.load("../artifacts/model.pkl")

# Feature order (must match training)
FEATURES = [
    "co",
    "no2",
    "nox",
    "pm10",
    "relativehumidity",
    "temperature"
]


def predict(sample_dict):
    df = pd.DataFrame([sample_dict], columns=FEATURES)
    return model.predict(df)[0]


if __name__ == "__main__":

    sample = {
        "co": 2.14,
        "no2": 203.5,
        "nox": 0.1449,
        "pm10": 266.0,
        "relativehumidity": 86.2,
        "temperature": 9.2
    }

    result = predict(sample)

    print("Predicted PM2.5:", result)