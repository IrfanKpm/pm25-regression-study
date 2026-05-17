import numpy as np
import pandas as pd
import joblib
import os

from model import build_ridge_model
from sklearn.metrics import mean_squared_error



# Step 1: Load processed data
train_df = pd.read_csv("../data/processed/train.csv")
val_df   = pd.read_csv("../data/processed/val.csv")
test_df  = pd.read_csv("../data/processed/test.csv")

# Step 2: Split features/target
TARGET = "pm25"

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]


# Step 3: Build model
print("\nBuilding Ridge Regression model...")
model = build_ridge_model(degree=5, alpha=0.01)
print("Model created successfully.")
print(f"Configuration -> degree=5, alpha=0.01")


# Step 4: Train model
model.fit(X_train, y_train)

# Step 5: Generate predictions
train_pred = model.predict(X_train)
val_pred   = model.predict(X_val)
test_pred   = model.predict(X_test)


# Step 6: Define evaluation metric
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Step 7: Evaluate performance
train_rmse = rmse(y_train, train_pred)
val_rmse   = rmse(y_val, val_pred)
test_rmse  = rmse(y_test, test_pred)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")


# Step 8: Save model artifact
print("\nSaving trained model...")
os.makedirs("../artifacts", exist_ok=True)
model_path = "../artifacts/model.pkl"
joblib.dump(model, model_path)
print(f"Model successfully saved at: {model_path}")

# -----------------------------
# Final confirmation
# -----------------------------
print("✅ Final Model Pipeline Complete")
print("Model is trained, evaluated, and saved to artifacts.")
