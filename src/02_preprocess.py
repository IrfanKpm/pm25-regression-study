import os
import pandas as pd


def preprocess_data(input_path,output_dir="../data/processed"):
    """
    Load, clean, engineer features,
    shuffle, split, and save datasets.
    """
    # Load dataset
    df = pd.read_csv(input_path)
    # Drop unnecessary column
    if "datetime_utc" in df.columns:
        df = df.drop(columns=["datetime_utc"])

    # Remove weak / unwanted features
    drop_cols = [
        "no",
        "o3",
        "so2",
        "wind_direction",
        "wind_speed"
    ]

    existing_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=existing_cols)
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)

    # Train / Validation / Test split
    train_size = int(0.75 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save processed datasets
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print("Datasets saved successfully.")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")



if __name__ == "__main__":
    INPUT_PATH = "../data/raw/openaq_ml_format.csv"
    preprocess_data(INPUT_PATH)