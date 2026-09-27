import pandas as pd
import os

from preprocessing import preprocess_text


# FILE PATHS

DATASET_PATH = "Complete_Quran_data.csv"

# MAIN

def main():
    print("URDU DATASET PREPROCESSING")

    # 1. Load dataset

    print("\nLoading dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")

    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    # 2. Check required columns
    # required for Urdu Text verse analyzer
    required_columns = [
        "Urdu",
        "Category",
        "Emotion",
        "Context"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column '{column}' not found."
            )
            
    # 3. Keep relevant columns

    df = df[
        [
            "Surah",
            "Ayah",
            "Urdu",
            "Category",
            "Emotion",
            "Context"
        ]
    ].copy()

    # 4. Remove missing values

    df = df.dropna(
        subset=[
            "Urdu",
            "Category"
        ]
    )

    # 5. Preprocess Urdu text

    print("\nPreprocessing Urdu text...")
    
    df["Processed_Urdu"] = df["Urdu"].apply(preprocess_text)

    # 6. Remove empty text
    df = df[df["Processed_Urdu"].str.strip() != ""]


    # 7. Save processed dataset

    df.to_csv(
        "processed_quran.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 8. Display results

    print("PREPROCESSING COMPLETE")

    print("\nFinal dataset size:",df.shape)

    print("\nExample records:")

    print(
        df[
            [
                "Urdu",
                "Processed_Urdu",
                "Category"
            ]
        ].head(5).to_string()
    )

    print("\nSaved to: \n processed_quran.csv")

# RUN

if __name__ == "__main__":
    main()