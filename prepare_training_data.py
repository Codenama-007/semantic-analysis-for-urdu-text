import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


INPUT_FILE = "final_dataset.csv"

def main():
    print("=" * 70)
    print("PREPARING DATA FOR TENSORFLOW")
    print("=" * 70)

    # 1. Load final dataset

    df = pd.read_csv(INPUT_FILE)

    print(f"\nDataset shape: {df.shape}")

    # 2. Define the 12 semantic categories

    categories = [
        "Divine Attributes & Signs (Asma wa Sifat)",
        "Eschatology (Akhirah)",
        "Ethics & Morality (Akhlaq)",
        "Faith (Aqeedah)",
        "History & Stories (Qasas al-Anbiya)",
        "Law (Ahkam)",
        "Moral teaching context",
        "Revelation",
        "Social Relations (Mu‘amalat)",
        "Spiritual reminder",
        "Supplication & Spirituality (Dua, Dhikr, Tazkiyah)",
        "Worship (‘Ibadah)"
    ]

    # 3. Convert text to NumPy array

    X = df["Processed_Urdu"].fillna("").astype(str).to_numpy()

    # 4. Create multi-label target matrix

    y = df[categories].to_numpy(dtype=np.float32)

    print("\nInput shape:")
    print(X.shape)

    print("\nTarget shape:")
    print(y.shape)

    # 5. First split
    # 80% Training
    # 20% Temporary

    splitter_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_indices, temp_indices = next(
        splitter_1.split(X, y)
    )

    X_train = X[train_indices]
    y_train = y[train_indices]

    X_temp = X[temp_indices]
    y_temp = y[temp_indices]

    # 6. Second split
    # Temporary → 50% Validation + 50% Test

    splitter_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )

    val_indices, test_indices = next(
        splitter_2.split(X_temp, y_temp)
    )

    X_val = X_temp[val_indices]
    y_val = y_temp[val_indices]

    X_test = X_temp[test_indices]
    y_test = y_temp[test_indices]
    
    # 7. Save train / validation / test datasets

    train_df = pd.DataFrame(
        X_train,
        columns=["Processed_Urdu"]
    )

    val_df = pd.DataFrame(
        X_val,
        columns=["Processed_Urdu"]
    )

    test_df = pd.DataFrame(
        X_test,
        columns=["Processed_Urdu"]
    )

    # Add category labels
    for i, category in enumerate(categories):

        train_df[category] = y_train[:, i]
        val_df[category] = y_val[:, i]
        test_df[category] = y_test[:, i]

    train_df.to_csv(
        "train.csv",
        index=False,
        encoding="utf-8-sig"
    )

    val_df.to_csv(
        "validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        "test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSaved datasets:")
    print("--------------------------------")
    print("train.csv")
    print("validation.csv")
    print("test.csv")

    # 8. Display dataset sizes

    print("\nDataset split:")
    print("--------------------------------")
    print("Training   :", len(X_train))
    print("Validation :", len(X_val))
    print("Testing    :", len(X_test))

    # 9. Display label distribution

    print("\nLabel distribution:")
    print("--------------------------------")

    for i, category in enumerate(categories):

        print(
            f"{category}: "
            f"Train={int(y_train[:, i].sum())}, "
            f"Val={int(y_val[:, i].sum())}, "
            f"Test={int(y_test[:, i].sum())}"
        )

    # 10. Display example

    print("\nExample training sample:")
    print("--------------------------------")

    print("Processed Urdu:")
    print(X_train[0])

    print("\nEncoded labels:")
    print(y_train[0])

    print("\nActive categories:")

    for i, category in enumerate(categories):

        if y_train[0][i] == 1:
            print("✓", category)

    print("\n" + "=" * 70)
    print("TENSORFLOW DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()