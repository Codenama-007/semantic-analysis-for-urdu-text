import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# 1. Load datasets

train_df = pd.read_csv(
    "train.csv",
    encoding="utf-8-sig"
)

validation_df = pd.read_csv(
    "validation.csv",
    encoding="utf-8-sig"
)

test_df = pd.read_csv(
    "test.csv",
    encoding="utf-8-sig"
)


TEXT_COLUMN = "Processed_Urdu"

category_columns = [
    column
    for column in train_df.columns
    if column != TEXT_COLUMN
]

# 2. Prepare text

X_train = (
    train_df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .to_numpy()
)

X_val = (
    validation_df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .to_numpy()
)

X_test = (
    test_df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .to_numpy()
)

# 3. Prepare labels

y_val = validation_df[category_columns].to_numpy(
    dtype="float32"
)

y_test = test_df[category_columns].to_numpy(
    dtype="float32"
)

# 4. Load trained model

model = tf.keras.models.load_model(
    "urdu_semantic_bilstm.keras"
)

# 5. Rebuild the SAME vectorizer

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None,
    split="whitespace"
)

vectorizer.adapt(X_train)

# 6. Generate validation probabilities

print("Generating validation predictions...")

val_probabilities = model.predict(
    vectorizer(X_val),
    batch_size=32,
    verbose=1
)

# 7. Find best threshold for each category

thresholds = []

print("\n" + "=" * 75)
print("THRESHOLD SEARCH")
print("=" * 75)

for i, category in enumerate(category_columns):

    best_threshold = 0.5
    best_f1 = 0.0

    # Try thresholds from 0.10 to 0.90
    for threshold in np.arange(0.10, 0.91, 0.05):

        predictions = (
            val_probabilities[:, i] >= threshold
        ).astype(int)

        score = f1_score(
            y_val[:, i],
            predictions,
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    thresholds.append(best_threshold)

    print(
        f"{category:<55} "
        f"Threshold={best_threshold:.2f} "
        f"Validation F1={best_f1:.3f}"
    )


thresholds = np.array(thresholds)

# 8. Generate test predictions

print("\nGenerating test predictions...")

test_probabilities = model.predict(
    vectorizer(X_test),
    batch_size=32,
    verbose=1
)

# 9. Apply category-specific thresholds

y_pred_tuned = np.zeros_like(test_probabilities, dtype=int)

for i in range(len(category_columns)):

    y_pred_tuned[:, i] = (
        test_probabilities[:, i] >= thresholds[i]
    ).astype(int)

# 10. Compare overall F1

baseline_predictions = (
    test_probabilities >= 0.5
).astype(int)


baseline_micro_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="micro",
    zero_division=0
)

tuned_micro_f1 = f1_score(
    y_test,
    y_pred_tuned,
    average="micro",
    zero_division=0
)


baseline_macro_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="macro",
    zero_division=0
)

tuned_macro_f1 = f1_score(
    y_test,
    y_pred_tuned,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 75)
print("THRESHOLD COMPARISON")
print("=" * 75)

print(f"Baseline Micro F1 : {baseline_micro_f1:.4f}")
print(f"Tuned Micro F1    : {tuned_micro_f1:.4f}")

print()

print(f"Baseline Macro F1 : {baseline_macro_f1:.4f}")
print(f"Tuned Macro F1    : {tuned_macro_f1:.4f}")

# 11. Per-category comparison

print("\n" + "=" * 75)
print("PER-CATEGORY F1")
print("=" * 75)

for i, category in enumerate(category_columns):

    baseline_f1 = f1_score(
        y_test[:, i],
        baseline_predictions[:, i],
        zero_division=0
    )

    tuned_f1 = f1_score(
        y_test[:, i],
        y_pred_tuned[:, i],
        zero_division=0
    )

    print(
        f"{category:<55} "
        f"0.5={baseline_f1:.3f} "
        f"Tuned={tuned_f1:.3f}"
    )