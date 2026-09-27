import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

# 1. Load test data

test_df = pd.read_csv(
    "test.csv",
    encoding="utf-8-sig"
)

TEXT_COLUMN = "Processed_Urdu"

category_columns = [
    column
    for column in test_df.columns
    if column != TEXT_COLUMN
]

X_test = (
    test_df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .to_numpy()
)

y_test = test_df[category_columns].to_numpy(dtype="float32")


print("Test samples:", len(X_test))
print("Number of categories:", len(category_columns))

# 2. Load trained model

model = tf.keras.models.load_model(
    "urdu_semantic_bilstm.keras"
)

# 3. Create the same vectorizer

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100

# IMPORTANT:
# The model was trained using a vocabulary learned
# from the training set.
#
# Therefore we must rebuild the same vocabulary
# using train.csv, NOT test.csv.

train_df = pd.read_csv(
    "train.csv",
    encoding="utf-8-sig"
)

X_train = (
    train_df[TEXT_COLUMN]
    .fillna("")
    .astype(str)
    .to_numpy()
)

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None,
    split="whitespace"
)

vectorizer.adapt(X_train)

# 4. Generate predictions

print("\nGenerating predictions...")

probabilities = model.predict(
    vectorizer(X_test),
    batch_size=32
)

print("Prediction shape:", probabilities.shape)


# 5. Convert probabilities to labels

THRESHOLD = 0.5

y_pred = (
    probabilities >= THRESHOLD
).astype(int)

# 6. Overall metrics

print("\n" + "=" * 60)
print("OVERALL METRICS")
print("=" * 60)

micro_precision = precision_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    y_test,
    y_pred,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print(f"Micro Precision : {micro_precision:.4f}")
print(f"Micro Recall    : {micro_recall:.4f}")
print(f"Micro F1        : {micro_f1:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")


# 7. Per-category metrics

print("\n" + "=" * 60)
print("PER-CATEGORY PERFORMANCE")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=category_columns,
    zero_division=0
)

print(report)