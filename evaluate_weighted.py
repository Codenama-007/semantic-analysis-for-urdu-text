import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# Configuration

# IMPORTANT:
# Check where your model was actually saved.
# Your previous output showed:
# urdu_semantic_bilstm_weighted.keras
#
# If the file is in the project root, keep this:
MODEL_PATH = "urdu_semantic_bilstm_weighted.keras"

# If it is inside models/, change to:
# MODEL_PATH = "models/urdu_semantic_bilstm_weighted.keras"

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100


# Load data

train_df = pd.read_csv(
    "train.csv"
)

test_df = pd.read_csv(
    "test.csv"
)

category_columns = [
    column
    for column in train_df.columns
    if column != "Processed_Urdu"
]

X_train = (
    train_df["Processed_Urdu"]
    .fillna("")
    .astype(str)
    .to_numpy()
)

X_test = (
    test_df["Processed_Urdu"]
    .fillna("")
    .astype(str)
    .to_numpy()
)

y_test = test_df[
    category_columns
].to_numpy(dtype=np.int32)

# Rebuild vectorizer

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None,
    split="whitespace"
)

# IMPORTANT:
# Vocabulary is learned ONLY from training data.
vectorizer.adapt(X_train)

# Load weighted model

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Vectorize test data

X_test_vectorized = vectorizer(X_test)

# Generate predictions

y_prob = model.predict(
    X_test_vectorized,
    verbose=1
)

# Same threshold used by the baseline evaluation
THRESHOLD = 0.5

y_pred = (
    y_prob >= THRESHOLD
).astype(np.int32)

# Overall metrics

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


print("\n" + "=" * 60)
print("WEIGHTED MODEL — TEST RESULTS")
print("=" * 60)

print(f"Test samples   : {len(y_test)}")
print(f"Number classes : {len(category_columns)}")

print(f"\nMicro Precision : {micro_precision:.4f}")
print(f"Micro Recall    : {micro_recall:.4f}")
print(f"Micro F1        : {micro_f1:.4f}")
print(f"Macro F1        : {macro_f1:.4f}")
print(f"Weighted F1     : {weighted_f1:.4f}")

# Per-category results

print("\n" + "=" * 60)
print("PER-CATEGORY RESULTS")
print("=" * 60)

report = classification_report(
    y_test,
    y_pred,
    target_names=category_columns,
    zero_division=0,
    output_dict=True
)

for category in category_columns:

    metrics = report[category]

    print(
        f"\n{category}"
    )

    print(
        f"  Precision: {metrics['precision']:.3f}"
    )

    print(
        f"  Recall   : {metrics['recall']:.3f}"
    )

    print(
        f"  F1       : {metrics['f1-score']:.3f}"
    )

    print(
        f"  Support  : {int(metrics['support'])}"
    )
