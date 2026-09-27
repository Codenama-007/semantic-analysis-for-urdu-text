import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

# o

# Your previous training output showed the model
# was saved in the project root.
MODEL_PATH = "urdu_semantic_bilstm_weighted.keras"

# If it is actually inside models/, use:
# MODEL_PATH = "models/urdu_semantic_bilstm_weighted.keras"

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100

# Load datasets

train_df = pd.read_csv(
    "train.csv"
)

val_df = pd.read_csv(
    "validation.csv"
)

test_df = pd.read_csv(
    "test.csv"
)

# Identify category columns

category_columns = [
    column
    for column in train_df.columns
    if column != "Processed_Urdu"
]

# Prepare text

X_train = (
    train_df["Processed_Urdu"]
    .fillna("")
    .astype(str)
    .to_numpy()
)

X_val = (
    val_df["Processed_Urdu"]
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

# Prepare labels

y_val = val_df[
    category_columns
].to_numpy(dtype=np.int32)

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

# Learn vocabulary ONLY from training data.
vectorizer.adapt(X_train)

# Load weighted model

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Vectorize validation and test data

X_val_vectorized = vectorizer(X_val)
X_test_vectorized = vectorizer(X_test)

# Generate probabilities

print("Generating validation predictions...")

val_prob = model.predict(
    X_val_vectorized,
    verbose=1
)

print("\nGenerating test predictions...")

test_prob = model.predict(
    X_test_vectorized,
    verbose=1
)

# Find best threshold for each category

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

best_thresholds = []


print("\n" + "=" * 70)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 70)


for index, category in enumerate(category_columns):

    best_threshold = 0.50
    best_f1 = 0.0

    # If validation has no positive examples,
    # there is nothing meaningful to tune.
    if y_val[:, index].sum() == 0:

        print(
            f"{category}: "
            f"No validation positives → threshold 0.50"
        )

        best_thresholds.append(0.50)
        continue


    for threshold in thresholds:

        predictions = (
            val_prob[:, index] >= threshold
        ).astype(int)

        score = f1_score(
            y_val[:, index],
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = threshold


    best_thresholds.append(
        best_threshold
    )

    print(
        f"{category}: "
        f"threshold={best_threshold:.2f}, "
        f"validation F1={best_f1:.3f}"
    )

# Apply thresholds to test predictions

test_pred_tuned = np.zeros_like(
    test_prob,
    dtype=np.int32
)

for index, threshold in enumerate(
    best_thresholds
):

    test_pred_tuned[:, index] = (
        test_prob[:, index] >= threshold
    ).astype(np.int32)

# Evaluate tuned weighted model

micro_precision = f1_score(
    y_test,
    test_pred_tuned,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    y_test,
    test_pred_tuned,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    test_pred_tuned,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    test_pred_tuned,
    average="weighted",
    zero_division=0
)

micro_recall = (
    np.sum(
        y_test * test_pred_tuned
    )
    /
    np.maximum(
        np.sum(y_test),
        1
    )
)


print("\n" + "=" * 70)
print("WEIGHTED MODEL + TUNED THRESHOLDS")
print("=" * 70)

print(
    f"Micro Precision : "
    f"{f1_score(y_test, test_pred_tuned, average='micro', zero_division=0):.4f}"
)

print(
    f"Micro Recall    : "
    f"{micro_recall:.4f}"
)

print(
    f"Micro F1        : "
    f"{micro_f1:.4f}"
)

print(
    f"Macro F1        : "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1     : "
    f"{weighted_f1:.4f}"
)

# Per-category test results

print("\n" + "=" * 70)
print("PER-CATEGORY TEST RESULTS")
print("=" * 70)


for index, category in enumerate(
    category_columns
):

    score = f1_score(
        y_test[:, index],
        test_pred_tuned[:, index],
        zero_division=0
    )

    print(
        f"{category}: "
        f"threshold={best_thresholds[index]:.2f}, "
        f"test F1={score:.3f}, "
        f"support={y_test[:, index].sum()}"
    )
