import os
import numpy as np
import pandas as pd
import tensorflow as tf

# Configuration



MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
LSTM_UNITS = 64

BATCH_SIZE = 32
EPOCHS = 15

# Prevent extremely rare labels from dominating training
MAX_POS_WEIGHT = 5.0


# Load data

train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("validation.csv")

category_columns = [
    column
    for column in train_df.columns
    if column != "Processed_Urdu"
]

X_train = train_df["Processed_Urdu"].fillna("").astype(str).to_numpy()
y_train = train_df[category_columns].to_numpy(dtype=np.float32)

X_val = val_df["Processed_Urdu"].fillna("").astype(str).to_numpy()
y_val = val_df[category_columns].to_numpy(dtype=np.float32)

NUM_CLASSES = len(category_columns)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Number of categories: {NUM_CLASSES}")

# Calculate positive class weights

positive_counts = y_train.sum(axis=0)
negative_counts = len(y_train) - positive_counts

# More weight for rare positive labels
pos_weights = negative_counts / np.maximum(positive_counts, 1)

# Prevent the extremely rare 2-example labels
# from dominating the training process.
pos_weights = np.minimum(pos_weights, MAX_POS_WEIGHT)

print("\nPositive class weights:")

for category, weight, count in zip(
    category_columns,
    pos_weights,
    positive_counts
):
    print(f"{category}: {weight:.2f} (positive samples: {int(count)})")

# Text vectorization

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None,
    split="whitespace"
)

# IMPORTANT:
# Vocabulary is learned ONLY from training text.
vectorizer.adapt(X_train)

# Weighted Binary Cross-Entropy

pos_weights_tensor = tf.constant(
    pos_weights,
    dtype=tf.float32
)


def weighted_binary_crossentropy(y_true, y_pred):

    # Prevent log(0)
    epsilon = tf.keras.backend.epsilon()

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    # Standard binary cross-entropy components
    positive_loss = -(
        y_true *
        tf.math.log(y_pred)
    )

    negative_loss = -(
        (1.0 - y_true) *
        tf.math.log(1.0 - y_pred)
    )

    # Give rare positive labels more importance
    weighted_loss = (
        pos_weights_tensor * positive_loss
        + negative_loss
    )

    return tf.reduce_mean(weighted_loss)

# Build model

model = tf.keras.Sequential([

    tf.keras.Input(
        shape=(SEQUENCE_LENGTH,)
    ),

    tf.keras.layers.Embedding(
        input_dim=len(
            vectorizer.get_vocabulary()
        ),
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(LSTM_UNITS)
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="sigmoid"
    )
])


# Compile

model.compile(
    optimizer="adam",

    loss=weighted_binary_crossentropy,

    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        )
    ]
)

# Convert text to sequences

X_train_vectorized = vectorizer(
    X_train
)

X_val_vectorized = vectorizer(
    X_val
)

# Early stopping

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# Train

history = model.fit(

    X_train_vectorized,
    y_train,

    validation_data=(
        X_val_vectorized,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ]
)

# Save model

model_path = (
    "urdu_semantic_bilstm_weighted.keras"
)

model.save(model_path)

print("\nWeighted model saved successfully!")
print(f"Path: {model_path}")