import pandas as pd
import tensorflow as tf

# 1. Load datasets

train_df = pd.read_csv(
    "train.csv",
    encoding="utf-8-sig"
)

validation_df = pd.read_csv(
    "validation.csv",
    encoding="utf-8-sig"
)

# 2. Separate text and labels

TEXT_COLUMN = "Processed_Urdu"

category_columns = [
    column
    for column in train_df.columns
    if column != TEXT_COLUMN
]

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

y_train = train_df[category_columns].to_numpy(dtype="float32")
y_val = validation_df[category_columns].to_numpy(dtype="float32")


print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Number of categories:", len(category_columns))

# 3. Text Vectorization

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None,
    split="whitespace"
)

# IMPORTANT:
# Learn vocabulary ONLY from training data.
vectorizer.adapt(X_train)


# 4. Build the BiLSTM model

model = tf.keras.Sequential([

    tf.keras.Input(shape=(SEQUENCE_LENGTH,)),

    tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=128,
        mask_zero=True
    ),

    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(category_columns),
        activation="sigmoid"
    )
])

# 5. Compile

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy"
        )
    ]
)

# 6. Callbacks

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 7. Train

history = model.fit(
    vectorizer(X_train),
    y_train,

    validation_data=(
        vectorizer(X_val),
        y_val
    ),

    epochs=15,
    batch_size=32,

    callbacks=[
        early_stopping
    ]
)

# 8. Save trained model

model.save(
    "urdu_semantic_bilstm.keras"
)

print("\nModel saved successfully!")
print("Path: models/urdu_semantic_bilstm.keras")