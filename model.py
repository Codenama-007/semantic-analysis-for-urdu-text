import pandas as pd
import tensorflow as tf

# 1. Load training data

train_df = pd.read_csv("train.csv" , encoding="utf-8-sig")

train_text = train_df["Processed_Urdu"].fillna("").astype(str).to_numpy()

# 2. Get the 12 semantic categories

category_columns = [
    column
    for column in train_df.columns
    if column != "Processed_Urdu"
]

NUM_CLASSES = len(category_columns)

print("Number of categories:", NUM_CLASSES)
print("Categories:")

for category in category_columns:
    print("-", category)


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

vectorizer.adapt(train_text)

# 4. Build the BiLSTM model

model = tf.keras.Sequential([
    
    # Convert Urdu words into dense numerical vectors
    tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=128,
        mask_zero=True
    ),

    # Understand the context from both directions
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    ),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.3),

    # One output for each semantic category
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="sigmoid"
    )
])

# 5. Compile the model

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
    ]
)

# Adding an input to the model 
model = tf.keras.Sequential([
    
    # Input is a sequence of 100 token IDs
    tf.keras.Input(shape=(SEQUENCE_LENGTH,)),

    # Convert token IDs into dense vectors
    tf.keras.layers.Embedding(
        input_dim=len(vectorizer.get_vocabulary()),
        output_dim=128,
        mask_zero=True
    ),

    # Understand context in both directions
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    ),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.3),

    # 12 independent semantic-category probabilities
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="sigmoid"
    )
])


# 6. Display model architecture

model.summary()