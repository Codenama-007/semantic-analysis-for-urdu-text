import pandas as pd
import tensorflow as tf

# 1. Load training data

train_df = pd.read_csv("train.csv" , encoding="utf-8-sig")

# We only need the processed Urdu text here.
train_text = train_df["Processed_Urdu"].fillna("").astype(str).to_numpy()


# 2. Create TextVectorization layer

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None,
    split="whitespace"
)

# 3. Learn vocabulary from training text

vectorizer.adapt(train_text)

# 4. Test the vectorizer

sample_text = train_text[0]

vectorized_text = vectorizer(
    tf.constant([sample_text])
)

print("\nOriginal text:")
print(sample_text)

print("\nVectorized text:")
print(vectorized_text.numpy()[0])

print("\nVectorized shape:")
print(vectorized_text.shape)

print("\nVocabulary size:")
print(len(vectorizer.get_vocabulary()))

print("\nFirst 20 vocabulary tokens:")
print(vectorizer.get_vocabulary()[:20])