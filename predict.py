import numpy as np
import pandas as pd
import tensorflow as tf
from LughaatNLP import LughaatNLP

# Configuration


# Weighted model
MODEL_PATH = "urdu_semantic_bilstm_weighted.keras"

# If the model is inside models/, use:
# MODEL_PATH = "models/urdu_semantic_bilstm_weighted.keras"

MAX_VOCAB_SIZE = 20000
SEQUENCE_LENGTH = 100

# Maximum number of categories shown to the user
TOP_K = 3

# Minimum probability required to appear
MIN_CONFIDENCE = 0.50

# Initialize Urdu processor

urdu_processor = LughaatNLP()


# Load category names

train_df = pd.read_csv(
    "train.csv"
)

category_columns = [
    column
    for column in train_df.columns
    if column != "Processed_Urdu"
]

# Load training text

train_text = (
    train_df["Processed_Urdu"]
    .fillna("")
    .astype(str)
    .to_numpy()
)

# Rebuild TextVectorization

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize=None,
    split="whitespace"
)

# The vocabulary must come only from training data.
vectorizer.adapt(train_text)

# Load trained model

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# Urdu preprocessing

def preprocess_text(text):
    """
    Apply the same preprocessing pipeline
    used during model training.
    """

    text = str(text).strip()

    # Normalize Urdu characters
    text = urdu_processor.normalize(text)

    # Tokenize Urdu text
    tokens = urdu_processor.urdu_tokenize(text)

    # Convert tokens back into a string
    tokenized_text = " ".join(tokens)

    # Remove stopwords
    without_stopwords = (
        urdu_processor.remove_stopwords(
            tokenized_text
        )
    )

    # Apply Urdu stemming
    stemmed = (
        urdu_processor.urdu_stemmer(
            without_stopwords
        )
    )

    return stemmed

# Prediction function

def predict_categories(text):
    """
    Predict the semantic categories of Urdu text.

    Returns:
        processed_text
        predictions
        probabilities
    """

    # Step 1: Preprocess

    processed_text = preprocess_text(text)


    # Step 2: Vectorize

    vectorized_text = vectorizer(
        [processed_text]
    )

    # Step 3: Predict probabilities

    probabilities = model.predict(
        vectorized_text,
        verbose=0
    )[0]

    # Step 4: Create category predictions

    predictions = []

    for category, probability in zip(
        category_columns,
        probabilities
    ):

        probability = float(probability)

        if probability >= MIN_CONFIDENCE:

            predictions.append({
                "category": category,
                "confidence": probability
            })


    # Step 5: Sort by confidence

    predictions.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    # Step 6: Keep only top K

    predictions = predictions[:TOP_K]


    return (
        processed_text,
        predictions,
        probabilities
    )
    
# Command-line interface

if __name__ == "__main__":

    print("=" * 60)
    print("URDU TEXT SEMANTIC ANALYSIS")
    print("=" * 60)

    print(
        "\nEnter Urdu text."
        "\nType 'exit' to quit.\n"
    )

    while True:

        text = input(
            "Enter Urdu text: "
        ).strip()
        
        # Exit

        if text.lower() == "exit":

            print("\nExiting...")
            break

        # Empty input

        if not text:

            print(
                "Please enter some Urdu text.\n"
            )

            continue

        # Prediction

        (
            processed_text,
            predictions,
            probabilities
        ) = predict_categories(text)

        # Display preprocessing

        print(
            "\n" + "-" * 60
        )

        print(
            "Preprocessed Text:"
        )

        print(
            processed_text
        )

        # Display predictions

        print(
            "\nPredicted Semantic Categories:"
        )


        if predictions:

            for index, prediction in enumerate(
                predictions,
                start=1
            ):

                category = (
                    prediction["category"]
                )

                confidence = (
                    prediction["confidence"]
                    * 100
                )

                print(
                    f"{index}. {category}"
                )

                print(
                    f"   Confidence: "
                    f"{confidence:.2f}%"
                )

        else:

            print(
                "No category reached the "
                f"{MIN_CONFIDENCE * 100:.0f}% "
                "confidence threshold."
            )


        print(
            "-" * 60
        )

        print()
