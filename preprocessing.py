import pandas as pd
import re
from LughaatNLP import LughaatNLP

# LUGHAAT NLP INITIALIZATION

urdu_processor = LughaatNLP()

# 1. NORMALIZATION

def normalize_text(text):
    """
    Normalize Urdu text using LughaatNLP.

    LughaatNLP's normalize() performs:
    - Character normalization
    - Diacritic removal
    - Punctuation handling
    - Digit conversion
    - Special-character handling
    """

    if pd.isna(text):
        return ""

    text = str(text).strip()

    # LughaatNLP normalization
    text = urdu_processor.normalize(text)

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# 2. TOKENIZATION

def tokenize_text(text):
    """
    Tokenize Urdu text using LughaatNLP.

    Example:
        میں پاکستان سے ہوں۔
        
    becomes:
        ['میں', 'پاکستان', 'سے', 'ہوں', '۔']
    """

    if not text:
        return []

    tokens = urdu_processor.urdu_tokenize(text)

    return tokens

# 3. STOPWORD REMOVAL

def remove_stopwords(text):
    """
    Remove Urdu stopwords using LughaatNLP.
    """

    if not text:
        return ""

    return urdu_processor.remove_stopwords(text)

# 4. STEMMING

def stem_text(text):
    """
    Perform Urdu stemming using LughaatNLP.
    """

    if not text:
        return ""

    return urdu_processor.urdu_stemmer(text)

# 5. LEMMATIZATION
def lemmatize_text(text):
    """
    Perform Urdu lemmatization using LughaatNLP.
    """

    if not text:
        return ""

    return urdu_processor.lemmatize_sentence(text)


# 6. COMPLETE PREPROCESSING PIPELINE

def preprocess_text(text):
    """
    Complete Urdu NLP preprocessing pipeline.

    Steps:

    1. Normalization
    2. Tokenization
    3. Stopword removal
    4. Stemming

    Returns:
        Processed Urdu text
    """
    # Step 1: Normalize

    normalized = normalize_text(text)

    # Step 2: Tokenization

    tokens = tokenize_text(normalized)

    # Convert tokens back into a sentence
    tokenized_text = " ".join(tokens)

    # Step 3: Stopword removal

    without_stopwords = remove_stopwords(
        tokenized_text
    )

    # Step 4: Stemming

    stemmed = stem_text(
        without_stopwords
    )

    return stemmed

# 7. COMPLETE NLP DEMONSTRATION

def analyze_text(text):
    """
    Returns every preprocessing stage.

    This function is useful for:
    - Streamlit
    - Project demonstration
    - Report
    - Viva
    """

    # Normalization
    normalized = normalize_text(text)

    # Tokenization
    tokens = tokenize_text(normalized)

    # Reconstruct text
    tokenized_text = " ".join(tokens)

    # Stopword removal
    without_stopwords = remove_stopwords(
        tokenized_text
    )

    # Stemming
    stemmed = stem_text(
        without_stopwords
    )

    # Lemmatization
    lemmatized = lemmatize_text(
        without_stopwords
    )

    return {
        "original": text,
        "normalized": normalized,
        "tokens": tokens,
        "stopword_removed": without_stopwords,
        "stemmed": stemmed,
        "lemmatized": lemmatized,
    }
    
# 8. TEST THE PIPELINE

if __name__ == "__main__":

    sample_text = (
        "میں پاکستان سے ہوں اور مجھے اردو زبان بہت پسند ہے۔"
    )

    result = analyze_text(sample_text)

    print("\n")
    print("=" * 60)
    print("URDU NLP PREPROCESSING USING LUGHAATNLP")
    print("=" * 60)

    print("\n1. ORIGINAL TEXT")
    print("-" * 60)
    print(result["original"])

    print("\n2. NORMALIZED TEXT")
    print("-" * 60)
    print(result["normalized"])

    print("\n3. TOKENIZATION")
    print("-" * 60)
    print(result["tokens"])

    print("\n4. STOPWORD REMOVAL")
    print("-" * 60)
    print(result["stopword_removed"])

    print("\n5. STEMMING")
    print("-" * 60)
    print(result["stemmed"])

    print("\n6. LEMMATIZATION")
    print("-" * 60)
    print(result["lemmatized"])

    print("\n")
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)