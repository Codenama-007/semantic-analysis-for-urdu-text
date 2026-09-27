# Urdu Text Semantic Analysis — Project Context

This file gives a short explanation of every important file in the project.

## `app.py`

Streamlit web application for the final project.

It provides a simple interface where users can enter Urdu text and receive the predicted semantic categories.

---

## `requirements.txt`

Contains the Python packages required to run the project.

---

# `src/`

The `src` directory contains the preprocessing, dataset preparation, model training, evaluation and prediction code.

---

## `src/preprocessing.py`

Handles Urdu text preprocessing using LughaatNLP.

The preprocessing pipeline includes:

* Normalization
* Tokenization
* Stopword removal
* Stemming

It creates the processed Urdu text used by the machine-learning pipeline.

---

## `src/encode_categories.py`

Processes the semantic category column.

The dataset contains multiple semantic categories for some verses, so the categories are converted into a multi-label binary representation.

Each category becomes a separate column containing:

```text
0 = Category not present
1 = Category present
```

---

## `src/prepare_training_data.py`

Splits the encoded dataset into:

* Training data
* Validation data
* Testing data

It uses multi-label stratified splitting to preserve category distributions as much as possible.

---

## `src/vectorization.py`

Creates the TensorFlow `TextVectorization` layer.

It converts processed Urdu text into numerical sequences that can be passed to the neural network.

---

## `src/model.py`

Defines the BiLSTM neural-network architecture.

The model contains:

```text
Embedding
↓
Bidirectional LSTM
↓
Dropout
↓
Dense + Sigmoid
```

The model predicts multiple semantic categories for a single input.

---

## `src/train.py`

Trains the baseline BiLSTM model.

It uses:

* Adam optimizer
* Binary cross-entropy
* Early stopping

The trained baseline model is saved for evaluation.

---

## `src/evaluate.py`

Evaluates the baseline model on the test dataset.

It calculates:

* Precision
* Recall
* F1-score
* Micro F1
* Macro F1
* Weighted F1

---

## `src/tune_thresholds.py`

Experiments with different classification thresholds for each semantic category.

The thresholds are selected using the validation dataset.

This is useful because the dataset contains heavily imbalanced categories.

---

## `src/train_weighted.py`

Trains the weighted version of the BiLSTM model.

Class weighting gives additional importance to categories with fewer training examples.

This helps the model recognize minority semantic categories more effectively.

---

## `src/evaluate_weighted.py`

Evaluates the weighted BiLSTM model on the test dataset.

It uses a default classification threshold of:

```text
0.50
```

---

## `src/tune_weighted_thresholds.py`

Experiments with category-specific thresholds for the weighted model.

Validation data is used to select thresholds, while the test data is used only for final evaluation.

---

## `src/predict.py`

Contains the final prediction system.

It:

1. Accepts Urdu text.
2. Preprocesses the text using LughaatNLP.
3. Vectorizes the text.
4. Loads the trained weighted BiLSTM model.
5. Predicts scores for all semantic categories.
6. Filters predictions using the configured threshold.
7. Displays the top 3 predicted categories.

It can be executed from the terminal using:

```bash
python src/predict.py
```

---

# `data/`

Contains the datasets used by the project.

## `data/processed/`

Contains the processed and prepared datasets.

### `final_dataset.csv`

Final encoded dataset containing the processed Urdu text and semantic category labels.

### `category_names.csv`

Contains the semantic category names used by the model.

### `train.csv`

Training dataset used to train the neural network.

### `validation.csv`

Validation dataset used during model development and threshold selection.

### `test.csv`

Unseen test dataset used for final evaluation.

---

# `models/`

Contains trained machine-learning models.

The baseline model is stored as:

```text
urdu_semantic_bilstm.keras
```

The weighted model is:

```text
urdu_semantic_bilstm_weighted.keras
```

---

# Project Architecture

The overall project can be viewed as:

```text
                    Urdu Dataset
                         │
                         ▼
                  Preprocessing
                         │
                         ▼
                Category Encoding
                         │
                         ▼
              Multi-label Dataset
                         │
                         ▼
              Train / Validation / Test
                         │
                         ▼
                Text Vectorization
                         │
                         ▼
                BiLSTM Neural Network
                         │
                         ▼
             Semantic Classification
                         │
                         ▼
                 Prediction System
                         │
                         ▼
                  Streamlit Web App
```

---

# Problem Type

This project is a:

**Multi-Label Urdu Semantic Text Classification System**

It is not a simple single-label classification problem because one Urdu text can belong to multiple semantic categories simultaneously.

The model therefore uses:

```text
Sigmoid activation
+
Binary Cross-Entropy
```

instead of softmax classification.

---

# Final Model

The final prediction system uses a weighted Bidirectional LSTM model.

The weighted training approach was used to give greater importance to underrepresented semantic categories in the dataset.

The final prediction interface displays the top three categories that cross the configured score threshold.
