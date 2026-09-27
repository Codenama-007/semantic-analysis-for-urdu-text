# Urdu Text Semantic Analysis

## 1. Project Overview

**Urdu Text Semantic Analysis** is a Natural Language Processing (NLP) and Deep Learning project designed to analyze the semantic meaning of Urdu text.

The project takes an Urdu text input and predicts one or more **semantic categories** that describe the subject or meaning of the text.

The project uses Quranic Urdu verses as the primary dataset. Each verse is associated with one or more semantic categories such as:

* Faith (Aqeedah)
* Eschatology (Akhirah)
* Ethics & Morality (Akhlaq)
* History & Stories (Qasas al-Anbiya)
* Divine Attributes & Signs (Asma wa Sifat)
* Law (Ahkam)
* Worship (‘Ibadah)
* Social Relations (Mu‘amalat)
* Supplication & Spirituality (Dua, Dhikr, Tazkiyah)
* Revelation
* Spiritual reminder
* Moral teaching context

The task is formulated as a **multi-label text classification problem**, because a single Urdu verse can belong to multiple semantic categories simultaneously.

---

# 2. Problem Statement

Urdu is a relatively under-resourced language in Natural Language Processing compared with languages such as English.

Understanding the semantic meaning of Urdu text requires more than simply identifying individual words. A system must learn patterns in the language that indicate the underlying topic or semantic category of a sentence.

This project aims to develop a machine-learning system that can:

1. Accept Urdu text as input.
2. Preprocess the Urdu text.
3. Convert the text into a numerical representation.
4. Analyze the semantic patterns in the text.
5. Predict one or more relevant semantic categories.
6. Display the predicted categories and their model scores.

---

# 3. Project Objective

The main objective is to build an Urdu semantic classification system using:

* Urdu NLP preprocessing
* Multi-label classification
* TensorFlow
* Bidirectional LSTM (BiLSTM)
* Weighted binary cross-entropy
* Multi-label stratified dataset splitting

The final system provides a simple interface where a user can enter Urdu text and receive its predicted semantic categories.

---

# 4. Dataset

## Dataset Used

The project uses:

```text
Complete_Quran_data.csv
```

The dataset contains Quranic verses in Urdu along with semantic information.

Important fields include:

* `Surah`
* `Ayah`
* `Urdu`
* `Category`
* `Emotion`
* `Context`

The `Urdu` column contains the actual Urdu text used for semantic classification.

The `Category` column contains the semantic labels.

---

# 5. Dataset Characteristics

The original dataset contains approximately **6,236 records**.

After removing records with missing required information, **6,233 usable records** remain.

The project contains **12 semantic categories**.

The categories are multi-label, meaning that one verse can have more than one active category.

For example:

```text
Faith (Aqeedah), Eschatology (Akhirah)
```

is treated as two labels for the same verse.

---

# 6. Dataset Processing Pipeline

The dataset is processed through the following stages:

```text
Complete_Quran_data.csv
        ↓
dataset_processing.py
        ↓
processed_quran.csv
        ↓
encode_cateogories.py
        ↓
final_dataset.csv
        ↓
prepare_training_data.py
        ↓
train.csv
validation.csv
test.csv
```

---

# 7. Urdu Text Preprocessing

The project uses **LughaatNLP** for Urdu text preprocessing.

The preprocessing pipeline consists of:

```text
Original Urdu Text
        ↓
Normalization
        ↓
Tokenization
        ↓
Stopword Removal
        ↓
Stemming
        ↓
Processed Urdu Text
```

## 7.1 Normalization

Normalization standardizes the Urdu text and reduces unnecessary variations.

The project uses:

```python
urdu_processor.normalize(text)
```

---

## 7.2 Tokenization

The normalized Urdu sentence is divided into individual tokens using LughaatNLP.

Example:

```text
میں پاکستان سے ہوں
```

becomes individual tokens representing the words in the sentence.

---

## 7.3 Stopword Removal

Common Urdu stopwords are removed so that the model can focus more on meaningful semantic information.

---

## 7.4 Stemming

Urdu words are stemmed to reduce variations of related words.

The project uses:

```python
urdu_processor.urdu_stemmer(text)
```

---

## 7.5 Final Preprocessed Text

The final processed text is stored in:

```text
Processed_Urdu
```

This processed representation is used as the input to the machine-learning model.

---

# 8. Semantic Category Encoding

The original `Category` column contains comma-separated labels.

Because some category names themselves contain commas inside parentheses, the project does not simply split the string at every comma.

The category encoding process:

1. Separates individual semantic categories.
2. Handles commas inside parentheses correctly.
3. Normalizes spelling and punctuation variations.
4. Converts the categories into binary labels.

The project uses:

```python
MultiLabelBinarizer
```

Each semantic category becomes a separate binary column:

```text
0 = category is not assigned
1 = category is assigned
```

Therefore, each verse can have multiple `1` values.

---

# 9. Multi-Label Classification

This project is **not a single-label classification problem**.

It is a **multi-label classification problem**.

For example, a verse could produce:

```text
Faith = 1
Eschatology = 1
Ethics = 0
Law = 0
Worship = 0
...
```

This allows the system to associate one Urdu text with multiple semantic meanings.

---

# 10. Dataset Splitting

The dataset is divided into:

```text
Training   : 80%
Validation : 10%
Testing    : 10%
```

The project uses:

```python
MultilabelStratifiedShuffleSplit
```

instead of a normal single-label stratified split.

This is important because the dataset contains multiple labels per sample.

The split produces:

```text
Training   : 4,987 samples
Validation : 624 samples
Testing    : 622 samples
```

A fixed random seed is used so that the split can be reproduced.

---

# 11. Text Vectorization

Machine-learning models cannot directly process raw Urdu strings.

The processed Urdu text is therefore converted into integer sequences using TensorFlow's:

```python
TextVectorization
```

Configuration:

```text
Maximum vocabulary size : 20,000
Sequence length         : 100
Output mode             : integer
```

Only the training text is used to adapt the vectorizer.

This helps prevent information from the validation and test sets from influencing the vocabulary.

---

# 12. Model Architecture

The primary model is a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network.

Architecture:

```text
Input Urdu Text
        ↓
TextVectorization
        ↓
Embedding Layer
        ↓
Bidirectional LSTM
        ↓
Dropout
        ↓
Dense Layer
        ↓
12 Semantic Category Scores
```

## Model Configuration

### Embedding

```text
Embedding dimension: 128
```

The embedding layer converts integer tokens into dense vector representations.

### BiLSTM

```text
LSTM units: 64
Bidirectional: Yes
```

The bidirectional LSTM processes the sequence in both directions, allowing the model to capture contextual information from both preceding and following words.

### Dropout

```text
Dropout: 0.3
```

Dropout is used to reduce overfitting.

### Output Layer

```text
Units: 12
Activation: sigmoid
```

Sigmoid activation is used because this is a multi-label classification problem.

Each output corresponds to one semantic category.

---

# 13. Loss Function

The baseline model uses:

```text
Binary Cross-Entropy
```

because each semantic category represents an independent binary classification decision.

For the weighted model, a **weighted binary cross-entropy** approach is used.

The purpose of weighting is to give greater importance to underrepresented semantic categories.

---

# 14. Class Imbalance

The dataset contains significant class imbalance.

Some semantic categories have many examples, while others have very few.

Examples of highly represented categories include:

* Eschatology (Akhirah)
* History & Stories (Qasas al-Anbiya)
* Faith (Aqeedah)
* Divine Attributes & Signs (Asma wa Sifat)

Some categories have only a very small number of examples.

Three categories contain only **2 examples each**:

* Moral teaching context
* Revelation
* Spiritual reminder

Because these categories have extremely limited data, meaningful validation and test evaluation is not possible for them.

This is an important limitation of the dataset.

---

# 15. Baseline Model

A baseline BiLSTM model was first trained using normal binary cross-entropy.

The baseline test results were:

```text
Micro Precision : 0.7381
Micro Recall    : 0.5387
Micro F1        : 0.6228
Macro F1        : 0.2351
Weighted F1      : 0.5779
```

The baseline model performed better on the majority categories but struggled with several minority categories.

---

# 16. Threshold Tuning

The baseline model was also tested with category-specific classification thresholds.

Instead of using:

```text
0.50
```

for every category, thresholds were selected using the validation dataset.

This improved the macro F1 score:

```text
Baseline Macro F1 : 0.2351
Tuned Macro F1    : 0.3178
```

However, micro F1 decreased:

```text
Baseline Micro F1 : 0.6228
Tuned Micro F1    : 0.5751
```

This demonstrates the trade-off between overall performance and performance on minority categories.

---

# 17. Weighted Model

A second model was trained using weighted binary cross-entropy.

The purpose was to reduce the effect of class imbalance and improve recognition of less represented categories.

The weighted model produced:

```text
Micro Precision : 0.5017
Micro Recall    : 0.7141
Micro F1        : 0.5893
Macro F1        : 0.3298
Weighted F1     : 0.6046
```

Compared with the baseline model, the weighted model produced higher recall for several minority categories.

The weighted model was selected for the final prediction system because the project aims to recognize multiple semantic categories, including less frequent categories.

---

# 18. Final Prediction System

The final prediction system is implemented in:

```text
predict.py
```

The prediction pipeline is:

```text
User Urdu Text
        ↓
LughaatNLP Preprocessing
        ↓
Text Vectorization
        ↓
Weighted BiLSTM Model
        ↓
12 Category Scores
        ↓
Threshold Filtering
        ↓
Top 3 Categories
```

The system uses a minimum score threshold of:

```text
0.50
```

and displays a maximum of:

```text
Top 3 categories
```

This keeps the output focused instead of displaying every category whose model score crosses the threshold.

---

# 19. Example Prediction

Example input:

```text
اللہ بہت مہربان اور رحم کرنے والا ہے
```

After preprocessing:

```text
اللہ مہربن رحم
```

The model produces its highest-scoring semantic categories, such as:

```text
Divine Attributes & Signs (Asma wa Sifat)
Spiritual reminder
Supplication & Spirituality (Dua, Dhikr, Tazkiyah)
```

The displayed scores are model output scores and should not be interpreted as perfectly calibrated probabilities.

---

# 20. Streamlit Application

The project provides a web interface using:

```text
Streamlit
```

The application is started with:

```bash
streamlit run app.py
```

The application allows the user to:

1. Enter Urdu text.
2. Submit the text for analysis.
3. View the preprocessed text.
4. View the predicted semantic categories.
5. View the corresponding model scores.

---

# 21. Project File Responsibilities

## Dataset Files

### `Complete_Quran_data.csv`

Original dataset containing Quranic Urdu text and semantic information.

### `processed_quran.csv`

Dataset after Urdu preprocessing.

### `final_dataset.csv`

Dataset containing the encoded semantic category labels.

### `category_names.csv`

Contains the semantic category names.

### `train.csv`

Training dataset.

### `validation.csv`

Validation dataset.

### `test.csv`

Testing dataset.

---

# 22. Python Files

### `preprocessing.py`

Contains the Urdu preprocessing functions:

* Normalization
* Tokenization
* Stopword removal
* Stemming
* Lemmatization

### `dataset_processing.py`

Loads the original dataset, applies Urdu preprocessing and generates:

```text
processed_quran.csv
```

This is the **first dataset-generation script that must be executed**.

### `encode_cateogories.py`

Converts the original semantic category field into multi-label binary columns.

Generates:

```text
final_dataset.csv
category_names.csv
```

### `prepare_training_data.py`

Splits the encoded dataset into:

```text
train.csv
validation.csv
test.csv
```

using multi-label stratification.

### `vectorization.py`

Creates and tests the TensorFlow text vectorization pipeline.

### `model.py`

Defines the BiLSTM model architecture.

### `train.py`

Trains the baseline BiLSTM model.

### `evaluate.py`

Evaluates the baseline model using precision, recall and F1-score.

### `tune_thresholds.py`

Tests different category-specific thresholds for the baseline model.

### `train_weighted.py`

Trains the weighted BiLSTM model to address class imbalance.

### `evaluate_weighted.py`

Evaluates the weighted model.

### `tune_weighted_thresholds.py`

Tests category-specific thresholds for the weighted model.

### `predict.py`

Loads the trained model and performs semantic category prediction on new Urdu text.

### `app.py`

Provides the Streamlit web interface.

---

# 23. Technologies Used

## Programming Language

```text
Python
```

## NLP

```text
LughaatNLP
```

## Data Processing

```text
Pandas
NumPy
Scikit-learn
```

## Deep Learning

```text
TensorFlow / Keras
```

## Model

```text
Bidirectional LSTM
```

## Web Application

```text
Streamlit
```

## Dataset Splitting

```text
iterative-stratification
MultilabelStratifiedShuffleSplit
```

---

# 24. Complete System Architecture

```text
                    ┌─────────────────────────┐
                    │ Complete_Quran_data.csv │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ dataset_processing.py   │
                    │                         │
                    │ Urdu preprocessing      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ processed_quran.csv     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ encode_cateogories.py   │
                    │                         │
                    │ Multi-label encoding    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ final_dataset.csv       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ prepare_training_data   │
                    │                         │
                    │ Multi-label stratified  │
                    │ train/validation/test   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
                 Train        Validation      Test
                    │
                    ▼
          ┌──────────────────────┐
          │ TextVectorization    │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Embedding            │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Bidirectional LSTM   │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Dropout              │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Dense + Sigmoid      │
          │ 12 Categories        │
          └──────────┬───────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │ Semantic Predictions │
          └──────────┬───────────┘
                     │
                     ▼
              Streamlit App
```

---

# 25. Main Contributions

The project demonstrates:

1. Urdu-specific NLP preprocessing.
2. Multi-label semantic classification of Urdu text.
3. Handling of category strings containing nested commas.
4. Multi-label stratified dataset splitting.
5. Neural text representation using embeddings.
6. Contextual sequence modeling using BiLSTM.
7. Handling class imbalance using weighted loss.
8. Evaluation using precision, recall and F1-score.
9. Threshold analysis for multi-label prediction.
10. Deployment of the model through a Streamlit interface.

---

# 26. Limitations

The project has several important limitations.

### Dataset Size

The dataset contains approximately 6,233 usable samples, which is relatively small for training a deep learning NLP model.

### Class Imbalance

The semantic categories are highly imbalanced.

Some categories have hundreds or thousands of examples while others have only a few.

### Extremely Rare Categories

Three categories have only two examples:

```text
Moral teaching context
Revelation
Spiritual reminder
```

Therefore, the model cannot be reliably evaluated on these categories using the current validation and test split.

### Model Architecture

The system uses a relatively small BiLSTM architecture and does not use a large pretrained Urdu language model.

### Score Calibration

The model's sigmoid outputs are treated as model scores for thresholding. They are not guaranteed to represent calibrated probabilities.

---

# 27. Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the Urdu dataset.
* Collecting more examples for minority categories.
* Using pretrained Urdu or multilingual transformer models.
* Experimenting with Urdu-specific embeddings.
* Improving category annotation quality.
* Performing more extensive hyperparameter tuning.
* Calibrating model output scores.
* Saving the trained vectorizer together with the model.
* Adding more evaluation metrics.
* Testing the system on Urdu text outside the Quranic domain.
* Deploying the application as a production web service.

---

# 28. One-Line Project Description

**Urdu Text Semantic Analysis is a multi-label NLP system that uses Urdu preprocessing and a weighted BiLSTM neural network to identify the semantic categories of Urdu text.**

---

# 29. Short Report Abstract

This project presents an Urdu Text Semantic Analysis system based on Natural Language Processing and Deep Learning. The system processes Urdu text using normalization, tokenization, stopword removal and stemming through LughaatNLP. Quranic Urdu verses are used as the primary dataset and are classified into multiple semantic categories using a multi-label classification approach. A Bidirectional LSTM model with an embedding layer, dropout and sigmoid output layer is trained using TensorFlow. Due to significant class imbalance in the dataset, a weighted binary cross-entropy approach is also investigated to improve recognition of minority categories. The final system accepts Urdu text as input and predicts its relevant semantic categories. A Streamlit interface is provided for interactive demonstration.

---

# 30. Keywords

```text
Urdu NLP
Natural Language Processing
Semantic Analysis
Urdu Text Classification
Multi-Label Classification
Deep Learning
BiLSTM
TensorFlow
LughaatNLP
Text Classification
Semantic Categories
Urdu Language Processing
Class Imbalance
Streamlit
```
