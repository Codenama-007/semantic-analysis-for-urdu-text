# Execution Guide

## 1. Create Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

# 3. Generate the Required Dataset

Run these files **in this exact order**.

### Step 1 — Process the Original Dataset

Make sure `Complete_Quran_data.csv` is present in the project root.

Then run:

```bash
python dataset_processing.py
```

This creates:

```text
processed_quran.csv
```

### Step 2 — Encode the Semantic Categories

After `processed_quran.csv` has been created, run:

```bash
python encode_cateogories.py
```

This creates:

```text
final_dataset.csv
category_names.csv
```

### Step 3 — Prepare Training, Validation and Test Data

After `final_dataset.csv` has been created, run:

```bash
python prepare_training_data.py
```

This creates:

```text
train.csv
validation.csv
test.csv
```

At this point, the required datasets for model training are ready.

# 4. Model Training and Evaluation

After generating the datasets, run the following files as required:

### Vectorization

```bash
python vectorization.py
```

### Create Model

```bash
python model.py
```

### Train Baseline Model

```bash
python train.py
```

### Evaluate Baseline Model

```bash
python evaluate.py
```

### Tune Baseline Thresholds

```bash
python tune_thresholds.py
```

### Train Weighted Model

```bash
python train_weighted.py
```

### Evaluate Weighted Model

```bash
python evaluate_weighted.py
```

### Tune Weighted Thresholds

```bash
python tune_weighted_thresholds.py
```

# 5. Run Prediction

After the model is trained:

```bash
python predict.py
```

# 6. Run Streamlit Application

```bash
streamlit run app.py
```

## Important

For a fresh setup, the **dataset generation must start with**:

```bash
python dataset_processing.py
```

Do **not** start with `preprocessing.py`. That file contains the preprocessing functions used by `dataset_processing.py`.

The dataset generation flow is:

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
