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

## 3. Run the Project Files in This Order

### Step 1 — Preprocessing

```bash
python src/preprocessing.py
```

### Step 2 — Encode Categories

```bash
python src/encode_categories.py
```

### Step 3 — Prepare Training Data

```bash
python src/prepare_training_data.py
```

### Step 4 — Vectorization

```bash
python src/vectorization.py
```

### Step 5 — Create Model

```bash
python src/model.py
```

### Step 6 — Train Baseline Model

```bash
python src/train.py
```

### Step 7 — Evaluate Baseline Model

```bash
python src/evaluate.py
```

### Step 8 — Tune Baseline Thresholds

```bash
python src/tune_thresholds.py
```

### Step 9 — Train Weighted Model

```bash
python src/train_weighted.py
```

### Step 10 — Evaluate Weighted Model

```bash
python src/evaluate_weighted.py
```

### Step 11 — Tune Weighted Thresholds

```bash
python src/tune_weighted_thresholds.py
```

### Step 12 — Run Prediction System

```bash
python src/predict.py
```

### Step 13 — Run Streamlit Application

```bash
streamlit run app.py
```

## Final Application

After completing the training steps, the application can be started directly with:

```bash
streamlit run app.py
```
