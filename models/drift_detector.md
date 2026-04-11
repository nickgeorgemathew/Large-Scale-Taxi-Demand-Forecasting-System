Good. Now we will design **`drift_detector.py` properly**, the way a production ML system would. We will focus on **architecture and reasoning**, not just code.

We will cover:

1. What drift actually means
2. What types of drift we detect
3. Where this module sits in the pipeline
4. Data flow through the module
5. Internal architecture of the file
6. Pseudocode for each component
7. What inputs and outputs it should have

---

# 1. Intuition — What Drift Means

Imagine your model was trained on data like:

| hour | demand |
| ---- | ------ |
| 8    | 120    |
| 9    | 140    |
| 18   | 210    |

But months later the environment changes:

| hour | demand |
| ---- | ------ |
| 8    | 200    |
| 9    | 220    |
| 18   | 300    |

The **input distribution changed**.

Even if the model initially works, eventually:

```text
model predictions become unreliable
```

This is called **data drift**.

Drift detection systems continuously ask:

```text
Does current data still look like training data?
```

---

# 2. Types of Drift You Should Detect

Your `drift_detector.py` should detect **three signals**.

### 1️⃣ Feature Drift

Distribution of input features changed.

Example:

```text
training hour_of_day distribution ≠ production distribution
```

---

### 2️⃣ Residual Drift

Prediction errors shift.

Residual:

```
residual = actual − prediction
```

Expected:

```
mean(residual) ≈ 0
```

If residual mean shifts → model bias.

---

### 3️⃣ Performance Drift

Already handled partly in `metrics_monitor`.

Example:

```
rolling RMSE > training RMSE
```

But drift_detector can combine signals.

---

# 3. Where drift_detector Fits

Monitoring pipeline:

```
prediction_logger
      ↓
metrics_monitor
      ↓
drift_detector
      ↓
alert_manager
      ↓
retrain_pipeline
```

metrics_monitor answers:

```
Is performance degrading?
```

drift_detector answers:

```
Is the data distribution changing?
```

---

# 4. Data Inputs to drift_detector

The module should receive two datasets.

### Training Dataset

Used to compute **reference distributions**.

Example:

| hour_of_day | lag_24h | demand |
| ----------- | ------- | ------ |

---

### Current Production Dataset

Usually derived from prediction logs.

Example:

| hour_of_day | lag_24h | prediction | actual |

---

# 5. Data Flow Through drift_detector

Conceptually:

```
load training data
        ↓
load recent production data
        ↓
calculate feature distributions
        ↓
compare distributions
        ↓
calculate drift scores
        ↓
return drift signals
```

---

# 6. Internal Architecture of drift_detector.py

The module should contain a class.

```
DriftDetector
    load_reference_data()
    load_current_data()
    detect_feature_drift()
    detect_residual_drift()
    detect_overall_drift()
```

Each function handles **one responsibility**.

---

# 7. Feature Drift Detection Logic

We use **Population Stability Index (PSI)**.

PSI compares two distributions.

Formula conceptually:

```
PSI = Σ (actual − expected) * ln(actual / expected)
```

Interpretation:

| PSI     | Meaning        |
| ------- | -------------- |
| <0.1    | stable         |
| 0.1–0.2 | moderate drift |

> 0.2 | significant drift |

---

# Pseudocode — Feature Drift

```
function detect_feature_drift(train_df, current_df):

    drift_results = {}

    for feature in monitored_features:

        train_distribution = histogram(train_df[feature])

        current_distribution = histogram(current_df[feature])

        psi = calculate_psi(train_distribution, current_distribution)

        if psi > threshold:
            drift_results[feature] = True
        else:
            drift_results[feature] = False

    return drift_results
```

---

# 8. Residual Drift Detection

Residual drift detects **bias shifts**.

Steps:

1️⃣ Compute residuals

```
residual = actual − prediction
```

2️⃣ Measure mean and variance

Expected:

```
mean ≈ 0
```

If residual mean shifts:

```
model systematically underpredicting or overpredicting
```

---

# Pseudocode — Residual Drift

```
function detect_residual_drift(df):

    residuals = df.actual − df.prediction

    mean_residual = mean(residuals)

    if abs(mean_residual) > threshold:
         drift_detected = True
    else:
         drift_detected = False

    return drift_detected
```

---

# 9. Combining Drift Signals

The final drift detection should combine signals.

Example:

```
feature drift
residual drift
performance drift
```

---

# Pseudocode — Overall Drift

```
function detect_overall_drift():

    feature_drift = detect_feature_drift()

    residual_drift = detect_residual_drift()

    if feature_drift OR residual_drift:
         drift_detected = True
    else:
         drift_detected = False

    return drift_detected
```

---

# 10. Expected Output Format

Your detector should return structured signals.

Example:

```
{
 "feature_drift": {
     "hour_of_day": False,
     "lag_24h": True
 },
 "residual_drift": False,
 "overall_drift": True
}
```

This feeds directly into:

```
alert_manager
```

---

# 11. Important Engineering Details

Your module should also:

### Handle missing data

Example:

```
actual values may be missing
```

---

### Use monitoring window

Only analyze recent data:

```
last 7 days
```

---

### Store drift history

Save results:

```
monitoring/drift_history.parquet
```

This allows plotting **drift trends**.

---

# 12. Libraries Needed

You will likely use:

```
pandas
numpy
scipy.stats
```

Optionally:

```
evidently
alibi-detect
```

But for your project manual PSI implementation is fine.

---

# 13. Final Data Flow Diagram

```
Prediction Logs
        ↓
metrics_monitor
        ↓
rolling metrics
        ↓
drift_detector
        ↓
drift signals
        ↓
alert_manager
        ↓
retrain_pipeline
```

This is essentially **a mini MLOps monitoring system**.

---

# Quick Summary

`drift_detector.py` should:

1. Load training reference distributions
2. Load recent production data
3. Detect feature distribution drift (PSI)
4. Detect residual drift (error bias)
5. Combine signals into overall drift

Output should be a **structured drift report**.

---

# Understanding Check

Suppose drift detection reports:

```
feature_drift = False
residual_drift = True
```

What does this suggest?

A) input data changed
B) model relationship with target changed
C) logging error

Explain your reasoning.

---

# Mini Architecture Exercise

You plan to monitor features:

```
hour_of_day
zone_id
lag_24h
rolling_mean
```

Which of these features would you **expect to drift most frequently in a real demand forecasting system**, and why?
