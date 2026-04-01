# **COMPLETE BREAKDOWN: MODEL TRAINING FOR TIME SERIES FORECASTING**

---

## **1. WHEN TO USE `self` IN PYTHON CLASSES**

### **The Rule**

Use `self` when you want to **store something for later use** across different methods in the class.

### **In Your Code:**

```python
class ModelTrainer:
    def __init__(self):
        self.best_model = {}  # ✅ self - want to access in other methods
    
    def split_data(self, df):
        self.train = df[...]  # ✅ self - store for train_model()
        self.val = df[...]    # ✅ self - store for train_model()
        self.test = df[...]   # ✅ self - store for evaluation later
    
    def train_model(self, train_df, val_df):
        start = time.perf_counter()  # ❌ NO self - only used in this function
        X_train = train_df[...]      # ❌ NO self - only used here
        
        self.model = LGBMRegressor() # ✅ self - need in plot_feature() and run()
```

### **Mental Model:**

```python
# WITHOUT self (local variable):
def method1(self):
    x = 5  # Dies when method1 finishes

def method2(self):
    print(x)  # ❌ ERROR: x doesn't exist

# WITH self (instance variable):
def method1(self):
    self.x = 5  # Stored on the object

def method2(self):
    print(self.x)  # ✅ Works: self.x persists
```

### **Interview Answer:**

"I use `self.variable` when the data needs to persist across multiple method calls or when other methods need access to it. Local variables without `self` are temporary and disappear when the function ends. In this class, `self.model` is used because `train_model()` creates the model and `plot_feature()` needs to access it later."

---

## **2. WHAT IS LightGBM AND LGBMRegressor?**

### **LightGBM = Light Gradient Boosting Machine**

**High-level:** A fast, efficient implementation of gradient boosting for machine learning.

**What is gradient boosting?**

It builds an ensemble of decision trees sequentially, where each new tree tries to correct the errors of the previous trees.

**Analogy:**
```
You're guessing taxi demand.

Tree 1: "I think it's 100 rides"
Actual: 150 rides
Error: +50 (you underestimated)

Tree 2: "I'll add 40 to fix the error"
Combined prediction: 100 + 40 = 140
Actual: 150
Error: +10 (getting closer)

Tree 3: "I'll add 8 more"
Combined prediction: 140 + 8 = 148
Actual: 150
Error: +2 (very close!)

Final model = Tree1 + Tree2 + Tree3 + ... + Tree2000
```

### **Why LightGBM Over XGBoost/CatBoost/RandomForest?**

| Algorithm | Speed | Memory | Accuracy | Best For |
|-----------|-------|--------|----------|----------|
| LightGBM  | ⚡⚡⚡ | 🟢🟢🟢 | ⭐⭐⭐ | Large datasets, time series |
| XGBoost   | ⚡⚡   | 🟢🟢   | ⭐⭐⭐⭐ | Smaller datasets, competitions |
| CatBoost  | ⚡     | 🟢     | ⭐⭐⭐⭐ | Categorical features |
| RandomForest | ⚡  | 🟢     | ⭐⭐   | Simple baselines |

**For your taxi demand project:**
- **Dataset size:** Millions of rows (every hour × 263 zones × months of data)
- **Feature types:** Mix of numerical (lags, rolling) and categorical (zone, hour)
- **Training time:** Need fast iterations

**LightGBM wins** because it's specifically optimized for large datasets and trains 10-100x faster than XGBoost.

### **LGBMRegressor vs LGBMClassifier**

```python
# REGRESSION (your case):
LGBMRegressor()
# Predicts continuous values: 0, 1, 2.5, 100, 450.7, etc.
# Used for: demand (any positive number)

# CLASSIFICATION (not your case):
LGBMClassifier()
# Predicts categories: 0, 1, 2, 3 (discrete classes)
# Used for: spam/not spam, cat/dog/bird, sentiment (positive/negative)
```

Your problem is **regression** because demand can be any value (45.3 rides, 120.7 rides, etc.), not discrete categories.

### **Key Hyperparameters Explained:**

```python
LGBMRegressor(
    n_estimators=2000,        # Build 2000 trees (more = better but slower)
    learning_rate=0.05,       # How much each tree contributes (smaller = more careful)
    num_leaves=63,            # Max leaves per tree (higher = more complex)
    min_child_samples=20,     # Min samples per leaf (prevents overfitting)
    subsample=0.8,            # Use 80% of data per tree (randomness helps generalization)
    colsample_bytree=0.8      # Use 80% of features per tree (prevents overreliance)
)
```

**Interview question: "Why learning_rate=0.05 instead of 0.1 or 0.01?"**

**Answer:** "Learning rate controls the step size in gradient descent. Lower values (0.01-0.05) make the model more cautious, training slowly but achieving better accuracy. Higher values (0.1-0.3) train faster but might overshoot optimal solutions. With 2000 trees and early stopping, 0.05 balances training time and accuracy - we can afford slower learning because we have many trees to refine predictions."

---

## **3. WHY USE A VALIDATION SET?**

### **The Three-Way Split**

```python
def split_data(self, df, val_start, test_start):
    self.train = df[df.hour_timestamp < val_start]
    self.val = df[(df.hour_timestamp >= val_start) & (df.hour_timestamp < test_start)]
    self.test = df[df.hour_timestamp >= test_start]
```

**Example with real dates:**
```python
Train: 2023-01-01 to 2023-10-31  (10 months) - Learn patterns
Val:   2023-11-01 to 2023-11-30  (1 month)   - Tune hyperparameters
Test:  2023-12-01 to 2023-12-31  (1 month)   - Final evaluation
```

### **Why Three Sets Instead of Two?**

**BAD: Only train + test**
```python
# Without validation set:
train_model(X_train, y_train)
test_score = evaluate(X_test, y_test)
# Score: 0.85

# Try different hyperparameters:
for lr in [0.01, 0.05, 0.1]:
    model = LGBMRegressor(learning_rate=lr)
    model.fit(X_train, y_train)
    score = evaluate(X_test, y_test)  # ❌ Testing on final test set!

# You pick lr=0.05 because it scored best on test set
# But now test set has influenced your model choice
# Test score is no longer a fair estimate of real-world performance
```

**GOOD: train + validation + test**
```python
# Tune on validation set:
for lr in [0.01, 0.05, 0.1]:
    model = LGBMRegressor(learning_rate=lr)
    model.fit(X_train, y_train)
    score = evaluate(X_val, y_val)  # ✅ Testing on validation set

# Pick lr=0.05 (best validation score)

# NOW evaluate on test set (ONCE):
final_model = LGBMRegressor(learning_rate=0.05)
final_model.fit(X_train, y_train)
test_score = evaluate(X_test, y_test)  # ✅ Unbiased estimate
```

### **The Role of Validation in Early Stopping**

```python
self.model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # Monitor validation performance
    callbacks=[early_stopping(100)]  # Stop if no improvement for 100 rounds
)
```

**What happens:**
```
Round   Train Error   Val Error   Decision
────────────────────────────────────────────
1       100.0         105.0       Continue
10      50.0          52.0        Continue
100     10.0          12.0        Continue
500     2.0           8.0         Continue (val still improving)
1000    0.5           7.8         Continue
1100    0.3           7.9         ⚠️ Val got worse (7.8 → 7.9)
1200    0.2           8.0         ⚠️ Val still worse
...
1300    0.1           8.1         🛑 STOP! No improvement for 100 rounds

# Model saved from round 1000 (best val error: 7.8)
```

**Without validation:** Model would keep training on train set, getting train error to 0.0 but overfitting terribly (memorizing training data instead of learning patterns).

**With validation:** Model stops when it starts overfitting (train error drops but val error increases).

### **Why Temporal Split for Time Series?**

```python
assert self.train.hour_timestamp.max() < self.val.hour_timestamp.min()
assert self.val.hour_timestamp.max() < self.test.hour_timestamp.min()
```

**Critical:** Train must come BEFORE validation, validation BEFORE test.

**Why?** In production, you predict the **future** using the **past**. If you randomly shuffle train/val/test, you're using future data to predict the past, which is impossible in reality.

**Bad (random split):**
```
Train: Jan, Mar, May, Jul, Sep, Nov (random months)
Test:  Feb, Apr, Jun, Aug, Oct, Dec

Problem: Training on November to predict February → time travel!
```

**Good (temporal split):**
```
Train: Jan-Oct
Val:   Nov
Test:  Dec

Realistic: Use Jan-Oct to predict Nov, then Nov to predict Dec
```

---

## **4. WHAT IS BASELINE NAIVE SEASONAL?**

### **The Concept**

A **baseline** is the simplest possible model to compare against. If your fancy LightGBM model doesn't beat the baseline, it's useless.

**Naive Seasonal Baseline** = "Tomorrow will be the same as last week"

### **Implementation:**

```python
def baseline_naive_seasonal(df):
    preds = df['lag_168h']  # Use last week's demand as prediction
    return compute_metrics(df['demand'], preds, label='Naive Seasonal Baseline')
```

**Example:**
```python
Date/Time           | Actual Demand | Prediction (lag_168h) | Error
─────────────────────────────────────────────────────────────────────
Fri Dec 15, 8pm    | 450           | 430 (from Dec 8)      | +20
Sat Dec 16, 8pm    | 500           | 480 (from Dec 9)      | +20
Sun Dec 17, 8pm    | 520           | 490 (from Dec 10)     | +30
```

**Why lag_168h specifically?**

168 hours = 7 days = 1 week. Weekly seasonality is strong in taxi demand:
- Fridays look like previous Fridays
- Monday 8am looks like previous Monday 8am
- Weekend patterns repeat weekly

### **Why This Baseline?**

**Alternative baselines:**

1. **Naive (lag_1h):** "Next hour = this hour"
   - Bad: No daily/weekly patterns
   
2. **Daily seasonal (lag_24h):** "Tomorrow 8am = today 8am"
   - Better: Captures daily cycle
   - Missing: Friday vs Monday difference

3. **Weekly seasonal (lag_168h):** "Next Friday = last Friday"
   - Best simple baseline for taxi data
   - Captures both daily AND weekly patterns

### **How to Beat the Baseline**

Your LightGBM model should achieve lower error because it uses:
- Multiple lags (1h, 24h, 168h) instead of just one
- Rolling windows (trends)
- Zone features (spatial patterns)
- Temporal features (holiday effects, rush hours)

**Interview answer:** "The naive seasonal baseline sets a minimum bar. If my model can't beat 'use last week's value,' then all my feature engineering and model complexity is worthless. In time series, a good baseline is often surprisingly hard to beat - I've seen projects where a simple lag_168h outperformed poorly-tuned neural networks."

---

## **5. CRITICAL BUGS IN YOUR CODE**

### **Bug #1: Missing Import**

```python
from sklearn.metrics import compute_metrics  # ❌ Doesn't exist in sklearn!
```

**Fix:**
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true, y_pred, label=''):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}
```

### **Bug #2: Variable Name Typo**

```python
assert self.val.hour_timestamp.max() < test.hour_timestamp.min()
#                                      ^^^^ Should be self.test
```

**Fix:**
```python
assert self.val.hour_timestamp.max() < self.test.hour_timestamp.min()
```

### **Bug #3: Wrong Time Calculation**

```python
end = (start - time.perf_counter()) * 1000  # ❌ Negative time!
```

**Fix:**
```python
end = (time.perf_counter() - start) * 1000  # ✅ Elapsed time
```

### **Bug #4: Missing Imports**

```python
from lightgbm import early_stopping, log_evaluation  # Add these
```

### **Bug #5: Hardcoded Paths**

```python
joblib.dump(self.model, "C:/Users/nikhi/...")  # ❌ Only works on your machine
```

**Fix:**
```python
from pathlib import Path

# At top of file:
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# In method:
model_path = MODEL_DIR / "lgbm_demand_v1.pkl"
joblib.dump(self.model, model_path)
```

### **Bug #6: JSON vs Pickle for Feature Columns**

```python
joblib.dump(feature_cols, "...feature_cols.json")  # ❌ joblib creates .pkl, not .json
```

**Fix:**
```python
import json
with open(MODEL_DIR / "feature_cols.json", 'w') as f:
    json.dump(feature_cols, f)
```

---

## **6. ESSENTIAL IMPROVEMENTS TO MAKE**

### **Improvement #1: Proper Metrics Computation**

```python
def evaluate_model(self, df, model, split_name='Test'):
    """Evaluate model on a dataset split."""
    feature_cols = [...]  # Same as training
    X = df[feature_cols]
    y_true = df['demand']
    y_pred = model.predict(X)
    
    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Print nicely
    print(f"\n{'='*60}")
    print(f"  {split_name} Set Performance")
    print(f"{'='*60}")
    print(f"  MAE:  {mae:.2f} rides/hour")
    print(f"  RMSE: {rmse:.2f} rides/hour")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    
    return {'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2}
```

### **Improvement #2: Feature Importance Analysis**

```python
def analyze_feature_importance(self, top_n=20):
    """Show which features matter most."""
    importance_df = pd.DataFrame({
        'feature': self.feature_cols,
        'importance': self.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"  Top {top_n} Most Important Features")
    print(f"{'='*60}")
    print(importance_df.head(top_n).to_string(index=False))
    
    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'].head(top_n), 
             importance_df['importance'].head(top_n))
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'feature_importance.png', dpi=150)
    plt.close()
    
    return importance_df
```

### **Improvement #3: Residual Analysis**

```python
def analyze_residuals(self, df, split_name='Test'):
    """Analyze prediction errors to find patterns."""
    X = df[self.feature_cols]
    y_true = df['demand']
    y_pred = self.model.predict(X)
    
    residuals = y_true - y_pred
    
    # By hour
    df['residual'] = residuals
    hourly_error = df.groupby('hour_of_day')['residual'].agg(['mean', 'std'])
    
    print(f"\n{'='*60}")
    print(f"  Error by Hour of Day ({split_name})")
    print(f"{'='*60}")
    print(hourly_error)
    
    # By zone (find worst zones)
    zone_error = df.groupby('zone_id').agg({
        'residual': ['mean', 'std', 'count']
    }).round(2)
    zone_error.columns = ['mean_error', 'std_error', 'count']
    zone_error = zone_error.sort_values('mean_error', key=abs, ascending=False)
    
    print(f"\n{'='*60}")
    print(f"  Top 10 Worst Zones (Highest Error)")
    print(f"{'='*60}")
    print(zone_error.head(10))
```

### **Improvement #4: Hyperparameter Tuning with Optuna**

```python
import optuna

def tune_hyperparameters(self, n_trials=50):
    """Use Optuna to find best hyperparameters."""
    
    def objective(trial):
        params = {
            'n_estimators': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        }
        
        model = LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[early_stopping(100, verbose=False)]
        )
        
        y_pred = model.predict(self.X_val)
        rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best RMSE: {study.best_value:.4f}")
    
    return study.best_params
```

### **Improvement #5: Complete Pipeline**

```python
def run_complete_pipeline(self, df, val_start, test_start):
    """Full training and evaluation pipeline."""
    
    # 1. Split data
    print("\n" + "="*60)
    print("  STEP 1: SPLITTING DATA")
    print("="*60)
    self.split_data(df, val_start, test_start)
    
    # 2. Baseline
    print("\n" + "="*60)
    print("  STEP 2: BASELINE MODEL")
    print("="*60)
    baseline_metrics = self.baseline_naive_seasonal(self.test)
    
    # 3. Train model
    print("\n" + "="*60)
    print("  STEP 3: TRAINING LIGHTGBM")
    print("="*60)
    self.train_model(self.train, self.val)
    
    # 4. Evaluate
    print("\n" + "="*60)
    print("  STEP 4: EVALUATION")
    print("="*60)
    train_metrics = self.evaluate_model(self.train, self.model, 'Train')
    val_metrics = self.evaluate_model(self.val, self.model, 'Validation')
    test_metrics = self.evaluate_model(self.test, self.model, 'Test')
    
    # 5. Feature importance
    print("\n" + "="*60)
    print("  STEP 5: FEATURE ANALYSIS")
    print("="*60)
    importance_df = self.analyze_feature_importance()
    
    # 6. Residual analysis
    print("\n" + "="*60)
    print("  STEP 6: ERROR ANALYSIS")
    print("="*60)
    self.analyze_residuals(self.test, 'Test')
    
    # 7. Save everything
    self.save_artifacts()
    
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    
    return {
        'baseline': baseline_metrics,
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'feature_importance': importance_df
    }
```

---

## **7. INTERVIEW-READY EXPLANATIONS**

### **Q: "Why LightGBM for this problem?"**

**A:** "I chose LightGBM because this is a large-scale time series problem with millions of rows (hourly data across 263 zones for months). LightGBM is optimized for large datasets through leaf-wise tree growth and histogram-based splitting, making it 10-100x faster than XGBoost while maintaining similar accuracy. For time series specifically, its ability to handle diverse feature types - continuous lags, rolling statistics, and categorical zone IDs - makes it ideal. The built-in early stopping on validation set prevents overfitting without manual intervention."

### **Q: "Walk me through your train/val/test split."**

**A:** "I use a temporal split, not random, because this is time series. Train is Jan-Oct 2023, validation is November, test is December. This mimics production where we predict future using past. Validation serves two purposes: (1) early stopping during training to prevent overfitting, and (2) hyperparameter tuning without touching the test set. The assertions ensure no temporal leakage - training data must come strictly before validation, which comes before test. This is critical because using future data to predict past would artificially inflate metrics."

### **Q: "Explain your baseline and why it matters."**

**A:** "The naive seasonal baseline uses lag_168h - last week's demand at the same hour. This is surprisingly effective for taxi data because of strong weekly seasonality: Friday nights look like previous Friday nights. It sets a minimum performance bar - if my LightGBM model with 40+ features can't beat 'use last week's value,' then all my complexity is worthless. In production, I'd actually ensemble the baseline with the ML model - weight them based on validation performance - because sometimes simple approaches win for stable patterns."

### **Q: "What would you monitor in production?"**

**A:** "I'd track three categories of metrics:

**Performance metrics:**
- MAE, RMSE by zone and hour (detect if certain zones/times degrade)
- MAPE for percentage error (easier for stakeholders to understand)
- Prediction intervals to quantify uncertainty

**Data drift:**
- Feature distributions (is lag_24h shifting? new peak hours?)
- Target distribution (is overall demand changing?)
- Alert if current week's pattern differs from training distribution

**Business metrics:**
- Fleet utilization (are predictions actually reducing wait times?)
- Revenue impact (did better predictions increase rides?)
- A/B test against baseline to prove value

I'd retrain weekly with latest data, keeping a rolling window of last 6-12 months to adapt to changing patterns while maintaining enough history for robust statistics."

---

This code forms the core of your ML system. Master these concepts and you'll ace any time series forecasting interview.