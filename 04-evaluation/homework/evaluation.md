# 04-evaluation

In this homework, we will use the lead scoring dataset. [Download it from: here](https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv)
In this dataset our desired target for classification task will be converted variable - has the client signed up to the platform or not.

## Importing necessary libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
```

## Data preparation

- Check if the missing values are presented in the features.
- If there are missing values:
  - For caterogiral features, replace them with 'NA'
  - For numerical features, replace with with 0.0

- Split the data into 3 parts: train/validation/test with 60%/20%/20% distribution. Use train_test_split function for that with random_state=1

```python
# Load the dataset
df = pd.read_csv('course_lead_scoring.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
df.head()

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check for empty strings (which are also missing values)
print("\nEmpty strings per column:")
for col in df.columns:
    if df[col].dtype == 'object':
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            print(f"{col}: {empty_count}")

# Identify categorical and numerical features
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove target variable from features
if 'converted' in categorical_features:
    categorical_features.remove('converted')
if 'converted' in numerical_features:
    numerical_features.remove('converted')

print(f"\nCategorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Handle missing values
# For categorical features: replace empty strings and NaN with 'NA'
for col in categorical_features:
    df[col] = df[col].replace('', 'NA')
    df[col] = df[col].fillna('NA')

# For numerical features: replace NaN with 0.0
for col in numerical_features:
    df[col] = df[col].fillna(0.0)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Split the data into train/validation/test with 60%/20%/20% distribution
# First split: 60% train, 40% temp (validation + test)
df_train_full, df_temp = train_test_split(df, test_size=0.4, random_state=1)

# Second split: split temp into 50%/50% which gives us 20%/20% of original
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=1)

print(f"\nTrain set size: {len(df_train_full)} ({len(df_train_full)/len(df)*100:.1f}%)")
print(f"Validation set size: {len(df_val)} ({len(df_val)/len(df)*100:.1f}%)")
print(f"Test set size: {len(df_test)} ({len(df_test)/len(df)*100:.1f}%)")

# Reset indices
df_train_full = df_train_full.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("\nData preparation complete!")
```

## Question 1: ROC AUC feature importance

ROC AUC could also be used to evaluate feature importance of numerical variables.

Let's do that

For each numerical variable, use it as score (aka prediction) and compute the AUC with the y variable as ground truth.
Use the training dataset for that

If your AUC is < 0.5, invert this variable by putting "-" in front

(e.g. -df_train['balance'])

AUC can go below 0.5 if the variable is negatively correlated with the target variable. You can change the direction of the correlation by negating this variable - then negative correlation becomes positive.

Which numerical variable (among the following 4) has the highest AUC?

- lead_score
- number_of_courses_viewed
- interaction_count
- annual_income

```python
# Extract target variable for training set
y_train = df_train_full['converted']

# Numerical variables to evaluate
numerical_vars = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']

# Compute AUC for each numerical variable
auc_scores = {}

for var in numerical_vars:
    # Use the variable as prediction score
    score = df_train_full[var]

    # Compute AUC
    auc = roc_auc_score(y_train, score)

    # If AUC < 0.5, invert the variable
    if auc < 0.5:
        score = -df_train_full[var]
        auc = roc_auc_score(y_train, score)
        print(f"{var}: {auc:.4f} (inverted)")
    else:
        print(f"{var}: {auc:.4f}")

    auc_scores[var] = auc

# Find the variable with highest AUC
best_var = max(auc_scores, key=auc_scores.get)
print(f"\nVariable with highest AUC: {best_var} (AUC = {auc_scores[best_var]:.4f})")
```

**Output:**

```
lead_score: 0.6111
number_of_courses_viewed: 0.7652
interaction_count: 0.7272
annual_income: 0.5446

Variable with highest AUC: number_of_courses_viewed (AUC = 0.7652)
```

**Answer: number_of_courses_viewed**

## Question 2: Training the model

Apply one-hot-encoding using DictVectorizer and train the logistic regression with these parameters:

`LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`

What's the AUC of this model on the validation dataset? (round to 3 digits)

- 0.32
- 0.52
- 0.72
- 0.92

```python
# Prepare features and target for train and validation sets
# Remove target variable from dataframes
X_train = df_train_full.drop('converted', axis=1)
y_train = df_train_full['converted']

X_val = df_val.drop('converted', axis=1)
y_val = df_val['converted']

# Convert dataframes to dictionaries for DictVectorizer
train_dicts = X_train.to_dict(orient='records')
val_dicts = X_val.to_dict(orient='records')

# Apply one-hot encoding using DictVectorizer
dv = DictVectorizer(sparse=False)
X_train_encoded = dv.fit_transform(train_dicts)
X_val_encoded = dv.transform(val_dicts)

print(f"Training set shape after encoding: {X_train_encoded.shape}")
print(f"Validation set shape after encoding: {X_val_encoded.shape}")

# Train logistic regression model
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_encoded, y_train)

# Predict probabilities on validation set
y_val_pred = model.predict_proba(X_val_encoded)[:, 1]

# Calculate AUC on validation set
auc_val = roc_auc_score(y_val, y_val_pred)

print(f"\nAUC on validation set: {auc_val:.3f}")
```

**Output:**

```
Training set shape after encoding: (877, 31)
Validation set shape after encoding: (292, 31)

AUC on validation set: 0.794
```

**Answer: 0.794** (Note: Among the given options, this would be closest to 0.72, though the actual value is 0.794)

## Question 3: Precision and Recall

Now let's compute precision and recall for our model.

- Evaluate the model on all thresholds from 0.0 to 1.0 with step 0.01
- For each threshold, compute precision and recall
- Plot them

At which threshold precision and recall curves intersect?

- 0.145
- 0.345
- 0.545
- 0.745

```python
# Compute precision and recall for different thresholds
thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []

for threshold in thresholds:
    # Apply threshold to predictions
    y_pred_threshold = (y_val_pred >= threshold).astype(int)

    # Compute precision and recall
    # Handle edge cases where there are no positive predictions
    if y_pred_threshold.sum() == 0:
        precision = 0
    else:
        precision = precision_score(y_val, y_pred_threshold, zero_division=0)

    recall = recall_score(y_val, y_pred_threshold, zero_division=0)

    precisions.append(precision)
    recalls.append(recall)

# Convert to numpy arrays
precisions = np.array(precisions)
recalls = np.array(recalls)

# Find the intersection point
# The intersection occurs where the absolute difference is minimized
diff = np.abs(precisions - recalls)
intersection_idx = np.argmin(diff)
intersection_threshold = thresholds[intersection_idx]

print(f"Threshold at intersection: {intersection_threshold:.3f}")
print(f"Precision at intersection: {precisions[intersection_idx]:.3f}")
print(f"Recall at intersection: {recalls[intersection_idx]:.3f}")

# Plot precision and recall curves
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions, label='Precision', linewidth=2)
plt.plot(thresholds, recalls, label='Recall', linewidth=2)
plt.axvline(x=intersection_threshold, color='red', linestyle='--',
            label=f'Intersection at {intersection_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as 'precision_recall_curves.png'")
```

**Output:**

```
Threshold at intersection: 0.590
Precision at intersection: 0.807
Recall at intersection: 0.807

Plot saved as 'precision_recall_curves.png'
```

**Answer: 0.545** (The actual intersection is at 0.590, closest to the option 0.545)

## Question 4: F1 score

Precision and recall are conflicting - when one grows, the other goes down. That's why they are often combined into the F1 score - a metrics that takes into account both

This is the formula for computing F1:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

Where P is precision and R is recall.

Let's compute F1 for all thresholds from 0.0 to 1.0 with increment 0.01

At which threshold F1 is maximal?

- 0.14
- 0.34
- 0.54
- 0.74

```python
# Compute F1 score for all thresholds
thresholds = np.arange(0.0, 1.01, 0.01)
f1_scores = []

for threshold in thresholds:
    # Apply threshold to predictions
    y_pred_threshold = (y_val_pred >= threshold).astype(int)

    # Compute precision and recall
    if y_pred_threshold.sum() == 0:
        precision = 0
        recall = 0
    else:
        precision = precision_score(y_val, y_pred_threshold, zero_division=0)
        recall = recall_score(y_val, y_pred_threshold, zero_division=0)

    # Compute F1 score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    f1_scores.append(f1)

# Convert to numpy array
f1_scores = np.array(f1_scores)

# Find the threshold with maximum F1 score
max_f1_idx = np.argmax(f1_scores)
max_f1_threshold = thresholds[max_f1_idx]
max_f1_score = f1_scores[max_f1_idx]

print(f"Threshold with maximum F1: {max_f1_threshold:.2f}")
print(f"Maximum F1 score: {max_f1_score:.3f}")
print(f"Precision at max F1: {precisions[max_f1_idx]:.3f}")
print(f"Recall at max F1: {recalls[max_f1_idx]:.3f}")

# Plot F1 score vs threshold
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, linewidth=2, color='green')
plt.axvline(x=max_f1_threshold, color='red', linestyle='--',
            label=f'Max F1 at {max_f1_threshold:.2f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('F1 Score vs Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('f1_score_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as 'f1_score_curve.png'")
```

**Output:**

```
Threshold with maximum F1: 0.47
Maximum F1 score: 0.848
Precision at max F1: 0.768
Recall at max F1: 0.948

Plot saved as 'f1_score_curve.png'
```

**Answer: 0.54** (The actual maximum F1 is at 0.47, closest to the option 0.54)

## Question 5: 5-Fold CV

Use the KFold class from Scikit-Learn to evaluate our model on 5 different folds:

`KFold(n_splits=5, shuffle=True, random_state=1)`

- Iterate over different folds of df_full_train
- Split the data into train and validation
- Train the model on train with these parameters: LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
- Use AUC to evaluate the model on validation

How large is standard deviation of the scores across different folds?

- 0.0001
- 0.006
- 0.06
- 0.36

```python
# Set up 5-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

# Store AUC scores from each fold
auc_scores_cv = []

# Iterate over different folds
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df_train_full)):
    print(f"\nFold {fold_idx + 1}:")

    # Split the data into train and validation for this fold
    df_fold_train = df_train_full.iloc[train_idx]
    df_fold_val = df_train_full.iloc[val_idx]

    # Prepare features and target
    X_fold_train = df_fold_train.drop('converted', axis=1)
    y_fold_train = df_fold_train['converted']

    X_fold_val = df_fold_val.drop('converted', axis=1)
    y_fold_val = df_fold_val['converted']

    # Convert to dictionaries for DictVectorizer
    fold_train_dicts = X_fold_train.to_dict(orient='records')
    fold_val_dicts = X_fold_val.to_dict(orient='records')

    # Apply one-hot encoding
    dv_fold = DictVectorizer(sparse=False)
    X_fold_train_encoded = dv_fold.fit_transform(fold_train_dicts)
    X_fold_val_encoded = dv_fold.transform(fold_val_dicts)

    # Train the model
    model_fold = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
    model_fold.fit(X_fold_train_encoded, y_fold_train)

    # Predict on validation set
    y_fold_val_pred = model_fold.predict_proba(X_fold_val_encoded)[:, 1]

    # Calculate AUC
    auc_fold = roc_auc_score(y_fold_val, y_fold_val_pred)
    auc_scores_cv.append(auc_fold)

    print(f"  Train size: {len(df_fold_train)}, Val size: {len(df_fold_val)}")
    print(f"  AUC: {auc_fold:.4f}")

# Convert to numpy array
auc_scores_cv = np.array(auc_scores_cv)

# Calculate statistics
mean_auc = auc_scores_cv.mean()
std_auc = auc_scores_cv.std()

print(f"\n{'='*50}")
print(f"Cross-Validation Results:")
print(f"AUC scores: {auc_scores_cv}")
print(f"Mean AUC: {mean_auc:.4f}")
print(f"Standard Deviation: {std_auc:.4f}")
print(f"{'='*50}")
```

**Output:**

```
Fold 1:
  Train size: 701, Val size: 176
  AUC: 0.8117

Fold 2:
  Train size: 701, Val size: 176
  AUC: 0.8232

Fold 3:
  Train size: 702, Val size: 175
  AUC: 0.8364

Fold 4:
  Train size: 702, Val size: 175
  AUC: 0.8392

Fold 5:
  Train size: 702, Val size: 175
  AUC: 0.8283

==================================================
Cross-Validation Results:
AUC scores: [0.81172965 0.82324219 0.8364     0.83922922 0.82826667]
Mean AUC: 0.8278
Standard Deviation: 0.0098
==================================================
```

**Answer: 0.006** (The actual standard deviation is 0.0098, closest to the option 0.006)

## Question 6: Hyperparameter Tuning

Now let's use 5-Fold cross-validation to find the best parameter C

- Iterate over the following C values: [0.000001, 0.001, 1]
- Initialize KFold with the same parameters as previously
- Use these parameters for the model: LogisticRegression(solver='liblinear', C=C, max_iter=1000)
- Compute the mean score as well as the std (round the mean and std to 3 decimal digits)

Which C leads to the best mean score?

- 0.000001
- 0.001
- 1

If you have ties, select the score with the lowest std. If you still have ties, select the smallest C.

```python
# C values to test
C_values = [0.000001, 0.001, 1]

# Store results for each C
results = []

# Iterate over different C values
for C in C_values:
    print(f"\n{'='*60}")
    print(f"Testing C = {C}")
    print(f"{'='*60}")

    # Initialize KFold with same parameters as before
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)

    # Store AUC scores for this C value
    auc_scores = []

    # Perform 5-fold cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df_train_full)):
        # Split the data
        df_fold_train = df_train_full.iloc[train_idx]
        df_fold_val = df_train_full.iloc[val_idx]

        # Prepare features and target
        X_fold_train = df_fold_train.drop('converted', axis=1)
        y_fold_train = df_fold_train['converted']

        X_fold_val = df_fold_val.drop('converted', axis=1)
        y_fold_val = df_fold_val['converted']

        # Convert to dictionaries
        fold_train_dicts = X_fold_train.to_dict(orient='records')
        fold_val_dicts = X_fold_val.to_dict(orient='records')

        # Apply one-hot encoding
        dv_fold = DictVectorizer(sparse=False)
        X_fold_train_encoded = dv_fold.fit_transform(fold_train_dicts)
        X_fold_val_encoded = dv_fold.transform(fold_val_dicts)

        # Train model with current C value
        model_fold = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model_fold.fit(X_fold_train_encoded, y_fold_train)

        # Predict and calculate AUC
        y_fold_val_pred = model_fold.predict_proba(X_fold_val_encoded)[:, 1]
        auc_fold = roc_auc_score(y_fold_val, y_fold_val_pred)
        auc_scores.append(auc_fold)

        print(f"  Fold {fold_idx + 1}: AUC = {auc_fold:.4f}")

    # Calculate mean and std
    auc_scores = np.array(auc_scores)
    mean_auc = np.round(auc_scores.mean(), 3)
    std_auc = np.round(auc_scores.std(), 3)

    print(f"\n  Mean AUC: {mean_auc:.3f}")
    print(f"  Std AUC: {std_auc:.3f}")

    # Store results
    results.append({
        'C': C,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'scores': auc_scores
    })

# Display summary
print(f"\n{'='*60}")
print("SUMMARY OF RESULTS")
print(f"{'='*60}")
print(f"{'C':<12} {'Mean AUC':<12} {'Std AUC':<12}")
print(f"{'-'*60}")

for result in results:
    print(f"{result['C']:<12} {result['mean_auc']:<12.3f} {result['std_auc']:<12.3f}")

# Find the best C
# Sort by mean (descending), then by std (ascending), then by C (ascending)
sorted_results = sorted(results, key=lambda x: (-x['mean_auc'], x['std_auc'], x['C']))
best_result = sorted_results[0]

print(f"\n{'='*60}")
print(f"Best C: {best_result['C']}")
print(f"Mean AUC: {best_result['mean_auc']:.3f}")
print(f"Std AUC: {best_result['std_auc']:.3f}")
print(f"{'='*60}")
```

**Output:**

```
============================================================
Testing C = 1e-06
============================================================
  Fold 1: AUC = 0.5231
  Fold 2: AUC = 0.6092
  Fold 3: AUC = 0.5775
  Fold 4: AUC = 0.4885
  Fold 5: AUC = 0.5435

  Mean AUC: 0.548
  Std AUC: 0.042

============================================================
Testing C = 0.001
============================================================
  Fold 1: AUC = 0.8440
  Fold 2: AUC = 0.8817
  Fold 3: AUC = 0.8817
  Fold 4: AUC = 0.8875
  Fold 5: AUC = 0.8667

  Mean AUC: 0.872
  Std AUC: 0.016

============================================================
Testing C = 1
============================================================
  Fold 1: AUC = 0.8117
  Fold 2: AUC = 0.8232
  Fold 3: AUC = 0.8364
  Fold 4: AUC = 0.8392
  Fold 5: AUC = 0.8283

  Mean AUC: 0.828
  Std AUC: 0.010

============================================================
SUMMARY OF RESULTS
============================================================
C            Mean AUC     Std AUC
------------------------------------------------------------
1e-06        0.548        0.042
0.001        0.872        0.016
1            0.828        0.010

============================================================
Best C: 0.001
Mean AUC: 0.872
Std AUC: 0.016
============================================================
```

**Answer: 0.001**

The hyperparameter C=0.001 leads to the best mean AUC score of 0.872 with a standard deviation of 0.016.
