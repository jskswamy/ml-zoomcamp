---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# 03. Classification

## Imports

```python
# Standard library imports
import csv
import math
import random
from collections import Counter

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mutual_info_score, classification_report

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')
```

---

## Dataset

We will be using lead scoring dataset Bank Marketing dataset

```sh
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/course_lead_scoring.csv
```

## Data Preparation

### Check if missing values are present in the features

```python
# Load the dataset
df = pd.read_csv('course_lead_scoring.csv')

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())

# Check for missing values
print("\nMissing values analysis:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Count': missing_values,
    'Missing Percentage': missing_percentage
})

print(missing_df[missing_df['Missing Count'] > 0])
```

### Output

```sh
Dataset shape: (1462, 9)

Dataset info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1462 entries, 0 to 1461
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   lead_source               1334 non-null   object
 1   industry                  1328 non-null   object
 2   number_of_courses_viewed  1462 non-null   int64
 3   annual_income             1281 non-null   float64
 4   employment_status         1362 non-null   object
 5   location                  1399 non-null   object
 6   interaction_count         1462 non-null   int64
 7   lead_score                1462 non-null   float64
 8   converted                 1462 non-null   int64
dtypes: float64(2), int64(3), object(4)
memory usage: 102.9+ KB
None

Missing values analysis:
                   Missing Count  Missing Percentage
lead_source                  128            8.755130
industry                     134            9.165527
annual_income                181           12.380301
employment_status            100            6.839945
location                      63            4.309166
```

---

### Identify categorical and numerical columns

```python
# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Show data types for each column
print("\nColumn data types:")
for col in df.columns:
    print(f"{col}: {df[col].dtype}")
```

### Output

```sh
Categorical columns: ['lead_source', 'industry', 'employment_status', 'location']
Numerical columns: ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score', 'converted']

Column data types:
lead_source: object
industry: object
number_of_courses_viewed: int64
annual_income: float64
employment_status: object
location: object
interaction_count: int64
lead_score: float64
converted: int64

```

---

### Replace missing values in categorical features with 'NA'

```python
# Create a copy for cleaning
df_clean = df.copy()

# For categorical features, replace missing values with 'NA'
categorical_cols = ['lead_source', 'industry', 'employment_status', 'location']

print("Before cleaning - missing values in categorical columns:")
for col in categorical_cols:
    missing_count = df_clean[col].isnull().sum()
    print(f"{col}: {missing_count} missing")

print("\nReplacing missing values in categorical features with 'NA'...")

for col in categorical_cols:
    if df_clean[col].isnull().any():
        print(f"Replacing missing values in {col} with 'NA'")
        df_clean[col] = df_clean[col].fillna('NA')

print("\nAfter cleaning - missing values in categorical columns:")
for col in categorical_cols:
    missing_count = df_clean[col].isnull().sum()
    print(f"{col}: {missing_count} missing")
```

### Output

```sh
Before cleaning - missing values in categorical columns:
lead_source: 128 missing
industry: 134 missing
employment_status: 100 missing
location: 63 missing

Replacing missing values in categorical features with 'NA'...
Replacing missing values in lead_source with 'NA'
Replacing missing values in industry with 'NA'
Replacing missing values in employment_status with 'NA'
Replacing missing values in location with 'NA'

After cleaning - missing values in categorical columns:
lead_source: 0 missing
industry: 0 missing
employment_status: 0 missing
location: 0 missing
```

---

### Replace missing values in numerical features with 0.0

```python
# For numerical features, replace missing values with 0.0
numerical_cols = ['number_of_courses_viewed', 'annual_income', 'interaction_count', 'lead_score', 'converted']

print("Before cleaning - missing values in numerical columns:")
for col in numerical_cols:
    missing_count = df_clean[col].isnull().sum()
    print(f"{col}: {missing_count} missing")

print("\nReplacing missing values in numerical features with 0.0...")

for col in numerical_cols:
    if df_clean[col].isnull().any():
        print(f"Replacing missing values in {col} with 0.0")
        df_clean[col] = df_clean[col].fillna(0.0)

print("\nAfter cleaning - missing values in numerical columns:")
for col in numerical_cols:
    missing_count = df_clean[col].isnull().sum()
    print(f"{col}: {missing_count} missing")

# Verify no missing values remain
print(f"\nTotal missing values after cleaning: {df_clean.isnull().sum().sum()}")
```

### Output

```sh
Before cleaning - missing values in numerical columns:
number_of_courses_viewed: 0 missing
annual_income: 181 missing
interaction_count: 0 missing
lead_score: 0 missing
converted: 0 missing

Replacing missing values in numerical features with 0.0...
Replacing missing values in annual_income with 0.0

After cleaning - missing values in numerical columns:
number_of_courses_viewed: 0 missing
annual_income: 0 missing
interaction_count: 0 missing
lead_score: 0 missing
converted: 0 missing

Total missing values after cleaning: 0
```

---

### Verify the cleaned dataset

```python
# Display cleaned dataset info
print("Cleaned dataset info:")
print(df_clean.info())

print("\nFirst 5 rows of cleaned data:")
print(df_clean.head())

print("\nSummary of data cleaning:")
print(f"Original dataset shape: {df.shape}")
print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
print(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")

# Show unique values in categorical columns to verify 'NA' replacement
print("\nUnique values in categorical columns after cleaning:")
categorical_cols = ['lead_source', 'industry', 'employment_status', 'location']
for col in categorical_cols:
    unique_vals = df_clean[col].unique()
    print(f"{col}: {unique_vals}")
```

### Output

```sh
Cleaned dataset info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1462 entries, 0 to 1461
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype
---  ------                    --------------  -----
 0   lead_source               1462 non-null   object
 1   industry                  1462 non-null   object
 2   number_of_courses_viewed  1462 non-null   int64
 3   annual_income             1462 non-null   float64
 4   employment_status         1462 non-null   object
 5   location                  1462 non-null   object
 6   interaction_count         1462 non-null   int64
 7   lead_score                1462 non-null   float64
 8   converted                 1462 non-null   int64
dtypes: float64(2), int64(3), object(4)
memory usage: 102.9+ KB
None

First 5 rows of cleaned data:
    lead_source    industry  number_of_courses_viewed  annual_income  \
0      paid_ads          NA                         1        79450.0
1  social_media      retail                         1        46992.0
2        events  healthcare                         5        78796.0
3      paid_ads      retail                         2        83843.0
4      referral   education                         3        85012.0

  employment_status       location  interaction_count  lead_score  converted
0        unemployed  south_america                  4        0.94          1
1          employed  south_america                  1        0.80          0
2        unemployed      australia                  3        0.69          1
3                NA      australia                  1        0.87          0
4     self_employed         europe                  3        0.62          1

Summary of data cleaning:
Original dataset shape: (1462, 9)
Cleaned dataset shape: (1462, 9)
Missing values before cleaning: 606
Missing values after cleaning: 0

Unique values in categorical columns after cleaning:
lead_source: ['paid_ads' 'social_media' 'events' 'referral' 'organic_search' 'NA']
industry: ['NA' 'retail' 'healthcare' 'education' 'manufacturing' 'technology'
 'other' 'finance']
employment_status: ['unemployed' 'employed' 'NA' 'self_employed' 'student']
location: ['south_america' 'australia' 'europe' 'africa' 'middle_east' 'NA'
 'north_america' 'asia']
```

---

## Reusable Configuration and Functions

```python
def identify_column_types(df):
    """Reusable function to identify categorical and numerical columns"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return categorical_cols, numerical_cols

LOGISTIC_REGRESSION_PARAMS = {
    'solver': 'liblinear',
    'C': 1.0,
    'max_iter': 1000,
    'random_state': 42
}

def split_data(df, test_size=0.2, random_state=42):
    """Reusable data splitting function"""
    return train_test_split(df, test_size=test_size, random_state=random_state)

def prepare_features(df_train, df_val, categorical_cols, numerical_cols):
    """Reusable feature preparation pipeline"""

    # Prepare numerical features
    X_train_num = df_train[numerical_cols].values
    X_val_num = df_val[numerical_cols].values

    # Apply one-hot encoding to categorical features
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_train_cat = encoder.fit_transform(df_train[categorical_cols])
    X_val_cat = encoder.transform(df_val[categorical_cols])

    # Combine features
    X_train_combined = np.hstack([X_train_num, X_train_cat])
    X_val_combined = np.hstack([X_val_num, X_val_cat])

    return X_train_combined, X_val_combined, encoder

def train_and_evaluate_model(X_train, y_train, X_val, y_val, model_params):
    """Reusable model training and evaluation"""

    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return model, y_pred, accuracy
```

---

## Question 1

What is the most frequent observation (mode) for the column industry?

- NA
- technology
- healthcare
- retail

```python
categorical_cols, numerical_cols = identify_column_types(df_clean)

# Find the most frequent observation (mode) for industry
industry_mode = df_clean['industry'].mode()
industry_value_counts = df_clean['industry'].value_counts()

print("Industry column value counts:")
print(industry_value_counts)
print(f"\nMost frequent observation (mode) for industry: {industry_mode[0]}")
print(f"Count: {industry_value_counts.iloc[0]}")
print(f"Percentage: {(industry_value_counts.iloc[0] / len(df_clean)) * 100:.2f}%")
```

### Output

```sh
Industry column value counts:
retail          203
healthcare      187
education       187
manufacturing   174
technology      179
other           198
finance         200
NA              134
Name: industry, dtype: int64

Most frequent observation (mode) for industry: retail
Count: 203
Percentage: 13.89%
```

---

## Question 2

Create the correlation matrix for the numerical features of your dataset. In a correlation matrix, you compute the correlation coefficient between every pair of features.

What are the two features that have the biggest correlation?

- `interaction_count` and `lead_score`
- `number_of_courses_viewed` and `lead_score`
- `number_of_courses_viewed` and `interaction_count`
- `annual_income` and `interaction_count`

Only consider the pairs above when answering this question.

```python
categorical_cols, numerical_cols = identify_column_types(df_clean)

# Create correlation matrix for numerical features
correlation_matrix = df_clean[numerical_cols].corr()

print("Correlation matrix for numerical features:")
print(correlation_matrix)

# Find the correlation coefficients for the specific pairs mentioned in the question
print("\nCorrelation coefficients for specific pairs:")
print(f"interaction_count and lead_score: {correlation_matrix.loc['interaction_count', 'lead_score']:.4f}")
print(f"number_of_courses_viewed and lead_score: {correlation_matrix.loc['number_of_courses_viewed', 'lead_score']:.4f}")
print(f"number_of_courses_viewed and interaction_count: {correlation_matrix.loc['number_of_courses_viewed', 'interaction_count']:.4f}")
print(f"annual_income and interaction_count: {correlation_matrix.loc['annual_income', 'interaction_count']:.4f}")

# Find the pair with the highest absolute correlation
pairs = [
    ('interaction_count', 'lead_score'),
    ('number_of_courses_viewed', 'lead_score'),
    ('number_of_courses_viewed', 'interaction_count'),
    ('annual_income', 'interaction_count')
]

max_corr = 0
max_pair = None

for col1, col2 in pairs:
    corr_value = abs(correlation_matrix.loc[col1, col2])
    if corr_value > max_corr:
        max_corr = corr_value
        max_pair = (col1, col2)

print(f"\nThe two features with the biggest correlation: {max_pair[0]} and {max_pair[1]}")
print(f"Correlation coefficient: {correlation_matrix.loc[max_pair[0], max_pair[1]]:.4f}")
```

### Output

```sh
Correlation matrix for numerical features:
                          number_of_courses_viewed  annual_income  \
number_of_courses_viewed                  1.000000       0.009770
annual_income                             0.009770       1.000000
interaction_count                        -0.023565       0.027036
lead_score                               -0.004879       0.015610
converted                                 0.435914       0.053131

                          interaction_count  lead_score  converted
number_of_courses_viewed          -0.023565   -0.004879   0.435914
annual_income                      0.027036    0.015610   0.053131
interaction_count                  1.000000    0.009888   0.374573
lead_score                         0.009888    1.000000   0.193673
converted                          0.374573    0.193673   1.000000

Correlation coefficients for specific pairs:
interaction_count and lead_score: 0.0099
number_of_courses_viewed and lead_score: -0.0049
number_of_courses_viewed and interaction_count: -0.0236
annual_income and interaction_count: 0.0270

The two features with the biggest correlation: annual_income and interaction_count
Correlation coefficient: 0.0270
```

---

## Question 3

- Calculate the mutual information score between converted and other categorical variables in the dataset. Use the training set only.
- Round the scores to 2 decimals using round(score, 2).

Which of these variables has the biggest mutual information score?

- industry
- location
- lead_source
- employment_status

```python
df_train, df_test = split_data(df_clean, test_size=0.2, random_state=42)

print(f"Training set size: {len(df_train)}")
print(f"Test set size: {len(df_test)}")

categorical_cols, numerical_cols = identify_column_types(df_clean)

# Calculate mutual information scores
mi_scores = {}

for var in categorical_cols:
    # Calculate mutual information between converted and each categorical variable
    mi_score = mutual_info_score(df_train['converted'], df_train[var])
    mi_scores[var] = round(mi_score, 2)
    print(f"Mutual information between 'converted' and '{var}': {mi_score:.4f} (rounded: {round(mi_score, 2)})")

# Find the variable with the biggest mutual information score
max_mi_var = max(mi_scores, key=mi_scores.get)
max_mi_score = mi_scores[max_mi_var]

print(f"\nVariable with the biggest mutual information score: {max_mi_var}")
print(f"Mutual information score: {max_mi_score}")

# Show all scores for comparison
print(f"\nAll mutual information scores:")
for var, score in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {var}: {score}")
```

### Output

```sh
Training set size: 1169
Test set size: 293
Mutual information between 'converted' and 'lead_source': 0.0257 (rounded: 0.03)
Mutual information between 'converted' and 'industry': 0.0117 (rounded: 0.01)
Mutual information between 'converted' and 'employment_status': 0.0133 (rounded: 0.01)
Mutual information between 'converted' and 'location': 0.0023 (rounded: 0.0)

Variable with the biggest mutual information score: lead_source
Mutual information score: 0.03

All mutual information scores:
  lead_source: 0.03
  industry: 0.01
  employment_status: 0.01
  location: 0.0
```

---

## Question 4

- Now let's train a logistic regression.
- Remember that we have several categorical variables in the dataset. Include them using one-hot encoding.
- Fit the model on the training dataset.
  - To make sure the results are reproducible across different versions of Scikit-Learn, fit the model with these parameters:
  - model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)
- Calculate the accuracy on the validation dataset and round it to 2 decimal digits.

What accuracy did you get?

- 0.64
- 0.74
- 0.84
- 0.94

```python
df_train, df_val = split_data(df_clean, test_size=0.2, random_state=42)

print(f"Training set size: {len(df_train)}")
print(f"Validation set size: {len(df_val)}")

# Separate features and target
X_train = df_train.drop('converted', axis=1)
y_train = df_train['converted']
X_val = df_val.drop('converted', axis=1)
y_val = df_val['converted']

# Identify column types using reusable function
categorical_cols, numerical_cols = identify_column_types(df_clean)

# Prepare features using reusable function
X_train_combined, X_val_combined, encoder = prepare_features(df_train, df_val, categorical_cols, numerical_cols)

print(f"Training features shape: {X_train_combined.shape}")
print(f"Validation features shape: {X_val_combined.shape}")

# Train and evaluate model using reusable function
model, y_val_pred, accuracy = train_and_evaluate_model(
    X_train_combined, y_train,
    X_val_combined, y_val,
    LOGISTIC_REGRESSION_PARAMS
)

accuracy_rounded = round(accuracy, 2)

print(f"\nValidation accuracy: {accuracy:.4f}")
print(f"Validation accuracy (rounded to 2 decimals): {accuracy_rounded}")

# Show some additional metrics
from sklearn.metrics import classification_report
print(f"\nClassification Report:")
print(classification_report(y_val, y_val_pred))
```

### Output

```sh

Training set size: 1169
Validation set size: 293
Training features shape: (1169, 27)
Validation features shape: (293, 27)

Validation accuracy: 0.7372
Validation accuracy (rounded to 2 decimals): 0.74

Classification Report:
              precision    recall  f1-score   support

           0       0.73      0.34      0.46        98
           1       0.74      0.94      0.83       195

    accuracy                           0.74       293
   macro avg       0.74      0.64      0.64       293
weighted avg       0.74      0.74      0.70       293

```

## Question 5

- Let's find the least useful feature using the feature elimination technique.
- Train a model using the same features and parameters as in Q4 (without rounding).
- Now exclude each feature from this set and train a model without it. Record the accuracy for each model.
- For each feature, calculate the difference between the original accuracy and the accuracy without the feature.

Which of following feature has the smallest difference?

- 'industry'
- 'employment_status'
- 'lead_score'

Note: The difference doesn't have to be positive.

```python
df_train, df_val = split_data(df_clean, test_size=0.2, random_state=42)

# Separate features and target
X_train = df_train.drop('converted', axis=1)
y_train = df_train['converted']
X_val = df_val.drop('converted', axis=1)
y_val = df_val['converted']

# Identify column types using reusable function
categorical_cols, numerical_cols = identify_column_types(df_clean)

# Prepare features using reusable function
X_train_combined, X_val_combined, encoder = prepare_features(df_train, df_val, categorical_cols, numerical_cols)

# Train the original model (with all features) using reusable function
model_original, y_val_pred_original, accuracy_original = train_and_evaluate_model(
    X_train_combined, y_train,
    X_val_combined, y_val,
    LOGISTIC_REGRESSION_PARAMS
)

print(f"Original model accuracy (all features): {accuracy_original:.4f}")

# Feature elimination for each feature
features_to_test = ['industry', 'employment_status', 'lead_score']
accuracy_differences = {}

for feature in features_to_test:
    print(f"\nTesting without feature: {feature}")

    # Create modified datasets without the feature
    if feature in categorical_cols:
        # Remove categorical feature
        modified_categorical_cols = [col for col in categorical_cols if col != feature]
        modified_numerical_cols = numerical_cols.copy()
    else:
        # Remove numerical feature
        modified_categorical_cols = categorical_cols.copy()
        modified_numerical_cols = [col for col in numerical_cols if col != feature]

    X_train_modified, X_val_modified, _ = prepare_features(df_train, df_val, modified_categorical_cols, modified_numerical_cols)

    _, y_val_pred_modified, accuracy_modified = train_and_evaluate_model(
        X_train_modified, y_train,
        X_val_modified, y_val,
        LOGISTIC_REGRESSION_PARAMS
    )

    # Calculate difference
    difference = accuracy_original - accuracy_modified
    accuracy_differences[feature] = difference

    print(f"Accuracy without {feature}: {accuracy_modified:.4f}")
    print(f"Difference: {difference:.4f}")

# Find the feature with smallest difference (least useful)
min_difference = min(accuracy_differences.values())
least_useful_feature = min(accuracy_differences, key=accuracy_differences.get)

print(f"\nFeature elimination results:")
for feature, diff in accuracy_differences.items():
    print(f"  {feature}: {diff:.4f}")

print(f"\nFeature with smallest difference (least useful): {least_useful_feature}")
print(f"Smallest difference: {min_difference:.4f}")
```

### Output

```sh
Original model accuracy (all features): 0.7372

Testing without feature: industry
Accuracy without industry: 0.7372
Difference: 0.0000

Testing without feature: employment_status
Accuracy without employment_status: 0.7304
Difference: 0.0068

Testing without feature: lead_score
Accuracy without lead_score: 0.7406
Difference: -0.0034

Feature elimination results:
  industry: 0.0000
  employment_status: 0.0068
  lead_score: -0.0034

Feature with smallest difference (least useful): lead_score
Smallest difference: -0.0034
```

## Question 6

- Now let's train a regularized logistic regression.
- Let's try the following values of the parameter C: [0.01, 0.1, 1, 10, 100].
- Train models using all the features as in Q4.
- Calculate the accuracy on the validation dataset and round it to 3 decimal digits.

Which of these C leads to the best accuracy on the validation set?

- 0.01
- 0.1
- 1
- 10
- 100

> Note: If there are multiple options, select the smallest C.

```python
df_train, df_val = split_data(df_clean, test_size=0.2, random_state=42)

# Separate features and target
X_train = df_train.drop('converted', axis=1)
y_train = df_train['converted']
X_val = df_val.drop('converted', axis=1)
y_val = df_val['converted']

categorical_cols, numerical_cols = identify_column_types(df_clean)

X_train_combined, X_val_combined, encoder = prepare_features(df_train, df_val, categorical_cols, numerical_cols)

# Test different C values
C_values = [0.01, 0.1, 1, 10, 100]
results = {}

print("Testing different C values for regularized logistic regression:")
print("=" * 60)

for C in C_values:
    # Create model parameters with current C value
    model_params = LOGISTIC_REGRESSION_PARAMS.copy()
    model_params['C'] = C

    # Train and evaluate model using reusable function
    model, y_val_pred, accuracy = train_and_evaluate_model(
        X_train_combined, y_train,
        X_val_combined, y_val,
        model_params
    )

    accuracy_rounded = round(accuracy, 3)
    results[C] = accuracy_rounded

    print(f"C = {C:5.2f}: Accuracy = {accuracy:.6f} (rounded: {accuracy_rounded})")

# Find the best C value
best_accuracy = max(results.values())
best_C_values = [C for C, acc in results.items() if acc == best_accuracy]

# If multiple C values have the same best accuracy, select the smallest
best_C = min(best_C_values)

print(f"\nResults summary:")
print(f"Best accuracy: {best_accuracy}")
print(f"C values with best accuracy: {best_C_values}")
print(f"Selected C (smallest among best): {best_C}")

# Show all results sorted by accuracy
print(f"\nAll results (sorted by accuracy):")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for C, acc in sorted_results:
    print(f"  C = {C:5.2f}: {acc}")
```

### Output

```sh
Testing different C values for regularized logistic regression:
============================================================
C =  0.01: Accuracy = 0.860068 (rounded: 0.86)
C =  0.10: Accuracy = 0.781570 (rounded: 0.782)
C =  1.00: Accuracy = 0.778157 (rounded: 0.778)
C = 10.00: Accuracy = 0.778157 (rounded: 0.778)
C = 100.00: Accuracy = 0.778157 (rounded: 0.778)

Results summary:
Best accuracy: 0.86
C values with best accuracy: [0.01]
Selected C (smallest among best): 0.01

All results (sorted by accuracy):
  C =  0.01: 0.86
  C =  0.10: 0.782
  C =  1.00: 0.778
  C = 10.00: 0.778
  C = 100.00: 0.778

```
