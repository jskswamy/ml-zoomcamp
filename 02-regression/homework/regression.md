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

# 02. Regression

## Import Necessary Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
```

## Question 1

There's one column with missing values. What is it?

'engine_displacement'
'horsepower'
'vehicle_weight'
'model_year'

```python
data = pd.read_csv('car_fuel_efficiency.csv')
print('Dataset shape:', data.shape)
print('\nMissing values per column:')
print(data.isnull().sum())
```

Based on the output, the column missing values are in `horsepower`

## Question 2

What's the median (50% percentile) for variable 'horsepower'?

- 49
- 99
- 149
- 199

```python
# Calculate median horsepower
median_horsepower = data['horsepower'].median()
print(f'Median horsepower: {median_horsepower}')
```

**Answer: 149**

### Prepare and split the dataset

- Shuffle the dataset (the filtered one you created above), use seed 42.
- Split your data in train/val/test sets, with 60%/20%/20% distribution

```python
# Set random seed for reproducibility
np.random.seed(42)

# Shuffle the dataset
data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train/val/test sets (60%/20%/20%)
# First split: 60% train, 40% temp (val + test)
train_data, temp_data = train_test_split(data_shuffled, test_size=0.4, random_state=42)

# Second split: split the 40% temp into 20% val and 20% test
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f'Train set size: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)')
print(f'Validation set size: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)')
print(f'Test set size: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)')
print(f'Total: {len(train_data) + len(val_data) + len(test_data)}')
```

### Question 3

- We need to deal with missing values for the column from Q1.
- We have two options: fill it with 0 or with the mean of this variable.
- Try both options. For each, train a linear regression model without regularization using the code from the lessons.
- For computing the mean, use the training only!
- Use the validation dataset to evaluate the models and compare the RMSE of each option.
- Round the RMSE scores to 2 decimal digits using round(score, 2)
- Which option gives better RMSE?

Options:

- With 0
- With mean
- Both are equally good

```python
# Linear regression functions from the lecture
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

# target variable (log transformation)
y_train = np.log1p(train_data['fuel_efficiency_mpg'].values)
y_val = np.log1p(val_data['fuel_efficiency_mpg'].values)

# Option 1: Fill missing horsepower with 0
train_data_0 = train_data.copy()
val_data_0 = val_data.copy()

train_data_0['horsepower'] = train_data_0['horsepower'].fillna(0)
val_data_0['horsepower'] = val_data_0['horsepower'].fillna(0)

# Prepare features for option 1
X_train_0 = train_data_0[['horsepower']].values
X_val_0 = val_data_0[['horsepower']].values

# Train model for option 1
w0_0, w_0 = train_linear_regression(X_train_0, y_train)
y_pred_0 = w0_0 + X_val_0.dot(w_0)
rmse_0 = rmse(y_val, y_pred_0)

print(f'RMSE with 0: {round(rmse_0, 2)}')

# Option 2: Fill missing horsepower with mean (calculated from training set only)
mean_horsepower = train_data['horsepower'].mean()
print(f'Mean horsepower from training set: {mean_horsepower}')

train_data_mean = train_data.copy()
val_data_mean = val_data.copy()

train_data_mean['horsepower'] = train_data_mean['horsepower'].fillna(mean_horsepower)
val_data_mean['horsepower'] = val_data_mean['horsepower'].fillna(mean_horsepower)

# Prepare features for option 2
X_train_mean = train_data_mean[['horsepower']].values
X_val_mean = val_data_mean[['horsepower']].values

# Train model for option 2
w0_mean, w_mean = train_linear_regression(X_train_mean, y_train)
y_pred_mean = w0_mean + X_val_mean.dot(w_mean)
rmse_mean = rmse(y_val, y_pred_mean)

print(f'RMSE with mean: {round(rmse_mean, 2)}')

print(f'\nComparison:')
print(f'RMSE with 0: {round(rmse_0, 2)}')
print(f'RMSE with mean: {round(rmse_mean, 2)}')
```

**Answer: With mean**

The results show:

- RMSE with 0: 0.17
- RMSE with mean: 0.16

Filling missing horsepower values with the mean gives a better (lower) RMSE score.
