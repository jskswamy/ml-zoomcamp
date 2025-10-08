---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.17.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# 01 - Introduction

## Q1. Pandas version

What version of Pandas did you install?

You can get the version information using the **version** field:

```python
import pandas as pd
import numpy as np
```

```python
pd.__version__
```

## Getting the data

<!-- #region -->

```sh
wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv
```

<!-- #endregion -->

```python
data = pd.read_csv("car_fuel_efficiency.csv")
```

```python
data.info()
```

## Q2. Records Count

How many records are in the dataset?

```python
len(data)
```

## Q3. Fuel Types

How many fuel types are presented in the dataset?

```python
data["fuel_type"].unique().size
```

## Q4. Missing Values

How many columns in the dataset have missing values?

```python
data.isnull().sum()

# Filter the columns with missing values
missing_values = data.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values.size
```

## Q5. Max Fuel Efficiency

What's the maximum fuel efficiency of cars from Asia?

```python
max_fuel_efficiency = data['fuel_efficiency_mpg'].max()
print(f"Exact max value: {max_fuel_efficiency}")

print(f"Top 5 highest values:")
print(data['fuel_efficiency_mpg'].nlargest(5))

options = [13.75, 23.75, 33.75, 43.75]
print(f"\nClosest option to {max_fuel_efficiency}: {min(options, key=lambda x: abs(x - max_fuel_efficiency))}")

max_fuel_efficiency
```

## Q6. Median value of horsepower

1. Find the median value of the horsepower column in the dataset.
2. Next, calculate the most frequent value of the same horsepower column.
3. Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
4. Now, calculate the median value of horsepower once again.

Has it changed?

```python
# Step 1: Find the median value of the horsepower column
median_original = data['horsepower'].median()
print(f"Original median horsepower: {median_original}")

# Step 2: Calculate the most frequent value (mode) of horsepower
mode_horsepower = data['horsepower'].mode()[0]  # mode() returns a Series, take first value
print(f"Most frequent horsepower value (mode): {mode_horsepower}")

# Step 3: Fill missing values with the most frequent value
data_filled = data.copy()
data_filled['horsepower'] = data_filled['horsepower'].fillna(mode_horsepower)

# Step 4: Calculate the median again
median_after_fill = data_filled['horsepower'].median()
print(f"Median after filling with mode: {median_after_fill}")

# Step 5: Check if it changed
print(f"\nComparison:")
if median_after_fill > median_original:
    print("Yes, it increased")
elif median_after_fill < median_original:
    print("Yes, it decreased")
else:
    print("No")

median_original
```

## Q7. Sum of Weights

1. Select all the cars from Asia
2. Select only columns vehicle_weight and model_year
3. Select the first 7 values
4. Get the underlying NumPy array. Let's call it X.
5. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
6. Invert XTX.
7. Create an array y with values [1100, 1300, 800, 900, 1000, 1100, 1200].
8. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
9. What's the sum of all the elements of the result?

```python
asia_cars = data[data['origin'] == 'Asia'][['vehicle_weight', 'model_year']].head(7)
X = asia_cars.to_numpy()
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print(f"Sum of all elements in w: {w.sum()}")
```
