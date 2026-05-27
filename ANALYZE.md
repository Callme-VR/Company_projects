# Natural Gas Price Analysis

This document explains the analysis performed in `Jpmc_virtual_internship_programe_project_solutions.ipynb` for the JPMC virtual internship task.

## Project Objective

The objective of this project is to analyze historical monthly natural gas prices and estimate the natural gas price for any given date.

The notebook solves this by:

1. Loading the historical natural gas dataset.
2. Cleaning and preparing the data.
3. Visualizing historical price movement.
4. Building a time-series forecasting model.
5. Forecasting the next 12 months of prices.
6. Creating a function that estimates price for any input date.

## Dataset Description

The dataset is expected to be stored in a file named `Nat_Gas.csv`.

It contains monthly natural gas price records.

| Column | Description |
| --- | --- |
| `Dates` | The monthly date for each natural gas price observation |
| `Prices` | The natural gas price recorded for that date |

This is a time-series dataset because each price value is connected to a date and the order of dates matters.

Natural gas prices are affected by seasonal demand. For example, demand may rise during winter because gas is used for heating. Because of this, the dataset can show repeating yearly patterns.

## Step 1: Import Libraries

```python
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.tsa.holtwinters import ExponentialSmoothing
```

The notebook imports the libraries needed for the project.

`Path` is used to manage file paths.

`pandas` is used to load and clean the dataset.

`numpy` is used for numerical operations.

`matplotlib` is used to draw graphs.

`interp1d` is used to estimate prices between known monthly dates.

`ExponentialSmoothing` is used to build the Holt-Winters forecasting model.

## Step 2: Load and Prepare the Data

```python
local_path = Path('Nat_Gas.csv')
colab_path = Path('/content/Nat_Gas.csv')
```

The notebook first checks where the dataset is located.

If the notebook is running locally, it expects `Nat_Gas.csv` in the same folder.

If the notebook is running in Google Colab, it checks `/content/Nat_Gas.csv`.

```python
df = pd.read_csv(csv_path)
```

This line loads the CSV file into a pandas DataFrame called `df`.

```python
df.columns = df.columns.str.strip()
```

This removes extra spaces from column names. It helps avoid errors if the CSV has column names like `" Dates "` instead of `"Dates"`.

```python
df['Dates'] = pd.to_datetime(df['Dates'])
```

This converts the `Dates` column into a proper datetime format.

This is important because forecasting models need dates to be handled correctly.

```python
df['Prices'] = pd.to_numeric(df['Prices'], errors='coerce')
```

This converts the `Prices` column into numeric values.

If a value cannot be converted into a number, it becomes missing data.

```python
df = df.dropna(subset=['Dates', 'Prices']).sort_values('Dates')
```

This removes rows where either the date or price is missing.

Then it sorts all records by date.

Sorting is necessary because time-series data must be in chronological order.

```python
df = df.set_index('Dates')
```

This sets the date column as the DataFrame index.

This makes it easier to plot and forecast the data as a time series.

## Step 3: Visualize Historical Prices

```python
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['Prices'], marker='o', linewidth=2)
```

This creates a line chart of historical natural gas prices.

The x-axis represents dates.

The y-axis represents natural gas prices.

The graph helps identify:

- Overall price trend
- Seasonal movement
- Sudden price increases or decreases
- Cyclical behavior across months

This visualization is important because it gives an initial understanding of the dataset before forecasting.

## Step 4: Build the Forecasting Model

```python
model = ExponentialSmoothing(
    df['Prices'],
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    initialization_method='estimated'
).fit(optimized=True)
```

This creates a Holt-Winters Exponential Smoothing model.

The model is appropriate for this task because natural gas prices can contain both trend and seasonality.

`df['Prices']` tells the model to learn from the price column.

`trend='add'` means the model includes an additive trend. This allows prices to move upward or downward over time.

`seasonal='add'` means the model includes additive seasonality. This allows the model to capture repeating monthly patterns.

`seasonal_periods=12` means one complete seasonal cycle has 12 observations. Since the dataset is monthly, 12 observations represent one year.

`initialization_method='estimated'` allows the model to estimate its initial values.

`.fit(optimized=True)` trains the model and chooses good smoothing parameters automatically.

## Step 5: Forecast Future Prices

```python
forecast_steps = 12
forecast = model.forecast(forecast_steps)
```

This forecasts natural gas prices for the next 12 months.

The forecast continues from the last date available in the dataset.

```python
future_df = forecast.to_frame(name='Prices')
```

This converts the forecast output into a DataFrame with a column named `Prices`.

```python
combined_df = pd.concat([df, future_df])
```

This combines the historical prices and forecasted prices into one DataFrame.

This combined data is used later for estimating prices on any requested date.

## Step 6: Plot Historical and Forecasted Prices

```python
ax.plot(df.index, df['Prices'], label='Historical Prices', marker='o', linewidth=2)
ax.plot(future_df.index, future_df['Prices'], label='Forecasted Prices', marker='o', linestyle='--', linewidth=2)
```

This graph shows historical and forecasted prices together.

The historical prices show actual data from the dataset.

The forecasted prices show the model's predicted values for the next 12 months.

The dashed forecast line makes it easy to separate actual values from predicted values.

This plot helps check whether the forecast follows a reasonable continuation of the historical trend and seasonal pattern.

## Step 7: Create a Price Estimation Function

```python
x = combined_df.index.map(pd.Timestamp.toordinal).to_numpy()
y = combined_df['Prices'].to_numpy()
```

The dates are converted into ordinal numbers because interpolation works with numerical inputs.

`x` stores the numeric date values.

`y` stores the corresponding natural gas prices.

```python
price_function = interp1d(x, y, kind='linear', fill_value='extrapolate')
```

This creates a linear interpolation function.

The function estimates a price between known monthly dates.

For example, if the dataset has prices for January 31 and February 28, the function can estimate a price for February 15.

`fill_value='extrapolate'` allows the function to estimate prices outside the known date range.

## Step 8: Estimate Price for Any Date

```python
def estimate_price(date_string):
```

This defines a function that accepts a date as input.

```python
date = pd.to_datetime(date_string)
```

This converts the input date into a pandas datetime value.

```python
estimated_price = float(price_function(date.toordinal()))
```

This converts the date into a numeric ordinal value and estimates the price using the interpolation function.

```python
return round(estimated_price, 2)
```

This returns the estimated price rounded to two decimal places.

## Step 9: Test the Function

```python
test_dates = ['2025-03-15', '2023-07-10', '2024-12-31']
```

These dates are used to test the estimation function.

```python
for test_date in test_dates:
    print(f"Estimated price on {test_date}: {estimate_price(test_date)}")
```

This prints the estimated price for each test date.

Dates inside the historical range are estimated using known historical data.

Dates in the forecast range are estimated using forecasted data.

## Final Analysis

The notebook successfully creates a complete workflow for natural gas price analysis and estimation.

It starts by cleaning monthly historical data, then visualizes price behavior, builds a seasonal forecasting model, and creates a reusable function for estimating gas prices on custom dates.

The Holt-Winters model is suitable because the data is monthly and natural gas prices can show seasonal behavior.

The interpolation step is useful because the dataset contains monthly observations, but users may ask for prices on any date, not only month-end dates.

## Limitations

The forecast is based only on historical price patterns.

It does not include external market factors such as:

- Weather shocks
- Supply disruptions
- Storage levels
- Geopolitical events
- Changes in demand
- Regulatory changes

Because of this, the results should be treated as estimates, not guaranteed future prices.

Also, extrapolated values outside the historical and forecasted range may be less reliable.

## Conclusion

This project provides a practical solution for estimating natural gas prices using time-series forecasting.

The notebook is useful for understanding past price behavior, forecasting near-term future prices, and building a simple price lookup function for contract or trading analysis.
