# JPMC Task 1 Detailed Analysis

This file explains the natural gas price analysis according to the attached example answer document and the project notebook `Jpmc_virtual_internship_programe_project_solutions.ipynb`.

The main purpose of Task 1 is to create a function that can estimate the natural gas price for any given date.

The attached example answer and the notebook both solve the same business problem:

> Use historical monthly natural gas prices to estimate prices on custom dates, including dates between known observations and possible future dates.

## 1. Business Problem

Natural gas prices change over time because of demand, supply, weather, storage levels, and market conditions.

For this task, the available data contains monthly natural gas prices. Since only monthly values are available, the goal is to estimate prices for any specific date requested by a user.

For example:

```python
estimate_price("2025-03-15")
```

The model should return an estimated price for that date.

## 2. Dataset Explanation

The dataset contains two important columns:

| Column | Meaning |
| --- | --- |
| `Dates` | Date of the monthly natural gas price observation |
| `Prices` | Natural gas price recorded for that date |

This is time-series data because every price depends on time.

In the attached example answer, the dataset is loaded as:

```python
df = pd.read_csv('natgas_R.csv', parse_dates=['Dates'])
prices = df['Prices'].values
dates = df['Dates'].values
```

In the notebook, the dataset is loaded as:

```python
df = pd.read_csv(csv_path)
df['Dates'] = pd.to_datetime(df['Dates'])
df['Prices'] = pd.to_numeric(df['Prices'], errors='coerce')
```

Both versions do the same core task:

- load the CSV file,
- convert dates into date format,
- extract prices,
- prepare the data for analysis.

The only naming difference is that the attached answer uses `natgas_R.csv`, while the notebook uses `Nat_Gas.csv`.

## 3. Initial Data Analysis

The first step is to plot prices against dates.

Attached example:

```python
fig, ax = plt.subplots()
ax.plot_date(dates, prices, '-')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Natural Gas Prices')
ax.tick_params(axis='x', rotation=45)
plt.show()
```

Notebook:

```python
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df.index, df['Prices'], marker='o', linewidth=2)
ax.set_title('Historical Natural Gas Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
plt.show()
```

This graph is important because it gives the first understanding of the data.

From the attached example answer, the expected observation is:

> Natural gas prices have a natural frequency of around one year, but trend upwards.

This means the data likely has two patterns:

1. A long-term upward trend.
2. A repeating yearly seasonal pattern.

This makes sense because natural gas demand is often seasonal. Demand is usually higher in colder months because natural gas is used for heating.

## 4. Date Conversion for Modeling

Machine learning and mathematical models cannot directly understand dates like `2020-10-31`.

So the attached example converts dates into number of days from the starting date.

```python
start_date = date(2020,10,31)
end_date = date(2024,9,30)
```

Here:

- `start_date` is the first date in the dataset.
- `end_date` is the last date in the dataset.

The attached answer then creates all monthly dates between the start and end dates.

```python
months = []
year = start_date.year
month = start_date.month + 1
```

The loop creates month-end dates.

```python
current = date(year, month, 1) + timedelta(days=-1)
months.append(current)
```

This logic creates the last day of each month.

Example:

- If `year = 2020` and `month = 11`, then `date(2020, 11, 1) - 1 day` gives `2020-10-31`.
- This matches the monthly price date format.

Then the dates are converted into days from the start date:

```python
days_from_start = [(day - start_date ).days for day in months]
```

This creates numeric values like:

```text
0, 30, 61, 91, ...
```

These numbers are easier to use in regression and sine models.

In the notebook, a similar conversion happens later:

```python
x = combined_df.index.map(pd.Timestamp.toordinal).to_numpy()
```

The difference is:

- Attached answer uses days from the start date.
- Notebook uses ordinal date numbers.

Both methods convert dates into numeric form.

## 5. Linear Trend Analysis

The attached example first models the long-term trend using simple linear regression.

The assumed trend equation is:

```text
y = A x + B
```

Where:

- `y` is the natural gas price,
- `x` is time in days,
- `A` is the slope,
- `B` is the intercept.

The attached answer defines:

```python
def simple_regression(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar)**2)
    intercept = ybar - slope * xbar
    return slope, intercept
```

Line-by-line meaning:

```python
xbar = np.mean(x)
```

This calculates the average time value.

```python
ybar = np.mean(y)
```

This calculates the average price.

```python
slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar)**2)
```

This calculates how much the price changes as time increases.

If the slope is positive, prices generally increase over time.

If the slope is negative, prices generally decrease over time.

```python
intercept = ybar - slope * xbar
```

This calculates the starting value of the regression line.

```python
return slope, intercept
```

This returns the trend model.

Then the attached answer applies the function:

```python
time = np.array(days_from_start)
slope, intercept = simple_regression(time, prices)
```

This fits the trend line to the natural gas price data.

## 6. Trend Visualization

The attached answer plots the original prices and the linear trend:

```python
plt.plot(time, prices)
plt.plot(time, time * slope + intercept)
plt.xlabel('Days from start date')
plt.ylabel('Price')
plt.title('Linear Trend of Monthly Input Prices')
plt.show()
```

This graph compares:

- actual monthly prices,
- estimated long-term trend.

The purpose is to check whether the linear trend captures the general movement of the data.

The attached answer says:

> From this plot we see the linear trend has been captured.

That means the model identifies the overall upward or downward movement in prices.

## 7. Seasonal Pattern Analysis

After removing the linear trend, the attached answer models the remaining variation as a yearly sine wave.

```python
sin_prices = prices - (time * slope + intercept)
```

This removes the trend from the original prices.

The result, `sin_prices`, contains the seasonal variation.

If the trend explains the general direction, the remaining values explain the repeating yearly ups and downs.

The attached answer assumes one yearly cycle:

```python
sin_time = np.sin(time * 2 * np.pi / (365))
cos_time = np.cos(time * 2 * np.pi / (365))
```

Meaning:

- `365` represents one year.
- `sin` and `cos` represent repeating yearly movement.
- This captures the idea that prices may rise and fall once per year.

This is useful because natural gas prices often have yearly seasonality.

## 8. Bilinear Regression for Seasonality

The attached answer defines:

```python
def bilinear_regression(y, x1, x2):
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return(slope1, slope2)
```

This fits the seasonal variation using sine and cosine terms.

```python
slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)
```

This calculates the sine and cosine coefficients.

The model then calculates amplitude and phase shift:

```python
amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift = np.arctan2(slope2, slope1)
```

Meaning:

- `amplitude` controls how large the seasonal price movement is.
- `shift` controls where the seasonal cycle starts.

In simple terms:

```text
Trend explains long-term movement.
Sine wave explains yearly seasonal movement.
```

## 9. Full Model in Attached Example

The attached answer combines trend and seasonality.

The final model is:

```text
estimated price = trend + seasonal component
```

Or:

```text
price = slope * days + intercept + amplitude * sin(days * 2pi / 365 + shift)
```

This model says:

- price changes gradually over time,
- price also rises and falls in a yearly pattern.

This is a strong approach for this task because it directly matches the observed behavior of natural gas prices.

## 10. Interpolation and Extrapolation

The attached answer creates a function:

```python
def interpolate(date):
    days = (date - pd.Timestamp(start_date)).days
    if days in days_from_start:
        return prices[days_from_start.index(days)]
    else:
        return amplitude * np.sin(days * 2 * np.pi / 365 + shift) + days * slope + intercept
```

Line-by-line meaning:

```python
days = (date - pd.Timestamp(start_date)).days
```

This converts the input date into days from the starting date.

```python
if days in days_from_start:
```

This checks whether the requested date exists exactly in the original monthly data.

```python
return prices[days_from_start.index(days)]
```

If the exact date exists, the function returns the original known price.

```python
else:
    return amplitude * np.sin(days * 2 * np.pi / 365 + shift) + days * slope + intercept
```

If the date is not directly available, the function estimates the price using the trend plus seasonal sine model.

This function is the final pricing tool in the attached example.

## 11. Notebook Method Compared With Attached Example

Your notebook uses Holt-Winters Exponential Smoothing:

```python
model = ExponentialSmoothing(
    df['Prices'],
    trend='add',
    seasonal='add',
    seasonal_periods=12,
    initialization_method='estimated'
).fit(optimized=True)
```

This is different from the attached example, but the idea is similar.

| Part | Attached Example | Notebook |
| --- | --- | --- |
| Trend | Manual linear regression | Holt-Winters additive trend |
| Seasonality | Sine and cosine yearly cycle | Holt-Winters additive seasonality |
| Frequency | 365 days | 12 monthly periods |
| Estimation | Custom sine + trend function | Forecast + interpolation |
| Final output | `interpolate(date)` | `estimate_price(date_string)` |

Both methods are valid for this task.

The attached answer explains the math manually.

The notebook uses a standard time-series library to do the same type of work more directly.

## 12. Why Seasonality Matters

Natural gas is not consumed evenly throughout the year.

Demand often increases during winter because gas is used for heating.

Demand may reduce during milder months.

Because of this, prices can show a repeating yearly cycle.

Ignoring seasonality may create poor forecasts because the model would only follow the long-term trend and miss yearly price changes.

## 13. Why Trend Matters

The trend captures whether prices are generally increasing or decreasing across the full dataset.

For example, if prices are gradually increasing from 2020 to 2024, the model should not treat every year as having the same average price.

The trend helps the model estimate future prices more realistically.

## 14. Why Interpolation Is Needed

The dataset contains monthly prices.

But a user may ask for a price on any date, such as:

```text
2023-07-10
```

This date may not exist exactly in the monthly dataset.

Interpolation estimates the value between known points.

In the notebook:

```python
price_function = interp1d(x, y, kind='linear', fill_value='extrapolate')
```

This creates a smooth date-to-price estimator.

## 15. Proper Final Analysis

The natural gas price data should be interpreted as monthly time-series data with both trend and seasonality.

The historical plot helps identify the overall behavior of the prices.

The attached example observes that prices follow a yearly seasonal pattern and have an upward trend.

The attached example then separates the problem into two parts:

1. Fit the long-term trend using linear regression.
2. Fit the yearly variation using a sine wave.

Your notebook performs the same conceptual task using Holt-Winters Exponential Smoothing.

The model learns:

- the general direction of prices,
- the repeating monthly seasonal pattern,
- the expected future values for the next 12 months.

After forecasting, the notebook combines historical and forecasted prices.

Then it builds an interpolation function so the user can estimate prices on any requested date.

## 16. Strengths of the Project

The project is strong because:

- It treats the data as time-series data.
- It converts dates properly.
- It visualizes historical behavior.
- It uses seasonality in the forecast.
- It provides a reusable price estimation function.
- It can estimate prices for dates that are not directly present in the dataset.

## 17. Limitations

The model is based only on historical price behavior.

It does not include external variables such as:

- weather changes,
- geopolitical events,
- natural gas storage levels,
- production changes,
- economic demand,
- supply disruptions,
- policy or regulation changes.

Because of this, the forecast should be treated as an estimate.

It is useful for analysis and approximation, but not guaranteed to predict real market prices exactly.

## 18. Final Conclusion

This JPMC Task 1 project builds a natural gas price estimation tool.

The attached example answer explains the solution using manual mathematical modeling:

```text
price = linear trend + yearly sinusoidal seasonality
```

The notebook solves the same problem using Holt-Winters forecasting:

```text
price = learned trend + learned monthly seasonality
```

Both approaches recognize the same important point:

> Natural gas prices are time-dependent and seasonal.

The final result is a function that can estimate the natural gas price for any date.

This is useful for contract pricing, financial analysis, and understanding future price exposure.
