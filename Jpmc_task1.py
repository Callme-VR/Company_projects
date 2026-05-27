import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Nat_Gas.csv")

# Convert dates
df['Dates'] = pd.to_datetime(df['Dates'])

# Sort values
df = df.sort_values('Dates')

# Convert dates into ordinal numbers for modeling
df['Ordinal'] = df['Dates'].map(pd.Timestamp.toordinal)

# Historical data
x = df['Ordinal'].values.reshape(-1, 1)
y = df['Prices'].values

# -------------------------------
# INTERPOLATION FOR PAST DATES
# -------------------------------

interp_function = interp1d(
    df['Ordinal'],
    y,
    kind='linear',
    fill_value='extrapolate'
)

# -------------------------------
# FUTURE FORECASTING
# -------------------------------

model = LinearRegression()
model.fit(x, y)

# Forecast next 12 months
future_dates = pd.date_range(
    start=df['Dates'].max(),
    periods=13,
    freq='M'
)[1:]

future_ordinals = np.array(
    [d.toordinal() for d in future_dates]
).reshape(-1, 1)

future_prices = model.predict(future_ordinals)

# -------------------------------
# PRICE ESTIMATION FUNCTION
# -------------------------------

def estimate_price(date_str):
    """
    Estimate natural gas price for any date.
    """
    input_date = pd.to_datetime(date_str)
    ordinal_date = input_date.toordinal()

    # If date is within historical range
    if input_date <= df['Dates'].max():
        estimated_price = float(interp_function(ordinal_date))
    else:
        estimated_price = float(
            model.predict([[ordinal_date]])[0]
        )

    return round(estimated_price, 2)

# Example usage
print("Estimated Price:",
      estimate_price("2025-06-30"))

# -------------------------------
# VISUALIZATION
# -------------------------------

plt.figure(figsize=(12, 6))

# Historical prices
plt.plot(
    df['Dates'],
    df['Prices'],
    marker='o',
    label='Historical Prices'
)

# Forecast prices
plt.plot(
    future_dates,
    future_prices,
    marker='x',
    linestyle='--',
    label='Forecast Prices'
)

plt.title("Natural Gas Price Analysis & Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.show()