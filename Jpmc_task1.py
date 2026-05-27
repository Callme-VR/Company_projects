from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def load_data(csv_path=None):
    """Load and clean the natural gas price dataset."""
    if csv_path is None:
        local_path = Path("Nat_Gas.csv")
        colab_path = Path("/content/Nat_Gas.csv")

        if local_path.exists():
            csv_path = local_path
        elif colab_path.exists():
            csv_path = colab_path
        else:
            raise FileNotFoundError(
                "Nat_Gas.csv was not found. Put it beside this script "
                "or upload it to /content/Nat_Gas.csv."
            )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df["Dates"] = pd.to_datetime(df["Dates"])
    df["Prices"] = pd.to_numeric(df["Prices"], errors="coerce")
    df = df.dropna(subset=["Dates", "Prices"])
    df = df.sort_values("Dates")
    df = df.set_index("Dates")

    return df


def build_forecast(df, forecast_steps=12):
    """Build a Holt-Winters seasonal model and forecast future prices."""
    model = ExponentialSmoothing(
        df["Prices"],
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True)

    forecast = model.forecast(forecast_steps)
    future_df = forecast.to_frame(name="Prices")
    combined_df = pd.concat([df, future_df])

    return model, future_df, combined_df


def create_price_function(combined_df):
    """Create an interpolation function for estimating prices by date."""
    x = combined_df.index.map(pd.Timestamp.toordinal).to_numpy()
    y = combined_df["Prices"].to_numpy()

    return interp1d(x, y, kind="linear", fill_value="extrapolate")


def estimate_price(date_string, price_function):
    """Estimate natural gas price for any date."""
    date = pd.to_datetime(date_string)
    estimated_price = float(price_function(date.toordinal()))
    return round(estimated_price, 2)


def plot_historical_prices(df):
    """Display historical natural gas prices."""
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df["Prices"], marker="o", linewidth=2)
    plt.title("Historical Natural Gas Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def plot_forecast(df, future_df):
    """Display historical and forecasted natural gas prices."""
    plt.figure(figsize=(14, 6))
    plt.plot(
        df.index,
        df["Prices"],
        label="Historical Prices",
        marker="o",
        linewidth=2,
    )
    plt.plot(
        future_df.index,
        future_df["Prices"],
        label="Forecasted Prices",
        marker="o",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Natural Gas Prices: Historical and Forecasted")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    df = load_data()

    print("Dataset preview:")
    print(df.head())

    _, future_df, combined_df = build_forecast(df, forecast_steps=12)
    price_function = create_price_function(combined_df)

    print("\nForecasted prices:")
    print(future_df)

    print("\nEstimated prices:")
    test_dates = ["2025-03-15", "2023-07-10", "2024-12-31"]
    for test_date in test_dates:
        price = estimate_price(test_date, price_function)
        print(f"{test_date}: {price}")

    plot_historical_prices(df)
    plot_forecast(df, future_df)


if __name__ == "__main__":
    main()
