# JPMC Virtual Internship Project Solutions

This repository contains a Jupyter Notebook solution for a natural gas price analysis and forecasting task from the JPMC virtual internship project.

## Project Overview

The notebook analyzes monthly natural gas prices and builds a simple forecasting workflow that can estimate the gas price for any given date.

The solution uses:

- Historical monthly natural gas price data
- Data cleaning and date parsing with Pandas
- Visualization with Matplotlib
- Holt-Winters seasonal forecasting
- Linear interpolation for date-based price estimation

## Files

| File | Description |
| --- | --- |
| `Jpmc_virtual_internship_programe_project_solutions.ipynb` | Main notebook containing the full analysis, forecast model, visualizations, and price estimation function |
| `Nat_Gas.csv` | Required dataset file. This file should contain `Dates` and `Prices` columns |

## Dataset

The notebook expects a CSV file named `Nat_Gas.csv`.

Required columns:

| Column | Description |
| --- | --- |
| `Dates` | Monthly date values for natural gas prices |
| `Prices` | Natural gas price values |

Place `Nat_Gas.csv` in the same folder as the notebook.

If running in Google Colab, upload the file to:

```text
/content/Nat_Gas.csv
```

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scipy statsmodels notebook
```

## How to Run

1. Open the notebook:

```bash
jupyter notebook Jpmc_virtual_internship_programe_project_solutions.ipynb
```

2. Make sure `Nat_Gas.csv` is available in the project folder.

3. Run each notebook cell from top to bottom.

## Notebook Workflow

The notebook performs the following steps:

1. Imports required libraries
2. Loads and prepares the natural gas price dataset
3. Visualizes historical prices
4. Builds a Holt-Winters seasonal forecasting model
5. Forecasts prices for the next 12 months
6. Combines historical and forecasted prices
7. Creates an interpolation function
8. Estimates prices for custom input dates

## Example Usage

After running the notebook cells, use:

```python
estimate_price("2025-03-15")
```

The function returns the estimated natural gas price for the given date, rounded to two decimal places.

## Notes

- The model uses monthly seasonality with `seasonal_periods=12`.
- Forecasts are estimates and should not be treated as financial advice.
- Dates outside the available historical and forecast range are extrapolated, so those estimates may be less reliable.
