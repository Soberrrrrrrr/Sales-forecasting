# Sales Forecasting App

This Streamlit app allows you to forecast sales using time series analysis. The app supports uploading a CSV file containing sales data, performs seasonal decomposition, and provides a sales forecast for the next few months.

## Features

- **File Upload**: Drag and drop a CSV file containing your sales data.
- **Sales Aggregation**: Aggregates sales data by month.
- **Seasonal Decomposition**: Decomposes the sales data into observed, trend, seasonal, and residual components.
- **Forecasting**: Uses the Holt-Winters Exponential Smoothing method to forecast sales for the next three months.
- **Visualization**: Plots the original sales data, the decomposed components, and the forecasted sales.

## Requirements

- Python 3.x
- Streamlit
- Pandas
- Numpy
- Matplotlib
- Statsmodels
- Scikit-learn
- Dateutil

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/colgate-forecasting.git
    cd colgate-forecasting
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. Upload your CSV file by using the drag-and-drop feature in the app.
3. View the aggregated sales data, decomposed components, and the forecasted sales.

## Sample CSV Format

The CSV file should contain at least two columns: `Date` and `Sales`. The `Date` column should be in a format that can be parsed by `pandas.to_datetime()`, and the `Sales` column should contain the sales data.

Example:
```csv
Date,Sales(kg)
01-01-2021,100
02-01-2021,150
...
