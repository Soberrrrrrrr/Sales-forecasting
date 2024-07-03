from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np    
import streamlit as st
from dateutil.relativedelta import relativedelta

st.title("Sales Forecasting")
st.write("This forecasting model expects sales dataset with two columns Date and Sales with atleast three years of data to correctly capture seasonality and trend.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    dataset = pd.read_csv(uploaded_file)
    # Convert the Date column to datetime format
    dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)
    # Set the Date column as the index
    dataset.set_index('Date', inplace=True)

    # Aggregate sales by month
    monthly_sales = dataset['Sales']

    # Plot the aggregated sales data
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(monthly_sales, label='Sales')
    ax1.set_title('Aggregated Sales Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales')
    ax1.legend()

    # Adding labels to each data point
    for i in range(len(monthly_sales)):
        ax1.text(monthly_sales.index[i], monthly_sales.iloc[i], str(round(monthly_sales.iloc[i], 2)), fontsize=8, ha='center')

    st.pyplot(fig1)

    # Aggregate sales by month
    monthly_sales = dataset['Sales'].resample('M').sum()  # Assuming Sales is daily data, resampling to monthly sum

    # Seasonal Decomposition
    result = seasonal_decompose(monthly_sales, model='additive')
    fig2 = result.plot()
    st.pyplot(fig2)

    # Optional: Plot the individual components
    fig3, axs = plt.subplots(4, 1, figsize=(12, 8))
    axs[0].plot(result.observed, label='Observed')
    axs[0].legend(loc='upper left')
    axs[1].plot(result.trend, label='Trend')
    axs[1].legend(loc='upper left')
    axs[2].plot(result.seasonal, label='Seasonal')
    axs[2].legend(loc='upper left')
    axs[3].plot(result.resid, label='Residual')
    axs[3].legend(loc='upper left')
    fig3.tight_layout()
    st.pyplot(fig3)

    # Create the train and test sets
    train_size = int(len(monthly_sales) * 0.90)
    train, test = monthly_sales[0:train_size], monthly_sales[train_size:]

    # Fit the model on the training set with seasonal component
    model_train = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()

    # Forecast the test set
    forecast_test = model_train.forecast(steps=len(test))

    # Calculate MAPE
    mape = np.mean(np.abs((test - forecast_test) / test)) * 100
    st.write('Mean Absolute Percentage Error (MAPE):', mape)

    # Define the number of steps to forecast
    forecast_steps = 3

    # Get the last date in your dataset
    last_date = monthly_sales.index[-1]

    # Define the future dates for forecasting
    future_dates = [last_date + relativedelta(months=i) for i in range(1, forecast_steps + 1)]

    # Forecast for the future dates
    forecast_future = model_train.forecast(steps=forecast_steps)

    # Print or use forecast_future as needed
    st.write("Forecasted values for the next", forecast_steps, "months:")
    for date, forecast_value in zip(future_dates, forecast_future):
        st.write(date.strftime('%Y-%m'), ':', forecast_value)

    # Create a DataFrame with forecasted values and dates
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': forecast_future})
    forecast_df.set_index('Date', inplace=True)

    # Print the forecast DataFrame
    st.write("Forecasted values for the next", forecast_steps, "months:")
    st.write(forecast_df)

    # Visualize the forecasted values
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(monthly_sales.index, monthly_sales.values, label='Actual Sales')
    ax4.plot(forecast_df.index, forecast_df['Forecasted_Sales'], marker='o', linestyle='-', color='r', label='Forecasted Sales')
    ax4.set_title('Actual vs Forecasted Sales')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sales')
    ax4.legend()
    ax4.grid(True)
    fig4.tight_layout()
    st.pyplot(fig4)
