import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go

st.title("Sales Forecasting and Price Elasticity Analysis")
st.write("This forecasting model expects a sales dataset with columns: Date, Sales, Region, and Sales_Price_Per_Unit, with at least three years of data to correctly capture seasonality and trend.")
st.write("Note: Sales should be in tonnes")

# Sample data
sample_data = pd.DataFrame({
    'Date': pd.date_range(start='2021-01-01', periods=3, freq='M'),
    'Sales': [1000, 1500, 1300],
    'Region': ['Region1'] * 3,
    'Sales_Price_Per_Unit': [10, 11, 12]
})

# Display sample data
st.write("Sample Data (Correct format with columns: Date, Sales, Region, and Sales_Price_Per_Unit):")
st.write(sample_data)

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    dataset = pd.read_csv(uploaded_file)
    
    # Input validation and error handling
    required_columns = ['Date', 'Sales', 'Region', 'Sales_Price_Per_Unit']
    if not all(column in dataset.columns for column in required_columns):
        st.error(f"Uploaded CSV must contain the following columns: {required_columns}")
    else:
        # Convert the Date column to datetime format
        dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True)
        # Set the Date column as the index
        dataset.set_index('Date', inplace=True)

        # Ensure numeric values
        dataset['Sales'] = pd.to_numeric(dataset['Sales'], errors='coerce')
        dataset['Sales_Price_Per_Unit'] = pd.to_numeric(dataset['Sales_Price_Per_Unit'], errors='coerce')
        dataset.dropna(inplace=True)

        # Select the region
        regions = dataset['Region'].unique()
        selected_region = st.selectbox("Select Region", regions)

        # Filter data by the selected region
        regional_data = dataset[dataset['Region'] == selected_region]
        monthly_sales = regional_data['Sales'].resample('M').sum()
        monthly_price = regional_data['Sales_Price_Per_Unit'].resample('M').mean()

        if not monthly_sales.empty:
            # Plot the aggregated sales data
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(monthly_sales, label='Sales')
            ax1.set_title(f'Aggregated Sales Over Time for {selected_region}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Sales')
            ax1.legend()

            for i in range(len(monthly_sales)):
                ax1.text(monthly_sales.index[i], monthly_sales.iloc[i], str(round(monthly_sales.iloc[i], 2)), fontsize=8, ha='center')

            st.pyplot(fig1)

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
            st.write(f'Mean Absolute Percentage Error (MAPE) for {selected_region}:', mape)

            # Define the number of steps to forecast
            forecast_steps = 3

            # Get the last date in your dataset
            last_date = monthly_sales.index[-1]

            # Define the future dates for forecasting
            future_dates = [last_date + relativedelta(months=i) for i in range(1, forecast_steps + 1)]

            # Forecast for the future dates
            forecast_future = model_train.forecast(steps=forecast_steps)

            # Print or use forecast_future as needed
            st.write(f"Forecasted values for the next {forecast_steps} months for {selected_region}:")
            for date, forecast_value in zip(future_dates, forecast_future):
                st.write(date.strftime('%Y-%m'), ':', forecast_value)

            # Create a DataFrame with forecasted values and dates
            forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': forecast_future})
            forecast_df.set_index('Date', inplace=True)

            # Print the forecast DataFrame
            st.write(forecast_df)

            # Visualize the forecasted values
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.plot(monthly_sales.index, monthly_sales.values, label='Actual Sales')
            ax4.plot(forecast_df.index, forecast_df['Forecasted_Sales'], marker='o', linestyle='-', color='r', label='Forecasted Sales')
            ax4.set_title(f'Actual vs Forecasted Sales for {selected_region}')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Sales')
            ax4.legend()
            ax4.grid(True)
            fig4.tight_layout()
            st.pyplot(fig4)

            # Calculate price elasticity
            change_in_qt_demand = monthly_sales.diff()
            change_in_price = monthly_price.diff()
            avg_price = monthly_price.rolling(window=2).mean()
            avg_quantity = monthly_sales.rolling(window=2).mean()

            price_elasticity = (change_in_qt_demand / change_in_price) * (avg_price / avg_quantity)

            # Combine all relevant metrics into a DataFrame
            metrics_df = pd.DataFrame({
                'Sales': monthly_sales,
                'Sales_Price_Per_Unit': monthly_price,
                'Change_in_Qty_Demand': change_in_qt_demand,
                'Change_in_Price': change_in_price,
                'Average_Price': avg_price,
                'Average_Quantity': avg_quantity,
                'Price_Elasticity': price_elasticity
            })

            st.write(f"Metrics for {selected_region}:")
            st.write(metrics_df)

            # Visualize the price elasticity
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            ax5.plot(price_elasticity, label='Price Elasticity', color='g')
            ax5.set_title(f'Price Elasticity Over Time for {selected_region}')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Price Elasticity')
            ax5.legend()
            ax5.grid(True)
            fig5.tight_layout()
            st.pyplot(fig5)

            # Correlation Analysis
            st.write("### Correlation Analysis")
            correlation_matrix = regional_data[['Sales', 'Sales_Price_Per_Unit']].corr()
            st.write("Correlation Matrix:")
            st.write(correlation_matrix)

            # Generate dynamic explanation
            correlation_explanation = ""
            if 'Sales' in correlation_matrix.columns and 'Sales_Price_Per_Unit' in correlation_matrix.columns:
                correlation_value = correlation_matrix.loc['Sales', 'Sales_Price_Per_Unit']
                if correlation_value > 0.5:
                    correlation_explanation = (
                        f"The correlation between Sales and Sales Price Per Unit is {correlation_value:.2f}, "
                        "which indicates a strong positive relationship. This means that as the sales price per unit increases, "
                        "the sales tend to increase as well, or vice versa."
                    )
                elif correlation_value < -0.5:
                    correlation_explanation = (
                        f"The correlation between Sales and Sales Price Per Unit is {correlation_value:.2f}, "
                        "which indicates a strong negative relationship. This means that as the sales price per unit increases, "
                        "the sales tend to decrease, or vice versa."
                    )
                else:
                    correlation_explanation = (
                        f"The correlation between Sales and Sales Price Per Unit is {correlation_value:.2f}, "
                        "which indicates a weak or negligible relationship. Changes in the sales price per unit have little to no impact "
                        "on sales."
                    )

            st.write("### Correlation Explanation")
            st.write(correlation_explanation)

            # Scenario Analysis
            st.write("### Scenario Analysis")
            price_change = st.slider("Change in Price", min_value=-20, max_value=20, value=0)
            price_change_df = pd.DataFrame({
                'Original_Price': monthly_price,
                'Change_in_Price': price_change
            })
            price_change_df['New_Price'] = price_change_df['Original_Price'] + price_change
            price_change_df['Projected_Sales'] = monthly_sales * (1 + (price_change_df['Change_in_Price'] / price_change_df['Original_Price']) * price_elasticity)

            st.write("Scenario Analysis Table:")
            st.write(price_change_df)

            # Revenue Impact
            st.write("### Revenue Impact")
            regional_data['Revenue'] = regional_data['Sales'] * regional_data['Sales_Price_Per_Unit']
            st.write("Revenue Impact Table:")
            st.write(regional_data[['Sales', 'Sales_Price_Per_Unit', 'Revenue']])

            # Revenue Forecasting
            forecast_revenue = forecast_df['Forecasted_Sales'] * price_change_df['New_Price'].iloc[-1]
            forecast_df['Forecasted_Revenue'] = forecast_revenue

            st.write(f"Forecasted Revenue for the next {forecast_steps} months:")
            st.write(forecast_df[['Forecasted_Sales', 'Forecasted_Revenue']])

            # Interactive Charts
            st.write("### Interactive Sales and Forecast Visualization")
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales.values, mode='lines', name='Actual Sales'))
            fig7.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecasted_Sales'], mode='lines', name='Forecasted Sales', line=dict(color='red')))
            fig7.update_layout(title=f'Interactive Sales and Forecast Visualization for {selected_region}', xaxis_title='Date', yaxis_title='Sales')
            st.plotly_chart(fig7)

            # Dynamic Tables
            st.write("### Metrics Table")
            st.dataframe(metrics_df)

            # Optional: Display historical vs. forecasted sales comparison
            st.write("### Historical vs Forecasted Sales Comparison")
            combined_df = pd.concat([monthly_sales, forecast_df], axis=1).rename(columns={'Sales': 'Actual_Sales'})
            combined_df.reset_index(inplace=True)
            st.write(combined_df[['Date', 'Actual_Sales', 'Forecasted_Sales']])

            # Convert DataFrame to CSV
            csv = combined_df.to_csv(index=False)

            # Add a download button
            st.download_button(
                label="Download Data for selected region with original sales and the next 3 months forecasted sales",
                data=csv,
                file_name=f'{selected_region}_sales_forecast.csv',
                mime='text/csv'
            )
        else:
            st.write("No data available for the selected region.")
else:
    st.write("No CSV file is uploaded for forecasting; hence no file is available to download.")
