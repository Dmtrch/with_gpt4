
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Load the file to read the data
file_path = 'book.xlsx'
df = pd.read_excel(file_path)

# Set the first column (dates) as the index
df.set_index(df.columns[0], inplace=True)

# Rename the index for clarity and infer the frequency
df.index.rename('Date', inplace=True)
df.index = pd.DatetimeIndex(df.index).to_period('M')

# Selecting exogenous and target time series based on specified columns
exog_series = df.iloc[:, [0, 1, 2, 3]]

# Target time series: columns G, J, L (7th, 10th, 12th columns)
target_series = df.iloc[:, [5, 8, 10]]


# Function to train and forecast using SARIMAX
def train_and_forecast(target, exog, steps=12):
    model = SARIMAX(target, exog=exog, order=(3, 1, 2), seasonal_order=(3, 1, 2, 4),
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False, maxiter=200)
    forecast = model_fit.forecast(steps=steps, exog=exog.iloc[-steps:])
    return forecast

# Training and forecasting for each target time series
forecasts = {}
for column in target_series.columns:
    print(f"Forecasting for {column}...")
    forecasts[column] = train_and_forecast(target_series[column], exog_series)

# Plotting the target time series and their forecasts
for column in target_series.columns:
    plt.figure(figsize=(10, 6))
    # Convert PeriodIndex to DateTimeIndex for plotting
    target_dates = target_series.index.to_timestamp()
    plt.plot(target_dates, target_series[column], label='Actual')

    # Ensure the forecast index starts immediately after the last actual data point
    last_actual_date = target_series.index[-1].to_timestamp()
    forecast_index = pd.date_range(start=last_actual_date, periods=len(forecasts[column]) + 1, freq='M')[1:]

    plt.plot(forecast_index, forecasts[column], label='Forecast', color='orange')

    plt.title(f"Time Series and Forecast for {column}")
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
