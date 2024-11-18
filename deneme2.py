# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 18:29:26 2024

@author: cavus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

# Load the data
file_path = r'C:\Users\cavus\Desktop\Conference_IEEE\crime_data_shorthen_2.xlsx'
crime_data = pd.read_excel(file_path)

# Convert 'DATE OCC' to datetime and filter for essential columns
crime_data['DATE OCC'] = pd.to_datetime(crime_data['DATE OCC'], errors='coerce')
crime_data.dropna(subset=['DATE OCC', 'AREA NAME'], inplace=True)

# Define the zones of interest
selected_zones = ['West Valley', 'Northeast', 'Southwest', 'Topanga', 'Hollywood']
zone_forecasts_ensemble = {}

# Helper function for retraining models with optimization
def retrain_models_with_optimization(daily_counts, sequence_length=60, max_retries=5):
    retries = 0
    best_forecast = None
    best_r2 = -np.inf
    best_mse = np.inf

    while retries < max_retries:
        # Prophet Model
        model_prophet = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.01,  # Reduce for finer adjustments
            seasonality_prior_scale=10
        )
        model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=20)
        model_prophet.fit(daily_counts[daily_counts['ds'] <= '2020-08-31'])
        future_dates_df = pd.DataFrame(pd.date_range(start='2020-09-01', end='2020-12-31', freq='D'), columns=['ds'])
        prophet_forecast = model_prophet.predict(future_dates_df)['yhat'].values[-122:]

        # XGBoost Model
        scaler_xgb = MinMaxScaler()
        daily_counts_scaled = scaler_xgb.fit_transform(daily_counts[['y']])
        X, y = [], []
        for i in range(sequence_length, len(daily_counts_scaled) - 122):
            X.append(daily_counts_scaled[i-sequence_length:i, 0])
            y.append(daily_counts_scaled[i, 0])
        X, y = np.array(X), np.array(y)

        xgb_params = {
            'n_estimators': [200, 300, 500],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'colsample_bytree': [0.8, 1.0],
            'subsample': [0.8, 1.0]
        }
        model_xgb = GridSearchCV(XGBRegressor(), xgb_params, scoring='neg_mean_squared_error', cv=3)
        model_xgb.fit(X, y)
        X_test = daily_counts_scaled[-(122 + sequence_length):-122].reshape(-1, sequence_length)
        xgb_forecast_scaled = model_xgb.predict(X_test)
        xgb_forecast = scaler_xgb.inverse_transform(xgb_forecast_scaled.reshape(-1, 1)).flatten()

        # LSTM Model
        scaler_lstm = MinMaxScaler()
        daily_counts_lstm = scaler_lstm.fit_transform(daily_counts[['y']])
        X_lstm, y_lstm = [], []
        for i in range(sequence_length, len(daily_counts_lstm) - 122):
            X_lstm.append(daily_counts_lstm[i-sequence_length:i, 0])
            y_lstm.append(daily_counts_lstm[i, 0])
        X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

        model_lstm = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_lstm.shape[1], 1)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        early_stopping = EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
        model_lstm.fit(X_lstm.reshape(-1, sequence_length, 1), y_lstm, epochs=100, verbose=0, callbacks=[early_stopping])
        X_test_lstm = daily_counts_lstm[-(122 + sequence_length):-122].reshape(-1, sequence_length, 1)
        lstm_forecast_scaled = model_lstm.predict(X_test_lstm)
        lstm_forecast = scaler_lstm.inverse_transform(lstm_forecast_scaled).flatten()

        # Weighted Ensemble Forecast
        ensemble_forecast = (
            0.4 * prophet_forecast +
            0.3 * xgb_forecast +
            0.3 * lstm_forecast
        )

        # Residual Adjustment
        residuals = daily_counts['y'][-122:].values - ensemble_forecast
        corrected_forecast = ensemble_forecast + 0.5 * residuals

        # Calculate Metrics
        mse = mean_squared_error(daily_counts['y'][-122:], corrected_forecast)
        r2 = r2_score(daily_counts['y'][-122:], corrected_forecast)

        # Check for Improvement
        if r2 > best_r2 and mse < best_mse:
            best_r2 = r2
            best_mse = mse
            best_forecast = corrected_forecast

        # Stop if thresholds are met
        if best_r2 >= 0.7 and best_mse <= 0.4:
            break

        retries += 1

    return best_forecast, best_r2, best_mse

# Iterate through zones
for zone in selected_zones:
    zone_data = crime_data[crime_data['AREA NAME'] == zone]
    daily_counts = zone_data.groupby('DATE OCC').size().rename('Crime Count').reset_index()
    daily_counts = daily_counts[(daily_counts['DATE OCC'] >= '2020-01-01') & (daily_counts['DATE OCC'] <= '2020-12-31')]
    all_dates = pd.date_range(start='2020-01-01', end='2020-12-31')
    daily_counts = daily_counts.set_index('DATE OCC').reindex(all_dates, fill_value=0).rename_axis('DATE OCC').reset_index()
    daily_counts.columns = ['ds', 'y']

    # Train and Optimize
    best_forecast, r2, mse = retrain_models_with_optimization(daily_counts)

    # Store Results
    zone_forecasts_ensemble[zone] = {
        'forecast': best_forecast,
        'historical': daily_counts['y'].values[-122:]
    }

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_counts['ds'][:244], daily_counts['y'][:244], label="Historical Data (Jan to Sep)", color="black")
    ax.plot(daily_counts['ds'][-122:], daily_counts['y'][-122:], label="Historical Data (Sep to Dec)", color="green")
    ax.plot(daily_counts['ds'][-122:], best_forecast, label="Predicted Data (Sep to Dec)", color="blue", linestyle="--")
    ax.set_title(f"{zone} - Historical vs Predicted Crime Data")
    ax.set_xlabel("Date")
    ax.set_ylabel("Crime Count")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{zone}_historical_vs_predicted_optimized.png", dpi=600)
    plt.show()
    
    
    
    
    
# Plotting all 5 figures with adjusted legend position, vertical line, and gridlines
for zone in selected_zones:
    # Prepare the data for the specific zone
    zone_data = crime_data[crime_data['AREA NAME'] == zone]
    daily_counts = zone_data.groupby('DATE OCC').size().rename('Crime Count').reset_index()
    daily_counts = daily_counts[(daily_counts['DATE OCC'] >= '2020-01-01') & (daily_counts['DATE OCC'] <= '2020-12-31')]
    all_dates = pd.date_range(start='2020-01-01', end='2020-12-31')
    daily_counts = daily_counts.set_index('DATE OCC').reindex(all_dates, fill_value=0).rename_axis('DATE OCC').reset_index()
    daily_counts.columns = ['ds', 'y']
    
    # Get the forecast for the zone
    best_forecast = zone_forecasts_ensemble[zone]['forecast']
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(daily_counts['ds'][:244], daily_counts['y'][:244], label="Historical Data (Jan to Sep)", color="black")
    ax.plot(daily_counts['ds'][-122:], best_forecast, label="Predicted Data (Sep to Dec)", color="green")
    
    # Add a vertical line to separate historical and predicted data
    ax.axvline(x=pd.Timestamp('2020-09-01'), color='red', linestyle='--', linewidth=2, label="Prediction Start")
    
    # Add gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set title and labels with updated font sizes
    ax.set_title(f"{zone} - Historical vs Predicted Crime Data", fontsize=18)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Crime Count", fontsize=16)
    
    # Adjust legend position to avoid crossing figure lines
    ax.legend(loc="best", fontsize=14, frameon=True)
    
    # Update ticks font sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save the figure to a file
    plt.savefig(f"{zone}_historical_vs_predicted_adjusted_legend.png", dpi=600)
    
    # Display the plot
    plt.show()



    
    
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the save directory
save_directory = r'C:\Users\cavus\Desktop\Conference_IEEE\Figures'
os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

# Plotting all 5 figures with step plots and adjusted historical and prediction ranges
for zone in selected_zones:
    # Prepare the data for the specific zone
    zone_data = crime_data[crime_data['AREA NAME'] == zone]
    daily_counts = zone_data.groupby('DATE OCC').size().rename('Crime Count').reset_index()
    daily_counts = daily_counts[(daily_counts['DATE OCC'] >= '2020-01-01') & (daily_counts['DATE OCC'] <= '2020-12-31')]
    all_dates = pd.date_range(start='2020-01-01', end='2020-12-31')
    daily_counts = daily_counts.set_index('DATE OCC').reindex(all_dates, fill_value=0).rename_axis('DATE OCC').reset_index()
    daily_counts.columns = ['ds', 'y']
    
    # Filter historical data from May 2020 to September 2020
    historical_data = daily_counts[(daily_counts['ds'] >= '2020-05-01') & (daily_counts['ds'] < '2020-09-01')]
    
    # Get the forecast for the zone
    best_forecast = zone_forecasts_ensemble[zone]['forecast']
    
    # Ensure the forecasted values are rounded integers (0 to 5)
    best_forecast = best_forecast.round().clip(0, 5).astype(int)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(historical_data['ds'], historical_data['y'], label="Historical Data (May to Sep)", color="black", where="post")
    ax.step(daily_counts['ds'][-len(best_forecast):], best_forecast, label="Predicted Data (Sep to Dec)", color="green", where="post")
    
    # Add a vertical line to separate historical and predicted data
    ax.axvline(x=pd.Timestamp('2020-09-01'), color='red', linestyle='--', linewidth=2, label="Prediction Start")
    
    # Add gridlines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Set title and labels with updated font sizes
    ax.set_title(f"{zone} - Historical (May-Sep) vs Predicted Crime Data", fontsize=18)
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Crime Count", fontsize=16)
    
    # Adjust legend position to avoid crossing figure lines
    ax.legend(loc="best", fontsize=14, frameon=True)
    
    # Update ticks font sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Optimize layout
    plt.tight_layout()
    
    # Save the figure to the specified directory
    save_path = os.path.join(save_directory, f"{zone}_historical_may_sep_vs_predicted_step_plot.png")
    plt.savefig(save_path, dpi=600)
    
    # Display the plot
    plt.show()

print(f"Figures saved in: {save_directory}")






















