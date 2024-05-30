import warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import hilbert, find_peaks
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor, DMatrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging


# Download earthquake data
"""
Fetches earthquake data from the USGS API within the specified parameters.
    Parameters:
    - start_time (str): Start date in the format 'YYYY-MM-DD'.
    - end_time (str): End date in the format 'YYYY-MM-DD'.
    - min_magnitude (float): Minimum magnitude of earthquakes to fetch.
    - min_latitude (float): Minimum latitude of the bounding box.
    - max_latitude (float): Maximum latitude of the bounding box.
    - min_longitude (float): Minimum longitude of the bounding box.
    - max_longitude (float): Maximum longitude of the bounding box.
"""
start_time = "1900-01-01"
end_time = "2024-05-29"
min_magnitude = 3.0
min_latitude = 36.6237
max_latitude = 40.2237
min_longitude = 24.8928
max_longitude = 29.3928

# Set up logging
logging.basicConfig(filename='earthquake_prediction.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Capture warnings and log them


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = logging.getLogger()
    log.warning('%s:%s: %s:%s', filename, lineno, category.__name__, message)


warnings.showwarning = warn_with_traceback

# Function to fetch earthquake data from USGS API


def fetch_earthquake_data(start_time, end_time, min_magnitude, min_latitude, max_latitude, min_longitude, max_longitude):
    """
    Returns:
    - pd.DataFrame: DataFrame containing the fetched earthquake data.
    """
    try:
        url = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query"
            "?format=geojson"
            f"&starttime={start_time}"
            f"&endtime={end_time}"
            f"&minmagnitude={min_magnitude}"
            f"&minlatitude={min_latitude}"
            f"&maxlatitude={max_latitude}"
            f"&minlongitude={min_longitude}"
            f"&maxlongitude={max_longitude}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        earthquakes = pd.DataFrame(data['features'])
        earthquakes['magnitude'] = earthquakes['properties'].apply(lambda x: x['mag'])
        earthquakes['place'] = earthquakes['properties'].apply(lambda x: x['place'])
        earthquakes['time'] = earthquakes['properties'].apply(lambda x: pd.to_datetime(x['time'], unit='ms'))
        earthquakes['latitude'] = earthquakes['geometry'].apply(lambda x: x['coordinates'][1])
        earthquakes['longitude'] = earthquakes['geometry'].apply(lambda x: x['coordinates'][0])
        earthquakes['depth'] = earthquakes['geometry'].apply(lambda x: x['coordinates'][2])
        logging.info(f'Fetched {len(earthquakes)} earthquake records from USGS API.')
        return earthquakes[['magnitude', 'place', 'time', 'latitude', 'longitude', 'depth']]
    except Exception as e:
        logging.error("Error fetching earthquake data", exc_info=True)
        raise e


# File paths
if not os.path.exists('data'):
    os.makedirs('data')

data_file = "earthquake_data.csv"
data_file_path = os.path.join('data', data_file)

# Check if data file exists
try:
    if not os.path.exists(data_file):

        earthquakes = fetch_earthquake_data(start_time, end_time, min_magnitude, min_latitude, max_latitude, min_longitude, max_longitude)
        print(f'Total Earthquakes Fetched: {len(earthquakes)}')

        # Save data to CSV
        earthquakes.to_csv(data_file_path, index=False)

        logging.info(f'Earthquake data saved to {data_file_path}')
    else:
        # Load data from CSV
        earthquakes = pd.read_csv(data_file_path)
        earthquakes['time'] = pd.to_datetime(earthquakes['time'])
        print(f'Earthquake data loaded from {data_file_path}')
        logging.info(f'Earthquake data loaded from {data_file_path}')
except Exception as e:
    logging.error("Error loading or saving earthquake data", exc_info=True)
    raise e

# Calculate time difference between consecutive earthquakes
earthquakes = earthquakes.sort_values('time')
earthquakes['time_diff'] = earthquakes['time'].diff().dt.total_seconds().fillna(0)

# Advanced Feature Engineering


def extract_advanced_features(df):
    """
    Extracts advanced features for earthquake prediction.
    Parameters:
    - df (pd.DataFrame): DataFrame containing the earthquake data.

    Returns:
    - pd.DataFrame: DataFrame containing the extracted features.
    """
    try:
        features = pd.DataFrame()
        features['max_acceleration'] = df['magnitude']
        features['min_acceleration'] = df['depth']
        features['mean_acceleration'] = df['magnitude'].expanding().mean()
        features['std_acceleration'] = df['magnitude'].expanding().std()

        zero_crossings = np.where(np.diff(np.sign(df['magnitude'])))[0]
        features['zero_crossings'] = len(zero_crossings)

        peaks, _ = find_peaks(df['magnitude'])
        features['peak_count'] = len(peaks)

        fft_values = fft(df['magnitude'].to_numpy())
        features['fft_mean'] = np.mean(np.abs(fft_values))
        features['fft_std'] = np.std(np.abs(fft_values))

        analytic_signal = hilbert(df['magnitude'].to_numpy())
        amplitude_envelope = np.abs(analytic_signal)
        features['envelope_mean'] = np.mean(amplitude_envelope)
        features['envelope_std'] = np.std(amplitude_envelope)

        features['latitude'] = df['latitude']
        features['longitude'] = df['longitude']

        features['month'] = df['time'].dt.month
        features['day_of_year'] = df['time'].dt.dayofyear

        logging.info("Advanced features extracted")
        return features
    except Exception as e:
        logging.error("Error in feature extraction", exc_info=True)
        raise e


try:
    features = extract_advanced_features(earthquakes)

    # Prepare data for model training and prediction
    earthquakes['previous_magnitude'] = earthquakes['magnitude'].shift(1).fillna(0)
    earthquakes['previous_time_diff'] = earthquakes['time_diff'].shift(1).fillna(0)
    earthquakes['next_magnitude'] = earthquakes['magnitude'].shift(-1).fillna(0)
    earthquakes['next_time_diff'] = earthquakes['time_diff'].shift(-1).fillna(0)

    data = earthquakes[['previous_magnitude', 'previous_time_diff', 'next_magnitude', 'next_time_diff']].dropna()
    X = data[['previous_magnitude', 'previous_time_diff']]
    y_magnitude = data['next_magnitude']
    y_time_diff = data['next_time_diff']

    # Split data into training and test sets
    X_train_mag, X_test_mag, y_train_mag, y_test_mag = train_test_split(X, y_magnitude, test_size=0.2, random_state=0)
    X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(X, y_time_diff, test_size=0.2, random_state=0)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_mag = scaler.fit_transform(X_train_mag)
    X_test_mag = scaler.transform(X_test_mag)
    X_train_time = scaler.fit_transform(X_train_time)
    X_test_time = scaler.transform(X_test_time)
    logging.info("Data preprocessing completed")
except Exception as e:
    logging.error("Error in data preprocessing", exc_info=True)
    raise e

try:
    # Model Training and Evaluation

    # Random Forest Regressor
    rf_params = {
        'n_estimators': [50, 100],  # Reduced number of estimators to match parameter grid size
        'max_depth': [10, 20],  # Limited max depth
        'min_samples_split': [5, 10]  # Increased min samples split
    }
    rf_grid = RandomizedSearchCV(RandomForestRegressor(random_state=0, n_jobs=-1), rf_params, cv=5, n_iter=6, scoring='neg_mean_squared_error', random_state=0)
    rf_grid.fit(X_train_mag, y_train_mag)
    best_rf_model = rf_grid.best_estimator_

    y_pred_mag = best_rf_model.predict(X_test_mag)
    mse_mag = mean_squared_error(y_test_mag, y_pred_mag)
    r2_mag = r2_score(y_test_mag, y_pred_mag)
    logging.info(f'Best Random Forest Model - Magnitude Prediction: MSE = {mse_mag}, R² = {r2_mag}')

    # Gradient Boosting Regressor
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5]
    }
    gb_grid = RandomizedSearchCV(GradientBoostingRegressor(random_state=0), gb_params, cv=5, n_iter=6, scoring='neg_mean_squared_error', random_state=0)
    gb_grid.fit(X_train_time, y_train_time)
    best_gb_model = gb_grid.best_estimator_

    y_pred_time = best_gb_model.predict(X_test_time)
    mse_time = mean_squared_error(y_test_time, y_pred_time)
    r2_time = r2_score(y_test_time, y_pred_time)
    logging.info(f'Best Gradient Boosting Model - Time Difference Prediction: MSE = {mse_time}, R² = {r2_time}')

    # XGBoost Regressor with GPU
    xgb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5],
        'tree_method': ['hist']
    }
    xgb_grid = RandomizedSearchCV(XGBRegressor(random_state=0), xgb_params, cv=5, n_iter=6, scoring='neg_mean_squared_error', random_state=0)
    xgb_grid.fit(X_train_mag, y_train_mag)
    best_xgb_model = xgb_grid.best_estimator_

    y_pred_xgb = best_xgb_model.predict(X_test_mag)
    mse_xgb = mean_squared_error(y_test_mag, y_pred_xgb)
    r2_xgb = r2_score(y_test_mag, y_pred_xgb)
    logging.info(f'Best XGBoost Model - Magnitude Prediction: MSE = {mse_xgb}, R² = {r2_xgb}')

    # New data prediction
    last_earthquake = earthquakes.iloc[-1]
    new_data = pd.DataFrame([{
        'previous_magnitude': last_earthquake['magnitude'],
        'previous_time_diff': last_earthquake['time_diff']
    }])

    # Feature Scaling
    new_data_scaled = scaler.transform(new_data)

    next_magnitude_pred = best_rf_model.predict(new_data_scaled)
    next_time_diff_pred = best_gb_model.predict(new_data_scaled)

    logging.info(f'Predicted Next Earthquake Magnitude: {next_magnitude_pred[0]}')
    logging.info(f'Predicted Time Difference to Next Earthquake (seconds): {next_time_diff_pred[0]}')

    # Convert predicted time difference to datetime
    predicted_next_time = last_earthquake['time'] + pd.to_timedelta(next_time_diff_pred[0], unit='s')
    logging.info(f'Predicted Date and Time of Next Earthquake: {predicted_next_time}')
    print(f'Predicted Next Earthquake Magnitude: {next_magnitude_pred[0]}')
    print(f'Predicted Time Difference to Next Earthquake (seconds): {next_time_diff_pred[0]}')
    print(f'Predicted Date and Time of Next Earthquake: {predicted_next_time}')
except Exception as e:
    logging.error("Error during model training or prediction", exc_info=True)
    raise e
