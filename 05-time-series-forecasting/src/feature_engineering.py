"""
Feature Engineering Module for Time Series Forecasting
======================================================
Creates advanced features for energy consumption prediction.

Author: Alexy Louis
Email: alexy.louis.scholar@gmail.com
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 3: FEATURE ENGINEERING")
print("="*70)

# Load data
DATA_PATH = Path('data/raw')
OUTPUT_PATH = Path('data/processed')
OUTPUT_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH / 'energy_consumption.csv', parse_dates=['timestamp'])
print(f"\nLoaded data: {df.shape}")

# =============================================================================
# 1. LAG FEATURES
# =============================================================================
print("\n📊 Creating lag features...")

# Target lags
lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 6h, 12h, 24h, 48h, 1 week
for lag in lag_periods:
    df[f'consumption_lag_{lag}h'] = df['consumption_mwh'].shift(lag)

# Same hour yesterday, same hour last week
df['consumption_same_hour_yesterday'] = df['consumption_mwh'].shift(24)
df['consumption_same_hour_last_week'] = df['consumption_mwh'].shift(168)

print(f"   Created {len(lag_periods) + 2} lag features")

# =============================================================================
# 2. ROLLING STATISTICS
# =============================================================================
print("📈 Creating rolling statistics...")

# Rolling windows
windows = [3, 6, 12, 24, 48, 168]  # hours

for window in windows:
    # Rolling mean
    df[f'consumption_rolling_mean_{window}h'] = df['consumption_mwh'].rolling(
        window=window, min_periods=1).mean()
    
    # Rolling std
    df[f'consumption_rolling_std_{window}h'] = df['consumption_mwh'].rolling(
        window=window, min_periods=1).std()
    
    # Rolling min/max
    df[f'consumption_rolling_min_{window}h'] = df['consumption_mwh'].rolling(
        window=window, min_periods=1).min()
    df[f'consumption_rolling_max_{window}h'] = df['consumption_mwh'].rolling(
        window=window, min_periods=1).max()

# Exponential weighted moving average
for span in [12, 24, 168]:
    df[f'consumption_ewma_{span}h'] = df['consumption_mwh'].ewm(span=span).mean()

print(f"   Created {len(windows) * 4 + 3} rolling features")

# =============================================================================
# 3. DIFFERENCE FEATURES
# =============================================================================
print("📉 Creating difference features...")

# First differences
df['consumption_diff_1h'] = df['consumption_mwh'].diff(1)
df['consumption_diff_24h'] = df['consumption_mwh'].diff(24)
df['consumption_diff_168h'] = df['consumption_mwh'].diff(168)

# Percent change
df['consumption_pct_change_1h'] = df['consumption_mwh'].pct_change(1)
df['consumption_pct_change_24h'] = df['consumption_mwh'].pct_change(24)

print(f"   Created 5 difference features")

# =============================================================================
# 4. CYCLICAL ENCODING (Fourier Features)
# =============================================================================
print("🔄 Creating cyclical encodings...")

# Hour of day (24-hour cycle)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Day of week (7-day cycle)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Month (12-month cycle)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Day of year (365-day cycle)
df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

print(f"   Created 8 cyclical features")

# =============================================================================
# 5. WEATHER INTERACTION FEATURES
# =============================================================================
print("🌡️ Creating weather interaction features...")

# Temperature squared (for non-linear relationship)
df['temperature_squared'] = df['temperature_c'] ** 2

# Temperature deviation from comfort zone (18°C)
df['temp_deviation'] = np.abs(df['temperature_c'] - 18)

# Wind chill effect (only when cold)
df['wind_chill_effect'] = np.where(
    df['temperature_c'] < 10,
    df['wind_speed_ms'] * np.abs(df['temperature_c'] - 10),
    0
)

# Heat index effect (only when hot and humid)
df['heat_index_effect'] = np.where(
    (df['temperature_c'] > 25) & (df['humidity_pct'] > 50),
    df['temperature_c'] * df['humidity_pct'] / 100,
    0
)

# Weather severity index
df['weather_severity'] = (
    df['is_heat_wave'] * 3 + 
    df['is_cold_snap'] * 3 + 
    df['is_storm'] * 2 +
    (df['temperature_c'] > 35).astype(int) + 
    (df['temperature_c'] < -10).astype(int) +
    (df['wind_speed_ms'] > 15).astype(int)
)

# Precipitation intensity
df['precip_intensity'] = np.where(
    df['precipitation_mm'] > 0,
    np.log1p(df['precipitation_mm']),
    0
)

# Cloud cover impact on solar
df['cloud_solar_interaction'] = df['cloud_cover_pct'] * df['uv_index'] / 100

print(f"   Created 8 weather interaction features")

# =============================================================================
# 6. TIME-BASED FEATURES
# =============================================================================
print("⏰ Creating time-based features...")

# Part of day
def get_part_of_day(hour):
    if 5 <= hour < 9:
        return 'Morning'
    elif 9 <= hour < 12:
        return 'Late_Morning'
    elif 12 <= hour < 14:
        return 'Noon'
    elif 14 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    elif 21 <= hour < 24:
        return 'Night'
    else:
        return 'Late_Night'

df['part_of_day'] = df['hour'].apply(get_part_of_day)

# Business hours flag
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & 
                           (df['is_weekend'] == 0)).astype(int)

# Peak hours (morning and evening peaks)
df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(int)
df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)

# Off-peak (late night)
df['is_off_peak'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)

# Days until/since holiday
df['timestamp_temp'] = pd.to_datetime(df['timestamp'])
holiday_dates = df[df['is_holiday'] == 1]['timestamp_temp'].dt.date.unique()
holiday_dates = pd.to_datetime(holiday_dates)

# Quarter
df['quarter'] = df['month'].apply(lambda x: (x - 1) // 3 + 1)

# Week of month
df['week_of_month'] = (df['day'] - 1) // 7 + 1

# Is month start/end
df['is_month_start'] = (df['day'] <= 3).astype(int)
df['is_month_end'] = (df['day'] >= 28).astype(int)

print(f"   Created 10 time-based features")

# =============================================================================
# 7. TARGET ENCODING (Historical averages)
# =============================================================================
print("📊 Creating target encoding features...")

# Average consumption by hour (historical)
hour_avg = df.groupby('hour')['consumption_mwh'].transform('mean')
df['hour_avg_consumption'] = hour_avg

# Average by day of week
dow_avg = df.groupby('day_of_week')['consumption_mwh'].transform('mean')
df['dow_avg_consumption'] = dow_avg

# Average by month
month_avg = df.groupby('month')['consumption_mwh'].transform('mean')
df['month_avg_consumption'] = month_avg

# Average by hour and day type
df['hour_daytype_avg'] = df.groupby(['hour', 'day_type'])['consumption_mwh'].transform('mean')

# Average by hour and season
df['hour_season_avg'] = df.groupby(['hour', 'season'])['consumption_mwh'].transform('mean')

print(f"   Created 5 target encoding features")

# =============================================================================
# 8. CATEGORICAL ENCODING
# =============================================================================
print("🏷️ Encoding categorical variables...")

# One-hot encode season
season_dummies = pd.get_dummies(df['season'], prefix='season')
df = pd.concat([df, season_dummies], axis=1)

# One-hot encode day_type
daytype_dummies = pd.get_dummies(df['day_type'], prefix='daytype')
df = pd.concat([df, daytype_dummies], axis=1)

# One-hot encode part_of_day
pod_dummies = pd.get_dummies(df['part_of_day'], prefix='pod')
df = pd.concat([df, pod_dummies], axis=1)

# Label encode for tree-based models
le = LabelEncoder()
df['season_encoded'] = le.fit_transform(df['season'])
df['day_type_encoded'] = le.fit_transform(df['day_type'])
df['part_of_day_encoded'] = le.fit_transform(df['part_of_day'])

print(f"   Created categorical encodings")

# =============================================================================
# 9. SPECIAL EVENT FEATURES
# =============================================================================
print("🎉 Creating special event features...")

# Days since last extreme weather
df['hours_since_heat_wave'] = df['is_heat_wave'].groupby(
    (df['is_heat_wave'] != df['is_heat_wave'].shift()).cumsum()
).cumcount()

df['hours_since_cold_snap'] = df['is_cold_snap'].groupby(
    (df['is_cold_snap'] != df['is_cold_snap'].shift()).cumsum()
).cumcount()

# Consecutive extreme weather hours
df['consecutive_heat_wave_hours'] = df['is_heat_wave'].groupby(
    (df['is_heat_wave'] != df['is_heat_wave'].shift()).cumsum()
).cumcount() * df['is_heat_wave']

df['consecutive_cold_snap_hours'] = df['is_cold_snap'].groupby(
    (df['is_cold_snap'] != df['is_cold_snap'].shift()).cumsum()
).cumcount() * df['is_cold_snap']

print(f"   Created 4 special event features")

# =============================================================================
# 10. PREPARE FINAL DATASET
# =============================================================================
print("\n💾 Preparing final dataset...")

# Drop temporary columns
df.drop(columns=['timestamp_temp', 'date_str'], inplace=True, errors='ignore')

# Count features by category
print(f"\n📊 FEATURE SUMMARY")
print(f"   Total features: {len(df.columns)}")

# For modeling, we need to drop the first 168 hours (1 week) due to lag features
# This is much better than dropping all rows with any NaN
df_model = df.iloc[168:].copy()  # Start from hour 169 (after 1 week of lags are available)

# Fill any remaining NaN values with forward fill then backward fill
df_model = df_model.ffill().bfill()

remaining_nan = df_model.isnull().sum().sum()
print(f"   Rows dropped (initial lag period): 168")
print(f"   Remaining NaN values: {remaining_nan}")

# Save full dataset with all features
df.to_csv(OUTPUT_PATH / 'energy_features_full.csv', index=False)
print(f"\n   ✅ Saved: energy_features_full.csv")

# Save modeling dataset
df_model.to_csv(OUTPUT_PATH / 'energy_features_model.csv', index=False)
print(f"   ✅ Saved: energy_features_model.csv ({len(df_model):,} rows)")

# =============================================================================
# FEATURE LIST FOR REFERENCE
# =============================================================================

feature_categories = {
    'target': ['consumption_mwh'],
    'lag_features': [c for c in df.columns if 'lag' in c or 'same_hour' in c],
    'rolling_features': [c for c in df.columns if 'rolling' in c or 'ewma' in c],
    'difference_features': [c for c in df.columns if 'diff' in c or 'pct_change' in c],
    'cyclical_features': [c for c in df.columns if '_sin' in c or '_cos' in c],
    'weather_features': ['temperature_c', 'humidity_pct', 'wind_speed_ms', 'cloud_cover_pct',
                        'precipitation_mm', 'pressure_hpa', 'uv_index', 'temperature_squared',
                        'temp_deviation', 'wind_chill_effect', 'heat_index_effect',
                        'weather_severity', 'precip_intensity', 'cloud_solar_interaction'],
    'time_features': ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
                     'is_holiday', 'is_business_hours', 'is_peak_hour', 'is_off_peak',
                     'quarter', 'week_of_month', 'is_month_start', 'is_month_end'],
    'solar_features': ['day_length_hours', 'is_daylight', 'hours_since_sunrise'],
    'extreme_weather': ['is_heat_wave', 'is_cold_snap', 'is_storm', 'is_extreme_weather',
                       'consecutive_heat_wave_hours', 'consecutive_cold_snap_hours'],
    'target_encodings': [c for c in df.columns if 'avg_consumption' in c],
}

print("\n📋 FEATURE CATEGORIES:")
for category, features in feature_categories.items():
    print(f"   {category}: {len(features)} features")

# Save feature list
import json
with open(OUTPUT_PATH / 'feature_list.json', 'w') as f:
    json.dump(feature_categories, f, indent=2)
print(f"\n   ✅ Saved: feature_list.json")

print("\n" + "="*70)
print("PHASE 3 COMPLETE!")
print("="*70)
print(f"\nTotal features created: {len(df.columns)}")
print(f"Modeling-ready rows: {len(df_model):,}")
