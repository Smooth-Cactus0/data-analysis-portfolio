"""
Energy Consumption Data Generator
=================================
Generates realistic hourly energy consumption data with:
- Multiple seasonalities (hourly, daily, weekly, annual)
- Weather data (temperature, humidity, wind, cloud cover)
- Extreme weather events (heat waves, cold snaps, storms)
- Solar features (day duration, sunrise/sunset)
- Calendar features (holidays, seasons, day types)

Author: Alexy Louis
Email: alexy.louis.scholar@gmail.com
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("ENERGY CONSUMPTION DATA GENERATOR")
print("="*70)

# =============================================================================
# Configuration
# =============================================================================

START_DATE = "2022-01-01"
END_DATE = "2023-12-31"  # 2 years of hourly data
LOCATION = "Chicago, IL"  # Midwest US - good seasonal variation
LATITUDE = 41.8781
LONGITUDE = -87.6298

# Base consumption parameters (in MWh for a small city/district)
BASE_LOAD = 500  # Base load in MWh
PEAK_LOAD = 1200  # Peak load capacity

# =============================================================================
# Generate Date Range
# =============================================================================

print("\n📅 Generating date range...")
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='h')
n_hours = len(date_range)
print(f"   Generated {n_hours:,} hourly timestamps ({n_hours/24:.0f} days)")

# Create base DataFrame
df = pd.DataFrame({'timestamp': date_range})
df['date'] = df['timestamp'].dt.date
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# =============================================================================
# Calendar Features
# =============================================================================

print("📆 Adding calendar features...")

# Season (meteorological)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

# US Federal Holidays (simplified)
us_holidays = {
    # 2022
    '2022-01-01': "New Year's Day",
    '2022-01-17': 'MLK Day',
    '2022-02-21': "Presidents' Day",
    '2022-05-30': 'Memorial Day',
    '2022-06-20': 'Juneteenth',
    '2022-07-04': 'Independence Day',
    '2022-09-05': 'Labor Day',
    '2022-10-10': 'Columbus Day',
    '2022-11-11': "Veterans Day",
    '2022-11-24': 'Thanksgiving',
    '2022-12-25': 'Christmas',
    '2022-12-26': 'Christmas (Observed)',
    # 2023
    '2023-01-01': "New Year's Day",
    '2023-01-02': "New Year's (Observed)",
    '2023-01-16': 'MLK Day',
    '2023-02-20': "Presidents' Day",
    '2023-05-29': 'Memorial Day',
    '2023-06-19': 'Juneteenth',
    '2023-07-04': 'Independence Day',
    '2023-09-04': 'Labor Day',
    '2023-10-09': 'Columbus Day',
    '2023-11-10': "Veterans Day (Observed)",
    '2023-11-23': 'Thanksgiving',
    '2023-12-25': 'Christmas',
}

df['date_str'] = df['date'].astype(str)
df['is_holiday'] = df['date_str'].isin(us_holidays.keys()).astype(int)
df['holiday_name'] = df['date_str'].map(us_holidays).fillna('')

# Day type classification
def get_day_type(row):
    if row['is_holiday']:
        return 'Holiday'
    elif row['is_weekend']:
        return 'Weekend'
    else:
        return 'Weekday'

df['day_type'] = df.apply(get_day_type, axis=1)

# =============================================================================
# Solar Features (Day Duration, Sunrise/Sunset)
# =============================================================================

print("☀️ Calculating solar features...")

def calculate_day_length(day_of_year, latitude):
    """Calculate day length in hours using astronomical formula."""
    # Convert latitude to radians
    lat_rad = np.radians(latitude)
    
    # Calculate solar declination
    declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
    dec_rad = np.radians(declination)
    
    # Calculate hour angle
    cos_hour_angle = -np.tan(lat_rad) * np.tan(dec_rad)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
    hour_angle = np.degrees(np.arccos(cos_hour_angle))
    
    # Day length in hours
    day_length = 2 * hour_angle / 15
    
    return day_length

df['day_length_hours'] = df['day_of_year'].apply(
    lambda x: calculate_day_length(x, LATITUDE)
)

# Approximate sunrise and sunset
df['sunrise_hour'] = 12 - df['day_length_hours'] / 2
df['sunset_hour'] = 12 + df['day_length_hours'] / 2

# Is it daylight?
df['is_daylight'] = ((df['hour'] >= df['sunrise_hour']) & 
                      (df['hour'] <= df['sunset_hour'])).astype(int)

# Hours since sunrise/until sunset
df['hours_since_sunrise'] = np.maximum(0, df['hour'] - df['sunrise_hour'])
df['hours_until_sunset'] = np.maximum(0, df['sunset_hour'] - df['hour'])

# =============================================================================
# Weather Data Generation
# =============================================================================

print("🌡️ Generating weather data...")

# Temperature (realistic for Chicago)
# Annual cycle + daily cycle + random variation

# Annual temperature cycle (Celsius)
annual_temp_mean = -5 + 25 * np.sin(2 * np.pi * (df['day_of_year'] - 100) / 365)

# Daily temperature cycle
daily_temp_variation = 5 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)

# Random weather variation (autocorrelated)
weather_noise = np.zeros(n_hours)
weather_noise[0] = np.random.normal(0, 3)
for i in range(1, n_hours):
    weather_noise[i] = 0.95 * weather_noise[i-1] + np.random.normal(0, 1)

df['temperature_c'] = annual_temp_mean + daily_temp_variation + weather_noise
df['temperature_f'] = df['temperature_c'] * 9/5 + 32

# Feels like temperature (wind chill / heat index simplified)
df['feels_like_c'] = df['temperature_c'].copy()

# Humidity (inverse relationship with temperature + seasonal)
base_humidity = 65 - 0.5 * df['temperature_c']
humidity_noise = np.random.normal(0, 10, n_hours)
df['humidity_pct'] = np.clip(base_humidity + humidity_noise, 20, 100)

# Wind speed (m/s) - higher in spring/fall
seasonal_wind = 3 + 2 * np.cos(2 * np.pi * df['day_of_year'] / 365)
wind_noise = np.abs(np.random.normal(0, 2, n_hours))
df['wind_speed_ms'] = seasonal_wind + wind_noise
df['wind_speed_mph'] = df['wind_speed_ms'] * 2.237

# Cloud cover (0-100%)
cloud_noise = np.zeros(n_hours)
cloud_noise[0] = 50
for i in range(1, n_hours):
    cloud_noise[i] = 0.9 * cloud_noise[i-1] + np.random.normal(0, 10)
df['cloud_cover_pct'] = np.clip(cloud_noise, 0, 100)

# Precipitation probability
df['precip_prob_pct'] = np.clip(
    df['cloud_cover_pct'] - 30 + np.random.normal(0, 15, n_hours), 0, 100
)

# Actual precipitation (mm)
df['precipitation_mm'] = np.where(
    (df['precip_prob_pct'] > 50) & (np.random.random(n_hours) < 0.3),
    np.abs(np.random.exponential(5, n_hours)),
    0
)

# Dew point
df['dew_point_c'] = df['temperature_c'] - ((100 - df['humidity_pct']) / 5)

# Pressure (hPa) - affects weather patterns
base_pressure = 1013 + 10 * np.sin(2 * np.pi * df['day_of_year'] / 365)
pressure_noise = np.random.normal(0, 5, n_hours)
df['pressure_hpa'] = base_pressure + pressure_noise

# UV Index (daylight hours, summer peak)
uv_seasonal = 6 * np.sin(np.pi * (df['day_of_year'] - 80) / 365)
uv_daily = np.where(df['is_daylight'], 
                    np.sin(np.pi * (df['hour'] - df['sunrise_hour']) / df['day_length_hours']),
                    0)
df['uv_index'] = np.maximum(0, uv_seasonal * uv_daily * (1 - df['cloud_cover_pct']/100))

# =============================================================================
# Extreme Weather Events
# =============================================================================

print("⚠️ Adding extreme weather events...")

# Heat waves (3+ consecutive days with temp > 35°C in summer)
heat_wave_periods = [
    ('2022-07-18', '2022-07-24'),  # Summer 2022 heat wave
    ('2022-08-01', '2022-08-05'),
    ('2023-06-25', '2023-06-30'),  # Summer 2023 heat wave
    ('2023-07-15', '2023-07-22'),
    ('2023-08-10', '2023-08-14'),
]

df['is_heat_wave'] = 0
for start, end in heat_wave_periods:
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    df.loc[mask, 'is_heat_wave'] = 1
    # Increase temperature during heat waves
    df.loc[mask, 'temperature_c'] += np.random.uniform(5, 12, mask.sum())

# Cold snaps (extreme cold in winter)
cold_snap_periods = [
    ('2022-01-15', '2022-01-20'),  # Polar vortex
    ('2022-02-01', '2022-02-05'),
    ('2022-12-20', '2022-12-26'),  # Christmas cold snap
    ('2023-01-28', '2023-02-03'),  # Winter 2023
    ('2023-02-15', '2023-02-18'),
]

df['is_cold_snap'] = 0
for start, end in cold_snap_periods:
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    df.loc[mask, 'is_cold_snap'] = 1
    # Decrease temperature during cold snaps
    df.loc[mask, 'temperature_c'] -= np.random.uniform(8, 15, mask.sum())

# Storms (high wind + precipitation)
storm_periods = [
    ('2022-03-15', '2022-03-16'),  # Spring storm
    ('2022-05-10', '2022-05-11'),
    ('2022-08-20', '2022-08-21'),  # Summer storm
    ('2022-11-05', '2022-11-06'),
    ('2023-04-01', '2023-04-02'),
    ('2023-06-18', '2023-06-19'),
    ('2023-09-25', '2023-09-26'),
    ('2023-11-15', '2023-11-16'),
]

df['is_storm'] = 0
for start, end in storm_periods:
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    df.loc[mask, 'is_storm'] = 1
    df.loc[mask, 'wind_speed_ms'] += np.random.uniform(10, 25, mask.sum())
    df.loc[mask, 'precipitation_mm'] += np.random.uniform(10, 50, mask.sum())
    df.loc[mask, 'cloud_cover_pct'] = np.minimum(100, df.loc[mask, 'cloud_cover_pct'] + 40)

# Extreme event flag
df['is_extreme_weather'] = ((df['is_heat_wave'] == 1) | 
                            (df['is_cold_snap'] == 1) | 
                            (df['is_storm'] == 1)).astype(int)

# Update feels like temperature
df.loc[df['temperature_c'] > 27, 'feels_like_c'] = (
    df.loc[df['temperature_c'] > 27, 'temperature_c'] + 
    df.loc[df['temperature_c'] > 27, 'humidity_pct'] * 0.1
)
df.loc[df['temperature_c'] < 10, 'feels_like_c'] = (
    df.loc[df['temperature_c'] < 10, 'temperature_c'] - 
    df.loc[df['temperature_c'] < 10, 'wind_speed_ms'] * 0.5
)

# =============================================================================
# Energy Consumption Generation
# =============================================================================

print("⚡ Generating energy consumption patterns...")

# Base load pattern
consumption = np.ones(n_hours) * BASE_LOAD

# 1. ANNUAL SEASONALITY (heating in winter, cooling in summer)
# Higher in extreme temperatures
temp_effect = np.where(
    df['temperature_c'] < 15,  # Heating needed
    (15 - df['temperature_c']) * 15,  # 15 MWh per degree below 15°C
    np.where(
        df['temperature_c'] > 22,  # Cooling needed
        (df['temperature_c'] - 22) * 20,  # 20 MWh per degree above 22°C
        0
    )
)
consumption += temp_effect

# 2. DAILY PATTERN (hourly profile)
# Different profiles for weekday vs weekend
hourly_weekday = {
    0: 0.70, 1: 0.65, 2: 0.60, 3: 0.58, 4: 0.60, 5: 0.70,
    6: 0.85, 7: 1.00, 8: 1.10, 9: 1.15, 10: 1.18, 11: 1.20,
    12: 1.15, 13: 1.18, 14: 1.20, 15: 1.18, 16: 1.15, 17: 1.20,
    18: 1.25, 19: 1.20, 20: 1.10, 21: 1.00, 22: 0.90, 23: 0.80
}

hourly_weekend = {
    0: 0.65, 1: 0.60, 2: 0.55, 3: 0.53, 4: 0.55, 5: 0.58,
    6: 0.65, 7: 0.75, 8: 0.90, 9: 1.00, 10: 1.08, 11: 1.12,
    12: 1.10, 13: 1.08, 14: 1.05, 15: 1.03, 16: 1.05, 17: 1.10,
    18: 1.15, 19: 1.12, 20: 1.05, 21: 0.95, 22: 0.85, 23: 0.75
}

hourly_factor = np.where(
    df['is_weekend'] == 1,
    df['hour'].map(hourly_weekend),
    df['hour'].map(hourly_weekday)
)
consumption *= hourly_factor

# 3. WEEKLY PATTERN
# Slightly lower on weekends overall
weekly_factor = np.where(df['is_weekend'] == 1, 0.92, 1.0)
consumption *= weekly_factor

# 4. HOLIDAY EFFECT (reduced commercial/industrial load)
holiday_factor = np.where(df['is_holiday'] == 1, 0.75, 1.0)
consumption *= holiday_factor

# 5. EXTREME WEATHER EFFECTS
# Heat waves: massive AC demand spike
heat_wave_effect = np.where(
    df['is_heat_wave'] == 1,
    consumption * 0.35,  # 35% increase
    0
)
consumption += heat_wave_effect

# Cold snaps: heating demand spike
cold_snap_effect = np.where(
    df['is_cold_snap'] == 1,
    consumption * 0.40,  # 40% increase
    0
)
consumption += cold_snap_effect

# Storms: some industrial shutdown, but residential up
storm_effect = np.where(
    df['is_storm'] == 1,
    consumption * -0.05,  # 5% decrease (industrial shutdown > residential increase)
    0
)
consumption += storm_effect

# 6. LIGHTING (daylight effect)
# More lighting needed when dark
lighting_factor = np.where(
    df['is_daylight'] == 0,
    50 + 20 * (1 - df['cloud_cover_pct']/100),  # More if cloudy
    10 + 30 * (df['cloud_cover_pct']/100)  # Less in daylight, more if cloudy
)
consumption += lighting_factor

# 7. RANDOM VARIATION (demand noise)
noise = np.random.normal(0, consumption * 0.03)  # 3% noise
consumption += noise

# 8. TREND (slight annual growth ~2%)
days_from_start = (df['timestamp'] - df['timestamp'].min()).dt.days
growth_factor = 1 + 0.02 * (days_from_start / 365)
consumption *= growth_factor

# Ensure non-negative and within capacity
consumption = np.clip(consumption, 100, PEAK_LOAD * 1.1)

df['consumption_mwh'] = consumption.round(2)

# =============================================================================
# Add Derived Consumption Features
# =============================================================================

print("📊 Adding derived features...")

# Consumption per capita (assuming 500,000 people in district)
df['consumption_kwh_per_capita'] = (df['consumption_mwh'] * 1000 / 500000).round(4)

# Heating/Cooling Degree Days (hourly approximation)
df['heating_degree_hours'] = np.maximum(0, 18 - df['temperature_c'])
df['cooling_degree_hours'] = np.maximum(0, df['temperature_c'] - 18)

# Load factor
df['load_factor'] = (df['consumption_mwh'] / PEAK_LOAD).round(4)

# =============================================================================
# Create Anomalies for Detection Demo
# =============================================================================

print("🔍 Injecting anomalies for detection demo...")

# Equipment failures (sudden drops)
equipment_failures = [
    ('2022-04-15 14:00', '2022-04-15 18:00', 0.6),  # 40% drop for 4 hours
    ('2022-09-10 10:00', '2022-09-10 14:00', 0.7),
    ('2023-03-20 08:00', '2023-03-20 12:00', 0.65),
    ('2023-08-05 16:00', '2023-08-05 20:00', 0.7),
]

df['is_equipment_failure'] = 0
for start, end, factor in equipment_failures:
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    df.loc[mask, 'consumption_mwh'] *= factor
    df.loc[mask, 'is_equipment_failure'] = 1

# Demand spikes (special events - sports games, concerts)
special_events = [
    ('2022-10-15 18:00', '2022-10-15 23:00', 1.3, 'Sports Event'),
    ('2022-12-31 20:00', '2023-01-01 02:00', 1.25, 'New Year Eve'),
    ('2023-07-04 19:00', '2023-07-04 23:00', 1.2, 'Independence Day'),
    ('2023-11-23 16:00', '2023-11-23 22:00', 1.15, 'Thanksgiving'),
]

df['is_special_event'] = 0
df['special_event_name'] = ''
for start, end, factor, name in special_events:
    mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    df.loc[mask, 'consumption_mwh'] *= factor
    df.loc[mask, 'is_special_event'] = 1
    df.loc[mask, 'special_event_name'] = name

# Anomaly flag
df['is_anomaly'] = ((df['is_equipment_failure'] == 1) | 
                    (df['is_special_event'] == 1)).astype(int)

# =============================================================================
# Save Data
# =============================================================================

print("\n💾 Saving data files...")

output_path = Path('/home/claude/data-analysis-portfolio/05-time-series-forecasting/data/raw')

# Main dataset
df.to_csv(output_path / 'energy_consumption.csv', index=False)
print(f"   ✅ energy_consumption.csv ({len(df):,} rows, {len(df.columns)} columns)")

# Weather data (for external data simulation)
weather_cols = ['timestamp', 'temperature_c', 'temperature_f', 'feels_like_c',
                'humidity_pct', 'wind_speed_ms', 'wind_speed_mph', 'cloud_cover_pct',
                'precip_prob_pct', 'precipitation_mm', 'dew_point_c', 'pressure_hpa',
                'uv_index', 'is_heat_wave', 'is_cold_snap', 'is_storm', 'is_extreme_weather']
df[weather_cols].to_csv(output_path / 'weather_data.csv', index=False)
print(f"   ✅ weather_data.csv")

# Calendar data
calendar_cols = ['timestamp', 'date', 'year', 'month', 'day', 'hour', 'day_of_week',
                 'day_of_year', 'week_of_year', 'is_weekend', 'season', 'is_holiday',
                 'holiday_name', 'day_type']
df[calendar_cols].to_csv(output_path / 'calendar_data.csv', index=False)
print(f"   ✅ calendar_data.csv")

# Solar data
solar_cols = ['timestamp', 'day_length_hours', 'sunrise_hour', 'sunset_hour',
              'is_daylight', 'hours_since_sunrise', 'hours_until_sunset']
df[solar_cols].to_csv(output_path / 'solar_data.csv', index=False)
print(f"   ✅ solar_data.csv")

# Metadata
metadata = {
    'dataset_name': 'Energy Consumption Dataset',
    'location': LOCATION,
    'latitude': LATITUDE,
    'longitude': LONGITUDE,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'frequency': 'hourly',
    'total_records': len(df),
    'base_load_mwh': BASE_LOAD,
    'peak_capacity_mwh': PEAK_LOAD,
    'features': {
        'consumption': ['consumption_mwh', 'consumption_kwh_per_capita', 'load_factor'],
        'weather': ['temperature_c', 'humidity_pct', 'wind_speed_ms', 'cloud_cover_pct', 
                   'precipitation_mm', 'pressure_hpa', 'uv_index'],
        'calendar': ['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 
                    'is_holiday', 'season', 'day_type'],
        'solar': ['day_length_hours', 'sunrise_hour', 'sunset_hour', 'is_daylight'],
        'extreme_events': ['is_heat_wave', 'is_cold_snap', 'is_storm', 'is_extreme_weather'],
        'anomalies': ['is_equipment_failure', 'is_special_event', 'is_anomaly']
    },
    'extreme_weather_events': {
        'heat_waves': len(heat_wave_periods),
        'cold_snaps': len(cold_snap_periods),
        'storms': len(storm_periods)
    },
    'anomalies_injected': {
        'equipment_failures': len(equipment_failures),
        'special_events': len(special_events)
    },
    'generated_at': datetime.now().isoformat()
}

with open(output_path / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✅ metadata.json")

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "="*70)
print("DATA GENERATION COMPLETE!")
print("="*70)

print(f"\n📊 DATASET SUMMARY")
print(f"   Period: {START_DATE} to {END_DATE}")
print(f"   Total hours: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")

print(f"\n⚡ CONSUMPTION STATISTICS")
print(f"   Mean: {df['consumption_mwh'].mean():.2f} MWh")
print(f"   Std: {df['consumption_mwh'].std():.2f} MWh")
print(f"   Min: {df['consumption_mwh'].min():.2f} MWh")
print(f"   Max: {df['consumption_mwh'].max():.2f} MWh")

print(f"\n🌡️ WEATHER STATISTICS")
print(f"   Temperature range: {df['temperature_c'].min():.1f}°C to {df['temperature_c'].max():.1f}°C")
print(f"   Mean humidity: {df['humidity_pct'].mean():.1f}%")

print(f"\n⚠️ EXTREME EVENTS")
print(f"   Heat wave hours: {df['is_heat_wave'].sum():,}")
print(f"   Cold snap hours: {df['is_cold_snap'].sum():,}")
print(f"   Storm hours: {df['is_storm'].sum():,}")

print(f"\n🔍 ANOMALIES")
print(f"   Equipment failures: {df['is_equipment_failure'].sum():,} hours")
print(f"   Special events: {df['is_special_event'].sum():,} hours")

print(f"\n📁 Files saved to: {output_path}")
