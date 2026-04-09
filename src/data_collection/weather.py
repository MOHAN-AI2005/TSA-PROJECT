import os
import requests
import pandas as pd
from datetime import datetime

# ---------------------------
# CONFIGURATION
# ---------------------------

LATITUDE = 13.0827   # Chennai
LONGITUDE = 80.2707

START_DATE = "2022-01-01"
END_DATE   = "2025-12-31"

OUTPUT_FILE = "../../data/raw/weather_data.csv"

# ---------------------------
# API URL
# ---------------------------

url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LATITUDE}&longitude={LONGITUDE}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
    "&timezone=Asia/Kolkata"
)

print("Fetching weather data...")

response = requests.get(url)
data = response.json()

# ---------------------------
# PARSE DATA
# ---------------------------

daily = data['daily']

df_weather = pd.DataFrame({
    'date': daily['time'],
    'temp_max': daily['temperature_2m_max'],
    'temp_min': daily['temperature_2m_min'],
    'precipitation': daily['precipitation_sum'],
    'wind_speed': daily['windspeed_10m_max']
})

# Convert date
df_weather['date'] = pd.to_datetime(df_weather['date'])

# Additional useful feature
df_weather['temp_avg'] = (df_weather['temp_max'] + df_weather['temp_min']) / 2

# ---------------------------
# SAVE
# ---------------------------

# ---------------------------
# SAVE (ROBUST PATH)
# ---------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAVE_FOLDER = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(SAVE_FOLDER, exist_ok=True)

OUTPUT_FILE = os.path.join(SAVE_FOLDER, "weather_data.csv")

df_weather.to_csv(OUTPUT_FILE, index=False)

print("Weather data saved at:", OUTPUT_FILE)