import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
from datetime import datetime

# ==========================================================
# --- CONFIGURATION (UPDATE THIS) ---
# NOTE: This key is used for weather data and location name lookups (OpenWeatherMap).
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f" 

# NOTE: For live traffic, obtain a key from HERE Technologies (Routing API).
# This key has been provided by the user and is assumed to be active.
HERE_API_KEY = "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw" 
PLACEHOLDER_CHECK = "PASTE_YOUR_API_KEY_HERE" 
# ==========================================================

# --- Helper Function: Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in km between two points on the earth."""
    R = 6371 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- Forward Geocoding Function (Location Name to Coordinates) ---
@st.cache_data(ttl=86400)
def fetch_coordinates(location_name, api_key):
    """Fetches coordinates (lat, lon) from a location name and reconstructs display name."""
    if not api_key or not location_name:
        return None, None, None

    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_name}&limit=1&appid={api_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data:
            location = data[0]
            lat = location.get('lat')
            lon = location.get('lon')
            display_name = f"{location.get('name', 'N/A')}, {location.get('country', 'N/A')}"
            return lat, lon, display_name
        else:
            return None, None, None

    except requests.exceptions.RequestException:
        return None, None, None
    except Exception:
        return None, None, None


# --- Real-time Weather Function (API Integration) ---
@st.cache_data(ttl=300)
def fetch_realtime_weather(latitude, longitude, api_key):
    """Fetches real-time weather data. Returns (temp, weather_main)."""
    
    if not api_key:
        return 25.0, 'Clear'

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        temp = data['main']['temp']
        weather_main = data['weather'][0]['main'] 
        return temp, weather_main
    
    except Exception:
        return 25.0, 'Clear'


# --- Live Traffic API Integration (Using HERE Routing API) ---
def fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, api_key):
    """Calls HERE Routing API with traffic enabled."""
    if not api_key or api_key == PLACEHOLDER_CHECK:
        return None, None

    try:
        url = "https://router.hereapi.com/v8/routes"
        origin_coords = f"{rest_lat},{rest_lon}"
        destination_coords = f"{del_lat},{del_lon}"

        params = {
            'transportMode': 'car',
            'origin': origin_coords,
            'destination': destination_coords,
            'routingMode': 'fast',
            'trafficMode': 'realtime',
            'return': 'summary',
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get('routes') and data['routes'][0].get('sections'):
            summary = data['routes'][0]['sections'][0]['summary']
            traffic_duration_sec = summary.get('duration')
            base_duration_sec = summary.get('baseDuration') 
            
            if traffic_duration_sec is None or base_duration_sec is None:
                return None, None
            
            traffic_duration_min = traffic_duration_sec / 60.0
            base_travel_time_min = base_duration_sec / 60.0
        
            st.success("✅ Live traffic data successfully retrieved from HERE API.")
            return traffic_duration_min, base_travel_time_min
        else:
            return None, None
            
    except Exception:
        return None, None


# --- Streamlit Application Layout ---
st.set_page_config(page_title="Delivery Predictor", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predict the probability of a late delivery based on real-time weather, **traffic-adjusted travel time**, and restaurant data.")

# 1. Load Model
try:
    st.session_state.tz = pd.Timestamp.now().tz_localize('Asia/Kolkata').tz
    model = joblib.load('late_delivery_predictor_model.pkl')
    st.success("Model loaded successfully for prediction.")
except FileNotFoundError:
    st.error("Error: Model file 'late_delivery_predictor_model.pkl' not found. Ensure it is uploaded.")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

if model:
    # 2. User Input Interface
    col1, col2 = st.columns(2)
    
    current_time_ts = datetime.now().astimezone(st.session_state.tz)
    order_hour = current_time_ts.hour

    BASE_SPEED_KM_PER_MIN = 0.5 
    
    if 17 <= order_hour <= 21:
        traffic_multiplier_sim = 1.67 
        traffic_label_sim = 'Jam (Multiplier: 1.67x)'
    elif 12 <= order_hour <= 14:
        traffic_multiplier_sim = 1.33 
        traffic_label_sim = 'High (Multiplier: 1.33x)'
    elif 8 <= order_hour <= 10:
        traffic_multiplier_sim = 1.18
        traffic_label_sim = 'Medium (Multiplier: 1.18x)'
    else:
        traffic_multiplier_sim = 1.0 
        traffic_label_sim = 'Low (Multiplier: 1.0x)'
    
    with col1:
        st.subheader("Restaurant & Delivery Location")
        rest_location_input = st.text_input("Restaurant Location (City, Country)", value="Bangalore, India")
        del_location_input = st.text_input("Delivery Location (City, Country)", value="Mangalore, India")
        
        rest_lat, rest_lon, rest_location_name = fetch_coordinates(rest_location_input, WEATHER_API_KEY)
        del_lat, del_lon, del_location_name = fetch_coordinates(del_location_input, WEATHER_API_KEY)

        # ✅ Removed st.info() lines — no blue info boxes shown now

        if not (rest_lat and rest_lon):
            st.error("⚠️ Restaurant Location not resolved. Check spelling/API Key.")

        if not (del_lat and del_lon):
            st.error("⚠️ Delivery Location not resolved. Check spelling/API Key.")

    with col2:
        st.subheader("Order Context")
        prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
        rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.9, 0.1)
        rating_del = st.slider("Delivery Person Rating (4.0 to 5.0)", 4.0, 5.0, 4.8, 0.1)

    st.markdown("---")
    
    if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
        with st.spinner("Fetching real-time data and calculating..."):
            if not all([rest_lat, rest_lon, del_lat, del_lon]):
                st.error("❌ Cannot predict: Please ensure both locations are resolved to coordinates before continuing.")
                st.stop()
            
            delivery_distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
            current_temp, weather_main = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
            
            api_result = fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, HERE_API_KEY)
            estimated_travel_time_traffic_adjusted, base_travel_time_min_api = api_result

            if estimated_travel_time_traffic_adjusted is None:
                base_travel_time_min = delivery_distance_km / BASE_SPEED_KM_PER_MIN
                estimated_travel_time_traffic_adjusted = base_travel_time_min * traffic_multiplier_sim
                traffic_density = traffic_label_sim.split(' ')[0]
            else:
                traffic_ratio = estimated_travel_time_traffic_adjusted / base_travel_time_min_api
                if traffic_ratio >= 1.5:
                    traffic_density = 'Jam'
                elif traffic_ratio >= 1.25:
                    traffic_density = 'High'
                elif traffic_ratio >= 1.05:
                    traffic_density = 'Medium'
                else:
                    traffic_density = 'Low'

            input_data_final = pd.DataFrame({
                'delivery_distance_km': [delivery_distance_km],
                'preparation_time_min': [prep_time],
                'restaurant_rating': [rating_rest],
                'delivery_person_rating': [rating_del],
                'Weather_Condition': [weather_main],
                'sin_hour': [np.sin(2 * np.pi * order_hour / 24)],
                'cos_hour': [np.cos(2 * np.pi * order_hour / 24)],
                'current_temp_c': [current_temp if not np.isnan(current_temp) else 25.0],
                'Road_Traffic_Density': [traffic_density]
            })
            
            try:
                prediction_proba = model.predict_proba(input_data_final)[:, 1][0] * 100
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                prediction_proba = 50.0 
            
            st.subheader("🎯 Prediction Result")

            def get_weather_icon(condition):
                if 'rain' in condition.lower() or 'drizzle' in condition.lower() or 'storm' in condition.lower(): return "🌧️"
                if 'cloud' in condition.lower(): return "☁️"
                if 'clear' in condition.lower() or 'sunny' in condition.lower(): return "☀️"
                if 'snow' in condition.lower() or 'hail' in condition.lower(): return "❄️"
                if 'haze' in condition.lower() or 'fog' in condition.lower(): return "🌫️"
                if 'wind' in condition.lower(): return "🌬️"
                return "❓"
            
            col_res1, col_res2, col_res3, col_res4 = st.columns(4) 

            with col_res1:
                st.metric("Probability of Being Late", f"{prediction_proba:.2f}%")
            with col_res2:
                st.metric("Traffic-Adj Travel Time", f"{estimated_travel_time_traffic_adjusted:.1f} min")
            with col_res3:
                st.metric("Current Weather", f"{get_weather_icon(weather_main)} {weather_main}")
            with col_res4:
                temp_display = f"{current_temp:.1f}" if isinstance(current_temp, (int, float)) and not np.isnan(current_temp) else 'N/A'
                st.metric("Temp (°C)", temp_display)
            
            if prediction_proba > 60:
                st.error("⚠️ **HIGH RISK:** A late delivery is highly probable. ")
            elif prediction_proba > 40:
                st.warning("🔶 **MODERATE RISK:** The chance of delay is significant.")
            else:
                st.success("✅ **LOW RISK:** Delivery is likely to be on time.")

            st.caption(f"Prediction based on: Restaurant @ **{rest_location_name}** to Delivery @ **{del_location_name}**")

if not model:
    st.stop()


