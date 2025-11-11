import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
from datetime import datetime

# ==========================================================
# --- CONFIGURATION (UPDATE THIS) ---
# NOTE: Replace the placeholder below with your *real* API key.
# This API key is required for both weather data and location name lookups.
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f" 
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

# --- Reverse Geocoding Function (API Integration) ---
@st.cache_data(ttl=86400) # Cache location names for 24 hours
def fetch_location_name(latitude, longitude, api_key):
    """Fetches location name (City, Country) using reverse geocoding."""
    if not api_key:
        return "Unknown Location (API Key Missing)"

    try:
        # OpenWeatherMap Geocoding API
        url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={latitude}&lon={longitude}&limit=1&appid={api_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data:
            location = data[0]
            # Prioritize City, then State, then Country
            name = location.get('name', 'N/A')
            country = location.get('country', 'N/A')
            
            # Use 'state' if available and different from 'name'
            state = location.get('state')
            if state and state != name:
                 return f"{name}, {state}, {country}"

            return f"{name}, {country}"
        else:
            return "No location found"

    except requests.exceptions.RequestException as e:
        # Handle connection errors, timeouts, etc.
        return f"Geocoding Error: {e}"
    except Exception:
        return "Unknown Location"


# --- Real-time Weather Function (API Integration) ---
# Use cache with Time-To-Live (TTL) of 300 seconds (5 minutes)
@st.cache_data(ttl=300)
def fetch_realtime_weather(latitude, longitude, api_key):
    """Fetches real-time weather data. Returns (temp, weather_main)."""
    
    if not api_key:
        st.warning("Weather API Key is missing. Using default weather: 25.0°C, 'Clear'")
        return 25.0, 'Clear'

    try:
        # OpenWeatherMap API call
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        temp = data['main']['temp']
        weather_main = data['weather'][0]['main'] 
        return temp, weather_main
    
    except requests.exceptions.HTTPError as e:
        # Handle API key errors (401), limits, etc.
        if response.status_code == 401:
            st.error("Invalid Weather API Key. Please check the key in the code. Using default weather.")
        else:
            st.error(f"Weather API HTTP Error: {e}. Using default weather.")
        return 25.0, 'Clear' # Fallback on API failure
        
    except requests.exceptions.RequestException:
        # Handle connection errors, timeouts, etc.
        st.error("Could not connect to the Weather API. Using default weather.")
        return 25.0, 'Clear'


# --- Streamlit Application Layout ---

st.set_page_config(page_title="Delivery Predictor", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Use real-time weather and estimated traffic data to predict the probability of an order being delivered late.")

# 1. Load Model
try:
    # The model file must be in the same folder as app.py on the deployment server
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
    
    # Use current time in a fixed timezone for consistency and feature generation
    current_time_ts = datetime.now().astimezone(pd.Timestamp.now().tz_localize('Asia/Kolkata').tz)
    order_hour = current_time_ts.hour

    # --- Traffic Estimation (Feature Engineering Preview) ---
    if 17 <= order_hour <= 21:
        traffic_estimate = 'Jam'  # Peak Evening Rush
        traffic_icon = "🛑"
    elif 12 <= order_hour <= 14:
        traffic_estimate = 'High'  # Lunch Rush
        traffic_icon = "🟡"
    elif 8 <= order_hour <= 10:
        traffic_estimate = 'Medium' # Morning period
        traffic_icon = "🟢"
    else:
        traffic_estimate = 'Low'  # Off-peak
        traffic_icon = "💨"
        
    st.markdown(f"**Current Hour Traffic Estimate ({current_time_ts.strftime('%H:%M')}):** {traffic_icon} `{traffic_estimate}`")
    st.markdown("---")


    with col1:
        st.subheader("Restaurant & Delivery Location")
        
        # Location Inputs
        rest_lat = st.number_input("Restaurant Latitude", value=28.6500, format="%.4f")
        rest_lon = st.number_input("Restaurant Longitude", value=77.2000, format="%.4f")
        
        # Reverse Geocoding for Restaurant
        rest_location_name = fetch_location_name(rest_lat, rest_lon, WEATHER_API_KEY)
        st.info(f"Restaurant Location Name: **{rest_location_name}**")
        
        del_lat = st.number_input("Delivery Latitude", value=28.7000, format="%.4f")
        del_lon = st.number_input("Delivery Longitude", value=77.2500, format="%.4f")
        
        # Reverse Geocoding for Delivery
        del_location_name = fetch_location_name(del_lat, del_lon, WEATHER_API_KEY)
        st.info(f"Delivery Location Name: **{del_location_name}**")


    with col2:
        st.subheader("Order Context")
        
        prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
        
        # Ratings
        rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.5, 0.1)
        rating_del = st.slider("Delivery Person Rating (4.0 to 5.0)", 4.0, 5.0, 4.8, 0.1)


    st.markdown("---")
    
    # 3. Prediction Button Logic
    if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
        with st.spinner("Fetching real-time data and calculating..."):
            
            # --- Feature Engineering ---
            
            # Distance
            delivery_distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
            
            # Weather (Use Delivery location as it impacts the final travel leg)
            current_temp, weather_main = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
            
            # Traffic feature is already determined above (traffic_estimate)
            traffic = traffic_estimate # Use the string variable
            
            # Time features (sin/cos for cyclical time dependency)
            sin_hour = np.sin(2 * np.pi * order_hour / 24)
            cos_hour = np.cos(2 * np.pi * order_hour / 24)
            
            # Create DataFrame for prediction
            input_data = pd.DataFrame({
                'delivery_distance_km': [delivery_distance_km],
                'preparation_time_min': [prep_time],
                'restaurant_rating': [rating_rest],
                'delivery_person_rating': [rating_del],
                'Road_Traffic_Density': [traffic], # Categorical feature
                'Weather_Condition': [weather_main], # Categorical feature
                'sin_hour': [sin_hour],
                'cos_hour': [cos_hour],
                'current_temp_c': [current_temp if not np.isnan(current_temp) else 25.0]
            })
            
            # --- Prediction ---
            try:
                # Assuming the model returns probability for class 1 (Late)
                prediction_proba = model.predict_proba(input_data)[:, 1][0] * 100
            except ValueError as e:
                st.error(f"Prediction Error: The input features may not match the model's expected format (e.g., missing categorical levels). Error: {e}")
                prediction_proba = 50.0 # Default to moderate risk on failure
            
            # 4. Display Results
            st.subheader("🎯 Prediction Result")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Probability of Being Late", f"{prediction_proba:.2f}%")
            
            with col_res2:
                st.metric("Distance", f"{delivery_distance_km:.2f} km")
                
            with col_res3:
                # Safely display temperature
                temp_display = f"{current_temp:.1f}" if isinstance(current_temp, (int, float)) and not np.isnan(current_temp) else 'N/A'
                st.metric("Current Weather", f"{weather_main}, {temp_display}°C")

            st.markdown(f"**Key Inputs:** Distance: {delivery_distance_km:.2f} km | Traffic: **{traffic}** | Weather: **{weather_main}** at {temp_display}°C")

            if prediction_proba > 60:
                st.error("⚠️ **HIGH RISK:** A late delivery is highly probable due to combined factors. Be proactive in notifying the customer.")
                
            elif prediction_proba > 40:
                st.warning("🔶 **MODERATE RISK:** The chance of delay is significant. Monitor this order closely.")
            else:
                st.success("✅ **LOW RISK:** Delivery is likely to be on time.")

            st.markdown("---")
            st.caption(f"Prediction based on: Restaurant @ **{rest_location_name}** to Delivery @ **{del_location_name}**")
            
# --- Final Check ---
if not model:
     st.stop()
