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
# This key is used for weather data and location name lookups (OpenWeatherMap).
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f" 

# NOTE: For live traffic, you must obtain a key from a provider like HERE. 
# Replace the placeholder with your actual HERE API Key.
LIVE_TRAFFIC_API_KEY = "PLACEHOLDER_FOR_LIVE_TRAFFIC_KEY"
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
        # Using OpenWeatherMap Geocoding API for forward geocoding
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_name}&limit=1&appid={api_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data:
            location = data[0]
            lat = location.get('lat')
            lon = location.get('lon')
            
            # Create a user-friendly display name (City, Country)
            display_name = f"{location.get('name', 'N/A')}, {location.get('country', 'N/A')}"
            
            return lat, lon, display_name
        else:
            st.warning(f"Could not find coordinates for: {location_name}")
            return None, None, None

    except requests.exceptions.RequestException as e:
        st.error(f"Geocoding Error: {e}")
        return None, None, None
    except Exception:
        return None, None, None


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
        if response.status_code == 401:
            st.error("Invalid Weather API Key. Please check the key in the code. Using default weather.")
        else:
            st.error(f"Weather API HTTP Error: {e}. Using default weather.")
        return 25.0, 'Clear' 
        
    except requests.exceptions.RequestException:
        st.error("Could not connect to the Weather API. Using default weather.")
        return 25.0, 'Clear'


# --- Placeholder for Live Traffic API Integration (Updated for HERE) ---
def fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, api_key):
    """
    Placeholder function for calling a real-time routing API (e.g., HERE Technologies).
    
    If successful, returns the travel duration in minutes that *includes* the current traffic.
    """
    if api_key == "PLACEHOLDER_FOR_LIVE_TRAFFIC_KEY":
        st.info("Using time-of-day simulation for traffic. A real HERE API key is required for true live traffic.")
        return None

    try:
        # --- HERE Routing API v8 Example Structure ---
        url = "https://router.hereapi.com/v8/routes"
        params = {
            'transportMode': 'car',
            'origin': f"{rest_lat},{rest_lon}",
            'destination': f"{del_lat},{del_lon}",
            'apiKey': api_key, 
            'return': 'summary', 
            'traffic': 'enabled', # Enables real-time traffic
            'departureTime': 'now' # Important for live traffic
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for valid route and extract duration in seconds from the summary
        if data.get('routes') and data['routes'][0].get('sections'):
            # The duration from HERE includes traffic when 'traffic=enabled'
            duration_sec = data['routes'][0]['sections'][0]['summary']['duration']
            st.success("✅ Live traffic data successfully retrieved from HERE API.")
            return duration_sec / 60.0 # Convert seconds to minutes
        else:
            st.warning("HERE API found no route or response structure was unexpected. Falling back to simulation.")
            return None
            
    except requests.exceptions.HTTPError as e:
        st.error(f"HERE API HTTP Error: {e}. Check your API Key/subscription. Falling back to simulation.")
        return None
    except requests.exceptions.RequestException:
        st.error("Could not connect to HERE API. Falling back to simulation.")
        return None
    except Exception:
        st.warning("An unexpected error occurred during HERE API call. Falling back to simulation.")
        return None

# --- Streamlit Application Layout ---

st.set_page_config(page_title="Delivery Predictor", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predict the probability of a late delivery based on real-time weather, **traffic-adjusted travel time**, and restaurant data.")

# 1. Load Model
try:
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

    # --- Traffic Estimation (Simulated Numerical Feature) ---
    BASE_SPEED_KM_PER_MIN = 0.5 # Corresponds to 30 km/h baseline speed
    
    # Define traffic multipliers for time-of-day simulation
    if 17 <= order_hour <= 21:
        traffic_multiplier = 1.67 
        traffic_label = 'Jam (Multiplier: 1.67x)'
        traffic_icon = "🛑"
    elif 12 <= order_hour <= 14:
        traffic_multiplier = 1.33 
        traffic_label = 'High (Multiplier: 1.33x)'
        traffic_icon = "🟡"
    elif 8 <= order_hour <= 10:
        traffic_multiplier = 1.18
        traffic_label = 'Medium (Multiplier: 1.18x)'
        traffic_icon = "🟢"
    else:
        traffic_multiplier = 1.0 
        traffic_label = 'Low (Multiplier: 1.0x)'
        traffic_icon = "💨"
        
    st.markdown(f"**Current Hour Traffic Adjustment ({current_time_ts.strftime('%H:%M')}):** {traffic_icon} `{traffic_label}`")
    st.markdown("---")


    # --- New Location Inputs (Text-based) ---
    with col1:
        st.subheader("Restaurant & Delivery Location")
        
        rest_location_input = st.text_input("Restaurant Location (City, Country)", value="New Delhi, India")
        del_location_input = st.text_input("Delivery Location (City, Country)", value="Ghaziabad, India")
        
        # Resolve coordinates from location names
        rest_lat, rest_lon, rest_location_name = fetch_coordinates(rest_location_input, WEATHER_API_KEY)
        del_lat, del_lon, del_location_name = fetch_coordinates(del_location_input, WEATHER_API_KEY)
        
        # Display resolved locations
        if rest_lat and rest_lon:
            st.info(f"Restaurant Coords: **{rest_location_name}** ({rest_lat:.4f}, {rest_lon:.4f})")
        else:
            st.error("⚠️ Restaurant Location not resolved. Check spelling/API Key.")

        if del_lat and del_lon:
            st.info(f"Delivery Coords: **{del_location_name}** ({del_lat:.4f}, {del_lon:.4f})")
        else:
            st.error("⚠️ Delivery Location not resolved. Check spelling/API Key.")


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
            
            # --- Coordinate Check ---
            if not all([rest_lat, rest_lon, del_lat, del_lon]):
                st.error("❌ Cannot predict: Please ensure both locations are resolved to coordinates before continuing.")
                st.stop()
            
            # --- Feature Engineering ---
            
            # 1. Distance
            delivery_distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
            
            # 2. Weather (fetches weather based on delivery location)
            current_temp, weather_main = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
            
            
            # 3. Traffic Adjusted Travel Time: Use API or Fallback to Simulation
            
            # Attempt to get real traffic time
            real_traffic_time_min = fetch_live_traffic_time(
                rest_lat, rest_lon, del_lat, del_lon, LIVE_TRAFFIC_API_KEY
            )
            
            if real_traffic_time_min is not None:
                # Use real API result if available
                estimated_travel_time_traffic_adjusted = real_traffic_time_min
            else:
                # Fallback to simulation using the traffic multiplier
                base_travel_time_min = delivery_distance_km / BASE_SPEED_KM_PER_MIN
                estimated_travel_time_traffic_adjusted = base_travel_time_min * traffic_multiplier
                
            
            # 4. Time features (sin/cos for cyclical time dependency)
            sin_hour = np.sin(2 * np.pi * order_hour / 24)
            cos_hour = np.cos(2 * np.pi * order_hour / 24)
            
            # Create DataFrame for prediction
            input_data = pd.DataFrame({
                'delivery_distance_km': [delivery_distance_km],
                'preparation_time_min': [prep_time],
                'restaurant_rating': [rating_rest],
                'delivery_person_rating': [rating_del],
                # Feature for the model
                'estimated_travel_time_traffic_adjusted_min': [estimated_travel_time_traffic_adjusted], 
                'Weather_Condition': [weather_main], 
                'sin_hour': [sin_hour],
                'cos_hour': [cos_hour],
                'current_temp_c': [current_temp if not np.isnan(current_temp) else 25.0]
            })
            
            # --- Prediction ---
            try:
                prediction_proba = model.predict_proba(input_data)[:, 1][0] * 100
            except ValueError as e:
                st.error(f"Prediction Error: Feature mismatch detected. The model likely needs retraining with the new numerical traffic feature. Defaulting to 50% risk. Full Error: {e}")
                prediction_proba = 50.0 
            
            # 4. Display Results
            st.subheader("🎯 Prediction Result")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Probability of Being Late", f"{prediction_proba:.2f}%")
            
            with col_res2:
                st.metric("Traffic-Adj Travel Time", f"{estimated_travel_time_traffic_adjusted:.1f} min")
                
            with col_res3:
                temp_display = f"{current_temp:.1f}" if isinstance(current_temp, (int, float)) and not np.isnan(current_temp) else 'N/A'
                st.metric("Current Weather", f"{weather_main}, {temp_display}°C")

            st.markdown(f"**Key Inputs:** Distance: {delivery_distance_km:.2f} km | Traffic: **{traffic_label}** | Weather: **{weather_main}** at {temp_display}°C")

            if prediction_proba > 60:
                st.error("⚠️ **HIGH RISK:** A late delivery is highly probable due to combined factors. Be proactive in notifying the customer. ")
                
            elif prediction_proba > 40:
                st.warning("🔶 **MODERATE RISK:** The chance of delay is significant. Monitor this order closely.")
            else:
                st.success("✅ **LOW RISK:** Delivery is likely to be on time.")

            st.markdown("---")
            st.caption(f"Prediction based on: Restaurant @ **{rest_location_name}** to Delivery @ **{del_location_name}**")
            
# --- Final Check ---
if not model:
     st.stop()
