
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time

# ==========================================================
# --- CONFIGURATION (UPDATE THIS) ---
# NOTE: In a real deployment, this should be stored securely as a Streamlit Secret.
# For simplicity in this standalone file, update the key below.
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f"
THRESHOLD_MIN = 10 
# ==========================================================

# --- Helper Function: Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in km between two points on the earth."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- Real-time Weather Function (API Integration) ---
def fetch_realtime_weather(latitude, longitude, api_key):
    """Fetches real-time weather data."""
    
    # Fallback if the user forgets to update the key
    if api_key == "5197fc88f5f846ee7566eb28d403c91f" or not api_key:
        return 25.0, 'Clear', 5.0 

    try:
        # Use OpenWeatherMap API
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        temp = data['main']['temp']
        weather_main = data['weather'][0]['main'] 
        return temp, weather_main
    except requests.exceptions.RequestException:
        return np.nan, 'Unknown'


# --- Streamlit Application Layout ---

st.set_page_config(page_title="Delivery Predictor", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Use the controls below to check the probability of an order being delivered late.")

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
    
    with col1:
        st.subheader("Location & Time")
        rest_lat = st.number_input("Restaurant Latitude", value=28.65, format="%.4f")
        rest_lon = st.number_input("Restaurant Longitude", value=77.20, format="%.4f")
        del_lat = st.number_input("Delivery Latitude", value=28.70, format="%.4f")
        del_lon = st.number_input("Delivery Longitude", value=77.25, format="%.4f")
        prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
        
    with col2:
        st.subheader("Ratings & Context")
        rating_rest = st.slider("Restaurant Rating", 3.0, 5.0, 4.5, 0.1)
        rating_del = st.slider("Delivery Person Rating", 4.0, 5.0, 4.8, 0.1)
        current_time_ts = pd.Timestamp(time.time(), unit='s', tz='Asia/Kolkata')
        st.info(f"Current Order Hour (Used for traffic estimation): {current_time_ts.hour}:00")

    st.markdown("---")
    
    # 3. Prediction Button Logic
    if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
        with st.spinner("Fetching real-time data and calculating..."):
            
            # --- Feature Engineering ---
            
            # Distance
            delivery_distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
            
            # Weather (Use Delivery location)
            

            current_temp, weather_main, _ = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
            
            # Time & Traffic (Simulated based on hour)
            order_hour = current_time_ts.hour
            if 17 <= order_hour <= 21:
                traffic = 'Jam' 
            elif 12 <= order_hour <= 14:
                traffic = 'High' 
            else:
                traffic = 'Medium' 
                
            sin_hour = np.sin(2 * np.pi * order_hour / 24)
            cos_hour = np.cos(2 * np.pi * order_hour / 24)
            
            # Create DataFrame for prediction
            input_data = pd.DataFrame({
                'delivery_distance_km': [delivery_distance_km],
                'preparation_time_min': [prep_time],
                'restaurant_rating': [rating_rest],
                'delivery_person_rating': [rating_del],
                'Road_Traffic_Density': [traffic],
                'Weather_Condition': [weather_main],
                'sin_hour': [sin_hour],
                'cos_hour': [cos_hour],
                'current_temp_c': [current_temp if not np.isnan(current_temp) else 25.0]
            })
            
            # --- Prediction ---
            prediction_proba = model.predict_proba(input_data)[:, 1][0] * 100
            
            # 4. Display Results
            st.subheader("🎯 Prediction Result")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Probability of Being Late", f"{prediction_proba:.2f}%")
            
            with col_res2:
                st.metric("Distance", f"{delivery_distance_km:.2f} km")
                
            with col_res3:
                st.metric("Current Weather", f"{weather_main}, {current_temp if not np.isnan(current_temp) else 'N/A'}°C")

            st.markdown(f"**Traffic Estimate:** The model used **{traffic}** traffic density based on the current hour.")

            if prediction_proba > 60:
                st.error("⚠️ **HIGH RISK:** A late delivery is highly probable. Consider adjusting ETA.")
            elif prediction_proba > 40:
                st.warning("🔶 **MODERATE RISK:** The chance of delay is significant. Monitor closely.")
            else:
                st.success("✅ **LOW RISK:** Delivery is likely to be on time.")
