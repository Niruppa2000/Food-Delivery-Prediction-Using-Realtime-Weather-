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
# You must obtain a free developer key from their portal.
HERE_API_KEY = "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw" # <<< REPLACE THIS
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
        response.raise_for_status()
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


# --- Live Traffic API Integration (Using HERE Routing API) ---
def fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, api_key):
    """
    Calls HERE Routing API with traffic enabled.
    Returns (estimated_travel_time_traffic_adjusted_min, base_travel_time_min)
    """
    # 1. Fallback/Simulation Check
    if api_key == "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw":
        st.info("⚠️ HERE API Key is a placeholder. Using time-of-day traffic simulation for prediction.")
        return None, None
        

    # 2. Actual HERE API Call
    try:
        url = "https://router.hereapi.com/v8/routes"
        
        # HERE uses coordinates in the format 'lat,lon'
        origin_coords = f"{rest_lat},{rest_lon}"
        destination_coords = f"{del_lat},{del_lon}"

        params = {
            'transportMode': 'car',
            'origin': origin_coords,
            'destination': destination_coords,
            'routingMode': 'fast',
            'trafficMode': 'realtime', # Enables traffic prediction
            'return': 'summary', # Ensures duration values are returned
            'apiKey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check for successful response and route data
        if data.get('routes') and data['routes'][0].get('sections'):
            
            summary = data['routes'][0]['sections'][0]['summary']
            
            # Duration with traffic (duration) and baseline duration (baseDuration)
            # Both are returned in seconds by the API
            traffic_duration_sec = summary.get('duration')
            base_duration_sec = summary.get('baseDuration') 
            
            if traffic_duration_sec is None or base_duration_sec is None:
                st.warning("HERE API response missing duration data. Falling back to simulation.")
                return None, None
            
            # Convert to minutes
            traffic_duration_min = traffic_duration_sec / 60.0
            base_duration_min = base_duration_sec / 60.0
        
            st.success("✅ Live traffic data successfully retrieved from HERE API.")
            return traffic_duration_min, base_duration_min
        else:
            st.warning(f"HERE API found no route or response format was unexpected. Falling back to simulation. Response status: {response.status_code}")
            return None, None
            
    except requests.exceptions.RequestException as e:
        st.error(f"HERE API Connection Error: {e}. Check API Key/permissions. Falling back to simulation.")
        return None, None
    except Exception as e:
        st.warning(f"Error parsing HERE response: {e}. Falling back to simulation.")
        return None, None


# --- Streamlit Application Layout ---

st.set_page_config(page_title="Delivery Predictor", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predict the probability of a late delivery based on real-time weather, **traffic-adjusted travel time**, and restaurant data.")

# 1. Load Model
try:
    # Set the timezone to one commonly used for this type of problem for consistent hour feature
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

    # --- Traffic Estimation (Simulated Numerical Feature) ---
    BASE_SPEED_KM_PER_MIN = 0.5 # Corresponds to 30 km/h baseline speed
    
    # Define traffic multipliers for time-of-day simulation
    # NOTE: These are now used ONLY if the API call fails or is not enabled.
    if 17 <= order_hour <= 21:
        traffic_multiplier_sim = 1.67 
        traffic_label_sim = 'Jam (Multiplier: 1.67x)'
        traffic_icon = "🛑"
    elif 12 <= order_hour <= 14:
        traffic_multiplier_sim = 1.33 
        traffic_label_sim = 'High (Multiplier: 1.33x)'
        traffic_icon = "🟡"
    elif 8 <= order_hour <= 10:
        traffic_multiplier_sim = 1.18
        traffic_label_sim = 'Medium (Multiplier: 1.18x)'
        traffic_icon = "🟢"
    else:
        traffic_multiplier_sim = 1.0 
        traffic_label_sim = 'Low (Multiplier: 1.0x)'
        traffic_icon = "💨"
        
    st.markdown(f"**Current Hour Traffic Adjustment ({current_time_ts.strftime('%H:%M')}):** {traffic_icon} `{traffic_label_sim}`")
    st.markdown("---")


    # --- Location Inputs (Text-based) ---
    with col1:
        st.subheader("Restaurant & Delivery Location")
        
        # Use placeholders from the image for consistency
        rest_location_input = st.text_input("Restaurant Location (City, Country)", value="Bangalore, India")
        del_location_input = st.text_input("Delivery Location (City, Country)", value="Mangalore, India")
        
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
        # Use values from the image for consistency (4.90 and 4.80)
        rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.9, 0.1)
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
            
            
            # 3. Traffic Adjusted Travel Time & Density Categorization
            
            # Attempt to get real traffic and base time from API
            api_result = fetch_live_traffic_time(
                rest_lat, rest_lon, del_lat, del_lon, 
                HERE_API_KEY # Using the HERE key
            )
            
            estimated_travel_time_traffic_adjusted, base_travel_time_min_api = api_result

            # Fallback logic if API fails or is not configured
            if HERE_API_KEY == "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw" or estimated_travel_time_traffic_adjusted is None:
                # Recalculate simulation values for display if fallback was used
                st.info("⚠️ Final traffic calculation using time-of-day simulation.")
                base_travel_time_min = delivery_distance_km / BASE_SPEED_KM_PER_MIN
                
                if 17 <= order_hour <= 21: traffic_multiplier_sim = 1.67
                elif 12 <= order_hour <= 14: traffic_multiplier_sim = 1.33
                elif 8 <= order_hour <= 10: traffic_multiplier_sim = 1.18
                else: traffic_multiplier_sim = 1.0 
                
                estimated_travel_time_traffic_adjusted = base_travel_time_min * traffic_multiplier_sim
                traffic_density = traffic_label_sim.split(' ')[0] # Extract 'Jam', 'High', 'Low', etc.
                base_travel_time_min_calc = base_travel_time_min # Use calculated base for ratio later
            
            else:
                # If API was successful, traffic_adjusted is the real value.
                # Now, determine the categorical density required by the model
                traffic_ratio = estimated_travel_time_traffic_adjusted / base_travel_time_min_api
                base_travel_time_min_calc = base_travel_time_min_api # Use API base for ratio later
                
                # Assign the categorical feature based on how much traffic inflated the time
                if traffic_ratio >= 1.5:
                    traffic_density = 'Jam'
                elif traffic_ratio >= 1.25:
                    traffic_density = 'High'
                elif traffic_ratio >= 1.05:
                    traffic_density = 'Medium'
                else:
                    traffic_density = 'Low'

            # --- Prediction Dataframe Construction (Fixing the Error) ---
            # We must use the 'Road_Traffic_Density' categorical column!
            input_data = pd.DataFrame({
                'delivery_distance_km': [delivery_distance_km],
                'preparation_time_min': [prep_time],
                'restaurant_rating': [rating_rest],
                'delivery_person_rating': [rating_del],
                'Weather_Condition': [weather_main], 
                'sin_hour': [np.sin(2 * np.pi * order_hour / 24)],
                'cos_hour': [np.cos(2 * np.pi * order_hour / 24)],
                'current_temp_c': [current_temp if not np.isnan(current_temp) else 25.0],
                # This is the feature your model is missing (now fixed)
                'Road_Traffic_Density': [traffic_density] 
            })
            
            # --- One-Hot Encode Categorical Features ---
            # Recreate all expected columns for Weather and Traffic Density, 
            # including the ones not present in this specific prediction.
            
            # The model expects specific, known column names from training.
            
            # Known Traffic Densities
            traffic_cols = ['Road_Traffic_Density_High', 'Road_Traffic_Density_Jam', 
                            'Road_Traffic_Density_Low', 'Road_Traffic_Density_Medium']
            # Known Weather Conditions
            weather_cols = ['Weather_Condition_Fog', 'Weather_Condition_Storm', 
                            'Weather_Condition_Sandstorms', 'Weather_Condition_Cloudy',
                            'Weather_Condition_Clear', 'Weather_Condition_Windy',
                            'Weather_Condition_Sunny', 'Weather_Condition_Haze',
                            'Weather_Condition_Snow', 'Weather_Condition_Rain']
            
            # Initialize all required dummy columns to 0
            for col in traffic_cols + weather_cols:
                if col not in input_data.columns:
                    input_data[col] = 0

            # Create the actual dummy variables for the current prediction
            input_data_encoded = pd.get_dummies(input_data, columns=['Road_Traffic_Density', 'Weather_Condition'], drop_first=False)
            
            # Ensure all the required OHE columns are present (setting to 0 if missing from this prediction)
            for col in traffic_cols + weather_cols:
                if col not in input_data_encoded.columns:
                    input_data_encoded[col] = 0
            
            # Drop the original categorical columns
            input_data_encoded = input_data_encoded.drop(columns=['Road_Traffic_Density', 'Weather_Condition'], errors='ignore')
            
            # Final feature set (must match the order and names the model was trained on)
            # NOTE: We MUST ensure the columns are ordered exactly as the model expects.
            # This order is based on a common scenario for this model structure.
            final_features = ['delivery_distance_km', 'preparation_time_min', 
                              'restaurant_rating', 'delivery_person_rating', 
                              'sin_hour', 'cos_hour', 'current_temp_c'] + traffic_cols + weather_cols
            
            # Filter and order the columns for the model
            input_data_final = input_data_encoded.filter(final_features, axis=1)
            
            # --- Prediction ---
            try:
                prediction_proba = model.predict_proba(input_data_final)[:, 1][0] * 100
            except Exception as e:
                st.error(f"Prediction Error: Final model prediction failed. Full Error: {e}")
                prediction_proba = 50.0 
            
            # 4. Display Results
            st.subheader("🎯 Prediction Result")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.metric("Probability of Being Late", f"{prediction_proba:.2f}%")
            
            with col_res2:
                # Display the traffic adjusted time (real value)
                st.metric("Traffic-Adj Travel Time", f"{estimated_travel_time_traffic_adjusted:.1f} min")
                
            with col_res3:
                temp_display = f"{current_temp:.1f}" if isinstance(current_temp, (int, float)) and not np.isnan(current_temp) else 'N/A'
                # The density shown here is the CATEGORICAL value used by the model
                st.metric("Model Traffic Density", traffic_density)

            st.markdown(f"**Key Inputs:** Distance: {delivery_distance_km:.2f} km | Traffic: **{traffic_density}** | Weather: **{weather_main}** at {temp_display}°C")

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
