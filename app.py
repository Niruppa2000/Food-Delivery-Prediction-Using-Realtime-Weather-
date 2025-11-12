import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
from datetime import datetime
from streamlit.components.v1 import html

# ==========================================================
# --- CONFIGURATION ---
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f"
HERE_API_KEY = "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw"
GOOGLE_MAPS_API_KEY = "AIzaSyDq1OSQrqwBN327aZNZVu0ho9saQvGGPxs"  # 🔑 Add your valid Google key here
PLACEHOLDER_CHECK = "PASTE_YOUR_API_KEY_HERE"
# ==========================================================


# --- Helper Function: Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


# --- Reverse Geocode with Google ---
def reverse_geocode_google(lat, lon, api_key):
    """Convert lat/lon → readable address using Google Maps Reverse Geocoding API."""
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        if data.get("results"):
            return data["results"][0]["formatted_address"]
        else:
            return f"Unknown Location ({lat:.3f}, {lon:.3f})"
    except Exception as e:
        return f"Error: {e}"


# --- Get browser GPS coordinates ---
def get_user_coordinates():
    """Injects JS to get live user location via browser and return lat/lon."""
    location_data = st.session_state.get("user_location", None)
    if location_data:
        return location_data
    get_location_script = """
        <script>
        function sendLocation() {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    const lat = position.coords.latitude;
                    const lon = position.coords.longitude;
                    const data = {lat: lat, lon: lon};
                    const json = JSON.stringify(data);
                    window.parent.postMessage({type: 'streamlit:setComponentValue', value: json}, '*');
                },
                (err) => { console.log(err); alert("Unable to get location. Please enable GPS."); }
            );
        }
        sendLocation();
        </script>
    """
    html(get_location_script, height=0)
    return None


# --- Weather Data ---
@st.cache_data(ttl=300)
def fetch_realtime_weather(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()
        return data["main"]["temp"], data["weather"][0]["main"]
    except Exception:
        return 25.0, "Clear"


# --- Live Traffic (HERE API) ---
def fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, api_key):
    if not api_key or api_key == PLACEHOLDER_CHECK:
        return None, None
    try:
        url = "https://router.hereapi.com/v8/routes"
        params = {
            "transportMode": "car",
            "origin": f"{rest_lat},{rest_lon}",
            "destination": f"{del_lat},{del_lon}",
            "routingMode": "fast",
            "trafficMode": "realtime",
            "return": "summary",
            "apiKey": api_key
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        d = r.json()
        if d.get("routes"):
            s = d["routes"][0]["sections"][0]["summary"]
            t_min = s["duration"] / 60
            b_min = s["baseDuration"] / 60
            st.success("✅ Live traffic data retrieved from HERE API.")
            return t_min, b_min
        return None, None
    except Exception:
        return None, None


# --- Streamlit Layout ---
st.set_page_config(page_title="Food Delivery Predictor", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predicts late delivery risk based on real-time weather, traffic & restaurant data.")


# --- Load Model ---
try:
    model = joblib.load("late_delivery_predictor_model.pkl")
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(f"Model load error: {e}")
    model = None


if model:
    col1, col2 = st.columns(2)
    BASE_SPEED_KM_PER_MIN = 0.5

    with col1:
        st.subheader("Restaurant & Delivery Location")

        rest_location_input = st.text_input("Restaurant Location", value="Bangalore, India", key="rest_in")
        del_location_input = st.text_input("Delivery Location", value="Mangalore, India", key="del_in")

        # --- Auto detect buttons ---
        if st.button("📍 Detect Restaurant Location Automatically"):
            coords = get_user_coordinates()
            if coords:
                st.session_state["rest_coords"] = coords
                rest_address = reverse_geocode_google(coords["lat"], coords["lon"], GOOGLE_MAPS_API_KEY)
                rest_location_input = rest_address
                st.session_state["rest_in"] = rest_address
                st.success(f"Restaurant Location: {rest_address}")

        if st.button("🏠 Detect Delivery Location Automatically"):
            coords = get_user_coordinates()
            if coords:
                st.session_state["del_coords"] = coords
                del_address = reverse_geocode_google(coords["lat"], coords["lon"], GOOGLE_MAPS_API_KEY)
                del_location_input = del_address
                st.session_state["del_in"] = del_address
                st.success(f"Delivery Location: {del_address}")

        # --- Convert to lat/lon using Google Geocoding ---
        def geocode_google(address):
            try:
                url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GOOGLE_MAPS_API_KEY}"
                r = requests.get(url)
                data = r.json()
                if data["results"]:
                    loc = data["results"][0]["geometry"]["location"]
                    return loc["lat"], loc["lng"]
                else:
                    return None, None
            except:
                return None, None

        rest_lat, rest_lon = geocode_google(rest_location_input)
        del_lat, del_lon = geocode_google(del_location_input)

        if not rest_lat or not del_lat:
            st.error("⚠️ Could not resolve one or both addresses.")

    with col2:
        st.subheader("Order Context")
        prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
        rating_rest = st.slider("Restaurant Rating", 3.0, 5.0, 4.9, 0.1)
        rating_del = st.slider("Delivery Person Rating", 4.0, 5.0, 4.8, 0.1)

    st.markdown("---")

    if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
        if not all([rest_lat, rest_lon, del_lat, del_lon]):
            st.error("❌ Please provide valid restaurant and delivery addresses.")
            st.stop()

        order_hour = datetime.now().hour
        dist = haversine(rest_lat, rest_lon, del_lat, del_lon)
        temp, weather = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
        traffic_time, base_time = fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, HERE_API_KEY)

        if not traffic_time:
            base_time = dist / BASE_SPEED_KM_PER_MIN
            traffic_time = base_time * 1.3
            traffic_density = "Medium"
        else:
            ratio = traffic_time / base_time
            if ratio >= 1.5:
                traffic_density = "Jam"
            elif ratio >= 1.25:
                traffic_density = "High"
            elif ratio >= 1.05:
                traffic_density = "Medium"
            else:
                traffic_density = "Low"

        sin_hour = np.sin(2*np.pi*order_hour/24)
        cos_hour = np.cos(2*np.pi*order_hour/24)

        X = pd.DataFrame([{
            "delivery_distance_km": dist,
            "preparation_time_min": prep_time,
            "restaurant_rating": rating_rest,
            "delivery_person_rating": rating_del,
            "Weather_Condition": weather,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "current_temp_c": temp,
            "Road_Traffic_Density": traffic_density
        }])

        pred = model.predict_proba(X)[0][1] * 100

        st.subheader("🎯 Prediction Result")
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        col_res1.metric("Late Delivery Probability", f"{pred:.2f}%")
        col_res2.metric("Distance (km)", f"{dist:.2f}")
        col_res3.metric("Weather", weather)
        col_res4.metric("Temp (°C)", f"{temp:.1f}")

        if pred > 60:
            st.error("⚠️ High Risk: Likely to be late.")
        elif pred > 40:
            st.warning("🔶 Moderate Risk: Possible delay.")
        else:
            st.success("✅ Low Risk: On-time delivery expected.")

        st.caption(f"From **{rest_location_input}** → **{del_location_input}**")

else:
    st.stop()

    

