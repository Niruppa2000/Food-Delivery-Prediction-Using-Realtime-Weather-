# app.py
# Real-Time Food Delivery Late Prediction
# - GOOGLE_MAPS_API_KEY read from environment (e.g., GitHub Secret or Streamlit Secrets)
# - WEATHER_API_KEY and HERE_API_KEY set inline per user request

import os
import uuid
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

# -------------------------
# Load env (safe to keep; if running in hosting with secrets, env will already be populated)
# -------------------------
load_dotenv()  # will read .env locally (no-op if not present)

# -------------------------
# Keys configuration
# -------------------------
# Google key is picked from environment (so GitHub Secret or host secret works)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

# Per your request: place the Weather & HERE API keys directly inside the script
# (Make sure you are comfortable with these being in the file)
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f"
HERE_API_KEY = "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw"

PLACEHOLDER_CHECK = "PASTE_YOUR_API_KEY_HERE"

# -------------------------
# Utility functions
# -------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# -------------------------
# Google Places / Geocoding helpers (server-side)
# -------------------------
@st.cache_data(ttl=180)
def places_autocomplete(query: str, key: str, session_token: str = None, country_code: str = "in"):
    if not key or not query or len(query.strip()) < 1:
        return []
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {
            "input": query,
            "key": key,
            "types": "address",
            "components": f"country:{country_code}",
        }
        if session_token:
            params["sessiontoken"] = session_token
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status not in ("OK", "ZERO_RESULTS"):
            return []
        preds = []
        for p in data.get("predictions", []):
            preds.append({"description": p.get("description"), "place_id": p.get("place_id")})
        return preds
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_place_details(place_id: str, key: str, session_token: str = None):
    if not place_id or not key:
        return None, None, None
    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {"place_id": place_id, "key": key, "fields": "formatted_address,geometry"}
        if session_token:
            params["sessiontoken"] = session_token
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        j = r.json()
        if j.get("status") != "OK":
            return None, None, None
        result = j.get("result", {})
        formatted = result.get("formatted_address")
        geom = result.get("geometry", {}).get("location", {})
        return formatted, geom.get("lat"), geom.get("lng")
    except Exception:
        return None, None, None

@st.cache_data(ttl=300)
def geocode_text_fallback(text: str, key: str):
    if not text or not key:
        return None, None, None
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": text, "key": key}
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "OK" and j.get("results"):
            res = j["results"][0]
            formatted = res.get("formatted_address")
            loc = res.get("geometry", {}).get("location", {})
            return formatted, loc.get("lat"), loc.get("lng")
        return None, None, None
    except Exception:
        return None, None, None

# -------------------------
# Weather & Traffic helpers
# -------------------------
@st.cache_data(ttl=300)
def fetch_realtime_weather(lat: float, lon: float, api_key: str):
    if not api_key:
        return 25.0, "Clear"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        d = r.json()
        temp = d["main"]["temp"]
        weather_main = d["weather"][0]["main"]
        return temp, weather_main
    except Exception:
        return 25.0, "Clear"

def fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, api_key):
    if not api_key or api_key == PLACEHOLDER_CHECK:
        return None, None
    try:
        url = "https://router.hereapi.com/v8/routes"
        origin = f"{rest_lat},{rest_lon}"
        dest = f"{del_lat},{del_lon}"
        params = {
            "transportMode": "car",
            "origin": origin,
            "destination": dest,
            "routingMode": "fast",
            "trafficMode": "realtime",
            "return": "summary",
            "apiKey": api_key
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        if j.get("routes") and j["routes"][0].get("sections"):
            s = j["routes"][0]["sections"][0]["summary"]
            traffic_min = s.get("duration") / 60.0
            base_min = s.get("baseDuration") / 60.0
            return traffic_min, base_min
        return None, None
    except Exception:
        return None, None

# -------------------------
# Streamlit app UI + logic
# -------------------------
st.set_page_config(page_title="Real-Time Delivery Late Prediction", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predict the probability of a late delivery based on real-time weather, traffic-adjusted travel time, and restaurant data.")

# Key status messages (do not print the key itself)
if not GOOGLE_MAPS_API_KEY:
    st.warning("Google API key not found in environment. Places Autocomplete & Geocoding will not work until you set the secret.")
if not WEATHER_API_KEY:
    st.info("Weather API key is empty or missing; simulated weather will be used.")
if not HERE_API_KEY:
    st.info("HERE API key is empty or missing; traffic will be simulated.")

# Load model
try:
    model = joblib.load("late_delivery_predictor_model.pkl")
    st.success("Model loaded successfully for prediction.")
except FileNotFoundError:
    st.error("Model file 'late_delivery_predictor_model.pkl' not found. Upload it to the app folder.")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

if not model:
    st.stop()

# session token for Google Places grouping
if "session_token" not in st.session_state:
    st.session_state["session_token"] = str(uuid.uuid4())

# UI columns
col1, col2 = st.columns(2)
order_hour = datetime.now().hour
BASE_SPEED_KM_PER_MIN = 0.5

with col1:
    st.subheader("Restaurant & Delivery Location")
    rest_query = st.text_input("Restaurant Location (start typing...)", value=st.session_state.get("rest_text", ""), key="rest_input")
    del_query = st.text_input("Delivery Location (start typing...)", value=st.session_state.get("del_text", ""), key="del_input")
    st.checkbox("Auto-fill top suggestion while typing (Google-like)", value=st.session_state.get("auto_fill_top", True), key="auto_fill_top_checkbox")

    rest_suggestions = []
    del_suggestions = []
    if rest_query and len(rest_query.strip()) >= 3 and GOOGLE_MAPS_API_KEY:
        rest_suggestions = places_autocomplete(rest_query.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
    if del_query and len(del_query.strip()) >= 3 and GOOGLE_MAPS_API_KEY:
        del_suggestions = places_autocomplete(del_query.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])

    rest_choice = None
    if rest_suggestions:
        top_desc = rest_suggestions[0]["description"]
        if st.session_state.get("auto_fill_top", True):
            typed = rest_query.strip().lower()
            if top_desc.lower().startswith(typed) and top_desc.lower() != typed:
                st.session_state["rest_input"] = top_desc
                formatted, lat, lng = get_place_details(rest_suggestions[0]["place_id"], GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
                if formatted:
                    st.session_state["rest_address"] = formatted
                    st.session_state["rest_lat"] = lat
                    st.session_state["rest_lon"] = lng
                    rest_choice = {"address": formatted, "lat": lat, "lon": lng}
        if not rest_choice:
            rest_opts = [s["description"] for s in rest_suggestions]
            sel = st.selectbox("Choose restaurant suggestion (optional)", options=["-- pick suggestion --"] + rest_opts, key="rest_select")
            if sel != "-- pick suggestion --":
                idx = rest_opts.index(sel)
                picked = rest_suggestions[idx]
                formatted, lat, lng = get_place_details(picked["place_id"], GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
                if formatted:
                    st.session_state["rest_address"] = formatted
                    st.session_state["rest_lat"] = lat
                    st.session_state["rest_lon"] = lng
                    rest_choice = {"address": formatted, "lat": lat, "lon": lng}
                    st.success(f"Selected: {formatted}")

    del_choice = None
    if del_suggestions:
        top_desc2 = del_suggestions[0]["description"]
        if st.session_state.get("auto_fill_top", True):
            typed2 = del_query.strip().lower()
            if top_desc2.lower().startswith(typed2) and top_desc2.lower() != typed2:
                st.session_state["del_input"] = top_desc2
                formatted2, lat2, lng2 = get_place_details(del_suggestions[0]["place_id"], GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
                if formatted2:
                    st.session_state["del_address"] = formatted2
                    st.session_state["del_lat"] = lat2
                    st.session_state["del_lon"] = lng2
                    del_choice = {"address": formatted2, "lat": lat2, "lon": lng2}
        if not del_choice:
            del_opts = [s["description"] for s in del_suggestions]
            sel2 = st.selectbox("Choose delivery suggestion (optional)", options=["-- pick suggestion --"] + del_opts, key="del_select")
            if sel2 != "-- pick suggestion --":
                idx2 = del_opts.index(sel2)
                picked2 = del_suggestions[idx2]
                formatted2, lat2, lng2 = get_place_details(picked2["place_id"], GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
                if formatted2:
                    st.session_state["del_address"] = formatted2
                    st.session_state["del_lat"] = lat2
                    st.session_state["del_lon"] = lng2
                    del_choice = {"address": formatted2, "lat": lat2, "lon": lng2}
                    st.success(f"Selected: {formatted2}")

    if ("rest_lat" not in st.session_state or "rest_lon" not in st.session_state) and rest_query and len(rest_query.strip()) >= 6 and GOOGLE_MAPS_API_KEY:
        formatted, la, lo = geocode_text_fallback(rest_query, GOOGLE_MAPS_API_KEY)
        if formatted:
            st.session_state["rest_address"] = formatted
            st.session_state["rest_lat"] = la
            st.session_state["rest_lon"] = lo

    if ("del_lat" not in st.session_state or "del_lon" not in st.session_state) and del_query and len(del_query.strip()) >= 6 and GOOGLE_MAPS_API_KEY:
        formatted2, la2, lo2 = geocode_text_fallback(del_query, GOOGLE_MAPS_API_KEY)
        if formatted2:
            st.session_state["del_address"] = formatted2
            st.session_state["del_lat"] = la2
            st.session_state["del_lon"] = lo2

    if "rest_lat" not in st.session_state or "rest_lon" not in st.session_state:
        st.info("Tip: Type 'Koramangala, Bangalore' or 'MG Road, Bangalore' and select a suggestion to resolve location.")
    if "del_lat" not in st.session_state or "del_lon" not in st.session_state:
        st.info("Tip: Type 'Hosur Road, Bangalore' or 'Electronic City, Bangalore' and select a suggestion to resolve location.")

with col2:
    st.subheader("Order Context")
    prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
    rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.5, 0.1)
    rating_del = st.slider("Delivery Person Rating (4.0 to 5.0)", 4.0, 5.0, 4.7, 0.1)

st.markdown("---")

if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
    rest_lat = st.session_state.get("rest_lat")
    rest_lon = st.session_state.get("rest_lon")
    del_lat = st.session_state.get("del_lat")
    del_lon = st.session_state.get("del_lon")

    if not all([rest_lat, rest_lon, del_lat, del_lon]):
        st.error("⚠️ Could not resolve one or both addresses. Try selecting a suggestion from the dropdown or type a clearer address.")
        st.stop()

    delivery_distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
    current_temp, weather_main = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
    api_result = fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, HERE_API_KEY)
    estimated_travel_time_traffic_adjusted, base_travel_time_min_api = api_result

    if estimated_travel_time_traffic_adjusted is None:
        base_travel_time_min = delivery_distance_km / BASE_SPEED_KM_PER_MIN
        if 17 <= order_hour <= 21:
            multiplier = 1.67
        elif 12 <= order_hour <= 14:
            multiplier = 1.33
        elif 8 <= order_hour <= 10:
            multiplier = 1.18
        else:
            multiplier = 1.0
        estimated_travel_time_traffic_adjusted = base_travel_time_min * multiplier
        traffic_density = "High" if multiplier > 1.2 else "Medium" if multiplier > 1.05 else "Low"
    else:
        ratio = estimated_travel_time_traffic_adjusted / (base_travel_time_min_api if base_travel_time_min_api else 1.0)
        if ratio >= 1.5:
            traffic_density = 'Jam'
        elif ratio >= 1.25:
            traffic_density = 'High'
        elif ratio >= 1.05:
            traffic_density = 'Medium'
        else:
            traffic_density = 'Low'

    sin_hour = np.sin(2 * np.pi * order_hour / 24)
    cos_hour = np.cos(2 * np.pi * order_hour / 24)

    input_data_final = pd.DataFrame([{
        'delivery_distance_km': delivery_distance_km,
        'preparation_time_min': prep_time,
        'restaurant_rating': rating_rest,
        'delivery_person_rating': rating_del,
        'Road_Traffic_Density': traffic_density,
        'Weather_Condition': weather_main,
        'sin_hour': sin_hour,
        'cos_hour': cos_hour,
        'current_temp_c': current_temp
    }])

    try:
        prediction_proba = model.predict_proba(input_data_final)[:, 1][0] * 100
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        prediction_proba = 50.0

    st.subheader("🎯 Prediction Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of Being Late", f"{prediction_proba:.2f}%")
    c2.metric("Traffic-Adj Travel Time", f"{estimated_travel_time_traffic_adjusted:.1f} min")
    c3.metric("Current Weather", weather_main)
    c4.metric("Temp (°C)", f"{current_temp:.1f}" if isinstance(current_temp, (int, float)) else "N/A")

    if prediction_proba > 60:
        st.error("⚠️ HIGH RISK: Likely to be late.")
    elif prediction_proba > 40:
        st.warning("🔶 MODERATE RISK: Monitor the order.")
    else:
        st.success("✅ LOW RISK: Delivery likely on-time.")

    st.caption(f"Prediction based on: Restaurant @ **{st.session_state.get('rest_address','(not set)')}** to Delivery @ **{st.session_state.get('del_address','(not set)')}**")

# End of file
