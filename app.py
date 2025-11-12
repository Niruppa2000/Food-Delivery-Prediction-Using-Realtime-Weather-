# app.py
# Cleaned UX: no debug JSON, no extra suggestion line.
# Google key from environment; WEATHER/HERE keys inline (per user request)

import os
import uuid
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Optional: load .env in local dev if python-dotenv present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Config: GOOGLE key from env (Streamlit secret), weather & here inline
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f"
HERE_API_KEY = "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw"
PLACEHOLDER_CHECK = "PASTE_YOUR_API_KEY_HERE"

# Utility
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Google helpers
@st.cache_data(ttl=180)
def places_autocomplete(query: str, key: str, session_token: str = None, country_code: str = "in"):
    if not key or not query or len(query.strip()) < 1:
        return {}
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {"input": query, "key": key, "types": "address", "components": f"country:{country_code}"}
        if session_token:
            params["sessiontoken"] = session_token
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

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
        res = j.get("result", {})
        formatted = res.get("formatted_address")
        geom = res.get("geometry", {}).get("location", {})
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

# Weather & traffic
@st.cache_data(ttl=300)
def fetch_realtime_weather(lat: float, lon: float, api_key: str):
    if not api_key:
        return 25.0, "Clear"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=6); r.raise_for_status()
        d = r.json()
        return d["main"]["temp"], d["weather"][0]["main"]
    except Exception:
        return 25.0, "Clear"

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
        r = requests.get(url, params=params, timeout=10); r.raise_for_status()
        j = r.json()
        if j.get("routes") and j["routes"][0].get("sections"):
            s = j["routes"][0]["sections"][0]["summary"]
            return s.get("duration") / 60.0, s.get("baseDuration") / 60.0
        return None, None
    except Exception:
        return None, None

# App UI
st.set_page_config(page_title="Real-Time Delivery Late Prediction", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predict the probability of a late delivery based on real-time weather, traffic-adjusted travel time, and restaurant data.")

# Show if Google key present
if not GOOGLE_MAPS_API_KEY:
    st.warning("Google API key not found in environment. Make sure secret name is EXACTLY: GOOGLE_MAPS_API_KEY.")
else:
    st.success("Google API key found — Autocomplete enabled.")

# load model
try:
    model = joblib.load("late_delivery_predictor_model.pkl")
    st.success("Model loaded successfully for prediction.")
except FileNotFoundError:
    st.error("Model file 'late_delivery_predictor_model.pkl' not found. Upload it.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# session token for grouping autocomplete requests
if "session_token" not in st.session_state:
    st.session_state["session_token"] = str(uuid.uuid4())

# UI inputs
col1, col2 = st.columns(2)
order_hour = datetime.now().hour
BASE_SPEED_KM_PER_MIN = 0.5

with col1:
    st.subheader("Restaurant & Delivery Location")
    # user-typed inputs (auto-filled by code where possible)
    rest_input = st.text_input("Restaurant Location (start typing...)", value=st.session_state.get("rest_input", ""))
    del_input = st.text_input("Delivery Location (start typing...)", value=st.session_state.get("del_input", ""))
    auto_fill = st.checkbox("Auto-fill top suggestion while typing (Google-like)", value=True)

    # request suggestions only when user typed >=3 chars and key present
    rest_json = {}
    del_json = {}
    if rest_input and len(rest_input.strip()) >= 3 and GOOGLE_MAPS_API_KEY:
        rest_json = places_autocomplete(rest_input.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
    if del_input and len(del_input.strip()) >= 3 and GOOGLE_MAPS_API_KEY:
        del_json = places_autocomplete(del_input.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])

    # Process restaurant suggestions: prefer auto-fill top suggestion
    rest_predictions = rest_json.get("predictions", []) if isinstance(rest_json, dict) else []
    if rest_predictions and auto_fill:
        top = rest_predictions[0].get("description")
        # if typed is a prefix of top suggestion, replace input with top (Google-like predictive)
        typed = rest_input.strip().lower()
        if top and top.lower().startswith(typed) and top.lower() != typed:
            # auto-update input - store back to session state and treat as accepted top suggestion
            st.session_state["rest_input"] = top
            rest_input = top
            # fetch place details for top suggestion
            formatted, lat, lon = get_place_details(rest_predictions[0].get("place_id"), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
            if formatted:
                st.session_state["rest_address"] = formatted
                st.session_state["rest_lat"] = lat
                st.session_state["rest_lon"] = lon

    # If user prefers to pick other suggestion, show compact expander (no persistent line)
    if rest_predictions and len(rest_predictions) > 1:
        with st.expander("Show other restaurant suggestions", expanded=False):
            # compact list of descriptions
            options = [p.get("description") for p in rest_predictions]
            pick = st.selectbox("Other suggestions", options=["-- pick --"] + options, key="rest_alt")
            if pick and pick != "-- pick --":
                idx = options.index(pick)
                pid = rest_predictions[idx].get("place_id")
                formatted, lat, lon = get_place_details(pid, GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
                if formatted:
                    st.session_state["rest_input"] = formatted
                    st.session_state["rest_address"] = formatted
                    st.session_state["rest_lat"] = lat
                    st.session_state["rest_lon"] = lon
                    st.success(f"Selected: {formatted}")

    # Delivery suggestions - same behavior
    del_predictions = del_json.get("predictions", []) if isinstance(del_json, dict) else []
    if del_predictions and auto_fill:
        top2 = del_predictions[0].get("description")
        typed2 = del_input.strip().lower()
        if top2 and top2.lower().startswith(typed2) and top2.lower() != typed2:
            st.session_state["del_input"] = top2
            del_input = top2
            formatted2, lat2, lon2 = get_place_details(del_predictions[0].get("place_id"), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
            if formatted2:
                st.session_state["del_address"] = formatted2
                st.session_state["del_lat"] = lat2
                st.session_state["del_lon"] = lon2

    if del_predictions and len(del_predictions) > 1:
        with st.expander("Show other delivery suggestions", expanded=False):
            options2 = [p.get("description") for p in del_predictions]
            pick2 = st.selectbox("Other suggestions", options=["-- pick --"] + options2, key="del_alt")
            if pick2 and pick2 != "-- pick --":
                idx2 = options2.index(pick2)
                pid2 = del_predictions[idx2].get("place_id")
                formatted2, lat2, lon2 = get_place_details(pid2, GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
                if formatted2:
                    st.session_state["del_input"] = formatted2
                    st.session_state["del_address"] = formatted2
                    st.session_state["del_lat"] = lat2
                    st.session_state["del_lon"] = lon2
                    st.success(f"Selected: {formatted2}")

with col2:
    st.subheader("Order Context")
    prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
    rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.5, 0.1)
    rating_del = st.slider("Delivery Person Rating (4.0 to 5.0)", 4.0, 5.0, 4.7, 0.1)

st.markdown("---")

# PREDICT
if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
    rest_lat = st.session_state.get("rest_lat")
    rest_lon = st.session_state.get("rest_lon")
    del_lat = st.session_state.get("del_lat")
    del_lon = st.session_state.get("del_lon")

    # Fallback geocoding if lat/lon not available but user typed something
    if not all([rest_lat, rest_lon]) and st.session_state.get("rest_input") and GOOGLE_MAPS_API_KEY:
        formatted_f, la_f, lo_f = geocode_text_fallback(st.session_state.get("rest_input"), GOOGLE_MAPS_API_KEY)
        if formatted_f:
            st.session_state["rest_address"] = formatted_f
            st.session_state["rest_lat"] = la_f
            st.session_state["rest_lon"] = lo_f
            rest_lat = la_f; rest_lon = lo_f

    if not all([del_lat, del_lon]) and st.session_state.get("del_input") and GOOGLE_MAPS_API_KEY:
        formatted_f2, la_f2, lo_f2 = geocode_text_fallback(st.session_state.get("del_input"), GOOGLE_MAPS_API_KEY)
        if formatted_f2:
            st.session_state["del_address"] = formatted_f2
            st.session_state["del_lat"] = la_f2
            st.session_state["del_lon"] = lo_f2
            del_lat = la_f2; del_lon = lo_f2

    if not all([rest_lat, rest_lon, del_lat, del_lon]):
        st.error("⚠️ Could not resolve one or both addresses. Try typing a clearer address or use the 'Show other suggestions' expander to pick one.")
        st.stop()

    # Feature engineering and predictions (unchanged)
    delivery_distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
    current_temp, weather_main = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
    api_res = fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, HERE_API_KEY)
    estimated_travel_time_traffic_adjusted, base_travel_time_min_api = api_res

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

    input_df = pd.DataFrame([{
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
        proba = model.predict_proba(input_df)[:,1][0] * 100
    except Exception as e:
        st.error(f"Prediction error: {e}")
        proba = 50.0

    st.subheader("🎯 Prediction Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Probability of Being Late", f"{proba:.2f}%")
    c2.metric("Traffic-Adj Travel Time", f"{estimated_travel_time_traffic_adjusted:.1f} min")
    c3.metric("Current Weather", weather_main)
    c4.metric("Temp (°C)", f"{current_temp:.1f}" if isinstance(current_temp, (int, float)) else "N/A")

    if proba > 60:
        st.error("⚠️ HIGH RISK: Likely to be late.")
    elif proba > 40:
        st.warning("🔶 MODERATE RISK: Monitor the order.")
    else:
        st.success("✅ LOW RISK: Delivery likely on-time.")

    st.caption(f"Prediction based on: Restaurant @ **{st.session_state.get('rest_address','(not set)')}** to Delivery @ **{st.session_state.get('del_address','(not set)')}**")

# End of app.py
