# app.py (UPDATED autocomplete + autofill top suggestion)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
import uuid
from datetime import datetime

# ================== CONFIG ==================
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f"
HERE_API_KEY = "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw"
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # <-- REPLACE with your key (Places + Geocoding enabled)
PLACEHOLDER_CHECK = "PASTE_YOUR_API_KEY_HERE"
# ============================================

# --- Helper: Haversine ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# --- Places Autocomplete (server-side) with session token ---
@st.cache_data(ttl=180)
def places_autocomplete(query, key, session_token=None, country_code="in"):
    """
    Calls Google Places Autocomplete API (server-side).
    Returns list of dicts: [{'description':..., 'place_id':...}, ...]
    """
    if not query or len(query.strip()) < 1:
        return []
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {
            "input": query,
            "key": key,
            "types": "address",            # focus on addresses
            "components": f"country:{country_code}",  # restrict to India (change/remove if global)
            "sessiontoken": session_token
        }
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            # e.g., OVER_QUERY_LIMIT, REQUEST_DENIED -> return empty gracefully
            return []
        preds = []
        for p in data.get("predictions", []):
            preds.append({"description": p.get("description"), "place_id": p.get("place_id")})
        return preds
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_place_details(place_id, key, sessiontoken=None):
    """
    Given a place_id, call Place Details to get formatted_address and lat/lng.
    Returns (formatted_address, lat, lng) or (None, None, None)
    """
    if not place_id:
        return None, None, None
    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {"place_id": place_id, "key": key, "fields": "formatted_address,geometry", "sessiontoken": sessiontoken}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            return None, None, None
        res = data.get("result", {})
        formatted = res.get("formatted_address")
        geom = res.get("geometry", {}).get("location", {})
        return formatted, geom.get("lat"), geom.get("lng")
    except Exception:
        return None, None, None

# --- Geocode fallback ---
@st.cache_data(ttl=300)
def geocode_text_fallback(text, key):
    """Use Google Geocoding API to convert typed address -> lat/lon + formatted"""
    if not text:
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

# --- Weather & HERE traffic (unchanged logic) ---
@st.cache_data(ttl=300)
def fetch_realtime_weather(latitude, longitude, api_key):
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

def fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, api_key):
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
            return traffic_duration_min, base_travel_time_min
        else:
            return None, None
    except Exception:
        return None, None

# --- Model loading ---
st.set_page_config(page_title="Delivery Predictor", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predict the probability of a late delivery based on real-time weather, traffic-adjusted travel time, and restaurant data.")

try:
    model = joblib.load('late_delivery_predictor_model.pkl')
    st.success("Model loaded successfully for prediction.")
except FileNotFoundError:
    st.error("Error: Model file 'late_delivery_predictor_model.pkl' not found. Ensure it is uploaded.")
    model = None
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

if not model:
    st.stop()

# --- Initialize session variables ---
if "session_token" not in st.session_state:
    st.session_state["session_token"] = str(uuid.uuid4())
if "rest_address" not in st.session_state:
    st.session_state["rest_address"] = ""
if "del_address" not in st.session_state:
    st.session_state["del_address"] = ""
if "auto_fill_top" not in st.session_state:
    st.session_state["auto_fill_top"] = True  # set True to auto-fill top suggestion

# ---- UI with improved autocomplete/autofill ----
col1, col2 = st.columns(2)
current_time_ts = datetime.now()
order_hour = current_time_ts.hour
BASE_SPEED_KM_PER_MIN = 0.5

with col1:
    st.subheader("Restaurant & Delivery Location")

    # Restaurant input
    rest_query = st.text_input("Restaurant Location (start typing...)", value=st.session_state.get("rest_text", ""), key="rest_input")

    # Only search when user typed >= 3 chars
    rest_suggestions = []
    if rest_query and len(rest_query.strip()) >= 3:
        rest_suggestions = places_autocomplete(rest_query.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])

    rest_choice = None
    if rest_suggestions:
        # Auto-fill top suggestion behavior:
        top_desc = rest_suggestions[0]["description"]
        # If auto_fill_top enabled and the typed text is a prefix of top suggestion, update input to top suggestion
        if st.session_state["auto_fill_top"]:
            typed = rest_query.strip().lower()
            if top_desc.lower().startswith(typed) and top_desc.lower() != typed:
                # update text input with top suggestion (this triggers rerun)
                st.session_state["rest_input"] = top_desc
                # also fetch details immediately
                formatted, lat, lng = get_place_details(rest_suggestions[0]["place_id"], GOOGLE_MAPS_API_KEY, sessiontoken=st.session_state["session_token"])
                if formatted:
                    st.session_state["rest_address"] = formatted
                    st.session_state["rest_lat"] = lat
                    st.session_state["rest_lon"] = lng
                    rest_choice = {"address": formatted, "lat": lat, "lon": lng}
        # If user wants to pick specifically, show selectbox
        if not rest_choice:
            rest_opts = [s["description"] for s in rest_suggestions]
            sel = st.selectbox("Choose restaurant from suggestions (optional)", options=["-- pick suggestion --"] + rest_opts, key="rest_select")
            if sel != "-- pick suggestion --":
                idx = rest_opts.index(sel)
                picked = rest_suggestions[idx]
                formatted, lat, lng = get_place_details(picked["place_id"], GOOGLE_MAPS_API_KEY, sessiontoken=st.session_state["session_token"])
                if formatted:
                    st.session_state["rest_address"] = formatted
                    st.session_state["rest_lat"] = lat
                    st.session_state["rest_lon"] = lng
                    rest_choice = {"address": formatted, "lat": lat, "lon": lng}
                    st.success(f"Selected: {formatted}")

    # If no suggestions or user typed fully, fallback to geocoding when user typed > 6 chars
    if not rest_choice and rest_query and len(rest_query.strip()) >= 6:
        # If user previously had a resolved address already and text hasn't changed, reuse it
        if st.session_state.get("rest_address") and st.session_state.get("rest_text") == rest_query:
            pass
        else:
            f, la, lo = geocode_text_fallback(rest_query, GOOGLE_MAPS_API_KEY)
            if f:
                st.session_state["rest_address"] = f
                st.session_state["rest_lat"] = la
                st.session_state["rest_lon"] = lo

    # Delivery input
    del_query = st.text_input("Delivery Location (start typing...)", value=st.session_state.get("del_text", ""), key="del_input")

    del_suggestions = []
    if del_query and len(del_query.strip()) >= 3:
        del_suggestions = places_autocomplete(del_query.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])

    del_choice = None
    if del_suggestions:
        top_desc2 = del_suggestions[0]["description"]
        if st.session_state["auto_fill_top"]:
            typed2 = del_query.strip().lower()
            if top_desc2.lower().startswith(typed2) and top_desc2.lower() != typed2:
                st.session_state["del_input"] = top_desc2
                formatted2, lat2, lng2 = get_place_details(del_suggestions[0]["place_id"], GOOGLE_MAPS_API_KEY, sessiontoken=st.session_state["session_token"])
                if formatted2:
                    st.session_state["del_address"] = formatted2
                    st.session_state["del_lat"] = lat2
                    st.session_state["del_lon"] = lng2
                    del_choice = {"address": formatted2, "lat": lat2, "lon": lng2}
        if not del_choice:
            del_opts = [s["description"] for s in del_suggestions]
            sel2 = st.selectbox("Choose delivery from suggestions (optional)", options=["-- pick suggestion --"] + del_opts, key="del_select")
            if sel2 != "-- pick suggestion --":
                idx2 = del_opts.index(sel2)
                picked2 = del_suggestions[idx2]
                formatted2, lat2, lng2 = get_place_details(picked2["place_id"], GOOGLE_MAPS_API_KEY, sessiontoken=st.session_state["session_token"])
                if formatted2:
                    st.session_state["del_address"] = formatted2
                    st.session_state["del_lat"] = lat2
                    st.session_state["del_lon"] = lng2
                    del_choice = {"address": formatted2, "lat": lat2, "lon": lng2}
                    st.success(f"Selected: {formatted2}")

    if not del_choice and del_query and len(del_query.strip()) >= 6:
        f2, la2, lo2 = geocode_text_fallback(del_query, GOOGLE_MAPS_API_KEY)
        if f2:
            st.session_state["del_address"] = f2
            st.session_state["del_lat"] = la2
            st.session_state["del_lon"] = lo2

    # optional toggle for user to disable auto-fill if they prefer manual typing
    st.checkbox("Auto-fill top suggestion while typing (Google-like)", value=st.session_state["auto_fill_top"], key="auto_fill_top_checkbox")

with col2:
    st.subheader("Order Context")
    prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
    rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.9, 0.1)
    rating_del = st.slider("Delivery Person Rating (4.0 to 5.0)", 4.0, 5.0, 4.8, 0.1)

st.markdown("---")

# Predict button
if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
    rest_lat = st.session_state.get("rest_lat")
    rest_lon = st.session_state.get("rest_lon")
    del_lat = st.session_state.get("del_lat")
    del_lon = st.session_state.get("del_lon")

    if not all([rest_lat, rest_lon, del_lat, del_lon]):
        st.error("⚠️ Could not resolve one or both addresses. Try selecting a suggestion or type a clearer address.")
        st.stop()

    # Feature engineering & prediction (same logic)
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
        ratio = estimated_travel_time_traffic_adjusted / base_travel_time_min_api if base_travel_time_min_api else 1.0
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
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    with col_res1:
        st.metric("Probability of Being Late", f"{prediction_proba:.2f}%")
    with col_res2:
        st.metric("Traffic-Adj Travel Time", f"{estimated_travel_time_traffic_adjusted:.1f} min")
    with col_res3:
        st.metric("Current Weather", f"{weather_main}")
    with col_res4:
        st.metric("Temp (°C)", f"{current_temp:.1f}" if isinstance(current_temp, (int, float)) else "N/A")

    if prediction_proba > 60:
        st.error("⚠️ HIGH RISK: Likely to be late.")
    elif prediction_proba > 40:
        st.warning("🔶 MODERATE RISK: Monitor the order.")
    else:
        st.success("✅ LOW RISK: Likely on-time.")

    st.caption(f"Prediction based on: Restaurant @ **{st.session_state.get('rest_address','(not set)')}** to Delivery @ **{st.session_state.get('del_address','(not set)')}**")

# END
