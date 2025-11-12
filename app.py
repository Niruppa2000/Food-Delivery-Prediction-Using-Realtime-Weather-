import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import time
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

# --- Google Places Autocomplete (server-side) ---
@st.cache_data(ttl=300)
def places_autocomplete(query, key, sessiontoken=None):
    """
    Call Google Places Autocomplete API to get text suggestions.
    Returns list of dicts: [{'description':..., 'place_id':...}, ...]
    """
    if not query or len(query.strip()) < 1:
        return []
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {
            "input": query,
            "key": key,
            "components": "country:in",  # restrict to India; remove or change if global
            "types": "geocode",  # restrict to addresses
        }
        if sessiontoken:
            params["sessiontoken"] = sessiontoken
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            # e.g., OVER_QUERY_LIMIT, REQUEST_DENIED
            return []
        results = []
        for pred in data.get("predictions", []):
            results.append({"description": pred.get("description"), "place_id": pred.get("place_id")})
        return results
    except Exception:
        return []

@st.cache_data(ttl=300)
def get_place_details(place_id, key):
    """
    Given a place_id, call Place Details to get formatted_address and lat/lng.
    Returns (formatted_address, lat, lng) or (None, None, None)
    """
    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {"place_id": place_id, "key": key, "fields": "formatted_address,geometry"}
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "OK":
            return None, None, None
        result = data.get("result", {})
        formatted = result.get("formatted_address")
        geom = result.get("geometry", {}).get("location", {})
        lat = geom.get("lat")
        lng = geom.get("lng")
        return formatted, lat, lng
    except Exception:
        return None, None, None

# --- Reverse Geocode fallback (Google) ---
def reverse_geocode_google(lat, lon, api_key):
    try:
        url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        d = r.json()
        if d.get("status") == "OK" and d.get("results"):
            return d["results"][0]["formatted_address"]
        return None
    except Exception:
        return None

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

# --- Model load ---
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

# ---- UI with autocomplete ----
col1, col2 = st.columns(2)
current_time_ts = datetime.now()
order_hour = current_time_ts.hour
BASE_SPEED_KM_PER_MIN = 0.5

with col1:
    st.subheader("Restaurant & Delivery Location")

    # Restaurant input with autocomplete logic
    rest_query = st.text_input("Restaurant Location (start typing...)",
                               value=st.session_state.get("rest_text", "Bangalore, India"),
                               key="rest_input")

    # When the user types 3+ chars, fetch suggestions
    rest_suggestions = []
    if rest_query and len(rest_query.strip()) >= 3:
        rest_suggestions = places_autocomplete(rest_query, GOOGLE_MAPS_API_KEY)

    rest_choice = None
    if rest_suggestions:
        # Build options list for selectbox
        rest_opts = [s["description"] for s in rest_suggestions]
        # Prepend a hint
        sel = st.selectbox("Choose restaurant from suggestions", options=["-- pick suggestion --"] + rest_opts, key="rest_select")
        if sel != "-- pick suggestion --":
            idx = rest_opts.index(sel)
            picked = rest_suggestions[idx]
            # Fetch details
            formatted, lat, lng = get_place_details(picked["place_id"], GOOGLE_MAPS_API_KEY)
            if formatted:
                st.session_state["rest_text"] = formatted
                st.session_state["rest_address"] = formatted
                st.session_state["rest_lat"] = lat
                st.session_state["rest_lon"] = lng
                rest_choice = {"address": formatted, "lat": lat, "lon": lng}
                st.success(f"Selected: {formatted}")

    # If user didn't pick but already had a previous selection in session_state, show it
    if "rest_address" in st.session_state and not rest_choice:
        st.markdown(f"**Restaurant (selected):** {st.session_state['rest_address']}")

    # Delivery input with autocomplete logic
    del_query = st.text_input("Delivery Location (start typing...)", value=st.session_state.get("del_text", "Mangalore, India"), key="del_input")

    del_suggestions = []
    if del_query and len(del_query.strip()) >= 3:
        del_suggestions = places_autocomplete(del_query, GOOGLE_MAPS_API_KEY)

    del_choice = None
    if del_suggestions:
        del_opts = [s["description"] for s in del_suggestions]
        sel2 = st.selectbox("Choose delivery from suggestions", options=["-- pick suggestion --"] + del_opts, key="del_select")
        if sel2 != "-- pick suggestion --":
            idx2 = del_opts.index(sel2)
            picked2 = del_suggestions[idx2]
            formatted2, lat2, lng2 = get_place_details(picked2["place_id"], GOOGLE_MAPS_API_KEY)
            if formatted2:
                st.session_state["del_text"] = formatted2
                st.session_state["del_address"] = formatted2
                st.session_state["del_lat"] = lat2
                st.session_state["del_lon"] = lng2
                del_choice = {"address": formatted2, "lat": lat2, "lon": lng2}
                st.success(f"Selected: {formatted2}")

    if "del_address" in st.session_state and not del_choice:
        st.markdown(f"**Delivery (selected):** {st.session_state['del_address']}")

    # Fallback: if user didn't pick suggestion but typed a full address, try geocoding it
    def geocode_text_fallback(text):
        if not text: return None, None, None
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {"address": text, "key": GOOGLE_MAPS_API_KEY}
            r = requests.get(url, params=params, timeout=5)
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

    # Try to ensure we have rest_lat/lon & del_lat/lon in session_state if user didn't pick suggestion
    if "rest_lat" not in st.session_state or "rest_lon" not in st.session_state:
        # attempt to geocode the typed text if user has entered something
        if rest_query and len(rest_query.strip()) >= 5:
            f, la, lo = geocode_text_fallback(rest_query)
            if f:
                st.session_state["rest_address"] = f
                st.session_state["rest_lat"] = la
                st.session_state["rest_lon"] = lo

    if "del_lat" not in st.session_state or "del_lon" not in st.session_state:
        if del_query and len(del_query.strip()) >= 5:
            f2, la2, lo2 = geocode_text_fallback(del_query)
            if f2:
                st.session_state["del_address"] = f2
                st.session_state["del_lat"] = la2
                st.session_state["del_lon"] = lo2

with col2:
    st.subheader("Order Context")
    prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
    rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.9, 0.1)
    rating_del = st.slider("Delivery Person Rating (4.0 to 5.0)", 4.0, 5.0, 4.8, 0.1)

st.markdown("---")

# Predict button
if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
    # Ensure coordinates exist
    rest_lat = st.session_state.get("rest_lat")
    rest_lon = st.session_state.get("rest_lon")
    del_lat = st.session_state.get("del_lat")
    del_lon = st.session_state.get("del_lon")

    if not all([rest_lat, rest_lon, del_lat, del_lon]):
        st.error("⚠️ Could not resolve one or both addresses. Try selecting a suggestion from the dropdown or type a clearer address.")
        st.stop()

    # Feature engineering
    delivery_distance_km = haversine(rest_lat, rest_lon, del_lat, del_lon)
    current_temp, weather_main = fetch_realtime_weather(del_lat, del_lon, WEATHER_API_KEY)
    api_result = fetch_live_traffic_time(rest_lat, rest_lon, del_lat, del_lon, HERE_API_KEY)
    estimated_travel_time_traffic_adjusted, base_travel_time_min_api = api_result

    # Fallback traffic simulation if HERE failed
    if estimated_travel_time_traffic_adjusted is None:
        base_travel_time_min = delivery_distance_km / BASE_SPEED_KM_PER_MIN
        # simple time-of-day multipliers
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

    # Build input df
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

    # Show resolved addresses
    st.caption(f"Prediction based on: Restaurant @ **{st.session_state.get('rest_address')}** to Delivery @ **{st.session_state.get('del_address')}**")

# END of file

