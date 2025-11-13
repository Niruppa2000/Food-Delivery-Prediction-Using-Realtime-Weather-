import os
import uuid
import requests
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# Optional local .env loader (won't fail if python-dotenv is missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------------- CONFIG ----------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
WEATHER_API_KEY = "5197fc88f5f846ee7566eb28d403c91f"   # per your request (kept inline)
HERE_API_KEY = "9JI9eOC0auXHPTmtQ5SohrGPp4WjOaq90TRCjfa-Czw"  # per your request
PLACEHOLDER_CHECK = "PASTE_YOUR_API_KEY_HERE"

# ---------------- Utilities ----------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ---------------- Google server-side helpers ----------------
@st.cache_data(ttl=180)
def places_autocomplete(query: str, key: str, session_token: str = None, country_code: str = "in"):
    """Return raw autocomplete JSON (server-side)."""
    if not key or not query or len(query.strip()) < 1:
        return {}
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {
            "input": query,
            "key": key,
            "types": "address",
            "components": f"country:{country_code}"
        }
        if session_token:
            params["sessiontoken"] = session_token
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

@st.cache_data(ttl=300)
def get_place_details(place_id: str, key: str, session_token: str = None):
    """Return (formatted_address, lat, lon) or (None, None, None)."""
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
        loc = res.get("geometry", {}).get("location", {})
        return formatted, loc.get("lat"), loc.get("lng")
    except Exception:
        return None, None, None

@st.cache_data(ttl=300)
def geocode_text_fallback(text: str, key: str):
    """Server-side geocode fallback for typed free-text addresses."""
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

# ---------------- Weather & Traffic helpers (unchanged) ----------------
@st.cache_data(ttl=300)
def fetch_realtime_weather(lat: float, lon: float, api_key: str):
    if not api_key:
        return 25.0, "Clear"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
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
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        if j.get("routes") and j["routes"][0].get("sections"):
            s = j["routes"][0]["sections"][0]["summary"]
            return s.get("duration") / 60.0, s.get("baseDuration") / 60.0
        return None, None
    except Exception:
        return None, None

# ---------------- UI layout ----------------
st.set_page_config(page_title="Real-Time Delivery Late Prediction", layout="wide")
st.title("🍔 Real-Time Food Delivery Late Prediction")
st.markdown("Predict late deliveries using real-time weather, traffic-adjusted travel time, and exact address autocompletion.")

# Show a clear status message about Google key presence
if not GOOGLE_MAPS_API_KEY:
    st.warning("Google API key not found in environment. Add secret 'GOOGLE_MAPS_API_KEY' in Streamlit Secrets.")
else:
    st.success("Google API key found — Autocomplete enabled.")

# Load the trained model (must be present)
try:
    model = joblib.load("late_delivery_predictor_model.pkl")
    st.success("Model loaded successfully for prediction.")
except FileNotFoundError:
    st.error("Model file 'late_delivery_predictor_model.pkl' not found. Upload it to the app folder.")
    st.stop()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Use a session token for grouping Google Autocomplete calls (best practice)
if "session_token" not in st.session_state:
    st.session_state["session_token"] = str(uuid.uuid4())

# Layout columns
col1, col2 = st.columns(2)
order_hour = datetime.now().hour
BASE_SPEED_KM_PER_MIN = 0.5

with col1:
    st.subheader("Restaurant & Delivery Location")

    # Text inputs (user types here)
    rest_typed = st.text_input("Restaurant Location (start typing...)", value=st.session_state.get("rest_typed", ""), key="rest_input")
    del_typed  = st.text_input("Delivery Location (start typing...)", value=st.session_state.get("del_typed", ""), key="del_input")
    auto_fill_option = st.checkbox("Auto-fill top suggestion while typing (Google-like)", value=True)

    # Get suggestions server-side when user typed >=3 chars
    rest_suggestions_json = {}
    del_suggestions_json  = {}
    if rest_typed and len(rest_typed.strip()) >= 3 and GOOGLE_MAPS_API_KEY:
        rest_suggestions_json = places_autocomplete(rest_typed.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
    if del_typed and len(del_typed.strip()) >= 3 and GOOGLE_MAPS_API_KEY:
        del_suggestions_json = places_autocomplete(del_typed.strip(), GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])

    # Extract descriptions for dropdown (if any)
    rest_preds = rest_suggestions_json.get("predictions", []) if isinstance(rest_suggestions_json, dict) else []
    rest_descriptions = [p.get("description") for p in rest_preds] if rest_preds else []

    del_preds = del_suggestions_json.get("predictions", []) if isinstance(del_suggestions_json, dict) else []
    del_descriptions = [p.get("description") for p in del_preds] if del_preds else []

    # ---------- RESTAURANT: auto-fill top suggestion or show inline dropdown ----------
    # Auto-fill top suggestion behavior
    if rest_descriptions and auto_fill_option:
        top_desc = rest_descriptions[0]
        typed_lower = rest_typed.strip().lower()
        if top_desc and top_desc.lower().startswith(typed_lower) and top_desc.lower() != typed_lower:
            # Accept top suggestion automatically (update session input and fetch details)
            st.session_state["rest_typed"] = top_desc
            rest_typed = top_desc
            # Get place details for the top suggestion
            top_place_id = rest_preds[0].get("place_id")
            formatted, lat, lon = get_place_details(top_place_id, GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
            if formatted:
                st.session_state["rest_address"] = formatted
                st.session_state["rest_lat"] = lat
                st.session_state["rest_lon"] = lon

    # Show compact inline dropdown (no label) for suggestions so user can pick another
    if rest_descriptions:
        sel = st.selectbox("", options=["(keep typed)"] + rest_descriptions, index=0, key="rest_select", label_visibility="collapsed")
        if sel != "(keep typed)":
            idx = rest_descriptions.index(sel)
            pid = rest_preds[idx].get("place_id")
            formatted, lat, lon = get_place_details(pid, GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
            if formatted:
                st.session_state["rest_typed"] = formatted
                st.session_state["rest_address"] = formatted
                st.session_state["rest_lat"] = lat
                st.session_state["rest_lon"] = lon
                # also set the visible text input to formatted
                # (st.experimental_rerun is not needed; session state update updates input next run)

    # If no suggestions, do not show anything (keeps UI clean)

    # ---------- DELIVERY: same behavior ----------
    if del_descriptions and auto_fill_option:
        top_desc2 = del_descriptions[0]
        typed_lower2 = del_typed.strip().lower()
        if top_desc2 and top_desc2.lower().startswith(typed_lower2) and top_desc2.lower() != typed_lower2:
            st.session_state["del_typed"] = top_desc2
            del_typed = top_desc2
            top_pid = del_preds[0].get("place_id")
            formatted2, lat2, lon2 = get_place_details(top_pid, GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
            if formatted2:
                st.session_state["del_address"] = formatted2
                st.session_state["del_lat"] = lat2
                st.session_state["del_lon"] = lon2

    if del_descriptions:
        sel2 = st.selectbox("", options=["(keep typed)"] + del_descriptions, index=0, key="del_select", label_visibility="collapsed")
        if sel2 != "(keep typed)":
            idx2 = del_descriptions.index(sel2)
            pid2 = del_preds[idx2].get("place_id")
            formatted2, lat2, lon2 = get_place_details(pid2, GOOGLE_MAPS_API_KEY, session_token=st.session_state["session_token"])
            if formatted2:
                st.session_state["del_typed"] = formatted2
                st.session_state["del_address"] = formatted2
                st.session_state["del_lat"] = lat2
                st.session_state["del_lon"] = lon2

with col2:
    st.subheader("Order Context")
    prep_time = st.slider("Preparation Time (min)", 5, 45, 20)
    rating_rest = st.slider("Restaurant Rating (3.0 to 5.0)", 3.0, 5.0, 4.5, 0.1)
    rating_del = st.slider("Delivery Person Rating (4.0 to 5.0)", 4.0, 5.0, 4.7, 0.1)

st.markdown("---")

# ---------- PREDICT ----------
if st.button("PREDICT LATE DELIVERY RISK", use_container_width=True, type="primary"):
    # Try to get lat/lon from session state first
    rest_lat = st.session_state.get("rest_lat")
    rest_lon = st.session_state.get("rest_lon")
    del_lat = st.session_state.get("del_lat")
    del_lon = st.session_state.get("del_lon")

    # If lat/lon missing, try geocode fallback using typed text (server-side)
    if not all([rest_lat, rest_lon]) and st.session_state.get("rest_typed") and GOOGLE_MAPS_API_KEY:
        f, la, lo = geocode_text_fallback(st.session_state.get("rest_typed"), GOOGLE_MAPS_API_KEY)
        if f:
            st.session_state["rest_address"] = f
            st.session_state["rest_lat"] = la
            st.session_state["rest_lon"] = lo
            rest_lat, rest_lon = la, lo

    if not all([del_lat, del_lon]) and st.session_state.get("del_typed") and GOOGLE_MAPS_API_KEY:
        f2, la2, lo2 = geocode_text_fallback(st.session_state.get("del_typed"), GOOGLE_MAPS_API_KEY)
        if f2:
            st.session_state["del_address"] = f2
            st.session_state["del_lat"] = la2
            st.session_state["del_lon"] = lo2
            del_lat, del_lon = la2, lo2

    if not all([rest_lat, rest_lon, del_lat, del_lon]):
        st.error("⚠️ Could not resolve one or both addresses. Try selecting a suggestion from the dropdown or type a clearer address.")
        st.stop()

    # Feature engineering & prediction (unchanged)
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

# End of file
