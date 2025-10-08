import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# =======================================
# CONFIG / THRESHOLDS
# =======================================

DOE_THRESHOLDS = {
    'bod': [1, 3, 6, 12],
    'cod': [10, 25, 50, 100],
    'do': [7, 5, 3, 1],
    'ph': [6.5, 8.5],
    'tss': [25, 50, 150, 300],
    'nh3_n': [0.1, 0.3, 0.9, 2.7],
}

WQI_WEIGHTS = {
    'bod': 0.16,
    'cod': 0.11,
    'do': 0.20,
    'nh3_n': 0.10,
    'tss': 0.10,
    'ph': 0.033,
}

# =======================================
# GLOBAL WATER QUALITY API SOURCES
# =======================================

WQ_APIS = {
    "Global - UNEP GEMS/Water": "https://gemstat.org/api/data",
    "USA - USGS Water Quality Portal": "https://www.waterqualitydata.us/data/Result/search?countrycode=US&mimeType=json",
    "Global - Copernicus Water Quality": "https://cds.climate.copernicus.eu/api/v2/",
    "Malaysia - DOE (manual)": "https://enviro.doe.gov.my/",
    "Malaysia - DID Public Info Banjir": "https://publicinfobanjir.water.gov.my/api/v1/locations",
    "Singapore - PUB Water Quality (demo)": "https://data.gov.sg/api/action/datastore_search?resource_id=2b03c745-57e8-4a96-9a8c-4a2577ad1fdd"
}

# =======================================
# FUNCTIONS
# =======================================

def fetch_api_data(base_url: str, params: dict = None) -> pd.DataFrame:
    """Fetch JSON or CSV data from API and return DataFrame."""
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        if 'application/json' in resp.headers.get('Content-Type', ''):
            data = resp.json()
            if isinstance(data, dict):
                data = data.get('data') or data.get('result') or data
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Flatten nested dict if possible
                return pd.json_normalize(data)
        else:
            from io import StringIO
            return pd.read_csv(StringIO(resp.text))
    except Exception as e:
        st.warning(f"⚠️ API fetch error: {e}")
        return None


def compute_subindex(param: str, value: float) -> float:
    thr = DOE_THRESHOLDS.get(param)
    if thr is None or pd.isna(value):
        return np.nan
    if param in ['do']:
        return max(0, min(100, (value / thr[0]) * 100))
    else:
        for i, bound in enumerate(thr):
            if value <= bound:
                prev = thr[i-1] if i > 0 else 0
                return 100 * (1 - (value - prev) / (bound - prev)) if bound != prev else 100
        return 0


def compute_wqi(df: pd.DataFrame) -> pd.Series:
    sub_indices = {}
    for p in WQI_WEIGHTS:
        if p in df.columns:
            sub_indices[p] = df[p].apply(lambda v: compute_subindex(p, v))
        else:
            sub_indices[p] = pd.Series([np.nan] * len(df))
    wqi = sum(WQI_WEIGHTS[p] * sub_indices[p].fillna(0) for p in WQI_WEIGHTS)
    return wqi


def classify_wqi(wqi: float) -> str:
    if wqi >= 92.7:
        return "Excellent"
    elif wqi >= 76.5:
        return "Good"
    elif wqi >= 51.9:
        return "Moderate"
    elif wqi >= 31.0:
        return "Poor"
    else:
        return "Very Poor"


def flag_threshold(param: str, value: float) -> str:
    thr = DOE_THRESHOLDS.get(param)
    if thr is None or pd.isna(value):
        return "gray"
    if param in ['do']:
        if value < thr[1]:
            return "red"
        elif value < thr[0]:
            return "orange"
        else:
            return "green"
    else:
        if value > thr[-1]:
            return "red"
        elif value > thr[-2]:
            return "orange"
        else:
            return "green"

# =======================================
# STREAMLIT APP
# =======================================

st.set_page_config(page_title="💧 Water Quality Dashboard", layout="wide")
st.title("💧 Water Quality Dashboard (CSV + API sources)")

# Sidebar: Data Source
st.sidebar.header("📦 Data Source")
source_option = st.sidebar.radio("Choose data source:", ["Upload CSV", "Use Built-in API"])

df = None

if source_option == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

elif source_option == "Use Built-in API":
    api_choice = st.sidebar.selectbox("Select API source", list(WQ_APIS.keys()))
    st.sidebar.markdown(f"**API URL:** `{WQ_APIS[api_choice]}`")
    if st.sidebar.button("Fetch Data"):
        df = fetch_api_data(WQ_APIS[api_choice])

# Stop if no data
if df is None or df.empty:
    st.info("Please upload a CSV or fetch from API to continue.")
    st.stop()

# Normalize columns
df.columns = df.columns.str.strip().str.lower()

# Detect date
date_cols = [c for c in df.columns if 'date' in c]
if not date_cols:
    st.warning("⚠️ No date column detected — using index as date.")
    df['date'] = pd.date_range(start='2020-01-01', periods=len(df))
    date_col = 'date'
else:
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Detect key columns
loc_col = next((c for c in df.columns if 'location' in c), None)
lat_col = next((c for c in df.columns if 'lat' in c), None)
lon_col = next((c for c in df.columns if 'lon' in c), None)

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# Filters
st.sidebar.header("🔍 Filters")
if loc_col:
    selected_locs = st.sidebar.multiselect("Filter by location", df[loc_col].unique())
    if selected_locs:
        df = df[df[loc_col].isin(selected_locs)]

# Date range filter
if date_col in df.columns:
    dmin, dmax = df[date_col].min(), df[date_col].max()
    start_date, end_date = st.sidebar.date_input("Date range", [dmin, dmax])
    if start_date and end_date:
        df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

# Parameter pick
param = st.sidebar.selectbox("Parameter to visualize", numeric_cols)

# Compute WQI
df['wqi'] = compute_wqi(df)
df['wqi_class'] = df['wqi'].apply(classify_wqi)

# ==========================
# MAP VIEW
# ==========================
if lat_col and lon_col:
    st.subheader("🗺️ Mean Parameter by Location")
    df_mean = df.groupby(loc_col, as_index=False)[param].mean()
    df_coords = df[[loc_col, lat_col, lon_col]].drop_duplicates()
    df_mean = df_mean.merge(df_coords, on=loc_col, how='left')
    fig_map = px.scatter_mapbox(
        df_mean, lat=lat_col, lon=lon_col, color=param, size=param,
        hover_name=loc_col, color_continuous_scale='Viridis', zoom=6, height=450
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ==========================
# TIME SERIES
# ==========================
st.subheader(f"📈 {param.upper()} Time Series")
df['flag_color'] = df[param].apply(lambda v: flag_threshold(param, v))

fig_ts = px.scatter(df, x=date_col, y=param, color=loc_col if loc_col else None, hover_data=numeric_cols)
if loc_col:
    for loc in df[loc_col].unique():
        dloc = df[df[loc_col] == loc]
        fig_ts.add_scatter(x=dloc[date_col], y=dloc[param], mode='lines', name=f"{loc} trend")

if param in DOE_THRESHOLDS and param not in ['do']:
    fig_ts.add_hline(y=DOE_THRESHOLDS[param][2], line_dash="dash", line_color="red", annotation_text="Class III boundary")

st.plotly_chart(fig_ts, use_container_width=True)

# ==========================
# SUMMARY & WQI
# ==========================
st.subheader("📊 Summary & WQI")

# FIXED GROUPBY (no None issue)
if loc_col:
    stats = df.groupby(loc_col)[param].agg(['mean','min','max','std','count']).reset_index()
else:
    stats = df[param].agg(['mean','min','max','std','count']).to_frame().T.reset_index(drop=True)

st.dataframe(stats, use_container_width=True)

fig_wqi = px.line(df, x=date_col, y='wqi', color=loc_col if loc_col else None, title="WQI Over Time")
st.plotly_chart(fig_wqi, use_container_width=True)

st.subheader("💦 Sample Readings & Classification")
show_cols = [date_col] + ([loc_col] if loc_col else []) + [param, 'wqi', 'wqi_class']
st.dataframe(df[show_cols].sort_values(by=date_col).reset_index(drop=True), use_container_width=True)

# ==========================
# DOWNLOAD
# ==========================
st.sidebar.markdown("### 💾 Download Filtered Data")
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name=f"water_quality_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime='text/csv'
)
