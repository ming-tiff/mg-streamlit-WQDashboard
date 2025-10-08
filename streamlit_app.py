import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# ==========================================================
# CONFIG / THRESHOLDS
# ==========================================================
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

# ==========================================================
# FUNCTIONS
# ==========================================================
def fetch_api_data(base_url: str, params: dict = None) -> pd.DataFrame:
    """Fetch JSON or CSV data from an API."""
    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        if 'application/json' in resp.headers.get('Content-Type', ''):
            data = resp.json()
            if isinstance(data, dict) and 'data' in data:
                data = data['data']
            return pd.DataFrame(data)
        else:
            return pd.read_csv(pd.compat.StringIO(resp.text))
    except Exception as e:
        st.warning(f"API fetch error: {e}")
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
                prev = thr[i - 1] if i > 0 else 0
                return 100 * (1 - (value - prev) / (bound - prev)) if bound != prev else 100
        return 0

def compute_wqi(df: pd.DataFrame) -> pd.Series:
    sub_indices = {}
    for p in WQI_WEIGHTS:
        sub_indices[p] = df[p].apply(lambda v: compute_subindex(p, v)) if p in df.columns else np.nan
    wqi = sum(WQI_WEIGHTS[p] * pd.Series(sub_indices[p]) for p in WQI_WEIGHTS)
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

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(page_title="Water Quality Dashboard", layout="wide")
st.title("💧 Water Quality Dashboard (with Real API Support)")

# Sidebar – Data Source
st.sidebar.header("Data Source")

source_option = st.sidebar.selectbox("Select source:", ["Upload CSV", "Use Built-in API"])

if source_option == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()
else:
    st.sidebar.markdown("### 🌐 Available APIs")
    api_choice = st.sidebar.selectbox("Choose API:", [
        "Global – Aquatic Informatics (Sample Data)",
        "Malaysia – DOE (Department of Environment, example open dataset)",
        "Southeast Asia – Open Data Portal (Environmental Monitoring)",
        "Global – Open-Meteo (Water Temperature Proxy)"
    ])
    API_URLS = {
        "Global – Aquatic Informatics (Sample Data)": "https://api.openaq.org/v2/measurements?parameter=pm25",
        "Malaysia – DOE (Department of Environment, example open dataset)": "https://environment.data.gov.my/water_quality",
        "Southeast Asia – Open Data Portal (Environmental Monitoring)": "https://data.humdata.org/dataset/environmental-quality",
        "Global – Open-Meteo (Water Temperature Proxy)": "https://archive-api.open-meteo.com/v1/era5"
    }
    url = API_URLS[api_choice]
    st.write(f"**Fetching data from:** {url}")
    df = fetch_api_data(url)

if df is None or df.empty:
    st.error("No data loaded. Please check the source or upload a valid CSV.")
    st.stop()

df.columns = df.columns.str.strip().str.lower()
date_col = next((c for c in df.columns if 'date' in c), None)
if date_col is None:
    st.error("Date column not found.")
    st.stop()

df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])
loc_col = next((c for c in df.columns if 'location' in c), None)
lat_col = next((c for c in df.columns if 'lat' in c), None)
lon_col = next((c for c in df.columns if 'lon' in c), None)
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# ==========================================================
# Filters
# ==========================================================
st.sidebar.header("Filters")
if loc_col:
    locations = st.sidebar.multiselect("Locations", sorted(df[loc_col].unique()))
    if locations:
        df = df[df[loc_col].isin(locations)]

# Date range
dmin, dmax = df[date_col].min(), df[date_col].max()
start_date, end_date = st.sidebar.date_input("Date Range", [dmin, dmax])
df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

# ==========================================================
# Parameter Visualization (NEW)
# ==========================================================
st.subheader("📊 Parameter Visualization")

params_to_plot = st.multiselect("Select parameters to visualize", numeric_cols, default=numeric_cols[:3])

if len(params_to_plot) == 1:
    fig_param = px.line(df, x=date_col, y=params_to_plot[0], color=loc_col if loc_col else None,
                        title=f"{params_to_plot[0].upper()} over time")
    st.plotly_chart(fig_param, use_container_width=True)
elif len(params_to_plot) == 2:
    fig_param = px.scatter(df, x=params_to_plot[0], y=params_to_plot[1],
                           color=loc_col if loc_col else None,
                           title=f"Relationship: {params_to_plot[0]} vs {params_to_plot[1]}")
    st.plotly_chart(fig_param, use_container_width=True)
elif len(params_to_plot) > 2:
    corr = df[params_to_plot].corr()
    st.write("### Correlation Matrix")
    st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)

# ==========================================================
# WQI COMPUTATION + SUMMARY
# ==========================================================
df['wqi'] = compute_wqi(df)
df['wqi_class'] = df['wqi'].apply(classify_wqi)

st.subheader("📈 WQI Over Time")
fig_wqi = px.line(df, x=date_col, y='wqi', color=loc_col if loc_col else None)
st.plotly_chart(fig_wqi, use_container_width=True)

# Summary Table
st.subheader("📋 Summary Stats")
if loc_col:
    stats = df.groupby(loc_col)[params_to_plot].agg(['mean', 'min', 'max', 'std']).reset_index()
else:
    stats = df[params_to_plot].agg(['mean', 'min', 'max', 'std']).reset_index()
st.dataframe(stats, use_container_width=True)

# ==========================================================
# DOWNLOAD
# ==========================================================
st.sidebar.markdown("### 💾 Download Data")
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download filtered data as CSV", data=csv_bytes,
                            file_name=f"water_quality_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime='text/csv')
