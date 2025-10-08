import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

# ========== CONFIG / THRESHOLDS ==========

# Malaysia DOE parameter thresholds by water quality class (Class I to V) — example subset
DOE_THRESHOLDS = {
    'bod': [1, 3, 6, 12],  # mg/L: boundaries for classes I–V
    'cod': [10, 25, 50, 100],
    'do': [7, 5, 3, 1],
    'ph': [6.5, 8.5],  # for “normal” range
    'tss': [25, 50, 150, 300],
    'nh3_n': [0.1, 0.3, 0.9, 2.7],
    # etc. add more
}

# Weights & formula for Malaysia DOE WQI (six parameters) — you may adjust based on literature
WQI_WEIGHTS = {
    'bod': 0.16,
    'cod': 0.11,
    'do': 0.20,
    'nh3_n': 0.10,
    'tss': 0.10,
    'ph': 0.033,
    # sum of weights should be 1 (or normalized)
}

# ========== FUNCTIONS ==========

def fetch_api_data(base_url: str, params: dict = None) -> pd.DataFrame:
    """Fetch JSON or CSV data from an API. Return DataFrame or None if fail."""
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        # Try JSON
        if 'application/json' in resp.headers.get('Content-Type', ''):
            j = resp.json()
            # Assumes the JSON is list-of-dict or has “data” key
            if isinstance(j, dict) and 'data' in j:
                data = j['data']
            else:
                data = j
            return pd.DataFrame(data)
        # Fallback: CSV
        else:
            return pd.read_csv(pd.compat.StringIO(resp.text))
    except Exception as e:
        st.warning(f"API fetch error: {e}")
        return None

def compute_subindex(param: str, value: float) -> float:
    """
    Convert a parameter reading to a sub-index score (0–100) based on threshold curves.
    This is a simplified approach — real DOE WQI uses curves / rating functions.
    """
    # Example: if parameter is “bod”, lower is better, so invert mapping
    # This simple linear mapping is just illustrative
    thr = DOE_THRESHOLDS.get(param)
    if thr is None or pd.isna(value):
        return np.nan
    # For “good” side
    if param in ['do']:  # larger is better
        # if value >= thr[0] => best (assign 100)
        # else linear down to 0 (when value = 0)
        return max(0, min(100, (value / thr[0]) * 100))
    else:
        # lower is better (for contaminants)
        # if ≤ first threshold => 100,
        # if ≥ last threshold => 0, else linear interpolation
        for i, bound in enumerate(thr):
            if value <= bound:
                # map between previous bound and this
                prev = thr[i-1] if i > 0 else 0
                # linear interpolation
                return 100 * (1 - (value - prev) / (bound - prev)) if bound != prev else 100
        return 0

def compute_wqi(df: pd.DataFrame) -> pd.Series:
    """Compute DOE WQI for rows in DataFrame (expects columns matching WQI_WEIGHTS keys)."""
    sub_indices = {}
    for p in WQI_WEIGHTS:
        if p in df.columns:
            sub_indices[p] = df[p].apply(lambda v: compute_subindex(p, v))
        else:
            sub_indices[p] = pd.Series([np.nan] * len(df))
    # Weighted sum
    wqi = sum(WQI_WEIGHTS[p] * sub_indices[p].fillna(0) for p in WQI_WEIGHTS)
    # Optionally normalize so that max possible = 100
    # sum_weights = sum(WQI_WEIGHTS.values())
    # wqi = wqi / sum_weights
    return wqi

def classify_wqi(wqi: float) -> str:
    """Classify WQI into categories (Excellent, Good, etc) — example thresholds."""
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
    """Return color / label depending on whether the reading is acceptable or exceeds limits."""
    thr = DOE_THRESHOLDS.get(param)
    if thr is None or pd.isna(value):
        return "gray"
    if param in ['do']:
        # do below second bound => warning
        if value < thr[1]:
            return "red"
        elif value < thr[0]:
            return "orange"
        else:
            return "green"
    else:
        # for contaminants (lower is better)
        if value > thr[-1]:
            return "red"
        elif value > thr[-2]:
            return "orange"
        else:
            return "green"


# ========== MAIN STREAMLIT APP ==========

st.set_page_config(page_title="Water Quality Dashboard with APIs", layout="wide")
st.title("💧 Water Quality Dashboard (with API support)")

# ========== Data Source Section ==========

st.sidebar.header("Data Source (choose one)")

source_option = st.sidebar.selectbox(
    "How to load data:",
    ["Upload CSV", "Use API URL"]
)

df = None
if source_option == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
elif source_option == "Use API URL":
    api_url = st.sidebar.text_input("API URL (GET endpoint)")
    api_params_text = st.sidebar.text_area("Query params (JSON)", value="{}")
    try:
        api_params = eval(api_params_text)
    except:
        api_params = {}
    if api_url:
        df = fetch_api_data(api_url, api_params)

if df is None:
    st.info("Please upload CSV or enter API URL to fetch data.")
    st.stop()

# Normalize columns
df.columns = df.columns.str.strip().str.lower()

# Detect date column
date_cols = [c for c in df.columns if 'date' in c]
if not date_cols:
    st.error("No date column detected. Please have a column with ‘date’ in its name.")
    st.stop()
date_col = date_cols[0]
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])

# Identify location / lat / lon
loc_col = next((c for c in df.columns if 'location' in c), None)
lat_col = next((c for c in df.columns if 'lat' in c), None)
lon_col = next((c for c in df.columns if 'lon' in c), None)

# Numeric / parameter columns
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

# ========== Filters ==========

st.sidebar.header("Filters")
if loc_col:
    selected_locs = st.sidebar.multiselect("Locations", sorted(df[loc_col].unique()), default=None)
    if selected_locs:
        df = df[df[loc_col].isin(selected_locs)]

# Date range
dmin, dmax = df[date_col].min(), df[date_col].max()
start_date, end_date = st.sidebar.date_input("Date range", [dmin, dmax])
if start_date and end_date:
    df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

# Parameter pick
param = st.sidebar.selectbox("Parameter to visualize", numeric_cols)

# ========== Compute WQI + Classify ==========

# Compute WQI for each row (if parameters exist)
df['wqi'] = compute_wqi(df)
df['wqi_class'] = df['wqi'].apply(classify_wqi)

# ========== Map View ==========

if lat_col and lon_col:
    st.subheader("🗺️ Mean parameter by location")
    df_mean = df.groupby(loc_col, as_index=False)[param].mean()
    # attach lat/lon
    df_coords = df[[loc_col, lat_col, lon_col]].drop_duplicates()
    df_mean = df_mean.merge(df_coords, on=loc_col, how='left')
    fig_map = px.scatter_mapbox(
        df_mean, lat=lat_col, lon=lon_col,
        color=param, size=param,
        hover_name=loc_col, color_continuous_scale='Viridis',
        zoom=6, height=500
    )
    fig_map.update_layout(mapbox_style='open-street-map', margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ========== Time Series + Highlighting ==========

st.subheader(f"📈 Time series: {param.upper()} by location")

# Add flag color
df['flag_color'] = df[param].apply(lambda v: flag_threshold(param, v))

fig = px.scatter(
    df, x=date_col, y=param, color=loc_col if loc_col else None,
    size_max=6, hover_data=numeric_cols
)
# Overlay lines
for loc in df[loc_col].unique() if loc_col else [None]:
    dloc = df[df[loc_col] == loc] if loc_col else df
    fig.add_scatter(x=dloc[date_col], y=dloc[param], mode='lines', name=f"{loc} trend")

# Optionally add threshold lines
# E.g. for contaminant, add line at the class III boundary
if param in DOE_THRESHOLDS and param not in ['do']:
    bound = DOE_THRESHOLDS[param][2]  # e.g. class III
    fig.add_hline(y=bound, line_dash="dash", line_color="red", annotation_text="Class III boundary")

st.plotly_chart(fig, use_container_width=True)

# ========== Summary Stats & WQI Plot ==========

st.subheader("📊 Summary & WQI")

# Stats
stats = df.groupby(loc_col if loc_col else None)[param].agg(['mean','min','max','std','count']).reset_index()
st.dataframe(stats, use_container_width=True)

# WQI over time
fig2 = px.line(df, x=date_col, y='wqi', color=loc_col if loc_col else None, title="WQI over time")
st.plotly_chart(fig2, use_container_width=True)

# Show sample table with classification
st.subheader("Sample readings with classification")
show_cols = [date_col] + ([loc_col] if loc_col else []) + [param, 'wqi', 'wqi_class']
st.dataframe(df[show_cols].sort_values(by=date_col).reset_index(drop=True), use_container_width=True)

# ========== Data Download ==========

st.sidebar.markdown("### 💾 Download Data")
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Download filtered data as CSV", data=csv_bytes,
                            file_name=f"water_quality_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime='text/csv')
