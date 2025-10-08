# ============================================================
# 🌊 WATER QUALITY DASHBOARD (CSV-based)
# Author: [Your Name]
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# 1. PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="Water Quality Dashboard 🌊", layout="wide")

st.title("💧 Water Quality Monitoring Dashboard")
st.markdown("Visualize, analyze, and interpret water quality data from CSV files.")

# ------------------------------------------------------------
# 2. FILE UPLOAD
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV data file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Convert date
    date_cols = [col for col in df.columns if 'date' in col.lower()]

    if date_cols:
        date_min, date_max = df[date_cols[0]].min(), df[date_cols[0]].max()
        date_range = st.sidebar.date_input("Date range:", [date_min, date_max])

        if len(date_range) == 2:
            df = df[
                (df[date_cols[0]] >= pd.to_datetime(date_range[0])) &
                (df[date_cols[0]] <= pd.to_datetime(date_range[1]))
            ]
    else:
        st.warning("⚠️ No date column found in your dataset. Skipping date filtering.")
        st.success(f"✅ Data loaded successfully — {len(df)} records found.")


    #####
   # date_cols = [c for c in df.columns if 'date' in c]
   # if date_cols:
       # df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors='coerce')
       # df = df.dropna(subset=[date_cols[0]])

    # Identify numeric columns (parameters)
    #numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
   # location_col = [c for c in df.columns if 'location' in c]
   # lat_col = [c for c in df.columns if 'lat' in c]
   # lon_col = [c for c in df.columns if 'lon' in c]

    #st.success(f"✅ Data loaded successfully — {len(df)} records found.")

    # ------------------------------------------------------------
    # 3. SIDEBAR FILTERS
    # ------------------------------------------------------------
    st.sidebar.header("🔍 Filters")
    if location_col:
        selected_location = st.sidebar.multiselect(
            "Select location(s):", sorted(df[location_col[0]].unique()), default=None
        )
        if selected_location:
            df = df[df[location_col[0]].isin(selected_location)]

    # Date range
    date_min, date_max = df[date_cols[0]].min(), df[date_cols[0]].max()
    date_range = st.sidebar.date_input("Date range:", [date_min, date_max])
    if len(date_range) == 2:
        df = df[(df[date_cols[0]] >= pd.to_datetime(date_range[0])) &
                (df[date_cols[0]] <= pd.to_datetime(date_range[1]))]

    # Parameter selection
    selected_parameter = st.sidebar.selectbox("Select parameter:", numeric_cols)

    # ------------------------------------------------------------
    # 4. MAP VIEW
    # ------------------------------------------------------------
    if lat_col and lon_col:
        st.subheader("🗺️ Water Quality by Location")
        df_mean = df.groupby(location_col[0], as_index=False)[selected_parameter].mean()
        df_mean = df_mean.merge(df[[location_col[0], lat_col[0], lon_col[0]]].drop_duplicates(), on=location_col[0])

        fig_map = px.scatter_mapbox(
            df_mean,
            lat=lat_col[0],
            lon=lon_col[0],
            color=selected_parameter,
            hover_name=location_col[0],
            color_continuous_scale="YlGnBu",
            zoom=5,
            size_max=12,
            height=500
        )
        fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # ------------------------------------------------------------
    # 5. TIME SERIES
    # ------------------------------------------------------------
    st.subheader(f"📈 {selected_parameter.upper()} Over Time")
    fig_line = px.line(
        df, x=date_cols[0], y=selected_parameter, color=location_col[0] if location_col else None,
        markers=True, title=f"Trend of {selected_parameter.upper()} over time"
    )
    fig_line.update_layout(legend_title="Location", yaxis_title=f"{selected_parameter}")
    st.plotly_chart(fig_line, use_container_width=True)

    # ------------------------------------------------------------
    # 6. STATISTICS SUMMARY
    # ------------------------------------------------------------
    st.subheader("📊 Summary Statistics")
    stats = df.groupby(location_col[0] if location_col else None)[selected_parameter].agg(
        ['mean', 'min', 'max', 'std', 'count']
    ).reset_index()
    st.dataframe(stats, use_container_width=True)

    # ------------------------------------------------------------
    # 7. PARAMETER CORRELATION
    # ------------------------------------------------------------
    st.subheader("🔗 Parameter Correlation Matrix")
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------------------------------------------------
    # 8. DOWNLOAD FILTERED DATA
    # ------------------------------------------------------------
    st.sidebar.markdown("### 💾 Download Filtered Data")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"filtered_water_quality_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime='text/csv'
    )

else:
    st.info("👆 Please upload a CSV file to begin.")
