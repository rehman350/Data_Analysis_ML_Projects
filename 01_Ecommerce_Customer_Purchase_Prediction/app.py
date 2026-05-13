# app.py — Streamlit Web Application
# Yeh file Jupyter se BAHAR banao — notepad/vscode mein
import os
import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd

# ---- Page Config ----
st.set_page_config(
    page_title="E-commerce Purchase Predictor",
    page_icon="🛒",
    layout="wide"
)

# ---- Load Model ----
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model  = joblib.load(os.path.join(base_path, 'purchase_model.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
    with open(os.path.join(base_path, 'feature_names.json')) as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, feature_names = load_model()

# ---- Header ----
st.title("🛒 E-commerce Purchase Predictor")
st.markdown("**Will this customer make a purchase?** — Random Forest Model (AUC: 0.918)")
st.divider()

# ---- Input Section ----
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📄 Page Behavior")
    page_values      = st.slider("Page Values",          0.0, 200.0, 10.0, step=0.5)
    bounce_rates     = st.slider("Bounce Rate",          0.0, 0.20,  0.05, step=0.01)
    exit_rates       = st.slider("Exit Rate",            0.0, 0.20,  0.05, step=0.01)
    product_related  = st.slider("Product Pages Visited",0,   100,   15)
    product_duration = st.slider("Product Time (sec)",   0.0, 3000.0,500.0, step=10.0)

with col2:
    st.subheader("🧭 Session Info")
    administrative   = st.slider("Admin Pages Visited",  0, 20, 2)
    admin_duration   = st.slider("Admin Time (sec)",     0.0, 500.0, 50.0, step=5.0)
    informational    = st.slider("Info Pages Visited",   0, 10, 0)
    info_duration    = st.slider("Info Time (sec)",      0.0, 500.0, 0.0,  step=5.0)
    special_day      = st.slider("Special Day (0=normal, 1=holiday)", 0.0, 1.0, 0.0, step=0.2)

with col3:
    st.subheader("👤 User Profile")
    month         = st.selectbox("Month", 
                    ['Feb','Mar','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    visitor_type  = st.selectbox("Visitor Type",
                    ['Returning_Visitor', 'New_Visitor', 'Other'])
    weekend       = st.selectbox("Weekend Visit?", ['No', 'Yes'])
    os_type       = st.selectbox("Operating System", [1,2,3,4,5,6,7,8])
    browser       = st.selectbox("Browser",          [1,2,3,4,5,6,7])
    region        = st.selectbox("Region",           [1,2,3,4,5,6,7,8,9])
    traffic_type  = st.selectbox("Traffic Type",     list(range(1,21)))

# ---- Encode inputs ----
month_map   = {'Feb':2,'Mar':5,'May':6,'Jun':4,'Jul':3,
               'Aug':0,'Sep':9,'Oct':8,'Nov':7,'Dec':1}
visitor_map = {'New_Visitor':0, 'Other':1, 'Returning_Visitor':2}

month_enc   = month_map[month]
visitor_enc = visitor_map[visitor_type]
weekend_enc = 1 if weekend == 'Yes' else 0

# Feature engineering (same as training)
total_pages      = administrative + informational + product_related
total_duration   = admin_duration + info_duration + product_duration
avg_time_per_pg  = total_duration / (total_pages + 1)
bounce_exit_avg  = (bounce_rates + exit_rates) / 2
is_high_value    = 1 if page_values > 5.89 else 0   # training median
engagement       = (1 - bounce_rates) * page_values * np.log1p(total_pages)

input_data = pd.DataFrame([[
    administrative, admin_duration, informational, info_duration,
    product_related, product_duration, bounce_rates, exit_rates,
    page_values, special_day, month_enc, os_type, browser,
    region, traffic_type, visitor_enc, weekend_enc,
    total_pages, total_duration, avg_time_per_pg,
    bounce_exit_avg, is_high_value, engagement
]], columns=feature_names)

# ---- Predict Button ----
st.divider()
if st.button("🔮 Predict Purchase Probability", use_container_width=True):
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        if prediction == 1:
            st.success(f"✅ WILL PURCHASE")
        else:
            st.error(f"❌ WILL NOT PURCHASE")

    with res_col2:
        st.metric("Purchase Probability", f"{probability*100:.1f}%")

    with res_col3:
        st.metric("Confidence", 
                  "High" if probability > 0.7 or probability < 0.3 else "Medium")

    # Probability bar
    st.progress(float(probability))
    st.caption(f"Model: Random Forest | AUC: 0.918 | Training data: 12,330 sessions")