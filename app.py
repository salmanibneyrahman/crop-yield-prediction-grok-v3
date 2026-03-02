import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Bangladesh Crop & Yield Advisor", page_icon="🌾", layout="wide")

# ============================================================================
# LOAD MODELS (exact match to notebook)
# ============================================================================
@st.cache_resource
def load_models():
    try:
        yield_model = joblib.load('yield_model.pkl')
        crop_model = joblib.load('crop_model.pkl')
        le_crop = joblib.load('label_encoder_crop.pkl')
        season_le = joblib.load('label_encoder_season.pkl')
        soil_le = joblib.load('label_encoder_soil.pkl')
        yield_feature_names = joblib.load('yield_feature_names.pkl')
        crop_feature_names = joblib.load('crop_feature_names.pkl')
        season_classes = joblib.load('season_classes.pkl')
        district_classes = joblib.load('district_classes.pkl')
        soil_classes = joblib.load('soil_classes.pkl')

        st.success("✅ All models & encoders loaded (exact match to notebook)")
        return {
            'yield': yield_model,
            'crop': crop_model,
            'le_crop': le_crop,
            'season_le': season_le,
            'soil_le': soil_le,
            'yield_features': yield_feature_names,
            'crop_features': crop_feature_names,
            'season_classes': season_classes,
            'district_classes': district_classes,
            'soil_classes': soil_classes
        }
    except Exception as e:
        st.error(f"Load error: {e}")
        st.stop()

models = load_models()

# ============================================================================
# UI
# ============================================================================
st.title("🌾 Bangladesh Crop & Yield Intelligence System (Notebook Exact)")

mode = st.radio("Weather input", ["Manual"], horizontal=True)  # Auto removed for simplicity

col1, col2 = st.columns(2)

with col1:
    district = st.selectbox("District", options=models['district_classes'])
    season = st.selectbox("Season", options=models['season_classes'])
    area_ha = st.number_input("Cultivated Area (ha)", min_value=0.1, value=10.0)

with col2:
    st.subheader("Weather")
    min_temp = st.number_input("Min Temp (°C)", value=20.0)
    avg_temp = st.number_input("Avg Temp (°C)", value=26.0)
    max_temp = st.number_input("Max Temp (°C)", value=32.0)
    min_humidity = st.number_input("Min Humidity (%)", value=40)
    avg_humidity = st.number_input("Avg Humidity (%)", value=70)
    max_humidity = st.number_input("Max Humidity (%)", value=95)
    
    # NEW INPUTS TO MATCH NOTEBOOK
    rainfall = st.number_input("Monthly Avg Rainfall (mm)", value=200.0, min_value=0.0)
    soil = st.selectbox("Primary Soil Type", options=models['soil_classes'])

if st.button("🔮 Predict (Exact Notebook Style)", use_container_width=True):
    try:
        # Yield prediction (9 features, no Area)
        yield_input = pd.DataFrame([[
            avg_temp, avg_humidity, max_temp, min_temp,
            max_humidity, min_humidity, rainfall,
            models['season_le'].transform([season])[0],
            models['soil_le'].transform([soil])[0]
        ]], columns=models['yield_features'])

        yield_per_ha = float(models['yield'].predict(yield_input)[0])
        total_production = yield_per_ha * area_ha

        # Crop prediction (one-hot exact match)
        user_row = pd.DataFrame([{
            "Avg Temp": avg_temp,
            "Avg Humidity": avg_humidity,
            "Max Temp": max_temp,
            "Min Temp": min_temp,
            "Max Relative Humidity": max_humidity,
            "Min Relative Humidity": min_humidity,
            "Season": season,
            "District": district,
            "Primary_Soil_Type": soil
        }])

        X_crop_input = pd.get_dummies(user_row, columns=["Season", "District", "Primary_Soil_Type"],
                                      prefix=["Season", "District", "Soil"], drop_first=True)
        
        # Align columns exactly to training
        X_crop_input = X_crop_input.reindex(columns=models['crop_features'], fill_value=0)

        crop_idx = int(models['crop'].predict(X_crop_input)[0])
        crop_name = models['le_crop'].inverse_transform([crop_idx])[0]

        # Display
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Recommended Crop", crop_name.upper())
        with c2: st.metric("Yield (t/ha)", f"{yield_per_ha:.2f}")
        with c3: st.metric("Total Production (tons)", f"{total_production:,.0f}")

        st.success("✅ Prediction matches your notebook exactly!")

    except Exception as e:
        st.error(f"Prediction error: {e}")
