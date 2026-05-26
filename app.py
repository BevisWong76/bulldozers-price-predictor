import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ===================================================================
# 1. Page Config
# ===================================================================
st.set_page_config(
    page_title="Heavy Equipment Price Predictor", 
    page_icon="🚜", 
    layout="wide"
)

st.title("Heavy Equipment Auction Price Predictor")
st.markdown("Predict bulldozer auction prices using our tuned **LightGBM Pipeline**.")
st.divider()


# ===================================================================
# 2. Load Model & Feature Lists
# ===================================================================
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load("models/full_pipeline.joblib")
        num_feats = joblib.load("models/numerical_features.joblib")
        cat_feats = joblib.load("models/categorical_features.joblib")
        return pipeline, num_feats, cat_feats
    except FileNotFoundError:
        st.error(
            "⚠️ Artifacts not found! Please ensure `.joblib` files exist in `models/`."
        )
        return None, [], []


pipeline, numerical_features, categorical_features = load_artifacts()

# ===================================================================
# 3. Sidebar Inputs (Raw Feature Names)
# ===================================================================
st.sidebar.header("📋 Machine Specifications")

year_made = st.sidebar.number_input(
    "Manufacturing Year (YearMade)", 1960, 2026, 2012, 1
)

sale_year = st.sidebar.number_input(
    "Auction Sale Year (saleYear)", 2000, 2026, 2024, 1
)

product_size = st.sidebar.selectbox(
    "Product Size",
    ["Medium", "Large / Medium", "Small", "Mini", "Large", "Compact", "missing"],
)

coupler_system = st.sidebar.selectbox(
    "Coupler System", ["None or Unspecified", "Yes", "No", "missing"]
)

enclosure = st.sidebar.selectbox(
    "Enclosure Type",
    ["OROPS", "EROPS", "EROPS w AC", "EROPS AC", "None or Unspecified"],
    index=2,
)

product_class = st.sidebar.selectbox(
    "Product Class Description",
    [
        "Track Type Tractor, Dozer - 20.0 to 75.0 Horsepower",
        "Track Type Tractor, Dozer - 105.0 to 130.0 Horsepower",
        "Hydraulic Excavator, Track - 12.0 to 14.0 Metric Tons",
        "Wheel Loader - 110.0 to 120.0 Horsepower",
    ],
)

tire_size = st.sidebar.selectbox(
    "Tire Size",
    ['None or Unspecified', '20.5', '23.5', '14"', '26.5', '17.5"', "missing"],
)

grouser_tracks = st.sidebar.selectbox(
    "Grouser Tracks", ["None or Unspecified", "Yes", "No", "missing"]
)

model_id = st.sidebar.number_input("Model ID", 1, 50000, 4601, 1)


# ===================================================================
# 4. Construct Input DataFrame Matching Training Schema
# ===================================================================
col1, col2 = st.columns([1, 1])

# Create a dictionary with ALL expected raw features initialized to NaN/missing
all_expected_features = list(numerical_features) + list(categorical_features)
input_data = {feat: np.nan for feat in all_expected_features}

# Update with user selected inputs (using exact raw column names)
user_inputs = {
    "YearMade": year_made,
    "saleYear": sale_year,
    "ProductSize": product_size,
    "Coupler_System": coupler_system,
    "Enclosure": enclosure,
    "fiProductClassDesc": product_class,
    "Tire_Size": tire_size,
    "Grouser_Tracks": grouser_tracks,
    "ModelID": model_id,
}

for k, v in user_inputs.items():
    if k in input_data:
        input_data[k] = v

input_df = pd.DataFrame([input_data])

with col1:
    st.subheader("Summary of Selected Inputs")
    display_df = pd.DataFrame(
        list(user_inputs.items()), columns=["Feature", "Selected Value"]
    )
    display_df["Selected Value"] = display_df["Selected Value"].astype(str)
    st.dataframe(display_df, hide_index=True, width='stretch')

with col2:
    st.subheader("Estimated Auction Value", anchor=False)

    if st.button("Predict Auction Price", type="primary", width='stretch'):
        if pipeline is not None:
            with st.spinner("Calculating estimated price..."):
                try:
                    # Full pipeline transforms raw input_df and predicts log_price
                    log_price_pred = pipeline.predict(input_df)[0]
                    actual_price_pred = np.expm1(log_price_pred)

                    st.metric(
                        label="Predicted Price (USD)",
                        value=f"${actual_price_pred:,.2f}",
                        delta=f"Machine Age: {sale_year - year_made} Years",
                        delta_color="off",
                    )
                    st.success("Prediction generated successfully!")

                    mae_val = 5092.09
                    st.caption(
                        f"**Estimated Price Range:** ${max(0, actual_price_pred - mae_val):,.2f} — ${actual_price_pred + mae_val:,.2f} *(MAE ±${mae_val:,.2f})*"
                    )

                except Exception as e:
                    st.error(f"Prediction Error:\n\n`{e}`")

