import streamlit as st
import pandas as pd
import joblib

# Load model and components
model = joblib.load("final_gradient_boosting_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("minmax_scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load reference data (already encoded)
df = pd.read_csv("processed_all_cars_data.csv")

st.title("🚗 Used Car Price Predictor")

# Use encoder classes for dropdowns (show real labels, not numbers)
brand_options = list(label_encoders['Brand'].classes_)
brand = st.selectbox("Brand", brand_options)

# Filter Car Models based on selected brand
brand_encoded = label_encoders['Brand'].transform([brand])[0]
model_mask = df['Brand'] == brand_encoded
car_model_encoded = df[model_mask]['Car_Model'].unique()
car_model_options = label_encoders['Car_Model'].inverse_transform(car_model_encoded)
car_model = st.selectbox("Car Model", sorted(car_model_options))

city = st.selectbox("City", list(label_encoders['City'].classes_))
ft = st.selectbox("Fuel Type", list(label_encoders['ft'].classes_))
bt = st.selectbox("Body Type", list(label_encoders['bt'].classes_))
trans = st.radio("Transmission", ["Manual", "Automatic"])
insurance = st.selectbox("Insurance Validity", list(label_encoders['Insurance Validity'].classes_))
mileage_cat = st.selectbox("Mileage Category", list(label_encoders['Mileage_Category'].classes_))

model_year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2020)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=50.0, value=15.0)
owners = st.number_input("No. of Owners", min_value=1, max_value=10, value=1)

# Preprocessing
def preprocess_input():
    input_data = {
        "Brand": label_encoders["Brand"].transform([brand])[0],
        "Car_Model": label_encoders["Car_Model"].transform([car_model])[0],
        "City": label_encoders["City"].transform([city])[0],
        "ft": label_encoders["ft"].transform([ft])[0],
        "bt": label_encoders["bt"].transform([bt])[0],
        "Transmission_Manual": 1 if trans == "Manual" else 0,
        "Insurance Validity": label_encoders["Insurance Validity"].transform([insurance])[0],
        "Mileage_Category": label_encoders["Mileage_Category"].transform([mileage_cat])[0],
        "Model_Year": model_year,
        "Mileage(kmpl)": mileage,
        "No_of_Owners": owners,
    }
    df_input = pd.DataFrame([input_data])
    df_input["Mileage(kmpl)"] = scaler.transform(df_input[["Mileage(kmpl)"]])
    feature_columns = [col for col in model_columns if col != 'Car_Price']
    return df_input[feature_columns]


# Prediction
if st.button("Predict Price"):
    final_input = preprocess_input()
    price = model.predict(final_input)[0]
    st.success(f"💰 Estimated Car Price: ₹{price:,.2f}")
