import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Smartphone Addiction Predictor", page_icon="📱", layout="centered")
# --- HIDE STREAMLIT BRANDING ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("📱 Smartphone Addiction Risk Predictor")
st.markdown("Developed by **Team CU_CP_Team_14387**")
st.write("Adjust the sliders below to see our Machine Learning model predict the risk of smartphone addiction in real-time based on usage patterns.")

# --- 2. TRAIN ML MODEL (Cached so it runs instantly) ---
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
    
    # Preprocessing (Match the original data cleaning)
    df_clean = df.drop(columns=['transaction_id', 'user_id', 'addiction_level'])
    cat_cols = ['gender', 'stress_level', 'academic_work_impact']
    
    # Get dummies for both training and structure reference
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    
    X = df_encoded.drop(columns=['addicted_label'])
    y = df_encoded['addicted_label']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Gradient Boosting (Top Performer: ~93.7% Accuracy)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, X.columns

try:
    model, scaler, feature_columns = train_model()
    st.success("✅ AI Model Loaded: Gradient Boosting Classifier (93.7% Accuracy)")
except Exception as e:
    st.error(f"Error loading dataset. Please make sure the CSV file is in the same folder. Detail: {e}")
    st.stop()

# --- 3. USER INPUT DASHBOARD ---
st.header("Enter Daily Habits")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=80, value=22)
    daily_screen_time = st.slider("Daily Screen Time (Hours)", 0.0, 24.0, 5.0)
    social_media = st.slider("Social Media (Hours)", 0.0, 24.0, 3.0)
    gaming = st.slider("Gaming (Hours)", 0.0, 24.0, 1.0)
    weekend_screen = st.slider("Weekend Screen Time (Hours)", 0.0, 24.0, 6.0)

with col2:
    work_study = st.slider("Work/Study (Hours)", 0.0, 24.0, 4.0)
    sleep = st.slider("Sleep (Hours)", 0.0, 24.0, 7.0)
    notifications = st.slider("Notifications per Day", 0, 500, 150)
    app_opens = st.slider("App Opens per Day", 0, 300, 80)
    
st.markdown("---")
col3, col4, col5 = st.columns(3)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with col4:
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
with col5:
    academic_impact = st.selectbox("Academic/Work Impact", ["Yes", "No"])

# --- 4. MAKE PREDICTION ---
if st.button("Predict Addiction Risk", type="primary"):
    
    # Create a dictionary matching the original dataframe structure
    input_data = {
        'age': age,
        'daily_screen_time_hours': daily_screen_time,
        'social_media_hours': social_media,
        'gaming_hours': gaming,
        'work_study_hours': work_study,
        'sleep_hours': sleep,
        'notifications_per_day': notifications,
        'app_opens_per_day': app_opens,
        'weekend_screen_time': weekend_screen,
        'gender_Male': 1 if gender == "Male" else 0,
        'gender_Other': 1 if gender == "Other" else 0,
        'stress_level_Low': 1 if stress_level == "Low" else 0,
        'stress_level_Medium': 1 if stress_level == "Medium" else 0,
        'academic_work_impact_Yes': 1 if academic_impact == "Yes" else 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure columns match training data exactly (in case any one-hot columns are missing)
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Order columns exactly as they were during training
    input_df = input_df[feature_columns]
    
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] # Probability of being addicted
    
    st.markdown("---")
    st.header("Results:")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Smartphone Addiction** (Confidence: {probability*100:.1f}%)")
        st.write("Recommendation: Consider utilizing app blockers and setting daily screen-time limits to improve well-being.")
    else:
        st.success(f"✅ **Healthy Usage Pattern** (Addiction Probability: {probability*100:.1f}%)")
        st.write("Great job maintaining a healthy balance between your digital and real life!")
