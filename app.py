import streamlit as st
import pandas as pd
import pickle

# === Load Model dan Tools ===
with open("mental_health_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Prediksi Gangguan Kecemasan", layout="centered")
st.title("🧠 Prediksi Gangguan Kecemasan Berdasarkan Data Pribadi")

st.markdown("Masukkan informasi berikut untuk memprediksi kemungkinan mengalami **gangguan kecemasan**:")

# === Input Form Manual ===
Gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
Age = st.text_input("Age", "25")
Occupation = st.selectbox("Occupation", label_encoders["Occupation"].classes_)
Sleep_Hours = st.text_input("Sleep Hours", "7")
Physical_Activity = st.text_input("Physical Activity (hrs/week)", "3")
Caffeine = st.text_input("Caffeine Intake (mg/day)", "100")
Alcohol = st.text_input("Alcohol Consumption (drinks/week)", "2")
Smoking = st.selectbox("Do you smoke?", label_encoders["Smoking"].classes_)
Family_History = st.selectbox("Family History of Anxiety?", label_encoders["Family History of Anxiety"].classes_)
Stress_Level = st.text_input("Stress Level (1-10)", "5")
Heart_Rate = st.text_input("Heart Rate (bpm)", "75")
Breathing_Rate = st.text_input("Breathing Rate (breaths/min)", "15")
Sweating_Level = st.text_input("Sweating Level (1-5)", "2")
Dizziness = st.selectbox("Do you experience dizziness?", label_encoders["Dizziness"].classes_)
Medication = st.selectbox("Are you on medication?", label_encoders["Medication"].classes_)
Therapy = st.text_input("Therapy Sessions (per month)", "1")
Major_Event = st.selectbox("Recent Major Life Event?", label_encoders["Recent Major Life Event"].classes_)
Diet_Quality = st.text_input("Diet Quality (1-10)", "5")

# === Susun Data Input ke DataFrame ===
input_dict = {
    'Age': int(Age),
    'Gender': Gender,
    'Occupation': Occupation,
    'Sleep Hours': float(Sleep_Hours),
    'Physical Activity (hrs/week)': float(Physical_Activity),
    'Caffeine Intake (mg/day)': float(Caffeine),
    'Alcohol Consumption (drinks/week)': float(Alcohol),
    'Smoking': Smoking,
    'Family History of Anxiety': Family_History,
    'Stress Level (1-10)': int(Stress_Level),
    'Heart Rate (bpm)': int(Heart_Rate),
    'Breathing Rate (breaths/min)': int(Breathing_Rate),
    'Sweating Level (1-5)': int(Sweating_Level),
    'Dizziness': Dizziness,
    'Medication': Medication,
    'Therapy Sessions (per month)': int(Therapy),
    'Recent Major Life Event': Major_Event,
    'Diet Quality (1-10)': int(Diet_Quality)
}

input_df = pd.DataFrame([input_dict])

# === Encode Kategorikal ===
for col in label_encoders:
    input_df[col] = label_encoders[col].transform(input_df[col])

# === Scaling ===
numerical_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# === Prediksi ===
if st.button("🔍 Prediksi"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Hasil Prediksi:")
    if pred == 1:
        st.error("⚠️ Anda berpotensi mengalami gangguan kecemasan.")
    else:
        st.success("✅ Anda tidak menunjukkan tanda-tanda gangguan kecemasan.")
    
