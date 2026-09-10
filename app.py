import streamlit as st
import pandas as pd
import joblib
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Stroke Risk Screening", page_icon="🩺")
st.title("🩺 Stroke Risk Screening")
st.caption(
    "This is a statistical screening tool trained on a public dataset, "
    "NOT a medical diagnosis. Always consult a doctor for real medical advice."
)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Ever Married", ["Yes", "No"])

with col2:
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    work_type = st.selectbox(
        "Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    )
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox(
        "Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"]
    )
raw_input = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}])
encoded = pd.get_dummies(raw_input, drop_first=True)
encoded = encoded.reindex(columns=feature_columns, fill_value=0)
if st.button("Check Risk"):
    scaled_input = scaler.transform(encoded)
    probability = model.predict_proba(scaled_input)[0, 1]

    st.subheader(f"Estimated stroke risk: {probability * 100:.1f}%")

    if probability >= 0.5:
        st.error(
            "⚠️ Higher risk pattern detected. This does not mean a stroke is certain — "
            "it means the input pattern is similar to past stroke cases in the data. "
            "Please consult a doctor."
        )
    else:
        st.success(
            "✅ Lower risk pattern based on this data. This is not a medical clearance — "
            "if you have symptoms or concerns, still see a doctor."
        )

    with st.expander("Why is this model tuned this way?"):
        st.write(
            "The model favors **catching more true stroke cases (higher recall)** over avoiding "
            "false alarms, because in this dataset only ~5% of people had a stroke. For a health "
            "screening tool, missing a real case is considered worse than a false alarm."
        )

#  Run : streamlit run app.py