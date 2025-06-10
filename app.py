import streamlit as st # type: ignore
import numpy as np # type: ignore
import joblib # Use only joblib # type: ignore
import os
# --- Load model and scaler ---
model_path = "model_compressed.pkl"
scaler_path = "scaler.pkl"

if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found.")
    st.stop()
if not os.path.exists(scaler_path):
    st.error(f"Error: Scaler file '{scaler_path}' not found.")
    st.stop()

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Failed to load model or scaler. Error: {e}")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Career Aspiration Predictor", layout="centered", initial_sidebar_state="collapsed")

# --- Custom CSS for enhanced UI ---
st.markdown("""
<style>
    body, .stApp { background: linear-gradient(to right, #141e30, #243b55); color: white; font-family: 'Segoe UI', sans-serif; }
    .main-header {
        font-size: 3.5em;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 20px;
        font-weight: bold;
        animation: slide 4s ease-in-out infinite alternate;
        text-shadow: 0 0 10px #e74c3c, 0 0 20px #c0392b;
    }
    @keyframes slide {
        0% { transform: translateX(-10px); }
        100% { transform: translateX(10px); }
    }
    label, .stMarkdown p, .stCaption, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        font-size: 1.2em;
        padding: 12px;
        border-radius: 10px;
        border: none;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        box-shadow: 0 0 10px #e74c3c;
    }
    .stProgress > div > div > div > div {
        background-color: #e74c3c;
    }
    .stExpander {
        background-color: #2c3e50;
        color: white;
        border-radius: 10px;
    }
    .stExpander button {
        color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown('<div class="main-header">🎯 Career Aspiration Predictor 🎯</div>', unsafe_allow_html=True)

st.write("""
Welcome! This interactive tool helps you explore career paths based on your academic strengths and personal background.
""")
st.info("💡 Tip: Enter your academic and personal details to see top career matches.")
st.markdown("---")

# --- Personal Info Section ---
st.subheader("Personal & Academic Profile")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], help="Your biological gender")
    part_time_job = st.selectbox("Has Part-Time Job?", ["Yes", "No"], help="Do you have a part-time job?")

with col2:
    absence_days = st.number_input("Absence Days", 0, 365, help="Number of days you were absent")
    extracurricular = st.selectbox("In Extracurricular Activities?", ["Yes", "No"], help="Do you take part in extracurriculars?")

weekly_self_study_hours = st.number_input("Weekly Self-Study Hours", 0, 168, 10, help="How many hours do you self-study weekly?")
st.markdown("---")

# --- Subject Scores ---
st.subheader("Academic Performance")
with st.expander("📚 Enter Subject Scores", expanded=True):
    subjects = ["Math", "History", "Physics", "Chemistry", "Biology", "English", "Geography"]
    scores = {}
    cols = st.columns(2)
    for i, sub in enumerate(subjects):
        with cols[i % 2]:
            scores[sub] = st.slider(f"{sub} Score", 0, 100, 75)

st.markdown("---")

# --- Prediction ---
if st.button("Predict Career Aspiration"):
    gender_map = {"Male": 0, "Female": 1}
    binary_map = {"Yes": 1, "No": 0}
    input_data = [
        gender_map[gender],
        binary_map[part_time_job],
        absence_days,
        binary_map[extracurricular],
        weekly_self_study_hours,
        scores["Math"], scores["History"], scores["Physics"],
        scores["Chemistry"], scores["Biology"], scores["English"], scores["Geography"]
    ]
    total = sum(scores.values())
    avg = total / len(scores)
    input_data.extend([total, avg])

    try:
        scaled = scaler.transform([input_data])
        probs = model.predict_proba(scaled)[0]
        top_idx = np.argsort(probs)[::-1][:5]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    career_labels = [
        "Artist", "Game Developer", "Real Estate Developer", "Business Owner", "Designer", "Doctor", "Engineer",
        "Teacher", "Lawyer", "Psychologist", "Scientist", "Chef", "Architect", "Writer", "Athlete", "Musician", "Entrepreneur"
    ]

    st.success("✨ Top Career Recommendations ✨")
    for i, idx in enumerate(top_idx):
        st.markdown(f"### {i+1}. {career_labels[idx]} — **{probs[idx]*100:.1f}%** confidence")
        st.progress(float(probs[idx]))

    st.info("🔍 These are AI-generated suggestions based on input data. Explore your interests deeply!")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Developed with ❤️ using Streamlit")
