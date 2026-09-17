import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = pickle.load(open("linear_model.pkl", "rb"))

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
    color: white;
}

/* Titles */
h1, h2, h3 {
    color: white !important;
    font-family: 'Segoe UI';
}

/* Glassmorphism Cards */
.card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 20px;
    font-weight: bold;
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(135deg, #14b8a6, #06b6d4);
    color: white;
    font-size: 18px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #06b6d4, #3b82f6);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
}

/* Slider Labels */
label {
    color: white !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* Slider Values */
.stSlider span {
    color: white !important;
}

/* Markdown Text */
p, div {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div style='text-align:center; padding:20px;'>
    <h1 style='font-size:60px;'>⚡ Smart Energy Predictor</h1>
    <p style='font-size:24px; color:#cbd5e1;'>
        AI-Based Smart Energy Consumption Prediction System
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙ Control Panel")

st.sidebar.info(
    "Adjust energy parameters and predict electricity usage."
)

# ---------------- INPUT SECTION ----------------

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("## 📥 Enter Input Values")

col1, col2 = st.columns(2)

with col1:

    temperature = st.slider(
        "🌡 Temperature (°C)",
        0,
        50,
        25
    )

    humidity = st.slider(
        "💧 Humidity (%)",
        0,
        100,
        50
    )

    day = st.slider(
        "📅 Day",
        1,
        31,
        1
    )

with col2:

    month = st.slider(
        "🗓 Month",
        1,
        12,
        1
    )

    hour = st.slider(
        "⏰ Hour",
        0,
        23,
        12
    )

predict_button = st.button("🧠 Predict Energy Usage")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------

if predict_button:

    # Prediction
    prediction = model.predict([
        [temperature, humidity, day, month, hour]
    ])[0]

    # Prevent negative values
    prediction = max(0, round(prediction, 2))

    # Bill Prediction
    bill = round(prediction * 6, 2)

    # ---------------- SAVE HISTORY ----------------

    history_data = {
        "Date": [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ],
        "Energy Usage": [prediction],
        "Estimated Bill": [bill]
    }

    history_df = pd.DataFrame(history_data)

    if os.path.exists("prediction_history.csv"):

        history_df.to_csv(
            "prediction_history.csv",
            mode='a',
            header=False,
            index=False
        )

    else:

        history_df.to_csv(
            "prediction_history.csv",
            index=False
        )

    # ---------------- RESULTS SECTION ----------------

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("## 📊 Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.markdown(f"""
        <div class='metric-card'>
            ⚡ Energy Usage<br><br>
            {prediction} Units
        </div>
        """, unsafe_allow_html=True)

    with result_col2:

        st.markdown(f"""
        <div class='metric-card'>
            💰 Estimated Bill<br><br>
            ₹ {bill}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- SUGGESTIONS ----------------

    if prediction > 200:

        st.warning("⚠ High energy usage expected!")

        st.markdown("""
        ### 💡 Smart Suggestions
        - Turn off unused appliances
        - Reduce AC temperature
        - Use appliances during off-peak hours
        - Switch to LED lighting
        """)

    else:

        st.success("✅ Energy consumption is efficient.")

        st.markdown("""
        ### 💡 Smart Suggestions
        - Current usage is optimized
        - Maintain efficient usage pattern
        - Continue smart energy practices
        """)

    # ---------------- FAULT DETECTION ----------------

    if prediction > 350:

        st.error(
            "⚠️ Abnormal Energy Usage Detected! Possible power leakage or faulty appliance."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- GRAPH SECTION ----------------

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("## 📈 Energy Trend Analysis")

    trend_data = [
        max(0, prediction - 20),
        max(0, prediction - 10),
        prediction,
        prediction + 10
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        trend_data,
        marker='o',
        linewidth=3
    )

    ax.set_title("Predicted Energy Trend")

    ax.set_xlabel("Time")

    ax.set_ylabel("Energy Usage")

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- HISTORY SECTION ----------------

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("## 📜 Prediction History")

    if os.path.exists("prediction_history.csv"):

        history = pd.read_csv("prediction_history.csv")

        st.dataframe(
            history.tail(10),
            use_container_width=True
        )

        csv = history.to_csv(
            index=False
        ).encode('utf-8')

        st.download_button(
            label="⬇ Download Prediction History",
            data=csv,
            file_name='prediction_history.csv',
            mime='text/csv'
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------

st.markdown("""
<hr>

<center>

<p style='color:#94a3b8;'>

Developed using Machine Learning & Streamlit

<br>

Smart Energy Consumption Prediction System

</p>

</center>
""", unsafe_allow_html=True)