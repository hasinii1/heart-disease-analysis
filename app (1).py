import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Load Dataset
# -----------------------------
data_path = os.path.join(os.path.dirname(__file__),
                         "heart_disease_data.csv")
df = pd.read_csv(data_path)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🤖 Models",
        "❤️ Prediction"
    ]
)

# ============================================
# Home PAGE
# ============================================
if page == "🏠 Home":

    st.markdown(
        """
        <h1 style='text-align:center;
        color:#C62828;
        font-size:45px;'>
        ❤️ Heart Disease Prediction System
        </h1>
        """,
        unsafe_allow_html=True
    )

    
    st.divider()
    
    st.subheader("🩺 About the Project")
    st.write("""
        This application predicts whether a patient is likely to have heart disease using Machine Learning.""")
    st.divider()
    
    st.subheader("📊 Exploratory Data Analysis")
    st.write(
        "Explore the dataset through interactive charts and statistics."
    )

    # -----------------------------
    # Dataset Overview
    # -----------------------------

    st.subheader("📋 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())
    
    st.divider()
    
    # -----------------------------
    # Heart Disease Distribution
    # -----------------------------

    st.subheader("❤️ Heart Disease Distribution")

    target_map = {
        0: "No Disease",
        1: "Heart Disease"
    }

    target_df = df.copy()
    target_df["Diagnosis"] = target_df["target"].map(target_map)

    fig = px.pie(
        target_df,
        names="Diagnosis",
        title="Heart Disease Distribution",
        hole=0.45
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Chest Pain Type
    # -----------------------------

    st.subheader("💓 Chest Pain Type Distribution")

    cp_names = {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal",
        3: "Asymptomatic"
    }

    cp_df = df.copy()
    cp_df["Chest Pain"] = cp_df["cp"].map(cp_names)

    fig = px.bar(
        cp_df["Chest Pain"].value_counts().reset_index(),
        x="Chest Pain",
        y="count",
        labels={"count":"Patients"},
        title="Chest Pain Types"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Age vs Cholesterol
    # -----------------------------

    st.subheader("📈 Age vs Cholesterol")

    fig = px.scatter(
        df,
        x="age",
        y="chol",
        color="target",
        title="Age vs Cholesterol",
        labels={"target":"Heart Disease"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------

    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------
    # Feature Explorer
    # -----------------------------

    st.subheader("🔍 Feature Explorer")

    numeric_columns = [
        col for col in df.columns
        if df[col].dtype != "object"
    ]

    feature = st.selectbox(
        "Select Feature",
        numeric_columns
    )

    fig = px.histogram(
        df,
        x=feature,
        color="target",
        marginal="box",
        title=f"{feature} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================
# MODELS PAGE
# ============================================

elif page == "🤖 Models":

    st.title("🤖 Machine Learning Models")

    st.write(
        "Multiple Machine Learning algorithms were trained and evaluated. "
        "Random Forest was selected as the final prediction model."
    )

    st.divider()

    # -----------------------------
    # Accuracy Table
    # -----------------------------

    model_results = pd.DataFrame({

        "Model":[
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Support Vector Machine",
            "Random Forest"
        ],

        "Accuracy (%)":[
            88.82,
            78.68,      
            73.77,     
            73.77,     
            88.82       
        ]

    })

    st.subheader("📋 Model Comparison")

    st.dataframe(model_results, use_container_width=True)

    st.divider()

    # -----------------------------
    # Best Model
    # -----------------------------

    st.subheader("🏆 Best Performing Model")

    st.success("""

### 🌟 Random Forest Classifier

Reasons for Selection:

✔ Highest Prediction Accuracy

✔ Handles Non-linear Relationships

✔ Less Overfitting

✔ Robust on Medical Dataset

✔ Used for Final Prediction

""")

    st.divider()

    # -----------------------------
    # Feature Importance
    # -----------------------------

    if hasattr(model, "feature_importances_"):

        st.subheader("📈 Feature Importance")

        importance = pd.DataFrame({

            "Feature":df.columns[:-1],
            "Importance":model.feature_importances_

        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        fig = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance"
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================================
# PREDICTION PAGE
# ============================================

elif page == "❤️ Prediction":

    st.title("❤️ Heart Disease Prediction")

    st.write("Enter the patient's medical details below.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("Age", 1, 120, 45)

        sex = st.selectbox(
            "Sex",
            [0,1],
            format_func=lambda x: "Female" if x==0 else "Male"
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0,1,2,3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            80,
            250,
            120
        )

        chol = st.number_input(
            "Serum Cholesterol",
            100,
            700,
            200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar >120 mg/dl",
            [0,1]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0,1,2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            60,
            250,
            150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0,1]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            0.0,
            10.0,
            1.0
        )

        slope = st.selectbox(
            "Slope",
            [0,1,2]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0,1,2,3,4]
        )

        thal = st.selectbox(
            "Thalassemia",
            [0,1,2,3]
        )

    st.subheader("📊 Patient Summary")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Age", age)
    c2.metric("Blood Pressure", trestbps)
    c3.metric("Cholesterol", chol)
    c4.metric("Heart Rate", thalach)

    if st.button("🔍 Predict Heart Disease", use_container_width=True):

        input_data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]])

        prediction = model.predict(input_data)

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.error("⚠️ The patient is likely to have Heart Disease.")

        else:

            st.success("✅ The patient is unlikely to have Heart Disease.")

        if hasattr(model,"predict_proba"):

            probability = model.predict_proba(input_data)[0]

            st.subheader("Prediction Confidence")

            st.progress(float(max(probability)))

            st.write(
                f"Healthy Probability : **{probability[0]*100:.2f}%**"
            )

            st.write(
                f"Heart Disease Probability : **{probability[1]*100:.2f}%**"
            )


    # ============================================
    # MEDICAL FEATURE GUIDE
    # ============================================

    with st.expander("📚 Medical Feature Guide", expanded=False):

        st.markdown("""

### 👤 Sex

- **0** → Female
- **1** → Male

---

### ❤️ Chest Pain Type (CP)

- **0** → Typical Angina
- **1** → Atypical Angina
- **2** → Non-anginal Pain
- **3** → Asymptomatic

---

### 🍬 Fasting Blood Sugar (FBS)

- **0** → ≤120 mg/dL
- **1** → >120 mg/dL

---

### 📈 Resting ECG (RestECG)

- **0** → Normal

- **1** → ST-T Wave Abnormality

- **2** → Left Ventricular Hypertrophy

---

### 🏃 Exercise Induced Angina (Exang)

- **0** → No

- **1** → Yes

---

### 📉 Oldpeak

ST depression induced by exercise relative to rest.

Higher values generally indicate a greater likelihood of heart disease.

---

### 📊 Slope

- **0** → Upsloping

- **1** → Flat

- **2** → Downsloping

---

### 🩸 Number of Major Vessels (CA)

Represents the number of major vessels detected by fluoroscopy.

- **0** → No vessels

- **1** → One vessel

- **2** → Two vessels

- **3** → Three vessels

- **4** → Four vessels

Higher values generally indicate more severe coronary artery disease.

---

### 🧬 Thalassemia (Thal)

- **0** → Unknown

- **1** → Normal

- **2** → Fixed Defect

- **3** → Reversible Defect

---

### ❤️ Maximum Heart Rate (Thalach)

Maximum heart rate achieved during exercise testing.

Higher values generally indicate better cardiovascular fitness.

---

### 🩺 Resting Blood Pressure (Trestbps)

Resting blood pressure measured in **mm Hg** before exercise.

---

### 🧪 Serum Cholesterol (Chol)

Measured in **mg/dL**.

Higher cholesterol levels are associated with increased cardiovascular risk.
""")

    st.divider()
    st.caption("""
    ⚠️ **Disclaimer:** This application is developed for educational purposes only.
    It should **not** be used as a substitute for professional medical diagnosis or treatment.
    Always consult a qualified healthcare professional for medical advice.
""")