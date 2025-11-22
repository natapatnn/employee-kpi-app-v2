import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# -----------------------------
# Load trained model + encoders
# -----------------------------
with open("model_kpi.pkl", "rb") as file:
    (
        model,
        department_encoder,
        region_encoder,
        education_encoder,
        gender_encoder,
        recruitment_channel_encoder,
    ) = pickle.load(file)

# -----------------------------
# Load original dataset (ใช้เพื่อทำกราฟ)
# -----------------------------
df = pd.read_csv("Uncleaned_employees_final_dataset.csv")

# เผื่อยังมี employee_id อยู่
if "employee_id" in df.columns:
    df = df.drop("employee_id", axis=1)

# เผื่อมี missing แบบเดียวกับตอนเทรน
if "education" in df.columns:
    df["education"] = df["education"].fillna("None")

if "previous_year_rating" in df.columns:
    df["previous_year_rating"] = df["previous_year_rating"].fillna(
        df["previous_year_rating"].mode()[0]
    )

# -----------------------------
# หน้าเว็บหลัก
# -----------------------------
st.title("Employee KPIs App")

if "tab_selected" not in st.session_state:
    st.session_state.tab_selected = 0

tabs = ["Predict KPIs", "Visualize Data", "Predict from CSV"]
selected_tab = st.radio("Select Tab:", tabs, index=st.session_state.tab_selected)

if selected_tab != tabs[st.session_state.tab_selected]:
    st.session_state.tab_selected = tabs.index(selected_tab)

# ============================================================
# TAB 1 : Predict KPIs (กรอกข้อมูลเองทีละคน)
# ============================================================
if st.session_state.tab_selected == 0:
    st.header("Predict KPIs")

    department = st.selectbox("Department", department_encoder.classes_)
    region = st.selectbox("Region", region_encoder.classes_)
    education = st.selectbox("Education", education_encoder.classes_)
    gender = st.radio("Gender", gender_encoder.classes_)
    recruitment_channel = st.selectbox(
        "Recruitment Channel", recruitment_channel_encoder.classes_
    )

    no_of_trainings = st.slider("Number of Trainings", 1, 10, 1)
    age = st.slider("Age", 18, 60, 30)
    previous_year_rating = st.slider("Previous Year Rating", 1.0, 5.0, 3.0, 0.1)
    length_of_service = st.slider("Length of Service (years)", 1, 20, 5)
    awards_won = st.checkbox("Awards Won")
    avg_training_score = st.slider("Average Training Score", 40, 100, 70)

    user_input = pd.DataFrame(
        {
            "department": [department],
            "region": [region],
            "education": [education],
            "gender": [gender],
            "recruitment_channel": [recruitment_channel],
            "no_of_trainings": [no_of_trainings],
            "age": [age],
            "previous_year_rating": [previous_year_rating],
            "length_of_service": [length_of_service],
            "awards_won": [1 if awards_won else 0],
            "avg_training_score": [avg_training_score],
        }
    )

    # encode ตาม encoder ที่เซฟมาใน model_kpi.pkl
    user_input["department"] = department_encoder.transform(user_input["department"])
    user_input["region"] = region_encoder.transform(user_input["region"])
    user_input["education"] = education_encoder.transform(user_input["education"])
    user_input["gender"] = gender_encoder.transform(user_input["gender"])
    user_input["recruitment_channel"] = recruitment_channel_encoder.transform(
        user_input["recruitment_channel"]
    )

    if st.button("Predict"):
        prediction = model.predict(user_input)[0]
        st.subheader("Prediction Result")
        st.write("KPIs_met_more_than_80:", int(prediction))

# ============================================================
# TAB 2 : Visualize Data จาก dataset เดิม
# ============================================================
elif st.session_state.tab_selected == 1:
    st.header("Visualize Data")

    categorical_columns = [
        "department",
        "region",
        "education",
        "gender",
        "recruitment_channel",
    ]
    categorical_columns = [c for c in categorical_columns if c in df.columns]

    condition_feature = st.selectbox(
        "Select Condition Feature:", categorical_columns
    )

    default_condition_values = ["Select All"] + df[condition_feature].unique().tolist()
    condition_values = st.multiselect(
        "Select Condition Values:", default_condition_values
    )

    if "Select All" in condition_values or len(condition_values) == 0:
        condition_values = df[condition_feature].unique().tolist()

    if len(condition_values) > 0:
        filtered_df = df[df[condition_feature].isin(condition_values)]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            x=condition_feature,
            hue="KPIs_met_more_than_80",
            data=filtered_df,
            ax=ax,
        )
        plt.title("Number of Employees based on KPIs")
        plt.xlabel(condition_feature)
        plt.ylabel("Number of Employees")
        st.pyplot(fig)

# ============================================================
# TAB 3 : Predict from CSV (อัปโหลดไฟล์พนักงานหลายคน)
# ============================================================
elif st.session_state.tab_selected == 2:
    st.header("Predict from CSV")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        csv_df_org = pd.read_csv(uploaded_file)
        csv_df_org = csv_df_org.dropna()

        csv_df = csv_df_org.copy()

        if "employee_id" in csv_df.columns:
            csv_df = csv_df.drop("employee_id", axis=1)

        csv_df["department"] = department_encoder.transform(csv_df["department"])
        csv_df["region"] = region_encoder.transform(csv_df["region"])
        csv_df["education"] = education_encoder.transform(csv_df["education"])
        csv_df["gender"] = gender_encoder.transform(csv_df["gender"])
        csv_df["recruitment_channel"] = recruitment_channel_encoder.transform(
            csv_df["recruitment_channel"]
        )

        predictions = model.predict(csv_df)
        csv_df_org["KPIs_met_more_than_80"] = predictions

        st.subheader("Predicted Results")
        st.write(csv_df_org)

        st.subheader("Visualize Predictions")
        feature_for_visualization = st.selectbox(
            "Select Feature for Visualization:", csv_df_org.columns
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(
            x=feature_for_visualization,
            hue="KPIs_met_more_than_80",
            data=csv_df_org,
            ax=ax,
        )
        plt.title(f"Number of Employees based on KPIs - {feature_for_visualization}")
        plt.xlabel(feature_for_visualization)
        plt.ylabel("Number of Employees")
        st.pyplot(fig)
