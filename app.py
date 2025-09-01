import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Load trained model
# ---------------------------
model_path = "E:/PYTHON/ML_PROJECT/model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🚀 Employee Resignation Prediction App")

st.write("Fill the details below to predict whether an employee is likely to resign.")


# ---------------------------
# User inputs (ordered same as dataset)
# ---------------------------

department = st.selectbox("Department", [
    "Sales", "Marketing", "Operations", "Customer Support",
    "Legal", "Finance", "IT", "HR", "Engineering"
])

gender = st.selectbox("Gender", ["Other", "Male", "Female"])

age = st.number_input("Age", min_value=18, max_value=70, step=1)

job_title = st.selectbox("Job Title", [
    "Manager", "Engineer", "Analyst", "Technician",
    "Specialist", "Developer", "Consultant"
])

years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, step=1)

education = st.selectbox("Education Level", [
    "High School", "Bachelor", "Master", "PhD"
])

performance_score = st.slider("Performance Score", 1, 10, 5)

monthly_salary = st.number_input("Monthly Salary", min_value=1000.0, max_value=50000.0, step=100.0)

work_hours = st.slider("Work Hours Per Week", 20, 80, 40)

projects = st.number_input("Projects Handled", min_value=0, max_value=50, step=1)

overtime = st.number_input("Overtime Hours", min_value=0, max_value=100, step=1)

sick_days = st.number_input("Sick Days", min_value=0, max_value=30, step=1)

remote_work = st.slider("Remote Work Frequency", 0, 5, 2)

team_size = st.number_input("Team Size", min_value=1, max_value=50, step=1)

training_hours = st.number_input("Training Hours", min_value=0, max_value=500, step=1)

promotions = st.number_input("Promotions", min_value=0, max_value=10, step=1)

satisfaction = st.slider("Employee Satisfaction Score", 0.0, 10.0, 5.0)


# ---------------------------
# Encode categorical features (same mapping as training)
# ---------------------------
dept_map = {
    'Sales': 0, 'Marketing': 1, 'Operations': 2, 'Customer Support': 3,
    'Legal': 4, 'Finance': 5, 'IT': 6, 'HR': 7, 'Engineering': 8
}
gender_map = {'Other': 0, 'Male': 1, 'Female': 2}
job_map = {
    'Manager': 0, 'Engineer': 1, 'Analyst': 2, 'Technician': 3,
    'Specialist': 4, 'Developer': 5, 'Consultant': 6
}
edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}

# ---------------------------
# Create feature vector (ORDERED SAME AS TRAINING DATASET)
# ---------------------------
features = np.array([[ 
    dept_map[department],          # Department
    gender_map[gender],            # Gender
    age,                           # Age
    job_map[job_title],            # Job Title
    years_at_company,              # Years at Company
    edu_map[education],            # Education Level
    performance_score,             # Performance Score
    monthly_salary,                # Monthly Salary
    work_hours,                    # Work Hours Per Week
    projects,                      # Projects Handled
    overtime,                      # Overtime Hours
    sick_days,                     # Sick Days
    remote_work,                   # Remote Work Frequency
    team_size,                     # Team Size
    training_hours,                # Training Hours
    promotions,                    # Promotions
    satisfaction                   # Employee Satisfaction Score
]])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Resignation"):
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Employee is likely to RESIGN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Employee is likely to STAY (Probability: {prob:.2f})")

