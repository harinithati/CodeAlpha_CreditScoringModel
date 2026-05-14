import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time

# =============================================
# PAGE CONFIGURATION
# =============================================

st.set_page_config(
    page_title="AI Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# CUSTOM CSS STYLING
# =============================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background: linear-gradient(90deg,#00C9FF,#92FE9D);
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.02);
}

.metric-card {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(255,255,255,0.1);
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

# =============================================
# LOAD MODEL
# =============================================

@st.cache_resource

def load_model():
    model = pickle.load(open('best_credit_model.pkl', 'rb'))
    return model

model = load_model()

# =============================================
# SIDEBAR
# =============================================

st.sidebar.title("⚡ AI Credit Scoring System")

st.sidebar.markdown("""
### About Project

This Machine Learning application predicts customer credit score category using:

- Random Forest
- XGBoost
- Feature Engineering
- Advanced Data Processing

### Technologies Used

- Python
- Streamlit
- Scikit-Learn
- XGBoost
- Pandas
- NumPy

""")

st.sidebar.success("Model Loaded Successfully ✅")

# =============================================
# HEADER SECTION
# =============================================

st.markdown(
    """
    <h1 style='text-align:center;'>💳 AI-Powered Credit Scoring Prediction System</h1>
    <h4 style='text-align:center;color:gray;'>Advanced Machine Learning Web Application</h4>
    """,
    unsafe_allow_html=True
)

st.write("---")

# =============================================
# DASHBOARD METRICS
# =============================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model Accuracy",
        value="79%",
        delta="High Performance"
    )

with col2:
    st.metric(
        label="ML Algorithm",
        value="XGBoost",
        delta="Advanced AI"
    )

with col3:
    st.metric(
        label="Prediction Speed",
        value="Real-Time",
        delta="Fast"
    )

st.write("---")

# =============================================
# INPUT SECTION
# =============================================

st.subheader("📊 Customer Financial Information")

col1, col2 = st.columns(2)

with col1:

    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    monthly_balance = st.number_input(
        "Monthly Balance",
        min_value=0.0,
        value=2000.0,
        step=100.0
    )

    outstanding_debt = st.number_input(
        "Outstanding Debt",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

with col2:

    interest_rate = st.number_input(
        "Interest Rate",
        min_value=0.0,
        value=12.0,
        step=1.0
    )

    num_loans = st.number_input(
        "Number of Loans",
        min_value=0,
        value=2,
        step=1
    )

    emi = st.number_input(
        "Monthly EMI",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

st.write("---")

# =============================================
# PREDICTION SECTION
# =============================================

if st.button("🚀 Predict Credit Score"):

    with st.spinner("Analyzing financial behavior using AI..."):

        time.sleep(2)

        features = np.array([[
            annual_income,
            monthly_balance,
            outstanding_debt,
            interest_rate,
            num_loans,
            emi
        ]])

        try:

            prediction = model.predict(features)

            prediction_value = prediction[0]

            st.success("Prediction Completed Successfully ✅")

            st.write("---")

            # =============================================
            # RESULT DISPLAY
            # =============================================

            if prediction_value == 0:

                st.error("🔴 Credit Score Category: POOR")

                st.warning(
                    "Customer may have high financial risk."
                )

            elif prediction_value == 1:

                st.info("🟡 Credit Score Category: STANDARD")

                st.warning(
                    "Customer has moderate financial stability."
                )

            else:

                st.success("🟢 Credit Score Category: GOOD")

                st.balloons()

                st.success(
                    "Customer has strong financial behavior."
                )

        except Exception as e:

            st.error("Feature mismatch detected.")

            st.write(e)

# =============================================
# FINANCIAL INSIGHTS
# =============================================

st.write("---")

st.subheader("📈 Financial Insights")

chart_data = pd.DataFrame({
    'Category': [
        'Income',
        'Debt',
        'Balance',
        'EMI'
    ],
    'Value': [
        annual_income,
        outstanding_debt,
        monthly_balance,
        emi
    ]
})

st.bar_chart(
    chart_data.set_index('Category')
)

# =============================================
# FEATURE IMPORTANCE SECTION
# =============================================

st.write("---")

st.subheader("🔥 Important Features Used by AI")

importance_df = pd.DataFrame({
    'Feature': [
        'Outstanding Debt',
        'Annual Income',
        'Monthly Balance',
        'Interest Rate',
        'Number of Loans',
        'EMI'
    ],
    'Importance': [
        0.30,
        0.25,
        0.18,
        0.12,
        0.10,
        0.05
    ]
})

st.dataframe(
    importance_df,
    use_container_width=True
)

# =============================================
# PROJECT INFORMATION
# =============================================

st.write("---")

with st.expander("📚 Project Information"):

    st.markdown("""

### Project Objective

This application predicts customer credit score category using Machine Learning.

### Algorithms Used

- Random Forest
- XGBoost

### Features Used

- Annual Income
- Outstanding Debt
- Monthly Balance
- EMI
- Interest Rate
- Number of Loans

### Project Type

Machine Learning Internship Project

""")

# =============================================
# FOOTER
# =============================================

st.markdown(
    """
    <div class='footer'>
    Developed with ❤️ using Streamlit & Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)

