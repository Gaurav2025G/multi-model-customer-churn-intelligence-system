import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import shap
import os

from joblib import load

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ENTERPRISE CSS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   GLOBAL
============================================================ */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.stApp{
background:#050816;
}

/* ============================================================
   TYPOGRAPHY
============================================================ */

html,
body,
[class*="css"]{

font-family:
'Inter',
sans-serif;
}

/* ============================================================
   HERO
============================================================ */

.hero-card{

background:
linear-gradient(
135deg,
#2563EB 0%,
#4F46E5 50%,
#7C3AED 100%
);

padding:40px;

border-radius:24px;

margin-bottom:25px;

box-shadow:
0px 20px 40px rgba(37,99,235,0.25);
}

.hero-title{

font-size:48px;

font-weight:800;

color:white;

margin-bottom:10px;
}

.hero-subtitle{

font-size:18px;

color:#E2E8F0;

line-height:1.7;
}

/* ============================================================
   KPI CARDS
============================================================ */

.metric-card{

background:
rgba(17,24,39,0.90);

backdrop-filter:
blur(12px);

padding:24px;

border-radius:20px;

border:
1px solid rgba(255,255,255,0.08);

transition:
all 0.3s ease;

text-align:center;
}

.metric-card:hover{

transform:
translateY(-5px);

box-shadow:
0px 15px 35px rgba(59,130,246,0.20);
}

.metric-label{

font-size:14px;

color:#94A3B8;

margin-bottom:8px;
}

.metric-value{

font-size:30px;

font-weight:800;

color:white;
}

/* ============================================================
   SECTION TITLES
============================================================ */

.section-title{

font-size:28px;

font-weight:700;

color:white;

margin-bottom:10px;
}

/* ============================================================
   SIDEBAR
============================================================ */

[data-testid="stSidebar"]{

background:
#0F172A;

border-right:
1px solid #1E293B;
}

[data-testid="stSidebar"] *{

color:white;
}

/* ============================================================
   INPUTS
============================================================ */

.stSelectbox div[data-baseweb="select"] > div{

background:#111827;

border:1px solid #334155;

color:white;
}

.stSelectbox div[data-baseweb="select"] span{

color:white;
}

.stNumberInput input{

background:#111827 !important;

color:white !important;
}

textarea{

background:#111827 !important;

color:white !important;
}

/* ============================================================
   BUTTON
============================================================ */

.stButton button{

width:100%;

height:58px;

border:none;

border-radius:16px;

font-size:18px;

font-weight:700;

background:
linear-gradient(
90deg,
#2563EB,
#7C3AED
);

color:white;

transition:
all 0.3s ease;
}

.stButton button:hover{

transform:
translateY(-3px);

box-shadow:
0px 15px 30px rgba(59,130,246,0.35);
}

/* ============================================================
   TABS
============================================================ */

.stTabs [role="tab"]{

font-size:16px;

font-weight:600;

color:#CBD5E1;
}

.stTabs [aria-selected="true"]{

color:#60A5FA !important;
}

/* ============================================================
   DATAFRAME
============================================================ */

[data-testid="stDataFrame"]{

border:
1px solid #1E293B;

border-radius:18px;
}

/* ============================================================
   FOOTER BUTTONS
============================================================ */

.footer-btn{

display:inline-block;

padding:12px 28px;

border-radius:30px;

font-weight:700;

text-decoration:none !important;

transition:
all 0.3s ease;

margin:5px;
}

.footer-btn:hover{

transform:
translateY(-4px);

box-shadow:
0px 10px 25px rgba(59,130,246,0.35);
}

/* ============================================================
   GLASS CARDS
============================================================ */

.glass-card{

background:
rgba(17,24,39,0.75);

backdrop-filter:
blur(15px);

padding:25px;

border-radius:22px;

border:
1px solid rgba(255,255,255,0.08);

margin-bottom:20px;
}

/* ============================================================
   RISK BADGES
============================================================ */

.high-risk{

background:#DC2626;

color:white;

padding:8px 18px;

border-radius:20px;

font-weight:700;
}

.medium-risk{

background:#F59E0B;

color:white;

padding:8px 18px;

border-radius:20px;

font-weight:700;
}

.low-risk{

background:#16A34A;

color:white;

padding:8px 18px;

border-radius:20px;

font-weight:700;
}

/* ============================================================
   DOWNLOAD BUTTON
============================================================ */

.download-box{

background:
linear-gradient(
135deg,
#2563EB,
#7C3AED
);

padding:20px;

border-radius:20px;

text-align:center;

color:white;
}

/* ============================================================
   ANIMATION
============================================================ */

.metric-card{

animation:
fadeIn 0.8s ease;
}

@keyframes fadeIn{

from{
opacity:0;
transform:translateY(20px);
}

to{
opacity:1;
transform:translateY(0px);
}
}

.github-btn{

background:
linear-gradient(
135deg,
#2563EB,
#1D4ED8
);

color:white !important;
}

.linkedin-btn{

background:
linear-gradient(
135deg,
#0077B5,
#005885
);

color:white !important;
}

/* ============================================================
   FORCE ALL TEXT TO WHITE
============================================================ */

html,
body,
p,
span,
label,
h1,
h2,
h3,
h4,
h5,
h6,
li,
ul,
ol,
a {

    color: white !important;
}

/* Streamlit markdown */

[data-testid="stMarkdownContainer"] {

    color: white !important;
}

/* Metrics */

[data-testid="stMetricLabel"] {

    color: white !important;
}

[data-testid="stMetricValue"] {

    color: white !important;
}

/* Tabs */

.stTabs [role="tab"] {

    color: white !important;
}

/* Dataframe */

[data-testid="stDataFrame"] {

    color: white !important;
}

/* Sidebar */

[data-testid="stSidebar"] * {

    color: white !important;
}

/* Success / Warning / Error boxes */

[data-testid="stAlert"] {

    color: white !important;
}

/* Selectbox labels */

.stSelectbox label {

    color: white !important;
}

/* Slider labels */

.stSlider label {

    color: white !important;
}

/* Number input labels */

.stNumberInput label {

    color: white !important;
}

/* Radio buttons */

.stRadio label {

    color: white !important;
}

/* Checkbox */

.stCheckbox label {

    color: white !important;
}

/* Dropdown popup background */
div[data-baseweb="popover"] {

    background: white !important;
}

/* Dropdown options text */
div[data-baseweb="menu"] * {

    color: black !important;
}

/* Individual option */
li[role="option"] {

    color: black !important;
    background: white !important;
}

/* Hover option */
li[role="option"]:hover {

    background: #E5E7EB !important;
    color: black !important;
}

/* Selected value shown in box */
.stSelectbox div[data-baseweb="select"] span {

    color: white !important;
}

/* ============================================================
   FOOTER BUTTONS
============================================================ */

.footer-btn{
    text-decoration:none !important;
    padding:12px 28px;
    border-radius:30px;
    color:white !important;
    font-weight:600;
    display:inline-block;
    transition:all 0.3s ease;
    margin:5px;
}

.github-btn{
    background:#2563EB;
    border:1px solid #374151;
}

.linkedin-btn{
    background:#2563EB;
    border:1px solid #374151;
}

.footer-btn:hover{
    transform:scale(1.05);
    box-shadow:0 10px 25px rgba(59,130,246,0.35);
}

/* ============================================================
   DOWNLOAD BUTTON
============================================================ */

.stDownloadButton button{

    width:100%;

    height:58px;

    border:none;

    border-radius:16px;

    background:linear-gradient(
        90deg,
        #2563EB,
        #7C3AED
    ) !important;

    color:white !important;

    font-size:18px;

    font-weight:700;

    transition:all 0.3s ease;
}

.stDownloadButton button:hover{

    transform:translateY(-3px);

    box-shadow:
        0px 15px 30px
        rgba(59,130,246,0.35);

    background:linear-gradient(
        90deg,
        #1D4ED8,
        #6D28D9
    ) !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL LOADING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logistic_model = load(
    os.path.join(
        BASE_DIR,
        "models",
        "logistic_model.pkl"
    )
)

rf_model = load(
    os.path.join(
        BASE_DIR,
        "models",
        "rf_model.pkl"
    )
)

xgb_model = load(
    os.path.join(
        BASE_DIR,
        "models",
        "xgb_model.pkl"
    )
)

# ============================================================
# PREPROCESSOR & RF CLASSIFIER
# ============================================================

preprocessor = rf_model.named_steps["preprocessor"]

rf_classifier = rf_model.named_steps["model"]


# ============================================================
# ENTERPRISE SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style='text-align:center;'>

        <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
        width="120">

        <h2 style="
        color:white;
        margin-top:10px;
        ">
        Churn Intelligence
        </h2>

        <p style="
        color:#94A3B8;
        ">
        Enterprise Customer Analytics
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("""
### 🚀 Platform Capabilities

✅ Churn Prediction

✅ Explainable AI (SHAP)

✅ Customer Risk Analysis

✅ Business Recommendations

✅ Model Benchmarking

✅ Executive Dashboard

✅ Retention Intelligence

✅ Real-Time Insights
""")

    st.markdown("---")

    st.markdown("""
### 🤖 Machine Learning Models

• Logistic Regression

• Random Forest

• XGBoost

### 📊 Performance

ROC-AUC: 86%

F1 Score: 65%

Features: 19

Explainability: SHAP
""")

    st.markdown("---")

    st.success(
        "Production Ready ML Solution"
    )

# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
"""
<div class="hero-card">

<div class="hero-title">

📊 Customer Churn Intelligence

</div>

<div class="hero-subtitle">

Predict customer churn, identify retention risks,
and drive proactive business decisions using
Machine Learning, Explainable AI and Business Analytics.

</div>

</div>
""",
unsafe_allow_html=True
)

# ============================================================
# EXECUTIVE KPI DASHBOARD
# ============================================================

st.markdown(
"""
<div class="section-title">

Executive Dashboard

</div>
""",
unsafe_allow_html=True
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:

    st.markdown(
    """
    <div class="metric-card">

    <div class="metric-label">

    Active Models

    </div>

    <div class="metric-value">

    3

    </div>

    </div>
    """,
    unsafe_allow_html=True
    )

with kpi2:

    st.markdown(
    """
    <div class="metric-card">

    <div class="metric-label">

    ROC-AUC Score

    </div>

    <div class="metric-value">

    86%

    </div>

    </div>
    """,
    unsafe_allow_html=True
    )

with kpi3:

    st.markdown(
    """
    <div class="metric-card">

    <div class="metric-label">

    Features

    </div>

    <div class="metric-value">

    19

    </div>

    </div>
    """,
    unsafe_allow_html=True
    )

with kpi4:

    st.markdown(
    """
    <div class="metric-card">

    <div class="metric-label">

    Explainability

    </div>

    <div class="metric-value">

    SHAP

    </div>

    </div>
    """,
    unsafe_allow_html=True
    )

st.write("")

# ============================================================
# BUSINESS OVERVIEW CARDS
# ============================================================

overview1, overview2, overview3 = st.columns(3)

with overview1:

    st.info(
        """
### 📈 Customer Retention

Identify customers likely to leave
before churn occurs and improve
retention strategy.
"""
    )

with overview2:

    st.info(
        """
### 🤖 Explainable AI

Understand prediction decisions
through SHAP explainability and
feature importance analysis.
"""
    )

with overview3:

    st.info(
        """
### 🎯 Business Impact

Drive revenue growth through
proactive customer engagement
and retention campaigns.
"""
    )

st.write("")

# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "🔮 Prediction Center",
        "📈 Analytics Dashboard"
    ]
)

# ============================================================
# ENTERPRISE PREDICTION CENTER
# ============================================================

with tab1:

    st.markdown("""
    <div class="section-title">
    🔮 Customer Churn Prediction Center
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Predict customer churn probability using
    Machine Learning and Explainable AI.
    """)
    
    st.write("")

    # ========================================================
    # MODEL SELECTION CARD
    # ========================================================

    st.markdown("""
    ### 🤖 Prediction Model
    """)

    model_choice = st.selectbox(
        "Choose Machine Learning Model",
        [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ]
    )

    st.write("")

    # ========================================================
    # CUSTOMER PROFILE
    # ========================================================

    st.markdown("""
    ### 👤 Customer Profile
    """)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    with c2:
        senior = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

    with c3:
        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with c4:
        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    st.write("")

    # ========================================================
    # SUBSCRIPTION DETAILS
    # ========================================================

    st.markdown("""
    ### 📅 Subscription Information
    """)

    col1, col2 = st.columns(2)

    with col1:

        tenure = st.slider(
            "Customer Tenure (Months)",
            0,
            72,
            12
        )

        contract = st.selectbox(
            "Contract Type",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

    with col2:

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=500.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=1000.0
        )

    st.write("")

    # ========================================================
    # TELECOM SERVICES
    # ========================================================

    st.markdown("""
    ### 📞 Telecom Services
    """)

    c1, c2, c3 = st.columns(3)

    with c1:

        phoneservice = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiplelines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

    with c2:

        internet = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        onlinesecurity = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with c3:

        onlinebackup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        techsupport = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    st.write("")

    # ========================================================
    # ENTERTAINMENT SERVICES
    # ========================================================

    st.markdown("""
    ### 🎬 Entertainment Services
    """)

    c1, c2, c3 = st.columns(3)

    with c1:

        deviceprotection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with c2:

        streamingtv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with c3:

        streamingmovies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    st.write("")

    # ========================================================
    # INPUT DATAFRAME
    # ========================================================

    data = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[senior],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phoneservice],
        "MultipleLines":[multiplelines],
        "InternetService":[internet],
        "OnlineSecurity":[onlinesecurity],
        "OnlineBackup":[onlinebackup],
        "DeviceProtection":[deviceprotection],
        "TechSupport":[techsupport],
        "StreamingTV":[streamingtv],
        "StreamingMovies":[streamingmovies],
        "Contract":[contract],
        "PaperlessBilling":[paperless],
        "PaymentMethod":[payment],
        "MonthlyCharges":[monthly],
        "TotalCharges":[total]

    })

    # ========================================================
    # MODEL LOGIC
    # ========================================================

    if model_choice == "Logistic Regression":
        model = logistic_model

    elif model_choice == "Random Forest":
        model = rf_model

    else:
        model = xgb_model

    st.write("")

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🚀 Generate Churn Intelligence Report"
    )

    if predict_button:

        prob = model.predict_proba(data)[0][1]

        prediction = model.predict(data)[0]

        st.session_state["prob"] = prob
        st.session_state["prediction"] = prediction
        st.session_state["customer_data"] = data
        

# ========================================================
# EXECUTIVE RISK DASHBOARD
# ========================================================

if predict_button:

    prob = model.predict_proba(data)[0][1]

    prediction = model.predict(data)[0]

    st.write("")
    st.markdown("## 📊 Executive Churn Intelligence Report")
    

    # ====================================================
    # EXECUTIVE KPI CARDS
    # ====================================================

    k1, k2, k3 = st.columns(3)

    with k1:

        st.metric(
            "Churn Probability",
            f"{prob*100:.2f}%"
        )

    with k2:

        st.metric(
            "Prediction",
            "Churn" if prediction == 1 else "No Churn"
        )

    with k3:

        if prob >= 0.60:

            risk_level = "High Risk"

        elif prob >= 0.30:

            risk_level = "Medium Risk"

        else:

            risk_level = "Low Risk"

        st.metric(
            "Risk Category",
            risk_level
        )

    st.write("")

    # ====================================================
    # GAUGE CHART
    # ====================================================

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=prob * 100,

            title={
                "text":"Customer Risk Score"
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":"#2563EB"
                },

                "steps":[

                    {
                        "range":[0,30],
                        "color":"#22C55E"
                    },

                    {
                        "range":[30,60],
                        "color":"#F59E0B"
                    },

                    {
                        "range":[60,100],
                        "color":"#EF4444"
                    }

                ]

            }

        )

    )

    gauge.update_layout(

        paper_bgcolor="#050816",

        font_color="white",

        height=450
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    # ====================================================
    # CUSTOMER STATUS ALERT
    # ====================================================

    if prob >= 0.60:

        st.error(
            """
🔴 HIGH RISK CUSTOMER

Immediate retention intervention recommended.
Customer is highly likely to churn.
"""
        )

    elif prob >= 0.30:

        st.warning(
            """
🟡 MEDIUM RISK CUSTOMER

Customer engagement monitoring recommended.
"""
        )

    else:

        st.success(
            """
🟢 LOW RISK CUSTOMER

Customer likely to remain active.
"""
        )

    # ====================================================
    # AI RECOMMENDATIONS
    # ====================================================

    st.write("")
    st.markdown("## 🤖 AI Retention Recommendations")

    recommendations = []

    if contract == "Month-to-month":

        recommendations.append(
            "Offer annual contract discounts."
        )

    if monthly > 80:

        recommendations.append(
            "Provide pricing incentives."
        )

    if techsupport == "No":

        recommendations.append(
            "Promote Tech Support package."
        )

    if onlinesecurity == "No":

        recommendations.append(
            "Upsell Online Security service."
        )

    if tenure < 12:

        recommendations.append(
            "Launch new customer engagement campaign."
        )

    if len(recommendations) == 0:

        recommendations.append(
            "Customer currently shows healthy retention profile."
        )

    for rec in recommendations:

        st.info(
            f"✅ {rec}"
        )

    # ====================================================
    # CUSTOMER PROFILE SUMMARY
    # ====================================================

    st.write("")
    st.markdown("## 👤 Customer Summary")

    summary_df = pd.DataFrame({

        "Attribute":[

            "Gender",
            "Tenure",
            "Contract",
            "Internet",
            "Monthly Charges",
            "Total Charges"

        ],

        "Value":[

            gender,
            tenure,
            contract,
            internet,
            monthly,
            total

        ]

    })

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    # ====================================================
    # SHAP EXPLAINABILITY
    # ====================================================

    st.write("")
    st.markdown("## 🧠 Explainable AI Analysis")

    try:

        X_transformed = (
            preprocessor.transform(data)
        )

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        X_df = pd.DataFrame(
            X_transformed,
            columns=feature_names
        )

        explainer = shap.Explainer(
            rf_classifier
        )

        shap_values = explainer(
            X_df
        )

        fig = plt.figure(
            figsize=(12,7)
        )

        shap.plots.waterfall(
            shap_values[0,:,1],
            show=False
        )

        st.pyplot(fig)

    except Exception as e:

        st.warning(
            f"SHAP Visualization Error: {e}"
        )

    # ====================================================
    # EXECUTIVE DECISION SUPPORT
    # ====================================================

    st.write("")
    st.markdown("## 🎯 Executive Decision Support")
    
    st.write("")
    st.markdown("""
                <div class="download-box">
                <h3>
                📄 Executive Churn Report
                </h3>
                Download customer risk assessment results.
                </div>
                """,
                unsafe_allow_html=True)
    report_df = pd.DataFrame({
        "Metric":[
            "Prediction",
            "Probability",
            "Risk Level",
            "Model"
            ],

    "Value":[
        "Churn" if prediction == 1 else "No Churn",
        f"{prob*100:.2f}%",
        risk_level,
        model_choice
    ]
    })
    csv = report_df.to_csv(index=False)
    st.download_button(
        "⬇ Download Report",
        csv,
        file_name="customer_churn_report.csv",
        mime="text/csv"
        )
    if prob >= 0.60:
        st.markdown("""
### Recommended Action Plan

1. Immediate customer outreach

2. Offer loyalty incentives

3. Provide contract upgrade options

4. Bundle premium services

5. Escalate to retention team
""")
    elif prob >= 0.30:

        st.markdown("""
### Recommended Action Plan

1. Monitor customer activity

2. Promote value-added services

3. Offer personalized discounts

4. Increase engagement campaigns
""")

    else:

        st.markdown("""
### Recommended Action Plan

1. Maintain service quality

2. Continue loyalty initiatives

3. Upsell premium services

4. Monitor customer satisfaction
""")
        
        

# ============================================================
# ENTERPRISE ANALYTICS DASHBOARD
# ============================================================

with tab2:

    st.markdown("""
    <div class="section-title">
    📈 Executive Analytics Dashboard
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Advanced model analytics, explainability, feature importance and business intelligence."
    )

    # ========================================================
    # MODEL PERFORMANCE DASHBOARD
    # ========================================================

    st.markdown("## 🏆 Model Benchmark Dashboard")

    performance_df = pd.DataFrame({

        "Model":[
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "ROC AUC":[
            0.86,
            0.85,
            0.85
        ],

        "F1 Score":[
            0.64,
            0.65,
            0.63
        ],

        "Precision":[
            0.52,
            0.56,
            0.55
        ],

        "Recall":[
            0.84,
            0.78,
            0.75
        ]

    })

    k1,k2,k3,k4 = st.columns(4)

    with k1:
        st.metric(
            "Best ROC-AUC",
            "0.86"
        )

    with k2:
        st.metric(
            "Best F1 Score",
            "0.65"
        )

    with k3:
        st.metric(
            "Models Compared",
            "3"
        )

    with k4:
        st.metric(
            "Features",
            "19"
        )

    st.write("")

    fig = px.bar(

        performance_df,

        x="Model",

        y="ROC AUC",

        color="Model",

        title="Model ROC-AUC Comparison"
    )

    fig.update_layout(

        template="plotly_dark",

        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # MODEL COMPARISON TABLE
    # ========================================================

    st.markdown(
        "## 📊 Model Performance Matrix"
    )

    st.dataframe(
        performance_df,
        use_container_width=True
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        "## 🎯 Feature Importance Analysis"
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    feature_importance = (
        rf_classifier.feature_importances_
    )

    feature_df = pd.DataFrame({

        "Feature":feature_names,

        "Importance":feature_importance

    })

    feature_df = (
        feature_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    top_features = feature_df.head(15)

    fig = px.bar(

        top_features,

        x="Importance",

        y="Feature",

        orientation="h",

        title="Top 15 Drivers of Customer Churn",

        color="Importance"
    )

    fig.update_layout(

        template="plotly_dark",

        height=700
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # SYNTHETIC ANALYTICS DATA
    # ========================================================

    st.markdown(
        "## 🔬 SHAP Explainability Dashboard"
    )

    sample_data = pd.DataFrame({

        "gender":
        np.random.choice(
            ["Male","Female"],
            250
        ),

        "SeniorCitizen":
        np.random.choice(
            [0,1],
            250
        ),

        "Partner":
        np.random.choice(
            ["Yes","No"],
            250
        ),

        "Dependents":
        np.random.choice(
            ["Yes","No"],
            250
        ),

        "tenure":
        np.random.randint(
            0,
            72,
            250
        ),

        "PhoneService":
        np.random.choice(
            ["Yes","No"],
            250
        ),

        "MultipleLines":
        np.random.choice(
            [
                "Yes",
                "No",
                "No phone service"
            ],
            250
        ),

        "InternetService":
        np.random.choice(
            [
                "DSL",
                "Fiber optic",
                "No"
            ],
            250
        ),

        "OnlineSecurity":
        np.random.choice(
            [
                "Yes",
                "No",
                "No internet service"
            ],
            250
        ),

        "OnlineBackup":
        np.random.choice(
            [
                "Yes",
                "No",
                "No internet service"
            ],
            250
        ),

        "DeviceProtection":
        np.random.choice(
            [
                "Yes",
                "No",
                "No internet service"
            ],
            250
        ),

        "TechSupport":
        np.random.choice(
            [
                "Yes",
                "No",
                "No internet service"
            ],
            250
        ),

        "StreamingTV":
        np.random.choice(
            [
                "Yes",
                "No",
                "No internet service"
            ],
            250
        ),

        "StreamingMovies":
        np.random.choice(
            [
                "Yes",
                "No",
                "No internet service"
            ],
            250
        ),

        "Contract":
        np.random.choice(
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            250
        ),

        "PaperlessBilling":
        np.random.choice(
            [
                "Yes",
                "No"
            ],
            250
        ),

        "PaymentMethod":
        np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            250
        ),

        "MonthlyCharges":
        np.random.uniform(
            20,
            120,
            250
        ),

        "TotalCharges":
        np.random.uniform(
            100,
            5000,
            250
        )

    })

    X_sample = (
        preprocessor.transform(
            sample_data
        )
    )

    X_sample_df = pd.DataFrame(

        X_sample,

        columns=feature_names
    )

    explainer = shap.Explainer(
        rf_classifier
    )

    shap_values = explainer(
        X_sample_df
    )

    # ========================================================
    # SHAP SUMMARY
    # ========================================================

    st.markdown(
        "### SHAP Summary Plot"
    )

    fig = plt.figure(
        figsize=(12,8)
    )

    shap.plots.beeswarm(

        shap_values[:,:,1],

        max_display=15,

        show=False

    )

    st.pyplot(fig)

    # ========================================================
    # SHAP BAR
    # ========================================================

    st.markdown(
        "### SHAP Feature Impact"
    )

    fig = plt.figure(
        figsize=(12,8)
    )

    shap.plots.bar(

        shap_values[:,:,1],

        max_display=15,

        show=False

    )

    st.pyplot(fig)

    # ========================================================
# BUSINESS INSIGHTS
# ========================================================

st.markdown(
    "## 💼 Executive Business Insights"
)

insight1, insight2 = st.columns(2)

with insight1:

    st.success("""
### 📉 Key Churn Risk Factors

#### Customer Lifecycle
• Customers with shorter tenure exhibit significantly higher churn rates.

#### Contract Structure
• Month-to-month subscriptions represent the highest-risk customer segment.

#### Pricing Sensitivity
• Customers with elevated monthly charges are more likely to discontinue services.

#### Service Portfolio
• Customers lacking value-added services demonstrate lower retention levels.

#### Internet Service Type
• Fiber optic customers show comparatively higher churn behavior than DSL subscribers.
""")

with insight2:

    st.info("""
### 🎯 Strategic Retention Recommendations

#### Increase Contract Commitment
• Incentivize migration from month-to-month plans to annual contracts.

#### Enhance Customer Engagement
• Launch targeted onboarding and loyalty programs for new customers.

#### Expand Service Adoption
• Promote Tech Support, Online Security, and Device Protection bundles.

#### Retention Campaigns
• Identify high-risk customers and provide personalized retention offers.

#### Revenue Protection
• Develop pricing strategies for customers with high monthly charges to reduce churn risk.
""")

# ========================================================
# EXECUTIVE SUMMARY
# ========================================================

st.markdown(
    "## 📋 Executive Summary"
)

st.markdown("""

### Overview

The Customer Churn Intelligence Platform leverages Machine Learning and Explainable AI to identify customers at risk of churn and provide actionable retention strategies.

---

### Executive Findings

#### 1️⃣ Customer Tenure is the Strongest Predictor
Customers in the early stages of their subscription lifecycle are substantially more likely to leave compared to long-term subscribers.

#### 2️⃣ Contract Type Directly Impacts Retention
Month-to-month contracts generate the highest churn probability, while long-term agreements significantly improve retention outcomes.

#### 3️⃣ Monthly Charges Influence Customer Behavior
Customers with higher recurring charges exhibit greater price sensitivity and increased churn risk.

#### 4️⃣ Service Adoption Drives Loyalty
Customers utilizing value-added services such as Online Security, Tech Support, and Device Protection demonstrate stronger retention rates.

#### 5️⃣ Internet Service Segmentation Matters
Fiber optic customers represent a key segment for retention initiatives due to elevated churn patterns.

---

### Business Impact

✅ Reduce customer acquisition costs through improved retention.

✅ Identify high-risk customers before churn occurs.

✅ Increase Customer Lifetime Value (CLV).

✅ Enable proactive, data-driven retention campaigns.

✅ Support executive decision-making through Explainable AI insights.

---

### Strategic Outcome

By integrating predictive analytics, explainability, and retention intelligence, organizations can transform customer churn management from a reactive process into a proactive growth strategy.

""")
    
# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
text-align:center;
padding:30px;
">

<h3 style="color:white;">
📊 Customer Churn Intelligence Platform
</h3>

<p style="
color:#CBD5E1;
margin-bottom:20px;
">
AI-Powered Customer Retention Analytics
</p>

<table style="margin:auto;">
<tr>

<td style="padding-right:15px;">

<a href="https://github.com/Gaurav2025G"
target="_blank"
class="footer-btn github-btn">
🐙 GitHub
</a>

</td>

<td>

<a href="https://www.linkedin.com/in/gaurav-raipurkardataanalyst/"
target="_blank"
class="footer-btn linkedin-btn">
💼 LinkedIn
</a>

</td>

</tr>
</table>

<br>

<p style="
color:#94A3B8;
">

© 2026 Gaurav Raipurkar

<br><br>

Built with ❤️ using

<br>

Streamlit • Scikit-Learn • XGBoost • SHAP • Plotly

</p>

</div>
""", unsafe_allow_html=True)