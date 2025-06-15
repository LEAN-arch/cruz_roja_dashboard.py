# cruz_roja_dashboard_ai_enhanced.py
# The definitive, AI-enhanced dashboard based on the 2013 Cruz Roja Tijuana Situational Diagnosis.
# V6 - IndexError Fix

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from prophet import Prophet
from datetime import timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Cruz Roja Tijuana - AI Command Center 2013",
    page_icon="⚕️",
    layout="wide",
)

# --- Data Loading & Simulation ---
@st.cache_data
def load_and_simulate_data():
    """
    Loads all data points from the 2013 report and simulates a granular daily
    time-series dataset for advanced analytics.
    """
    # Load original aggregated data
    original_data = {
        "population_projection": pd.DataFrame({"Year": [2005, 2010, 2015, 2020, 2030], "Population": [1410687, 1682160, 2005885, 2391915, 3401489]}),
        "marginalization_data": pd.DataFrame([{"Level": "Very High", "Percentage": 1.0}, {"Level": "High", "Percentage": 15.0}, {"Level": "Medium", "Percentage": 44.0}, {"Level": "Low", "Percentage": 24.0}, {"Level": "Very Low", "Percentage": 14.0}, {"Level": "N/A", "Percentage": 2.0}]),
        "funding_data": pd.DataFrame([{'Source': 'Donations & Projects', 'Percentage': 53.2},{'Source': 'General Services', 'Percentage': 25.9},{'Source': 'Fundraising', 'Percentage': 12.6},{'Source': 'Training Center', 'Percentage': 7.5},{'Source': 'Other', 'Percentage': 0.8}]),
        "uninsured_patients_pct": 89.4,
        "monthly_operating_costs": pd.DataFrame({'Month': ['Oct','Nov','Dec','Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep'], 'Medical': [3482131,3473847,3667978,2775683,2564990,2778673,3177997,2696104,2502781,2912605,3275804,3155497], 'Paramedic': [2127730,2651096,2076126,1996603,2039858,1862567,2301656,1914002,1952308,2210602,2321977,1936905]}),
        "weekly_costs": pd.DataFrame({
            "Department": ["Medical (Normal)", "Medical (Overtime)", "Paramedic (Normal)", "Paramedic (Overtime)"],
            "Weekly Cost (MXN)": [219139, 17081, 183169, 53914],
            "Category": ["Medical", "Medical", "Paramedic", "Paramedic"],
            "Type": ["Normal", "Overtime", "Normal", "Overtime"]
        }),
        "cost_per_patient_type": pd.DataFrame([{'Type': 'Deceased on Arrival', 'Cost': 792.77}, {'Type': 'Minor', 'Cost': 814.80}, {'Type': 'Non-Critical', 'Cost': 840.62}, {'Type': 'Critical (Trauma)', 'Cost': 1113.81}, {'Type': 'Critical (Medical)', 'Cost': 1164.57}]),
        "cost_per_patient_area": pd.DataFrame([{'Area': 'ER (Group I)', 'Cost': 902.04}, {'Area': 'ER (Group II)', 'Cost': 1031.31}, {'Area': 'ER (Group III)', 'Cost': 1434.81}, {'Area': 'Hospital', 'Cost': 1072.64}, {'Area': 'Pediatrics', 'Cost': 967.92}, {'Area': 'ICU', 'Cost': 2141.39}]),
        "c4_call_summary": pd.DataFrame([{"Category": "Real Calls", "Value": 21.8}, {"Category": "Prank Calls", "Value": 10.9}, {"Category": "Incomplete", "Value": 56.7}, {"Category": "Citizen Info", "Value": 10.6}]),
        "data_integrity_gap": {'values': [42264, 40809, 31409], 'stages': ["C-4 Calls Dispatched", "Services Logged (Bitácora)", "Patient Reports (FRAP)"]},
        "patient_acuity_prehospital": pd.DataFrame([{"Category": "Minor", "Percentage": 67.3}, {"Category": "Non-Critical", "Percentage": 19.5}, {"Category": "Critical", "Percentage": 3.3}]),
        "response_time_by_base": pd.DataFrame({"Base": ["Base 10", "Base 8", "Base 4", "Base 11", "Base 58", "Base 0"], "Avg Response Time (min)": [17.17, 15.17, 14.85, 14.35, 12.90, 12.22]}),
        "hospital_kpis": {"er_patients_annual": 33010, "avg_er_wait_time": "23:27", "avg_bed_occupancy_er": 45.4, "er_compliance_score": 87, "er_specialized_compliance": 95},
        "certification_data": {'Paramedics_BLS': 80, 'Paramedics_ACLS': 67, 'Doctors_ATLS': 13, 'Doctors_ACLS': 34, 'Nurses_BLS': 31, 'Nurses_ACLS': 16},
        "disaster_readiness": {"Hospital Safety Index": "C (Urgent Action Required)"},
        "staff_sentiment": {'strengths': {'Medical': 'Services Offered (58%)', 'Paramedic': 'Services Offered (59%)'},'opportunities': {'Medical': 'Training (42%)', 'Paramedic': 'Salary (45%)'},'motivation': {'Medical': 'Salary (58%)', 'Paramedic': 'Salary (69%)'}},
        "patient_sentiment": {'satisfaction_score': 8.6, 'main_reason': 'Accident (50%)', 'improvement_area': 'Information & Courtesy (26% each)'}
    }

    # --- Simulated Daily Data for AI Modules ---
    er_visits_monthly = [2829, 2548, 2729, 2780, 2306, 2775, 2744, 2774, 2754, 2934, 2985, 2852]
    dates = pd.date_range(start="2012-10-01", end="2013-09-30")
    daily_visits = []
    for i, month_total in enumerate(er_visits_monthly):
        month_start = pd.to_datetime("2012-10-01") + pd.DateOffset(months=i)
        days_in_month = month_start.days_in_month
        daily_avg = month_total / days_in_month if days_in_month > 0 else 0
        daily_counts = np.random.normal(loc=daily_avg, scale=daily_avg * 0.2, size=days_in_month).astype(int)
        daily_visits.extend(np.maximum(0, daily_counts))

    daily_df = pd.DataFrame({'date': dates[:len(daily_visits)], 'visits': daily_visits})
    diagnoses = ['Trauma', 'Medical Illness', 'Cardiac', 'Gyn.', 'Pediatric', 'Minor Injury']
    daily_df['diagnosis'] = np.random.choice(diagnoses, len(daily_df), p=[0.30, 0.40, 0.05, 0.05, 0.05, 0.15])
    daily_df['wait_time_min'] = np.maximum(5, np.random.normal(23, 8, len(daily_df)))
    daily_df['ai_risk_score'] = np.random.uniform(10, 95, len(daily_df))
    
    return original_data, daily_df

# --- AI & Statistical Functions ---
@st.cache_data
def get_prophet_forecast(_df, days_to_forecast=30):
    df_prophet = _df.rename(columns={'date': 'ds', 'visits': 'y'})
    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=days_to_forecast)
    forecast = model.predict(future)
    return forecast

def predict_resource_hotspots(df: pd.DataFrame):
    if df.empty: return pd.DataFrame()
    last_7_days = df[df['date'] >= df['date'].max() - timedelta(days=7)]
    if last_7_days.empty: return pd.DataFrame()
    weekly_proportions = last_7_days['diagnosis'].value_counts(normalize=True)
    total_predicted_visits = int(last_7_days['visits'].sum() * np.random.uniform(0.9, 1.1))
    predicted_cases = (weekly_proportions * total_predicted_visits).round().astype(int)
    resource_map = {"Trauma": "Splints/Bandages", "Medical Illness": "IV Kits", "Cardiac": "EKG Electrodes", "Gyn.": "OB Kits", "Pediatric": "Pediatric Supplies", "Minor Injury": "Basic First Aid"}
    hotspots_df = predicted_cases.reset_index(); hotspots_df.columns = ['diagnosis', 'predicted_cases']
    hotspots_df['resource_needed'] = hotspots_df['diagnosis'].map(resource_map)
    return hotspots_df.sort_values('predicted_cases', ascending=False)

# --- Load Data ---
original_data, daily_df = load_and_simulate_data()

# --- Dashboard UI ---
st.image("https://cruzrojatijuana.org.mx/wp-content/uploads/2022/10/logo.png", width=250)
st.title("AI-Enhanced Strategic Command Center: Cruz Roja Tijuana")
st.markdown("_Leveraging 2013 baseline data with predictive analytics for forward-looking decision making._")
st.divider()

# --- Date Filter ---
min_date = daily_df['date'].min().date()
max_date = daily_df['date'].max().date()
start_date, end_date = st.sidebar.date_input(
    "Select Date Range for Analysis:",
    value=(max_date - timedelta(days=89), max_date),
    min_value=min_date, max_value=max_date
)
period_df = daily_df[(daily_df['date'].dt.date >= start_date) & (daily_df['date'].dt.date <= end_date)]

# --- Main Tabs ---
# SME FIX: The number of tabs created now matches the number of `with tabs[]` blocks.
tab_list = [
    "📈 **Executive Summary**", 
    "🔮 **AI & Predictive Analytics**",
    "💰 **Financial Health**", 
    "🚑 **Prehospital Operations**", 
    "🏥 **Hospital Services**",
    "👥 **HR & Sentiment**",
    "📋 **Recommendations**"
]
tabs = st.tabs(tab_list)

# ============================ TAB 0: EXECUTIVE SUMMARY ============================
with tabs[0]:
    st.header("Top-Level Findings & Key Risks from 2013 Report")
    st.info("""
    - **Financial Vulnerability:** High dependence on donations (53%) and significant operational data gaps pose financial risks.
    - **Operational Mismatch:** A skilled dispatch system sends advanced life support units to a majority (67%) of minor incidents.
    - **Systemic Risk:** The main hospital has a critical 'C' safety rating, making it vulnerable in a major disaster.
    - **Skills Gap:** There are significant gaps in essential trauma and life-support certifications (e.g., only 13% of doctors ATLS certified).
    """, icon="❗")
    st.divider()
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Data Integrity & Operational Leakage")
        fig_gap = go.Figure(go.Funnel(y=original_data['data_integrity_gap']['stages'], x=original_data['data_integrity_gap']['values'], textposition="inside", textinfo="value+percent previous"))
        fig_gap.update_layout(title_text="23% of Incidents Lack Patient Reports", title_x=0.5, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_gap, use_container_width=True)
    with col2:
        st.subheader("Key Advanced Certification Gaps (%)")
        certs = original_data['certification_data']
        cert_data_df = pd.DataFrame([{'Role': 'Doctors (ATLS)', 'Percentage': certs['Doctors_ATLS']}, {'Role': 'Paramedics (ACLS)', 'Percentage': certs['Paramedics_ACLS']}, {'Role': 'Nurses (ACLS)', 'Percentage': certs['Nurses_ACLS']}])
        fig_certs = px.bar(cert_data_df, x='Role', y='Percentage', title="Advanced Certifications are Below Target", text='Percentage')
        fig_certs.update_traces(texttemplate='%{text}%', textposition='outside'); fig_certs.update_yaxes(range=[0,100])
        st.plotly_chart(fig_certs, use_container_width=True)

# ============================ TAB 1: AI & PREDICTIVE ANALYTICS ============================
with tabs[1]:
    st.header("🔮 AI & Predictive Analytics Hub")
    st.markdown("Use predictive forecasts and inferential statistics to guide strategic decisions.")
    
    st.subheader("Interactive Capacity & Staffing Forecast")
    col1, col2 = st.columns([2, 1])
    with col1:
        forecast_days = st.slider("Days to Forecast Ahead:", 7, 90, 30, key="forecast_days")
        forecast_df = get_prophet_forecast(daily_df, forecast_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,123,255,0.2)', name='Uncertainty'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,123,255,0.2)'))
        fig.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['visits'], mode='markers', name='Historical Data', marker=dict(color='black', opacity=0.6)))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast', line=dict(color='blue', width=3)))
        fig.update_layout(title="Forecasted Patient Demand", xaxis_title="Date", yaxis_title="Daily ER Visits")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Staffing Scenario")
        available_fte = st.slider("Number of Available Clinicians (FTE):", 1, 20, 10)
        avg_consult_time_min = 20; staff_hours_per_day = 8
        future_forecast = forecast_df[forecast_df['ds'] > daily_df['date'].max()]
        required_fte = (future_forecast['yhat'].sum() * avg_consult_time_min) / 60 / (staff_hours_per_day * forecast_days) if forecast_days > 0 else 0
        utilization_pct = (required_fte / available_fte * 100) if available_fte > 0 else 0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=utilization_pct, title={'text': "Projected Staff Utilization"}, number={'suffix': '%'},
            delta={'reference': 100}, gauge={'axis': {'range': [None, 120]}, 'steps': [{'range': [0, 85], 'color': 'lightgreen'}, {'range': [85, 100], 'color': 'orange'}],'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}}))
        st.plotly_chart(fig_gauge, use_container_width=True)
        if utilization_pct > 100: st.error(f"**Over-Capacity Alert:** Predicted workload requires {required_fte - available_fte:.1f} more FTEs.", icon="🔴")
        else: st.success(f"**Healthy Capacity:** Staffing levels are adequate.")

    st.divider()
    st.subheader("System Dynamics: Correlation Analysis")
    st.markdown("Uncover hidden relationships between operational factors. *Note: Correlation does not imply causation.*")
    corr_matrix = period_df[['visits', 'wait_time_min', 'ai_risk_score']].corr()
    st.plotly_chart(px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', range_color=[-1, 1], title="Correlation Between Daily Metrics for Selected Period"), use_container_width=True)

# ============================ TAB 2: FINANCIAL HEALTH ============================
with tabs[2]:
    st.header("💰 Financial Health Analysis")
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Uninsured Patients Served", f"{original_data['uninsured_patients_pct']}%", help="Source: Fig 5, p. 31")
        st.subheader("Funding Sources")
        st.plotly_chart(px.pie(original_data['funding_data'], names='Source', values='Percentage', hole=0.4, title="53% of Funding from Donations"), use_container_width=True)
        st.caption("Source: Table 2, p. 30")
    with col2:
        st.subheader("Monthly Operating Costs (MXN)")
        fig = px.bar(original_data['monthly_operating_costs'], x='Month', y=['Medical', 'Paramedic'], title="Operational Costs Fluctuate Monthly")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Source: Table 3, p. 30")
    st.divider()
    st.subheader("Cost Breakdowns")
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(px.bar(original_data['cost_per_patient_type'], x='Cost', y='Type', orientation='h', title="Cost per Patient by Acuity"), use_container_width=True)
        st.caption("Source: Table 18, p. 49")
    with colB:
        st.plotly_chart(px.bar(original_data['cost_per_patient_area'], x='Cost', y='Area', orientation='h', title="Cost per Patient by Hospital Area"), use_container_width=True)
        st.caption("Source: Table 27, p. 65")

# ============================ TAB 3: PREHOSPITAL OPERATIONS ============================
with tabs[3]:
    st.header("🚑 Prehospital Field Operations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("C4 Emergency Call Funnel")
        st.plotly_chart(px.funnel(original_data['c4_call_summary'], x='Value', y='Category', title="Only 22% of 066 Calls are Real Emergencies"), use_container_width=True)
        st.caption("Source: Table 7, p. 36")
    with col2:
        st.subheader("Prehospital Patient Acuity")
        st.plotly_chart(px.pie(original_data['patient_acuity_prehospital'], names='Category', values='Percentage', title="67% of Attended Patients have Minor Issues"), use_container_width=True)
        st.caption("Source: Table 16, p. 47")
    st.divider()
    st.subheader("Response Time by Ambulance Base")
    st.plotly_chart(px.bar(original_data['response_time_by_base'].sort_values("Avg Response Time (min)"), y="Base", x="Avg Response Time (min)", orientation='h', title="Response Times Vary Significantly by Base", text="Avg Response Time (min)").update_traces(texttemplate='%{text:.1f} min', textposition='inside'), use_container_width=True)
    st.caption("Source: Table 17, p. 48")

# ============================ TAB 4: HOSPITAL SERVICES ============================
with tabs[4]:
    st.header("🏥 Hospital Services")
    kpis = original_data['hospital_kpis']
    hosp_cols = st.columns(3)
    hosp_cols[0].metric("Annual ER Patients", f"{kpis['er_patients_annual']:,}")
    hosp_cols[1].metric("Avg. ER Wait Time", kpis['avg_er_wait_time'])
    hosp_cols[2].metric("Avg. ER Bed Occupancy", f"{kpis['avg_bed_occupancy_er']}%")
    st.divider()
    st.subheader("Facility Compliance Scores")
    st.progress(kpis['er_compliance_score'], text=f"ER General Compliance Score: {kpis['er_compliance_score']}%")
    st.progress(kpis['er_specialized_compliance'], text=f"ER Specialized Equipment Compliance: {kpis['er_specialized_compliance']}%")
    st.caption("Source: p. 70")
    
# ============================ TAB 5: HR & SENTIMENT ============================
with tabs[5]:
    st.header("👥 Human Resources & Stakeholder Sentiment")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Staff & Patient Survey Insights")
        st.markdown("##### Staff Sentiment (Source: p. 96-99)")
        st.info(f"**Main Strength:** {original_data['staff_sentiment']['strengths']['Paramedic']}")
        st.warning(f"**Top Improvement Opportunity:** {original_data['staff_sentiment']['opportunities']['Paramedic']} for paramedics.")
        st.error(f"**Primary Motivation Driver:** {original_data['staff_sentiment']['motivation']['Paramedic']} for paramedics.")
    with col2:
        st.markdown("##### Patient Sentiment (Source: p. 103-104)")
        st.info(f"**Overall Satisfaction:** High, with an average rating of **{original_data['patient_sentiment']['satisfaction_score']}/10**.")
        st.warning(f"**Top Improvement Area:** {original_data['patient_sentiment']['improvement_area']}.")
        st.success(f"**Primary Reason for Visit:** **{original_data['patient_sentiment']['main_reason']}**.")
    st.divider()
    st.subheader("Disaster Readiness")
    st.error(f"**Hospital Safety Index: {original_data['disaster_readiness']['Hospital Safety Index']}**", icon="🚨")
    st.caption("Source: p. 84")

# ============================ TAB 6: RECOMMENDATIONS ============================
with tabs[6]:
    st.header("📋 Summary of Report Recommendations")
    st.markdown("A complete list of actionable short and long-term recommendations proposed in the 2013 report.")
    st.subheader("Short-Term Priorities (Implement within 1 Year)")
    with st.expander("Show All Short-Term Recommendations"):
        st.markdown("""
        - **Legislation:** Propose municipal regulations for minimum EMT/paramedic education levels.
        - **Data Integrity & PPE:** Enforce mandatory use of Personal Protective Equipment (PPE) and accurate, complete FRAP documentation for every incident.
        - **Staffing:** Conduct a cost-benefit analysis of overtime vs. hiring new staff.
        - **Triage:** Establish and implement a formal triage system at the hospital.
        - **Training:** Mandate minimum certifications (BLS, ACLS, ATLS/PHTLS) for all clinical roles.
        - **Inventory:** Develop a system of maximums and minimums for supply management.
        - **First Responders:** Establish a system of First Responders including Fire Dept and Police.
        - **Radiocommunication:** Designate a supervisor for radio operators and monitor FRAP completion.
        """)
    st.subheader("Long-Term Strategic Goals (1-3+ Year Horizon)")
    with st.expander("Show All Long-Term Recommendations"):
        st.markdown("""
        - **System Integration:** Form a state-level commission for disaster management that integrates all emergency medical services.
        - **Disaster Funding:** Create mechanisms to mobilize dedicated funds for disaster response readiness.
        - **Professional Development:** Establish a robust continuing education program for all staff.
        - **Hospital Safety:** Implement the "Hospital Seguro" program to address the critical 'C' safety rating.
        - **Community Engagement:** Develop public education programs on proper use of emergency services.
        - **Technology:** Improve systems for real-time information exchange between services.
        - **Research:** Lay the groundwork for a professional prehospital research strategy.
        """)
