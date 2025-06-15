# cruz_roja_dashboard.py
# An interactable, actionable dashboard based on the 2013 Cruz Roja Tijuana Situational Diagnosis.
# Designed by an SME for strategic leadership. (V2 - Obsolete option removed)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Cruz Roja Tijuana - Strategic Dashboard 2013",
    page_icon="⚕️",
    layout="wide",
)

# --- Data Preparation (Extracted from the PDF Report) ---
# This section simulates loading the key data points from the report.

@st.cache_data
def load_data():
    """Loads all data points extracted from the 2013 report."""
    
    # Executive Summary & Financials (p. 30, 31, 36)
    kpi_data = {
        "total_annual_incidents": 42264,
        "prehospital_patients_attended": 31409,
        "avg_response_time": "14:03",
        "total_annual_operating_cost": 61855532,
        "hospital_safety_index": "C (Urgent Action)",
        "uninsured_patients_pct": 89.4,
    }

    funding_data = pd.DataFrame([
        {'Source': 'Donations & Projects', 'Percentage': 53.2},
        {'Source': 'General Services', 'Percentage': 25.9},
        {'Source': 'Fundraising', 'Percentage': 12.6},
        {'Source': 'Training Center', 'Percentage': 7.5},
        {'Source': 'Other', 'Percentage': 0.8}
    ])

    data_integrity_data = {
        'stages': ["C-4 Calls Dispatched", "Services Logged (Bitácora)", "Patient Reports (FRAP)"],
        'values': [42264, 40809, 31409]
    }

    # Operations Data (p. 38, 47, 48)
    dispatch_time_data = pd.DataFrame({
        "Priority": ["Priority 1", "Priority 2", "Priority 3"],
        "Goal (sec)": [180, 210, 240],
        "Actual (sec)": [157, 173, 177] # Converted from mm:ss
    })
    
    response_time_by_base = pd.DataFrame({
        "Base": ["Base 10 (El Refugio)", "Base 8 (Santa Fe)", "Base 4 (El Dorado)", "Base 11 (Playas)", "Base 5 (Otay)", "Base 58 (Cruz Roja)", "Base 0 (Centro)"],
        "Avg Response Time (min)": [17.17, 15.17, 14.85, 14.35, 13.55, 12.90, 12.22]
    })
    
    patient_acuity_data = pd.DataFrame([
        {"Category": "Leve (Minor)", "Percentage": 67.3},
        {"Category": "No Crítico (Non-Critical)", "Percentage": 19.5},
        {"Category": "Crítico (Critical)", "Percentage": 3.3},
        {"Category": "Deceased on Arrival", "Percentage": 3.6},
        {"Category": "No Record", "Percentage": 6.2},
    ])

    # Hospital Data (p. 62, 66)
    hospital_kpis = {
        "er_patients_annual": 33010,
        "avg_er_wait_time": "23:27",
        "avg_bed_occupancy_er": 45.4
    }

    # Human Resources Data (p. 55, 73, 76)
    certification_data = {
        "Paramedics_BLS": 80,
        "Paramedics_ACLS": 67,
        "Paramedics_PHTLS": 22,
        "Doctors_ATLS": 13,
        "Doctors_ACLS": 34,
        "Nurses_BLS": 31,
        "Nurses_ACLS": 16
    }
    
    return kpi_data, funding_data, data_integrity_data, dispatch_time_data, response_time_by_base, patient_acuity_data, hospital_kpis, certification_data

# --- Load all data ---
kpi_data, funding_data, data_integrity_data, dispatch_time_data, response_time_by_base, patient_acuity_data, hospital_kpis, certification_data = load_data()


# --- Dashboard UI ---
# The obsolete st.set_option line has been removed.

# --- Header ---
st.image("https://cruzrojatijuana.org.mx/wp-content/uploads/2022/10/logo.png", width=250)
st.title("Strategic Operations Dashboard: Cruz Roja Tijuana (2013 Analysis)")
st.markdown("_An interactive summary of the 2013 Situational Diagnosis Report for leadership._")
st.divider()

# --- Main Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Summary", 
    "💰 Financial Health", 
    "🚑 Field Operations (Prehospital)", 
    "🏥 Hospital Services & HR"
])


# ==============================================================================
# TAB 1: EXECUTIVE SUMMARY
# ==============================================================================
with tab1:
    st.header("Key Performance Indicators (Annual)")
    
    # --- Top-level KPIs ---
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Incidents (C-4)", f"{kpi_data['total_annual_incidents']:,}")
    kpi_cols[1].metric("Prehospital Patients", f"{kpi_data['prehospital_patients_attended']:,}")
    kpi_cols[2].metric("Avg. Response Time", kpi_data['avg_response_time'])
    kpi_cols[3].metric("Annual Operating Cost", f"${kpi_data['total_annual_operating_cost']:,.0f} MXN")
    kpi_cols[4].metric("Hospital Safety Index", kpi_data['hospital_safety_index'], delta="-CRITICAL-", delta_color="inverse")
    
    st.divider()
    
    # --- Visualizations ---
    viz_cols = st.columns([1, 1], gap="large")
    with viz_cols[0]:
        st.subheader("Funding Sources")
        fig_funding = px.pie(funding_data, names='Source', values='Percentage', 
                             hole=0.4, title="High Dependency on Donations (53%)")
        fig_funding.update_layout(showlegend=False, title_x=0.5, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_funding, use_container_width=True)

    with viz_cols[1]:
        st.subheader("Data Integrity & Operational Leakage")
        fig_gap = go.Figure(go.Funnel(
            y=data_integrity_data['stages'],
            x=data_integrity_data['values'],
            textposition="inside",
            textinfo="value+percent previous"
        ))
        fig_gap.update_layout(title_text="23% of Incidents Lack Patient Reports", title_x=0.5, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_gap, use_container_width=True)
        
    st.divider()
    
    # --- SME Strategic Insights ---
    st.subheader("SME Strategic Insights & Key Risks")
    st.info("""
    - **Financial Risk:** The organization is heavily reliant on donations. A downturn in fundraising could significantly impact operations.
    - **Data/Legal Risk:** The 23% gap between logged services and completed patient reports (FRAPs) represents a major operational and legal liability. It also hinders accurate billing and quality control.
    - **Safety Risk:** A 'C' Hospital Safety Index is a critical vulnerability. In a major disaster (e.g., earthquake), the primary facility may not be operational, crippling the city's emergency response.
    """, icon="⚠️")


# ==============================================================================
# TAB 2: FINANCIAL HEALTH
# ==============================================================================
with tab2:
    st.header("Financial Overview")
    st.metric("Uninsured Patients Served", f"{kpi_data['uninsured_patients_pct']}%")
    st.caption("The vast majority of services are provided to patients without insurance, underscoring the critical social mission but also the financial challenge.")
    
    st.divider()
    
    st.subheader("Note on Financials")
    st.markdown("The 2013 report provides aggregated annual and weekly costs. A live dashboard would feature interactive monthly cost trend charts here. Key insights from the report include:")
    
    cost_cols = st.columns(2)
    with cost_cols[0]:
        st.info("**High Overtime Burden (Paramedics)**")
        st.write("Weekly overtime costs for paramedics (`$54k`) were nearly 30% of their normal salary costs (`$183k`). This indicates significant understaffing, leading to higher operational costs and risk of staff burnout.")
    with cost_cols[1]:
        st.info("**Ambulance Fleet Inefficiency**")
        st.write("The analysis on page 40 showed that some ambulance models (e.g., Mercedes) had a significantly higher cost per service (`$178`) compared to others (e.g., Ford at `$100`). This highlights an opportunity for fleet optimization to reduce maintenance and fuel costs.")


# ==============================================================================
# TAB 3: FIELD OPERATIONS (PREHOSPITAL)
# ==============================================================================
with tab3:
    st.header("Prehospital Operations Analysis")
    
    op_cols = st.columns(2, gap="large")
    with op_cols[0]:
        st.subheader("Dispatch Time Performance")
        fig_dispatch = go.Figure()
        fig_dispatch.add_trace(go.Bar(name='Goal', x=dispatch_time_data['Priority'], y=dispatch_time_data['Goal (sec)'], marker_color='lightgrey'))
        fig_dispatch.add_trace(go.Bar(name='Actual', x=dispatch_time_data['Priority'], y=dispatch_time_data['Actual (sec)'], marker_color='#007BFF'))
        fig_dispatch.update_layout(barmode='group', title_text="Dispatch Times are Within Goals", title_x=0.5, yaxis_title="Time (seconds)")
        st.plotly_chart(fig_dispatch, use_container_width=True)

    with op_cols[1]:
        st.subheader("Patient Acuity Distribution")
        fig_acuity = px.pie(patient_acuity_data, names='Category', values='Percentage', hole=0.4, title="Majority of Calls (67%) are for Minor Issues")
        fig_acuity.update_layout(showlegend=False, title_x=0.5)
        st.plotly_chart(fig_acuity, use_container_width=True)

    st.info("**SME Insight:** While dispatch is efficient, the high volume of minor calls suggests a need for a tiered response system or public education campaign to reserve ambulance resources for true emergencies.", icon="💡")
    
    st.divider()

    st.subheader("Response Time by Ambulance Base")
    st.markdown("Significant variation exists between bases, highlighting opportunities for resource reallocation or base repositioning.")
    fig_response_base = px.bar(response_time_by_base.sort_values("Avg Response Time (min)"),
                               y="Base", x="Avg Response Time (min)",
                               orientation='h',
                               color="Avg Response Time (min)",
                               color_continuous_scale="RdYlGn_r",
                               text="Avg Response Time (min)")
    fig_response_base.update_layout(yaxis_title=None, title_x=0.5, height=400)
    fig_response_base.update_traces(texttemplate='%{text:.1f} min', textposition='inside')
    st.plotly_chart(fig_response_base, use_container_width=True)


# ==============================================================================
# TAB 4: HOSPITAL SERVICES & HR
# ==============================================================================
with tab4:
    st.header("Hospital & Human Resources Snapshot")
    
    hosp_cols = st.columns(3)
    hosp_cols[0].metric("Annual ER Patients", f"{hospital_kpis['er_patients_annual']:,}")
    hosp_cols[1].metric("Avg. ER Wait Time", hospital_kpis['avg_er_wait_time'])
    hosp_cols[2].metric("Avg. ER Bed Occupancy", f"{hospital_kpis['avg_bed_occupancy_er']}%")

    st.divider()
    
    st.subheader("Staff Certification & Skill Gaps")
    st.markdown("There are critical gaps in essential, life-saving certifications across all staff types. This represents a major risk to patient outcomes and a primary target for investment in training.")
    
    cert_cols = st.columns(3)
    
    with cert_cols[0]:
        st.markdown("##### Paramedic Certifications")
        st.progress(certification_data['Paramedics_BLS'], text=f"{certification_data['Paramedics_BLS']}% BLS Certified")
        st.progress(certification_data['Paramedics_ACLS'], text=f"{certification_data['Paramedics_ACLS']}% ACLS Certified")
        st.progress(certification_data['Paramedics_PHTLS'], text=f"{certification_data['Paramedics_PHTLS']}% PHTLS Certified")
        
    with cert_cols[1]:
        st.markdown("##### Doctor Certifications")
        st.progress(certification_data['Doctors_ATLS'], text=f"{certification_data['Doctors_ATLS']}% ATLS Certified (Trauma)")
        st.progress(certification_data['Doctors_ACLS'], text=f"{certification_data['Doctors_ACLS']}% ACLS Certified")
        
    with cert_cols[2]:
        st.markdown("##### Nurse Certifications")
        st.progress(certification_data['Nurses_BLS'], text=f"{certification_data['Nurses_BLS']}% BLS Certified")
        st.progress(certification_data['Nurses_ACLS'], text=f"{certification_data['Nurses_ACLS']}% ACLS Certified")

    st.warning("**Key Finding:** Only 13% of doctors hold a current Advanced Trauma Life Support (ATLS) certification, which is a critical standard for an organization specializing in trauma care.", icon="❗")
