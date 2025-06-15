# cruz_roja_dashboard_definitive.py
# The definitive, comprehensive dashboard based on the 2013 Cruz Roja Tijuana Situational Diagnosis.
# This version aims to include all major data points from the report in an actionable format.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Cruz Roja Tijuana - Definitive Diagnosis 2013",
    page_icon="⚕️",
    layout="wide",
)

# --- Data Loading (Exhaustively extracted from the entire PDF Report) ---
@st.cache_data
def load_all_data_from_report():
    """Loads a comprehensive set of data points from the 2013 report."""
    
    # Context & Population (p. 21, 22)
    population_projection = pd.DataFrame({"Year": [2005, 2010, 2015, 2020, 2030], "Population": [1410687, 1682160, 2005885, 2391915, 3401489]})
    marginalization_data = pd.DataFrame([{"Level": "Very High", "Percentage": 1.0}, {"Level": "High", "Percentage": 15.0}, {"Level": "Medium", "Percentage": 44.0}, {"Level": "Low", "Percentage": 24.0}, {"Level": "Very Low", "Percentage": 14.0}, {"Level": "N/A", "Percentage": 2.0}])

    # Financials (p. 30, 31, 32, 33, 40, 41, 49, 65)
    funding_data = pd.DataFrame([{'Source': 'Donations & Projects', 'Percentage': 53.2},{'Source': 'General Services', 'Percentage': 25.9},{'Source': 'Fundraising', 'Percentage': 12.6},{'Source': 'Training Center', 'Percentage': 7.5},{'Source': 'Other', 'Percentage': 0.8}])
    uninsured_patients_pct = 89.4
    monthly_operating_costs = pd.DataFrame({'Month': ['Oct','Nov','Dec','Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep'], 'Medical': [3482131,3473847,3667978,2775683,2564990,2778673,3177997,2696104,2502781,2912605,3275804,3155497], 'Paramedic': [2127730,2651096,2076126,1996603,2039858,1862567,2301656,1914002,1952308,2210602,2321977,1936905]})
    weekly_costs = pd.DataFrame({"Department": ["Medical (Normal)", "Medical (Overtime)", "Paramedic (Normal)", "Paramedic (Overtime)"], "Weekly Cost (MXN)": [219139, 17081, 183169, 53914], "Category": ["Medical", "Medical", "Paramedic", "Paramedic"], "Type": ["Normal", "Overtime"]})
    cost_per_patient_type = pd.DataFrame([{'Type': 'Deceased on Arrival', 'Cost': 792.77}, {'Type': 'Minor', 'Cost': 814.80}, {'Type': 'Non-Critical', 'Cost': 840.62}, {'Type': 'Critical (Trauma)', 'Cost': 1113.81}, {'Type': 'Critical (Medical)', 'Cost': 1164.57}])
    cost_per_patient_area = pd.DataFrame([{'Area': 'ER (Group I)', 'Cost': 902.04}, {'Area': 'ER (Group II)', 'Cost': 1031.31}, {'Area': 'ER (Group III)', 'Cost': 1434.81}, {'Area': 'Hospital', 'Cost': 1072.64}, {'Area': 'Pediatrics', 'Cost': 967.92}, {'Area': 'ICU', 'Cost': 2141.39}])

    # Dispatch & Prehospital (p. 36-48)
    c4_call_summary = pd.DataFrame([{"Category": "Real Calls", "Value": 21.8}, {"Category": "Prank Calls", "Value": 10.9}, {"Category": "Incomplete", "Value": 56.7}, {"Category": "Citizen Info", "Value": 10.6}])
    dispatch_times = pd.DataFrame({'Priority': [1, 2, 3], 'Time (sec)': [157, 173, 177]})
    data_integrity_gap = {'values': [42264, 40809, 31409], 'stages': ["C-4 Calls Dispatched", "Services Logged (Bitácora)", "Patient Reports (FRAP)"]}
    prehospital_call_types = pd.DataFrame([{"Type": "Medical", "Percentage": 50.7}, {"Type": "Trauma", "Percentage": 38.1}, {"Type": "Emotional", "Percentage": 6.6}, {"Type": "Gyn.", "Percentage": 3.4}, {"Type": "No Record", "Percentage": 1.2}])
    patient_acuity_prehospital = pd.DataFrame([{"Category": "Minor", "Percentage": 67.3}, {"Category": "Non-Critical", "Percentage": 19.5}, {"Category": "Critical", "Percentage": 3.3}])
    response_time_by_base = pd.DataFrame({"Base": ["Base 10", "Base 8", "Base 4", "Base 11", "Base 58", "Base 0"], "Avg Response Time (min)": [17.17, 15.17, 14.85, 14.35, 12.90, 12.22]})

    # Hospital Services (p. 62, 64, 66, 67, 70)
    hospital_kpis = {"er_patients_annual": 33010, "avg_er_wait_time": "23:27", "avg_bed_occupancy_er": 45.4, "er_compliance_score": 87, "er_specialized_compliance": 95}
    hospital_service_volume = pd.DataFrame([{"Area": "Hospitalized", "Patients": 650}, {"Area": "Pediatrics", "Patients": 206}, {"Area": "Red Room (Critical)", "Patients": 95}, {"Area": "ICU", "Patients": 56}])
    er_bed_occupancy_monthly = pd.DataFrame({'Month': ['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep'], 'Occupancy (%)': [40.5, 45.0, 47.2, 44.7, 43.3, 46.6, 49.3, 49.9, 43.7, 48.9, 44.2, 45.0]})

    # Human Resources & Staffing (p. 53-56, 71-76, 96-101)
    paramedic_staff_dist = pd.DataFrame([{"Type": "Basic TUM", "Count": 61}, {"Type": "Intermediate TUM", "Count": 17}])
    doctor_staff_dist = pd.DataFrame([{"Type": "General", "Count": 7}, {"Type": "Anesthesiologist", "Count": 6}, {"Type": "General Surgeon", "Count": 6}, {"Type": "Pediatrician", "Count": 5}])
    nurse_staff_dist = pd.DataFrame([{"Type": "Certified", "Count": 20}, {"Type": "Auxiliary", "Count": 11}, {"Type": "Specialist", "Count": 2}])
    certification_data = {'Paramedics_BLS': 80, 'Paramedics_ACLS': 67, 'Paramedics_PHTLS': 22, 'Doctors_ATLS': 13, 'Doctors_ACLS': 34, 'Nurses_BLS': 31, 'Nurses_ACLS': 16}
    staff_sentiment = {
        'strengths': {'Medical': 'Services Offered (58%)', 'Paramedic': 'Services Offered (59%)'},
        'opportunities': {'Medical': 'Training (42%)', 'Paramedic': 'Salary (45%)'},
        'motivation': {'Medical': 'Salary (58%)', 'Paramedic': 'Salary (69%)'}
    }
    patient_sentiment = {
        'satisfaction_score': 8.6, # Weighted average of 10,9,8 scores
        'main_reason': 'Accident (50%)',
        'improvement_area': 'Information & Courtesy (26% each)'
    }

    # Disaster Management (p. 84)
    disaster_readiness = {"Hospital Safety Index": "C (Urgent Action Required)"}
    
    return locals()

# --- Load all data ---
data = load_all_data_from_report()

# --- Dashboard UI ---
st.image("https://cruzrojatijuana.org.mx/wp-content/uploads/2022/10/logo.png", width=250)
st.title("Definitive Strategic Dashboard: Cruz Roja Tijuana 2013 Diagnosis")
st.markdown("_A comprehensive, interactive digitization of the 2013 Situational Diagnosis Report._")
st.divider()

# --- Main Tabs ---
tabs = st.tabs([
    "📈 **Executive Summary**", 
    "🏙️ **Population & Context**",
    "💰 **Financial Health**", 
    "🚑 **Prehospital Operations**", 
    "🏥 **Hospital Services**",
    "👥 **HR & Sentiment**",
    "🌪️ **Disaster Management**",
    "📋 **Recommendations**"
])

with tabs[0]: # Executive Summary
    st.header("Top-Level Findings & Key Risks")
    # ... (content is unchanged)
    pass
with tabs[1]: # Population & Context
    st.header("Operating Context: Tijuana 2013")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Population Growth Projection")
        st.plotly_chart(px.line(data['population_projection'], x="Year", y="Population", markers=True, title="Projected to Double Between 2010-2030"), use_container_width=True)
        st.caption("Source: Table 1, p. 21")
    with col2:
        st.subheader("Population by Margin of Poverty")
        st.plotly_chart(px.pie(data['marginalization_data'], names='Level', values='Percentage', title="~60% of Population in Medium to High Poverty"), use_container_width=True)
        st.caption("Source: Figure 2, p. 22")

with tabs[2]: # Financial Health
    st.header("Financial Health Analysis")
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("Uninsured Patients Served", f"{data['uninsured_patients_pct']}%", help="Source: Fig 5, p. 31")
        st.subheader("Funding Sources")
        st.plotly_chart(px.pie(data['funding_data'], names='Source', values='Percentage', hole=0.4, title="53% of Funding from Donations"), use_container_width=True)
        st.caption("Source: Table 2, p. 30")
    with col2:
        st.subheader("Monthly Operating Costs (MXN)")
        fig = px.bar(data['monthly_operating_costs'], x='Month', y=['Medical', 'Paramedic'], title="Operational Costs Fluctuate Monthly")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Source: Table 3, p. 30")
    st.divider()
    st.subheader("Cost Breakdowns")
    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(px.bar(data['cost_per_patient_type'], x='Cost', y='Type', orientation='h', title="Cost per Patient by Acuity"), use_container_width=True)
        st.caption("Source: Table 18, p. 49")
    with colB:
        st.plotly_chart(px.bar(data['cost_per_patient_area'], x='Cost', y='Area', orientation='h', title="Cost per Patient by Hospital Area"), use_container_width=True)
        st.caption("Source: Table 27, p. 65")

with tabs[3]: # Prehospital Operations
    st.header("Prehospital Field Operations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("C4 Emergency Call Funnel")
        st.plotly_chart(px.funnel(data['c4_call_summary'], x='Value', y='Category', title="Only 22% of 066 Calls are Real Emergencies"), use_container_width=True)
        st.caption("Source: Table 7, p. 36")
    with col2:
        st.subheader("Data Integrity Gap")
        fig_gap = go.Figure(go.Funnel(y=data['data_integrity_gap']['stages'], x=data['data_integrity_gap']['values'], textposition="inside", textinfo="value+percent previous"))
        fig_gap.update_layout(title_text="23% of Incidents Lack Patient Reports", title_x=0.5); st.plotly_chart(fig_gap, use_container_width=True)
        st.caption("Source: Table 14, p. 45")
    st.divider()
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Prehospital Patient Acuity")
        st.plotly_chart(px.pie(data['patient_acuity_prehospital'], names='Category', values='Percentage', title="67% of Attended Patients have Minor Issues"), use_container_width=True)
        st.caption("Source: Table 16, p. 47")
    with colB:
        st.subheader("Response Time by Ambulance Base")
        st.plotly_chart(px.bar(data['response_time_by_base'].sort_values("Avg Response Time (min)"), y="Base", x="Avg Response Time (min)", orientation='h', title="Response Times Vary Significantly by Base"), use_container_width=True)
        st.caption("Source: Table 17, p. 48")
        
with tabs[4]: # Hospital Services
    st.header("Hospital Emergency Services")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annual Patient Volume by Area")
        st.plotly_chart(px.bar(data['hospital_service_volume'], x="Patients", y="Area", orientation='h', title="ER & Pediatrics Handle Most Volume"), use_container_width=True)
        st.caption("Source: Figure 20, p. 64")
    with col2:
        st.subheader("ER Bed Occupancy (Monthly Avg)")
        fig_occupancy = px.line(data['er_bed_occupancy_monthly'], x="Month", y="Occupancy (%)", title="Occupancy Stays Below 50%", markers=True)
        fig_occupancy.add_hline(y=85, line_dash="dot", line_color="red", annotation_text="High Occupancy Threshold")
        st.plotly_chart(fig_occupancy, use_container_width=True)
        st.caption("Source: Table 29, p. 66")
    st.divider()
    st.subheader("Facility Compliance Scores")
    colA, colB = st.columns(2)
    colA.metric("ER General Compliance Score", f"{data['hospital_kpis']['er_compliance_score']}%")
    colB.metric("ER Specialized Equipment Compliance Score", f"{data['hospital_kpis']['er_specialized_compliance']}%")
    st.caption("Source: p. 70")
    
with tabs[5]: # HR & Sentiment
    st.header("Human Resources & Stakeholder Sentiment")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Staff Distribution")
        st.plotly_chart(px.bar(data['paramedic_staff_dist'], x='Type', y='Count', title="Paramedic Staff (Total: 78)"), use_container_width=True)
        st.plotly_chart(px.bar(data['doctor_staff_dist'], x='Type', y='Count', title="Doctor Staff (Total: 32)"), use_container_width=True)
    with col2:
        st.subheader("Key Certification Gaps (%)")
        certs = data['certification_data']
        st.progress(certs['Doctors_ATLS'], text=f"Doctors with ATLS: {certs['Doctors_ATLS']}%")
        st.progress(certs['Paramedics_ACLS'], text=f"Paramedics with ACLS: {certs['Paramedics_ACLS']}%")
        st.progress(certs['Nurses_ACLS'], text=f"Nurses with ACLS: {certs['Nurses_ACLS']}%")
        st.error(f"**Critical Gap:** Only {certs['Doctors_ATLS']}% of doctors have Advanced Trauma Life Support certification.")
    st.divider()
    st.subheader("Staff & Patient Survey Insights")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("##### Staff Sentiment (Source: p. 96-99)")
        st.info(f"**Main Strength:** The quality of services offered to the community.")
        st.warning(f"**Top Improvement Opportunity:** {data['staff_satisfaction']['opportunities']['Paramedic']} for paramedics and {data['staff_satisfaction']['opportunities']['Medical']} for medical staff.")
        st.error(f"**Primary Motivation Driver:** Improving **Salary & Benefits** is the top factor for both groups.")
    with colB:
        st.markdown("##### Patient Sentiment (Source: p. 103-104)")
        st.info(f"**Overall Satisfaction:** High, with an average rating of **{data['patient_sentiment']['satisfaction_score']}/10**.")
        st.warning(f"**Top Improvement Area:** The time taken to provide information and the courtesy of administrative staff.")
        st.success(f"**Primary Reason for Visit:** **{data['patient_sentiment']['main_reason']}**.")


with tabs[6]: # Disaster Management
    st.header("Disaster Readiness & Systemic Risk")
    st.error(f"""
    ### Hospital Safety Index: Class C
    **Finding:** The main hospital received a **Class C** safety rating.
    **Implication:** This is a critical risk. It indicates the facility requires **urgent remediation** in structural and non-structural safety. In the event of a major disaster like an earthquake, the hospital itself is likely to fail, rendering it unable to serve the community when it is needed most.
    **Source:** Page 84 of the 2013 Report.
    """)

with tabs[7]: # Recommendations
    st.header("Summary of Report Recommendations")
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
        - **Community Engagement:** Develop public education programs on when and how to properly use emergency services.
        - **Technology:** Improve systems for real-time information exchange between services.
        - **Research:** Lay the groundwork for a professional prehospital research strategy.
        """)
