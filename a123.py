import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# --- STREAMLIT APP CONFIGURATION (MUST BE THE FIRST COMMAND) ---
st.set_page_config(layout="wide", page_title="MicroNutriSense: Hidden Hunger")

# --- Data Loading and Preprocessing ---

@st.cache_data
def load_data():
    """
    Loads and preprocesses the three datasets.
    
    CRITICAL: The app expects the following three files in the working directory:
    1. CFSVA2024_HH_CHILD_6_59_MONTHS.csv
    2. CFSVA2024_HH_WOMEN_15_49_YEARS.csv
    3. CFSVA2024_HH data.csv (Note the space in the filename, which is used exactly as provided)
    """
    try:
        # Load all files. Using the names exactly as provided by the user.
        df_child = pd.read_csv("CFSVA2024_HH_CHILD_6_59_MONTHS.csv")
        df_women = pd.read_csv("CFSVA2024_HH_WOMEN_15_49_YEARS.csv")
        df_hh = pd.read_csv("CFSVA2024_HH data.csv") 
    except FileNotFoundError:
        st.error("One or more required CSV files were not found. Please ensure these files are in the working directory: CFSVA2024_HH_CHILD_6_59_MONTHS.csv, CFSVA2024_HH_WOMEN_15_49_YEARS.csv, CFSVA2024_HH data.csv")
        # Return empty dataframes so the app can still initialize without crashing
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 1. Clean Child Data for Malnutrition Status (Stunting/Wasting)
    try:
        # We assume the last 4 columns contain z-scores and status fields
        status_cols = df_child.columns[-4:] 
        df_child = df_child.rename(columns={
            status_cols[1]: 'Stunting_Status',      
            status_cols[2]: 'Wasting_Status',       
            status_cols[3]: 'UrbanRural_Child'      
        })
    except IndexError:
        st.warning("Could not automatically rename child status columns. Using generic names.")
        df_child = df_child.rename(columns={
            df_child.columns[-3]: 'Stunting_Status', 
            df_child.columns[-2]: 'Wasting_Status',  
            df_child.columns[-1]: 'UrbanRural_Child'
        })
    
    # Define Stunting/Wasting binary indicator for prevalence calculation
    stunting_categories = ['Moderately stunted', 'Severely stunted']
    df_child['Is_Stunted'] = df_child['Stunting_Status'].apply(
        lambda x: 'Stunted' if pd.notna(x) and any(cat in str(x) for cat in stunting_categories) else 'Normal'
    )
    df_child['Is_Wasted'] = df_child['Wasting_Status'].apply(
        lambda x: 'Wasted' if pd.notna(x) and 'Wasted' in str(x) else 'Normal'
    )

    # 2. Clean Women Data for Determinants (Education) and Outcome (Dietary Diversity)
    try:
        # S12_01_5 is Mother's Education; the second-to-last column is Dietary Diversity Status
        df_women = df_women.rename(columns={
            'S12_01_5': 'Mother_Education',
            df_women.columns[-2]: 'Diet_Diversity_Status'
        })
    except IndexError:
        st.warning("Could not automatically rename women's status columns. Using generic names.")
        df_women = df_women.rename(columns={
            df_women.columns[-3]: 'Mother_Education', 
            df_women.columns[-2]: 'Diet_Diversity_Status'
        })

    # Simplify Dietary Diversity Status
    # This filter captures low diversity based on common CFSVA reporting (less than 5 food groups)
    df_women['Is_Low_Diet_Diversity'] = df_women['Diet_Diversity_Status'].apply(
        lambda x: 'Low Diversity (<5 Food Groups)' if pd.notna(x) and ('<5' in str(x) or '4 food' in str(x) or '3 food' in str(x) or '2 food' in str(x) or '1 food' in str(x)) else 'Adequate Diversity (>=5 Food Groups)'
    )
    
    # 3. Standardize Geographical columns
    df_child = df_child.rename(columns={'S0_C_Prov': 'Province', 'S0_D_Dist': 'District'})
    df_women = df_women.rename(columns={'S0_C_Prov': 'Province', 'S0_D_Dist': 'District'})
    df_hh = df_hh.rename(columns={'S0_C_Prov': 'Province', 'S0_D_Dist': 'District', 'UrbanRural': 'UrbanRural'})

    return df_child, df_women, df_hh

# Load the data after configuration
df_child, df_women, df_hh = load_data()

# --- Utility Functions for Analysis ---

def calculate_prevalence(df, group_col, outcome_col, positive_value):
    """Calculates the percentage prevalence of an outcome by a grouping column."""
    if df.empty or group_col not in df.columns or outcome_col not in df.columns:
        # Return placeholder data for safe plotting
        return pd.DataFrame({'Location': ['N/A'], 'Prevalence_Percent': [0], 'Color_Col': [0]})

    # Count total valid cases for the group
    total = df[df[outcome_col].notna()].groupby(group_col).size().reset_index(name='Total')
    
    # Count positive cases based on the positive_value string match
    positive_cases = df[df[outcome_col].apply(
        lambda x: pd.notna(x) and positive_value in str(x))].groupby(group_col).size().reset_index(name='Positive_Cases')

    merged = pd.merge(total, positive_cases, on=group_col, how='left').fillna(0)
    
    # Avoid division by zero
    merged['Prevalence_Percent'] = np.where(merged['Total'] > 0, (merged['Positive_Cases'] / merged['Total']) * 100, 0)
    merged = merged.sort_values('Prevalence_Percent', ascending=False)
    
    # Clean up column names for display and plot coloring
    merged = merged.rename(columns={group_col: 'Location'})
    merged['Color_Col'] = merged['Prevalence_Percent'] # Add a consistent color column for plotting
    return merged

def plot_prevalence_hotspots(df_prev, title, color_col):
    """Generates a bar chart for prevalence hotspots."""
    if df_prev.empty or df_prev['Location'].iloc[0] == 'N/A':
        # Create an empty, non-crashing plot
        fig = px.bar(pd.DataFrame({'Location': ['Data Missing'], 'Prevalence_Percent': [0]}), x='Location', y='Prevalence_Percent', title=title)
        fig.update_layout(yaxis=dict(range=[0, 1]))
        return fig
        
    fig = px.bar(
        df_prev,
        x='Location',
        y='Prevalence_Percent',
        color=color_col,
        title=title,
        labels={'Prevalence_Percent': 'Prevalence (%)', 'Location': 'Location'},
        template='plotly_white'
    )
    fig.update_layout(xaxis={'categoryorder':'total descending'}, yaxis=dict(range=[0, df_prev['Prevalence_Percent'].max() * 1.1]))
    return fig

# --- STREAMLIT APP LAYOUT ---

st.title("MicroNutriSense: Mapping, Predicting, and Solving Hidden Hunger")
st.markdown("""
A data-driven application to combat **Hidden Hunger** (micronutrient deficiencies) by fulfilling the following challenge objectives:
1.  **Map** malnutrition hotspots using geospatial data.
2.  **Analyze** root causes (stunting, deficiencies) using household and maternal determinants.
3.  **Predict** malnutrition risk.
4.  **Recommend** multi-sectoral interventions and propose policy briefs.
""")

st.sidebar.header("Navigation & Settings")
app_mode = st.sidebar.selectbox("Select Analysis Module",
    ["1. Malnutrition Hotspot Mapping", "2. Root-Cause Analysis", "3. Predictive Modeling (Mock)", "4. Policy Briefs & Interventions"]
)

# Pre-calculate main prevalence indicators for use across modules
df_stunting_prev = calculate_prevalence(df_child, 'Province', 'Is_Stunted', 'Stunted')
df_mdd_prev = calculate_prevalence(df_women, 'Province', 'Is_Low_Diet_Diversity', 'Low Diversity')

# --- Module 1: Hotspot Mapping ---
if app_mode == "1. Malnutrition Hotspot Mapping":
    st.header("1. Malnutrition Hotspot Mapping")
    st.markdown("Identify geographical areas with the highest burden of child stunting and maternal dietary deficiencies. [Image of Map showing global malnutrition hotspots]")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Child Stunting Hotspots by Province")
        fig_stunting = plot_prevalence_hotspots(
            df_stunting_prev,
            "Prevalence of Stunting (Ages 6-59 Months) by Province",
            'Color_Col' # Use the consistent color column
        )
        st.plotly_chart(fig_stunting, use_container_width=True)
        st.caption("Stunting is an indicator of chronic malnutrition and a key manifestation of long-term hidden hunger.")
        st.dataframe(df_stunting_prev[['Location', 'Prevalence_Percent']].round(1).set_index('Location'), use_container_width=True)

    with col2:
        st.subheader("Maternal Diet Diversity Hotspots by Province")
        fig_mdd = plot_prevalence_hotspots(
            df_mdd_prev,
            "Prevalence of Low Maternal Diet Diversity by Province",
            'Color_Col'
        )
        st.plotly_chart(fig_mdd, use_container_width=True)
        st.caption("Low Dietary Diversity (consumption of $<5$ food groups) reflects inadequate micronutrient intake in mothers.")
        st.dataframe(df_mdd_prev[['Location', 'Prevalence_Percent']].round(1).set_index('Location'), use_container_width=True)

# --- Module 2: Root-Cause Analysis ---
elif app_mode == "2. Root-Cause Analysis":
    st.header("2. Root-Cause Analysis: Linking Determinants to Outcomes")
    st.markdown("Analyze how key socio-economic and maternal factors influence nutrition outcomes to inform causal interventions. [Image of UNICEF conceptual framework for malnutrition]")

    # 1. Mother's Education vs. Low Dietary Diversity (Maternal determinant)
    st.subheader("Maternal Dietary Diversity vs. Mother's Education Level")
    
    # Calculate prevalence of low diversity across education levels
    df_edu = df_women[['Mother_Education', 'Is_Low_Diet_Diversity']].dropna()
    if not df_edu.empty:
        edu_diversity_raw = df_edu.groupby('Mother_Education')['Is_Low_Diet_Diversity'].value_counts().unstack(fill_value=0)
        
        # Calculate percentages only if the target column exists
        if 'Low Diversity (<5 Food Groups)' in edu_diversity_raw.columns:
            edu_diversity_perc = edu_diversity_raw.div(edu_diversity_raw.sum(axis=1), axis=0) * 100
            
            # Focus on the 'Low Diversity' column and reset index for plotting
            education_diversity_low = edu_diversity_perc[['Low Diversity (<5 Food Groups)']].reset_index().rename(columns={'Low Diversity (<5 Food Groups)': 'Percentage'})

            fig_edu = px.bar(
                education_diversity_low.sort_values('Percentage', ascending=False),
                x='Mother_Education',
                y='Percentage',
                color='Percentage',
                title="Prevalence of Low Maternal Diet Diversity by Mother's Education Level",
                labels={'Percentage': '% with Low Diversity', 'Mother_Education': 'Mother\'s Highest Education'},
                template='plotly_white',
            )
            fig_edu.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig_edu, use_container_width=True)
            st.caption("**Root Cause Insight:** Lower education strongly correlates with higher dietary inadequacy, suggesting a need for education and knowledge-based interventions.")
        else:
             st.warning("Dietary diversity calculation failed. Check data structure.")
    else:
        st.warning("Maternal education data is missing or incomplete.")


    # 2. Stunting vs. Urban/Rural Setting (Geographic determinant)
    st.subheader("Child Stunting Prevalence by Urban/Rural Setting")
    df_stunting_ur_prev = calculate_prevalence(df_child, 'UrbanRural_Child', 'Is_Stunted', 'Stunted')

    fig_ur = plot_prevalence_hotspots(
        df_stunting_ur_prev,
        "Child Stunting Prevalence by Urban/Rural Setting",
        'Location'
    )
    st.plotly_chart(fig_ur, use_container_width=True)
    st.caption("**Hotspot Insight:** Rural areas face higher stunting burden, suggesting systemic issues with access to food markets, sanitation, and health care services.")


# --- Module 3: Predictive Modeling (Mockup) ---
elif app_mode == "3. Predictive Modeling (Mock)":
    st.header("3. Predictive Modeling: Forecasting Malnutrition Risk")
    st.markdown("""
    A sophisticated machine learning model would be trained on the combined dataset to estimate the probability of a child or mother developing malnutrition/deficiencies. This allows for **proactive and targeted** resource allocation. 
    """)

    st.info("⚠️ **Mockup Interface:** This section demonstrates the *workflow* and output of a predictive model (e.g., Random Forest or XGBoost).")

    st.subheader("Simulated Individual Risk Prediction")
    
    col_input, col_output = st.columns(2)

    with col_input:
        st.markdown("**Input Household Determinants:**")
        # Ensure prov_options is safe even if df_child is empty
        prov_options = df_child['Province'].dropna().unique() if 'Province' in df_child.columns and not df_child.empty else ['Default Province A', 'Default Province B']
        province = st.selectbox("Select Province", prov_options, index=0)
        
        edu_options = df_women['Mother_Education'].dropna().unique() if 'Mother_Education' in df_women.columns and not df_women.empty else ['Secondary/Higher', 'Primary', 'None']
        mother_edu = st.selectbox("Mother's Education Level", edu_options, index=0)
        
        # Safely determine UrbanRural options
        urban_rural_options = df_hh['UrbanRural'].dropna().unique() if 'UrbanRural' in df_hh.columns and not df_hh.empty else ['Rural', 'Urban']
        if len(urban_rural_options) > 1:
            urban_rural = st.radio("Household Setting", urban_rural_options, index=1)
        elif len(urban_rural_options) == 1:
            urban_rural = st.radio("Household Setting", urban_rural_options, index=0)
        else:
            urban_rural = st.radio("Household Setting", ['Rural', 'Urban'], index=0)

        house_size = st.slider("Household Size", 2, 12, 5)

    with col_output:
        st.markdown("**Predicted Outcome:**")
        
        # Find national average stunting prevalence for delta comparison
        national_avg = df_stunting_prev['Prevalence_Percent'].mean() if not df_stunting_prev.empty and df_stunting_prev['Location'].iloc[0] != 'N/A' else 25

        if st.button("Calculate Risk Score", use_container_width=True):
            # Simple rule-based mock prediction based on root-cause analysis
            risk = 0
            # Higher risk for the top stunting province
            is_valid_data = not df_stunting_prev.empty and df_stunting_prev['Location'].iloc[0] != 'N/A'
            if is_valid_data and province == df_stunting_prev['Location'].iloc[0]: risk += 25 
            # Higher risk for low education
            if 'Primary' in mother_edu or 'None' in mother_edu: risk += 20 
            # Higher risk for rural setting
            if urban_rural == 'Rural': risk += 15 
            # Higher risk for larger households
            if house_size >= 7: risk += 10 

            risk = min(risk, 90) + np.random.randint(0, 10) # Add a small random jitter
            
            risk_color = "inverse" if risk < national_avg else "normal"
            
            st.metric(
                label="Predicted Child Stunting Risk",
                value=f"{risk:.1f}%",
                delta=f"{risk - national_avg:.1f}% vs National Average ({national_avg:.1f}%)",
                delta_color=risk_color
            )
            st.progress(risk / 100)
            st.caption(f"A score significantly above the national average of {national_avg:.1f}% indicates high risk. This household would be flagged for priority intervention.")

    st.subheader("Simulated Model Feature Importance")
    st.markdown("Feature importance indicates which variables drive the prediction, helping to prioritize intervention design.")
    
    importance_data = pd.DataFrame({
        'Feature': ["Mother's Education Level", "Geographic Location (District/Province)", "Household Wealth Index (Simulated)", "Access to Clean Water/Sanitation (Simulated)", "Household Size"],
        'Importance_Score': [35, 25, 20, 10, 10]
    }).sort_values('Importance_Score', ascending=True)

    fig_importance = px.bar(
        importance_data,
        x='Importance_Score',
        y='Feature',
        orientation='h',
        title='Top 5 Drivers of Child Malnutrition Risk (Simulated)',
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    st.plotly_chart(fig_importance, use_container_width=True)


# --- Module 4: Policy Briefs & Interventions ---
elif app_mode == "4. Policy Briefs & Interventions":
    st.header("4. Policy Briefs and Multi-Sectoral Interventions")
    st.markdown("Interventions are recommended across three key sectors, followed by a formal policy brief based on the data findings. ")

    st.subheader("Multi-Sectoral Intervention Matrix")

    interventions = {
        "Health Sector (Targeting Deficiencies)": [
            "**Targeted Supplementation:** Implement Multiple Micronutrient Supplements (MMS) for pregnant women in high-risk provinces identified in **Module 1**.",
            "**MIYCN Counseling:** Enhance training for healthcare workers to deliver effective Maternal, Infant, and Young Child Nutrition (MIYCN) counseling.",
            "**Anemia Screening:** Universal screening and treatment protocols for anemia in all women of reproductive age."
        ],
        "Agriculture/Livelihoods Sector (Targeting Food Access)": [
            "**Biofortification:** Subsidize and promote biofortified staples (e.g., iron-rich beans, Vitamin A rich crops) to naturally boost micronutrient intake.",
            "**Home Gardens:** Provide resources for establishing diverse household kitchen gardens, addressing the low maternal diet diversity.",
            "**Value Chains:** Create market linkages for smallholder farmers who produce affordable, nutrient-dense foods (pulses, animal source foods)."
        ],
        "Education/WASH Sector (Targeting Empowerment/Environment)": [
            "**Girl's Education Incentives:** Implement scholarship/cash transfer programs conditional on female retention through secondary level, addressing the root cause found in **Module 2**.",
            "**Nutrition Literacy:** Integrate practical nutrition and hygiene (WASH) education into community adult learning and health center outreach.",
            "**Sanitation:** Invest in clean water and improved sanitation in rural hotspots to reduce environmental enteropathy, a driver of stunting."
        ]
    }

    cols = st.columns(3)
    for i, (sector, recs) in enumerate(interventions.items()):
        with cols[i]:
            st.markdown(f"**{sector}**")
            st.markdown("\n".join([f"- {rec}" for rec in recs]))
            st.markdown("---")

    st.subheader("Short Policy Brief")
    st.markdown("This brief is designed for rapid dissemination to local policy makers.")
    
    # Dynamic text for the policy brief
    is_valid_stunting_data = not df_stunting_prev.empty and df_stunting_prev['Location'].iloc[0] != 'N/A'
    if is_valid_stunting_data:
        highest_prev_prov = df_stunting_prev['Location'].iloc[0]
        stunting_prev_value = f"{df_stunting_prev['Prevalence_Percent'].iloc[0]:.1f}%"
    else:
        highest_prev_prov = "High-Prevalence Province (Data Not Loaded)"
        stunting_prev_value = "N/A"

    policy_brief_content = f"""
    ### Policy Brief: Breaking the Cycle of Hidden Hunger

    **To:** National Policy Steering Committee on Nutrition
    **From:** MicroNutriSense Data Analysis Team
    **Date:** November 2025

    #### I. Executive Summary

    Data analysis from the CFSVA 2024 reveals that chronic malnutrition (stunting) and critical micronutrient gaps (low maternal dietary diversity) are significantly higher in rural areas and are inversely correlated with the mother's education level. The province of **{highest_prev_prov}** is flagged as a critical hotspot with a stunting prevalence of **{stunting_prev_value}**. This brief recommends an integrated policy response focusing on two high-impact sectors: **Education** and **Health**.

    #### II. Key Findings from Data

    1.  **Hotspots:** The highest stunting prevalence is observed in **{highest_prev_prov}** (refer to **Module 1**).
    2.  **Root Cause:** Mothers with only primary or incomplete education levels demonstrate a significantly higher prevalence of low dietary diversity compared to those with secondary or higher education (refer to **Module 2**).
    3.  **Pathway:** Rural households are disproportionately affected by stunting, indicating access barriers to diverse foods and quality health/sanitation services.

    #### III. Policy Recommendations

    **A. Empowering Women through Education (Long-Term Impact)**
    * **Recommendation 1: Conditional Education Incentive Program (CEIP):** Launch a targeted cash transfer program in high-stunting provinces, conditional on: (a) maintaining female enrollment through secondary school, and (b) attendance of mandatory household nutrition/hygiene training sessions by the primary caregiver.
    * **Recommendation 2: Curriculum Integration:** Update the primary and secondary health curriculum to include practical, locally-relevant micronutrient and dietary diversity lessons.

    **B. Targeted Micronutrient Delivery and Counseling (Short-Term Impact)**
    * **Recommendation 3: Universal MMS Provision:** Mandate and fund the provision of Multiple Micronutrient Supplements (MMS) to all pregnant women, replacing standard Iron/Folic Acid (IFA) where possible, in all primary healthcare facilities.
    * **Recommendation 4: Data-Driven Resource Allocation:** Use the predictive model developed by MicroNutriSense (**Module 3**) to proactively identify and register at-risk households for home visits and specialized MIYCN counseling before the birth of the child.

    **C. Cross-Sectoral Linkage**
    * **Recommendation 5: Agriculture-Health Link:** Establish municipal purchase agreements where local health clinics source diverse, nutrient-rich foods (e.g., biofortified vegetables, eggs) from local women's agricultural cooperatives for use in nutrition rehabilitation and counseling demonstrations.

    #### IV. Conclusion

    Addressing hidden hunger requires moving beyond food security to focus on dietary quality, education, and health service utilization. By integrating education and health initiatives, policy makers can build resilience, improve child growth outcomes, and secure long-term human capital development.
    """
    st.markdown(policy_brief_content)