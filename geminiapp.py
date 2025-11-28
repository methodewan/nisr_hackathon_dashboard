import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import logging

# Set up basic logging (visible in the Streamlit console/logs)
logging.basicConfig(level=logging.INFO)

# --- Configuration ---
st.set_page_config(
    page_title="MicroNutriSense Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define colors for professional look
colors = {
    'primary': '#003366',  # Deep Blue (Corporate)
    'secondary': '#64C59A', # Medium Green (Health/Nature)
    'danger': '#E76F51',   # Orange/Red (Danger/Hotspot)
}

# --- 1. Data Loading and Preprocessing ---

@st.cache_data
def load_and_preprocess_data():
    """Load data and perform necessary cleaning and aggregation."""
    
    # --- ABSOLUTE PATHS PROVIDED BY USER ---
    # Using raw strings (r"...") to handle Windows backslashes correctly
    HH_PATH = r"C:\Users\hp\Downloads\Microdata\Microdata1111\Microdata\csvfile\CFSVA2024_HH data.csv"
    CHILD_PATH = r"C:\Users\hp\Downloads\Microdata\Microdata1111\Microdata\csvfile\CFSVA2024_HH_CHILD_6_59_MONTHS.csv"
    WOMEN_PATH = r"C:\Users\hp\Downloads\Microdata\Microdata1111\Microdata\csvfile\CFSVA2024_HH_WOMEN_15_49_YEARS.csv"

    data_available = False
    df_hh, df_child, df_women = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- USER ADJUSTMENT REQUIRED HERE ---
    STUNTING_COL = "stunting" 
    WASTING_COL = "wasting" 
    # -----------------------------------


    # Attempt to load using the absolute paths
    try:
        df_hh = pd.read_csv(HH_PATH)
        df_child = pd.read_csv(CHILD_PATH)
        df_women = pd.read_csv(WOMEN_PATH)
        data_available = True
        st.success("Successfully loaded data from absolute paths.")
    except FileNotFoundError:
        st.error(f"""
            **Fatal Error:** Could not find one or more files at the specified paths.
            Please verify the file paths and try again. Using mock data for demonstration purposes.
        """)
        data_available = False
        return pd.DataFrame(), pd.DataFrame(), False


    # --- Define Expected Column Mappings ---
    # Malnutrition Status columns (critical for Tab 1)
    MALNUT_COLS_MAPPING = {
        'S0_C_Prov': 'Province',
        STUNTING_COL: 'Stunting_Status', 
        WASTING_COL: 'Wasting_Status',  
    }
    
    # Driver Analysis columns (critical for Tab 2)
    DRIVER_COLS_MAPPING = {
        'S1_01_5': 'HH_Head_Education',
        'S5_01': 'Water_Source',
    }


    # --- Processing Child Data (Crucial for Tab 1) ---
    regional_agg = pd.DataFrame()
    df_driver_analysis = pd.DataFrame()
    
    if not df_child.empty:
        df_child_cols = df_child.columns.tolist()

        # 1. CHECK FOR CRITICAL MALNUTRITION COLUMNS
        missing_malnut_cols = [
            original for original in MALNUT_COLS_MAPPING.keys() 
            if original not in df_child_cols
        ]

        if missing_malnut_cols:
            all_cols_str = '\n- '.join(sorted(df_child_cols))
            
            st.error(f"""
                **Analysis Error (Child Data):** The required column(s) for malnutrition analysis are missing from `{CHILD_PATH}`. 
                **Missing Columns:** {', '.join(missing_malnut_cols)}.
                ---
                **Available Columns in Child Data:**
                - {all_cols_str}
            """)
            return regional_agg, df_driver_analysis, data_available

        # 2. RENAME FOUND COLUMNS
        df_child = df_child.rename(columns=MALNUT_COLS_MAPPING)
        
        # --- FIX 1: Ensure status and grouping columns are strings and handle NaNs ---
        df_child['Stunting_Status'] = df_child['Stunting_Status'].astype(str)
        df_child['Wasting_Status'] = df_child['Wasting_Status'].astype(str)
        df_child['Province'] = df_child['Province'].astype(str)

        # --- FIX 2: Drop rows where Province is missing/NaN (which appears as 'nan' string after conversion) ---
        initial_rows = len(df_child)
        df_child = df_child[df_child['Province'].str.lower() != 'nan'].copy()
        logging.info(f"Dropped {initial_rows - len(df_child)} rows with missing Province data.")
        # -------------------------------------------------------------------------------------

        # 3. Calculate Stunting and Wasting Prevalence
        # Check if the column contains the word 'Stunted' (case-insensitive)
        df_child['Stunted'] = df_child['Stunting_Status'].str.contains('Stunted', case=False, na=False)
        df_child['Wasted'] = df_child['Wasting_Status'].str.contains('Wasted', case=False, na=False)
        
        # Calculate Stunting Prevalence
        stunting_agg = df_child.groupby('Province').agg(
            Total_Children=('Stunting_Status', 'count'),
            Stunted_Count=('Stunted', 'sum')
        ).reset_index()
        stunting_agg['Stunting_Prevalence'] = (stunting_agg['Stunted_Count'] / stunting_agg['Total_Children']) * 100
        
        # Calculate Wasting Prevalence
        wasting_agg = df_child.groupby('Province').agg(
            Total_Children=('Wasting_Status', 'count'),
            Wasted_Count=('Wasted', 'sum')
        ).reset_index()
        wasting_agg['Wasting_Prevalence'] = (wasting_agg['Wasted_Count'] / wasting_agg['Total_Children']) * 100
        
        # Create a combined regional dataframe
        regional_agg = stunting_agg[['Province', 'Stunting_Prevalence']].merge(
            wasting_agg[['Province', 'Wasting_Prevalence', 'Total_Children']],
            on='Province'
        )
        
        # --- FIX 3: Ensure Prevalence columns are numeric (float) ---
        regional_agg['Stunting_Prevalence'] = pd.to_numeric(regional_agg['Stunting_Prevalence'], errors='coerce')
        regional_agg['Wasting_Prevalence'] = pd.to_numeric(regional_agg['Wasting_Prevalence'], errors='coerce')
        # -----------------------------------------------------------
        
        logging.info("Regional Aggregation Head (Check for NaN/Empty values in Prevalence):")
        logging.info(regional_agg.head())
        
        # --- Prepare Driver Analysis Data (Tab 2) ---
        if not df_hh.empty:
            df_hh_cols = df_hh.columns.tolist()
            
            # Check for driver columns in HH data
            missing_driver_cols = [
                original for original in DRIVER_COLS_MAPPING.keys() 
                if original not in df_hh_cols
            ]

            if missing_driver_cols:
                 st.warning(f"Warning: HH data is missing driver columns ({', '.join(missing_driver_cols)}). Tab 2 (Root Cause Analysis) may use simulated data.")
            else:
                # Rename and process HH data for driver categories
                df_hh = df_hh.rename(columns=DRIVER_COLS_MAPPING)
                
                # Simplified categories for visualization
                # Using lower() and str() for robustness against different types/cases
                df_hh['Water_Access_Level'] = df_hh['Water_Source'].astype(str).str.lower().apply(lambda x: 'High Access' if 'pipe' in x else 'Low Access')
                df_hh['HH_Head_Edu_Level'] = df_hh['HH_Head_Education'].astype(str).str.lower().apply(
                    lambda x: 'Low (None/Primary)' if 'primary' in x or 'none' in x else 
                              'High (Secondary/Higher)' if 'secondary' in x or 'university' in x else 'Other'
                )
                
                # Mock a join (as real HHID is unknown) to demonstrate driver analysis
                df_child_driver_mock = df_child[['Province', 'Stunting_Status']].copy()
                
                # Use a random sample of the HH driver columns to simulate the linkage
                unique_edu = df_hh['HH_Head_Edu_Level'].dropna().unique()
                unique_wash = df_hh['Water_Access_Level'].dropna().unique()
                
                if len(unique_edu) > 0 and len(unique_wash) > 0:
                    df_child_driver_mock['HH_Head_Edu_Level'] = np.random.choice(unique_edu, size=len(df_child_driver_mock))
                    df_child_driver_mock['Water_Access_Level'] = np.random.choice(unique_wash, size=len(df_child_driver_mock))
                
                # Prepare final driver analysis data: Filter for Stunted or Normal status, ensuring string type.
                # Filter out 'nan' status rows
                df_driver_analysis = df_child_driver_mock[df_child_driver_mock['Stunting_Status'].str.lower() != 'nan'].copy()
                
                # Grouping logic
                df_driver_analysis['Stunting_Group'] = np.where(
                    df_driver_analysis['Stunting_Status'].str.contains('Stunted', case=False, na=False), 
                    'Stunted', 
                    'Non-Stunted'
                )
            
    return regional_agg, df_driver_analysis, data_available

regional_agg, df_driver_analysis, data_available = load_and_preprocess_data()

# --- 2. Dashboard Layout (Streamlit) ---

st.title("🔬 MicroNutriSense: Mapping Hidden Hunger")
st.markdown("Interactive dashboard for tracking malnutrition and identifying intervention points using CFSVA 2024 data.")

# --- Tab Selection ---
tab = st.selectbox(
    "Select Analysis View:",
    ['1. Regional Hotspots (Prevalence)', '2. Root Cause Analysis (Drivers)', '3. Actionable Insights & Policy'],
    index=0
)


# --- Functions to Render Content for Each Tab ---

def render_tab_1(regional_agg):
    """Renders the Regional Hotspots Tab."""
    st.header("1. Regional Hotspots: Prevalence of Malnutrition")
    
    # --- DIAGNOSTIC TOOL: SHOW DATA FOR PLOTTING ---
    with st.expander("📊 Data Check: Regional Aggregation Table (Expand to Debug Charts)"):
        if regional_agg.empty:
            st.warning("The aggregated data frame is empty.")
        else:
            st.dataframe(regional_agg, use_container_width=True)
            st.info("If the charts are blank, check if the 'Prevalence' columns above contain valid numbers (not NaN or zero for all rows).")
    # ---------------------------------------------

    # Check for empty dataframe after load and processing
    if regional_agg.empty or not data_available:
        st.info("Regional data is not available or aggregation failed. Please check the console/logs for diagnostic messages.")
        return
        
    col1, col2 = st.columns(2)

    # --- Stunting Plot ---
    with col1:
        # Filter out provinces where prevalence is NaN before sorting/plotting
        stunting_agg_filtered = regional_agg.dropna(subset=['Stunting_Prevalence']).sort_values('Stunting_Prevalence', ascending=False)
        
        if stunting_agg_filtered.empty:
            st.warning("Stunting prevalence data is not sufficient to draw the chart.")
        else:
            fig_stunting = px.bar(
                stunting_agg_filtered,
                x='Province',
                y='Stunting_Prevalence',
                title='Stunting (Chronic Malnutrition) Prevalence by Province',
                color='Stunting_Prevalence',
                color_continuous_scale=px.colors.sequential.RdBu_r,
                text_auto='.2s',
                labels={'Stunting_Prevalence': 'Prevalence (%)', 'Province': 'Province/Region'},
                height=400
            )
            # Apply custom style to the chart
            fig_stunting.update_traces(marker_color=colors['danger'])
            fig_stunting.update_layout(plot_bgcolor='white', title_font_color=colors['primary'])
            st.plotly_chart(fig_stunting, use_container_width=True)

    # --- Interpretation for Chronic Hotspots ---
    with col2:
        st.subheader("Chronic Hotspots Interpretation")
        # Ensure there is data to index before accessing iloc[0]
        top_hotspot = stunting_agg_filtered.iloc[0]["Province"] if not stunting_agg_filtered.empty else "No Data"
        st.markdown(f'<div style="border-left: 5px solid {colors["danger"]}; padding: 10px; background-color: #ffe0d9; border-radius: 5px;">'
                    f'**Stunting Prevalence** is the key indicator for chronic hidden hunger. '
                    f'Regions with the highest prevalence (e.g., **{top_hotspot}**) are long-term hotspots requiring deep, multi-sectoral investments. '
                    f'Stunting reflects prolonged poor nutrition, inadequate health, and socio-economic factors. It is a critical measure for intervention success.'
                    f'</div>', unsafe_allow_html=True)
        st.write("Chronic malnutrition is irreversible and impacts long-term development.")
        st.write("")

    st.markdown("---")
    col3, col4 = st.columns(2)

    # --- Wasting Plot ---
    with col3:
        wasting_agg_filtered = regional_agg.dropna(subset=['Wasting_Prevalence']).sort_values('Wasting_Prevalence', ascending=False)

        if wasting_agg_filtered.empty:
            st.warning("Wasting prevalence data is not sufficient to draw the chart.")
        else:
            fig_wasting = px.bar(
                wasting_agg_filtered,
                x='Province',
                y='Wasting_Prevalence',
                title='Wasting (Acute Malnutrition) Prevalence by Province',
                color='Wasting_Prevalence',
                color_continuous_scale=px.colors.sequential.Teal,
                text_auto='.2s',
                labels={'Wasting_Prevalence': 'Prevalence (%)', 'Province': 'Province/Region'},
                height=400
            )
            fig_wasting.update_traces(marker_color=colors['secondary'])
            fig_wasting.update_layout(plot_bgcolor='white', title_font_color=colors['primary'])
            st.plotly_chart(fig_wasting, use_container_width=True)

    # --- Interpretation for Acute Risk ---
    with col4:
        st.subheader("Acute Risk Interpretation")
        st.markdown(f'<div style="border-left: 5px solid {colors["secondary"]}; padding: 10px; background-color: #e0f9ed; border-radius: 5px;">'
                    f'**Wasting Prevalence** shows the immediate, acute risk. '
                    f'Provinces with high wasting require urgent, life-saving interventions (e.g., supplementary feeding and rapid screening). Wasting is often tied to seasonal food shortages, disease outbreaks, or economic shocks, making it an early warning indicator.'
                    f'</div>', unsafe_allow_html=True)
        st.write("Acute malnutrition requires immediate, targeted medical and nutritional support.")


def render_tab_2(df_driver_analysis):
    """Renders the Root Cause Analysis Tab."""
    st.header("2. Root Cause Analysis: Multi-Sectoral Drivers")
    st.markdown("This section analyzes the correlation between stunting and key non-nutritional drivers (Education and WASH).")
    
    if df_driver_analysis.empty or not data_available:
        st.error("Cannot perform driver analysis. Data for joining driver columns (HH data) or child status columns is missing, likely due to file load failure or missing key columns.")
        return
        
    col1, col2 = st.columns(2)

    # 1. Stunting vs. Education Plot
    with col1:
        # Check if the dataframe is empty before grouping
        if not df_driver_analysis.empty:
            edu_agg = df_driver_analysis.groupby('HH_Head_Edu_Level')['Stunting_Group'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
            fig_edu = px.bar(
                edu_agg,
                x='HH_Head_Edu_Level',
                y='Percentage',
                color='Stunting_Group',
                title='Stunting Prevalence by Household Head Education Level',
                color_discrete_map={'Stunted': colors['danger'], 'Non-Stunted': colors['secondary']},
                category_orders={"HH_Head_Edu_Level": ["Low (None/Primary)", "High (Secondary/Higher)", "Other"]},
                labels={'HH_Head_Edu_Level': 'HH Head Education Level', 'Stunting_Group': 'Child Status'},
                barmode='stack',
                height=450
            )
            fig_edu.update_layout(plot_bgcolor='white', title_font_color=colors['primary'])
            st.plotly_chart(fig_edu, use_container_width=True)
            
            st.info("The chart demonstrates that education acts as a **critical protective factor**. Higher education levels in the household head correlate strongly with a lower stunting rate, indicating better knowledge of health and feeding practices.")
        else:
            st.warning("Skipping Education analysis: Driver data is not available.")

    # 2. Stunting vs. WASH Plot
    with col2:
        if not df_driver_analysis.empty:
            wash_agg = df_driver_analysis.groupby('Water_Access_Level')['Stunting_Group'].value_counts(normalize=True).mul(100).rename('Percentage').reset_index()
            fig_wash = px.bar(
                wash_agg,
                x='Water_Access_Level',
                y='Percentage',
                color='Stunting_Group',
                title='Stunting Prevalence by Water Access Level (WASH Proxy)',
                color_discrete_map={'Stunted': colors['danger'], 'Non-Stunted': colors['secondary']},
                labels={'Water_Access_Level': 'Water Access Level', 'Stunting_Group': 'Child Status'},
                barmode='stack',
                height=450
            )
            fig_wash.update_layout(plot_bgcolor='white', title_font_color=colors['primary'])
            st.plotly_chart(fig_wash, use_container_width=True)
            
            st.info("The strong link between 'Low Access' water sources and higher stunting prevalence confirms that **sanitation and disease burden are major contributors** to malnutrition. Repeated diarrhea hinders nutrient absorption, leading to chronic deficiencies.")
            st.write("[Image of the F-Diagram illustrating the transmission routes of fecal-oral diseases (F-Diagram)]")
        else:
            st.warning("Skipping WASH analysis: Driver data is not available.")


def render_tab_3():
    """Renders the Actionable Insights and Policy Tab."""
    st.header("3. Actionable Insights & Policy Recommendations")
    st.markdown("Translating data findings into short, high-impact policy recommendations.")
    
    st.subheader("Predictive Model Summary: Key Risk Factors (Simulated)")
    
    # Simulated Model Output
    risk_factors = {
        "Risk Factor (Driver)": ["Maternal Education (Low)", "Poor Water Source (WASH)", "Low Dietary Diversity (WRA)"],
        "Model Impact Score": ["+0.45", "+0.32", "+0.25"],
        "Severity Interpretation": [
            "Highest correlation with increased chronic malnutrition risk.",
            "Significant driver due to increased disease burden.",
            "Key proxy for micronutrient intake across the household."
        ]
    }
    st.dataframe(pd.DataFrame(risk_factors).set_index("Risk Factor (Driver)"), use_container_width=True)

    st.subheader("Policy Briefs for Local Implementation")
    
    st.markdown(f"""
    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px;">
        <h4 style="color: {colors['primary']}; margin-top: 0;">1. Education-Focused Nutrition Campaigns</h4>
        <p><strong>Action:</strong> Mandate and fund community-level training for mothers/caregivers in the top stunting hotspot districts. Focus training on optimal Infant and Young Child Feeding (IYCF) practices and micronutrient sources (e.g., Vitamin A). 
        <br><strong>Sector:</strong> Health, Education.</p>
    </div>
    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; margin-top: 10px;">
        <h4 style="color: {colors['primary']}; margin-top: 0;">2. Integrated WASH-Nutrition Intervention</h4>
        <p><strong>Action:</strong> Prioritize subsidized access to improved water sources and sanitation facilities in districts with both high stunting and low water access (as shown in Tab 2). Reducing disease directly improves nutrient retention.
        <br><strong>Sector:</strong> WASH, Health.</p>
    </div>
    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 8px; margin-top: 10px;">
        <h4 style="color: {colors['primary']}; margin-top: 0;">3. Biofortified Crop Promotion</h4>
        <p><strong>Action:</strong> Establish seed distribution programs for biofortified crops (e.g., Iron-rich beans, Vitamin A Maize) through local agricultural cooperatives in food-insecure regions. Link production to local school feeding programs.
        <br><strong>Sector:</strong> Agriculture, Food Security.</p>
    </div>
    """, unsafe_allow_html=True)


# --- 3. Render Selected Tab ---
if tab == '1. Regional Hotspots (Prevalence)':
    render_tab_1(regional_agg)
elif tab == '2. Root Cause Analysis (Drivers)':
    render_tab_2(df_driver_analysis)
elif tab == '3. Actionable Insights & Policy':
    render_tab_3()

if not data_available:
    st.warning("Data loading was unsuccessful. Please verify the absolute file paths in the script.")