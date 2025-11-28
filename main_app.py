import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.hh_data = None
    st.session_state.child_data = None
    st.session_state.women_data = None
    st.session_state.merged_data = None


@st.cache_data
def load_all_data():
    """Load all three datasets and merge them"""
    try:
        # Load household data
        hh_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH data.csv")
        
        # Load child data
        child_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv")
        
        # Load women data
        women_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv")
        
        # Clean and rename columns for household data
        if 'WI_cat' in hh_data.columns:
            hh_data = hh_data.rename(columns={'WI_cat': 'wealth_index'})
            hh_data['wealth_index'] = hh_data['wealth_index'].replace({
                'Wealth': 'Rich',
                'Wealthiest': 'Richest'
            })
        if 'FCS' in hh_data.columns:
            hh_data = hh_data.rename(columns={'FCS': 'dietary_diversity_score'})
        if 'UrbanRural' in hh_data.columns:
            hh_data = hh_data.rename(columns={'UrbanRural': 'urban_rural'})
        if 'S0_C_Prov' in hh_data.columns:
            hh_data = hh_data.rename(columns={'S0_C_Prov': 'province'})
        if 'S0_D_Dist' in hh_data.columns:
            hh_data = hh_data.rename(columns={'S0_D_Dist': 'district'})
        
        # Clean child data
        if 'S0_C_Prov' in child_data.columns:
            child_data = child_data.rename(columns={'S0_C_Prov': 'province'})
        if 'S0_D_Dist' in child_data.columns:
            child_data = child_data.rename(columns={'S0_D_Dist': 'district'})
        if 'UrbanRural' in child_data.columns:
            child_data = child_data.rename(columns={'UrbanRural': 'urban_rural'})
        if 'S13_01' in child_data.columns:
            child_data = child_data.rename(columns={'S13_01': 'age_months'})
        if 'S13_04_01' in child_data.columns:
            child_data = child_data.rename(columns={'S13_04_01': 'height_cm'})
        if 'S13_05' in child_data.columns:
            child_data = child_data.rename(columns={'S13_05': 'weight_kg'})
        if 'S13_01_3' in child_data.columns:
            child_data = child_data.rename(columns={'S13_01_3': 'sex'})
        
        # Convert Yes/No columns to binary
        yes_no_cols_child = ['S13_07', 'S13_20']
        for col in yes_no_cols_child:
            if col in child_data.columns:
                child_data[col] = child_data[col].map({
                    'Yes': 1, 'No': 0, "Don't know": 0, 1: 1, 0: 0
                }).fillna(0).astype(int)
        
        if 'S13_07' in child_data.columns:
            child_data = child_data.rename(columns={'S13_07': 'received_vitamin_a'})
        if 'S13_20' in child_data.columns:
            child_data = child_data.rename(columns={'S13_20': 'dietary_diversity'})
        
        # Create z-scores for child anthropometric data
        if 'height_cm' in child_data.columns:
            child_data['height_cm'] = pd.to_numeric(child_data['height_cm'], errors='coerce')
            child_data['stunting_zscore'] = (child_data['height_cm'] - child_data['height_cm'].mean()) / child_data['height_cm'].std()
        
        if 'weight_kg' in child_data.columns:
            child_data['weight_kg'] = pd.to_numeric(child_data['weight_kg'], errors='coerce')
            child_data['wasting_zscore'] = (child_data['weight_kg'] - child_data['weight_kg'].mean()) / child_data['weight_kg'].std()
            child_data['underweight_zscore'] = child_data['wasting_zscore'].copy()
        
        # Clean women data
        if 'S12_01_4' in women_data.columns:
            women_data = women_data.rename(columns={'S12_01_4': 'age_years'})
        if 'S12_05' in women_data.columns:
            women_data = women_data.rename(columns={'S12_05': 'education_level'})
        
        # Convert Yes/No columns to binary
        yes_no_cols_women = ['S12_03', 'S12_11', 'S12_15', 'S12_14_1']
        for col in yes_no_cols_women:
            if col in women_data.columns:
                women_data[col] = women_data[col].map({
                    'Yes': 1, 'No': 0, "Don't know": 0, 1: 1, 0: 0
                }).fillna(0).astype(int)
        
        if 'S12_03' in women_data.columns:
            women_data = women_data.rename(columns={'S12_03': 'pregnant'})
        if 'S12_11' in women_data.columns:
            women_data = women_data.rename(columns={'S12_11': 'breastfeeding'})
        if 'S12_15' in women_data.columns:
            women_data = women_data.rename(columns={'S12_15': 'received_ifa'})
        if 'S12_14_1' in women_data.columns:
            women_data = women_data.rename(columns={'S12_14_1': 'anemic'})
        
        if 'S12_12' in women_data.columns:
            women_data = women_data.rename(columns={'S12_12': 'bmi'})
            women_data['bmi'] = pd.to_numeric(women_data['bmi'], errors='coerce')
        
        if 'MDDWLess5' in women_data.columns:
            women_data['dietary_diversity'] = women_data['MDDWLess5'].map({
                '<5 food groups': 3, 
                '5 food groups or more': 6
            }).fillna(0).astype(int)
        
        # Convert numeric columns
        numeric_cols_hh = ['dietary_diversity_score']
        for col in numeric_cols_hh:
            if col in hh_data.columns:
                hh_data[col] = pd.to_numeric(hh_data[col], errors='coerce').fillna(0)
        
        numeric_cols_child = ['age_months']
        for col in numeric_cols_child:
            if col in child_data.columns:
                child_data[col] = pd.to_numeric(child_data[col], errors='coerce').fillna(0)
        
        numeric_cols_women = ['age_years']
        for col in numeric_cols_women:
            if col in women_data.columns:
                women_data[col] = pd.to_numeric(women_data[col], errors='coerce').fillna(0)
        
        return hh_data, child_data, women_data
    
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None, None


def merge_datasets(hh_data, child_data, women_data):
    """Merge household, child, and women datasets"""
    try:
        # Get the index column (adjust based on your actual column names)
        if '___index' in hh_data.columns and '___index' in child_data.columns:
            merged_data = child_data.merge(hh_data[['___index', 'wealth_index', 'urban_rural', 'province', 'district']], 
                                          on='___index', how='left')
        else:
            merged_data = child_data.copy()
        
        return merged_data
    
    except Exception as e:
        st.error(f"Error merging datasets: {e}")
        return child_data


def set_page_config():
    """Configure Streamlit page"""
    st.set_page_config(
        page_title="Ending Hidden Hunger Dashboard",
        page_icon="🥦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.2rem;
            color: #1a1a1a;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 2px solid #3b82f6;
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1a1a1a;
            margin: 0;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin: 0;
        }
        .section-header {
            font-size: 1.4rem;
            color: #1a1a1a;
            font-weight: 600;
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e5e7eb;
        }
        .chart-container {
            background:#2271b1;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


def create_sidebar():
    """Create sidebar navigation"""
    st.sidebar.markdown("""
    <style>
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 1rem;
            text-align: center;
        }
    </style>
    <div class="sidebar-header">🥦 Hidden Hunger Dashboard</div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # # Data loading section
    st.sidebar.markdown("")
    
    # Auto-load data on first run
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Auto-loading data..."):
            hh_data, child_data, women_data = load_all_data()
            
            if hh_data is not None and child_data is not None and women_data is not None:
                st.session_state.hh_data = hh_data
                st.session_state.child_data = child_data
                st.session_state.women_data = women_data
                st.session_state.merged_data = merge_datasets(hh_data, child_data, women_data)
                st.session_state.data_loaded = True
                st.sidebar.success("")
                # st.sidebar.success("✅ Data auto-loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to auto-load data")
    
    # # Manual reload button
  
    # Navigation menu
    st.sidebar.markdown("### 📖 Navigation")
    
    pages = [
        "🏠 Dashboard Overview",
        "🗺️ Micronutrient Hotspots",
        "🔮 Risk Prediction",
        "📊 Root Cause Analysis",
        "👶 Child Nutrition",
        "👩 Women's Health",
        "📋 Policy Recommendations"
    ]
    
    selected_page = st.sidebar.radio("Select Page", pages, label_visibility="collapsed")
    
    st.sidebar.markdown("")
    
    # Settings
    st.sidebar.markdown("### ⚙️ Settings")
    
    st.sidebar.markdown("""
    **About:**
    - Dashboard Version: 1.0
    - Last Updated: November 2025
    - Focus: Ending Hidden Hunger
    
    **Data Status:**
    - Auto-load: Enabled ✅
    - Cache: Active
    """)
    
    return selected_page


# Initialize page configuration
set_page_config()

# Import page modules
from dashboard_overview1 import dashboard_overview
from micronutrient_hotspots_page import micronutrient_hotspots
from risk_prediction import risk_prediction
from root_cause_analysis import root_cause_analysis
from child_nutrition import child_nutrition
from womens_health import womens_health
from policy_recommendations import policy_recommendations


def main():
    """Main application entry point"""
    # Create sidebar with navigation (includes auto-load)
    page = create_sidebar()
    
    # Display selected page
    if page == "🏠 Dashboard Overview":
        dashboard_overview()
    elif page == "🗺️ Micronutrient Hotspots":
        micronutrient_hotspots()
    elif page == "🔮 Risk Prediction":
        risk_prediction()
    elif page == "📊 Root Cause Analysis":
        root_cause_analysis()
    elif page == "👶 Child Nutrition":
        child_nutrition()
    elif page == "👩 Women's Health":
        womens_health()
    elif page == "📋 Policy Recommendations":
        policy_recommendations()


# Run the application
if __name__ == "__main__":
    main()