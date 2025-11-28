import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set page configuration FIRST
st.set_page_config(
    page_title="Ending Hidden Hunger Dashboard",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    
    .stApp > div {
        padding-top: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
        color: white;
        padding: 0;
    }
    
    .css-1d391kg .css-17eq0hr {
        background-color: #1a1a1a;
        padding: 20px;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 0;
        padding-left: 0;
        padding-right: 0;
        max-width: auto;
    }
    
    /* Header styling */
    .dashboard-header {
        background-color: #1a1a1a;
        color: white;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .user-info {
        display: flex;
        align-items: center;
    }
    
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #3498db;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-left: 15px;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 5px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 10px;
    }
    
    .metric-change {
        font-size: 0.8rem;
        display: flex;
        align-items: center;
    }
    
    .positive {
        color: #2ecc71;
    }
    
    .negative {
        color: #e74c3c;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chart-title {
        font-size: 1.1rem;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }
    
    /* Navigation items */
    .nav-item {
        
        align-items: center;
        padding: 12px 20px;
        margin-bottom: 5px;
        cursor: pointer;
        border-radius: 0;
        transition: background-color 0.2s;
    }
    
    .nav-item:hover {
        background-color: #333;
    }
    
    .nav-item.active {
        background-color: #3498db;
    }
    
    .nav-icon {
        margin-right: 15px;
        font-size: 1.2rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 5px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: white;
        border-radius: 5px;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Map container */
    .map-container {
        height: auto;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Custom radio button styling */
    .stRadio > div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0;
        background-color: transparent;
    }
    
    .stRadio > div[role="radiogroup"] > label {
        
        align-items: center;
        padding: 14px 20px !important;
        margin: 0 !important;
        cursor: pointer;
        transition: all 0.3s ease;
        border-radius: 0;
        color: #ffffff !important;
        background-color: transparent !important;
        border-left: 4px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #2a2a2a !important;
        border-left-color: #3498db;
        transform: translateX(4px);
    }
    
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio-checked"] {
        background-color: #3498db !important;
        border-left-color: #2980b9;
        box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
    }
    
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio-checked"]:hover {
        background-color: #2980b9 !important;
        transform: translateX(0);
    }
    
    /* Hide the default radio button circle */
    .stRadio > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Icon styling */
    .nav-icon {
        font-size: 1.3rem;
        margin-right: 15px;
        width: 30px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .stRadio > div[role="radiogroup"] > label:hover .nav-icon {
        transform: scale(1.1);
    }
    
    /* Active state icon animation */
    .stRadio > div[role="radiogroup"] > label[data-baseweb="radio-checked"] .nav-icon {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Text styling */
    .nav-text {
        font-weight: 500;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }
    
    /* Section divider */
    .nav-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #333, transparent);
        margin: 10px 20px;
    }
    
    /* Navigation title */
    .nav-title {
        color: #888;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 10px 20px 5px;
        margin-top: 10px;
    }
    .main-header {
        font-size: 2.2rem;
        color: #1a1a1a;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .css-1d391kg {
        background-color: #043959;
        color: white;
        padding: 0;
    }
    /* Header styling */
    .dashboard-header {
        background-color: #1a1a1a;
        color: white;
        padding: 15px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
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
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.hh_data = None
    st.session_state.child_data = None
    st.session_state.women_data = None
    st.session_state.merged_data = None
    st.session_state.models = {}
    st.session_state.feature_importance = None

# Load data function for all three datasets
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

        # Rename columns to match expected names
        if 'WI_cat' in hh_data.columns:
            hh_data = hh_data.rename(columns={'WI_cat': 'wealth_index'})
            # Standardize wealth index categories
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

        # For child data
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
        if 'S13_07' in child_data.columns:
            child_data = child_data.rename(columns={'S13_07': 'received_vitamin_a'})
            child_data['received_vitamin_a'] = child_data['received_vitamin_a'].map({'Yes': 1, 'No': 0, "Don't know": 0})
        if 'S13_20' in child_data.columns:
            child_data = child_data.rename(columns={'S13_20': 'dietary_diversity'})

        # For women data
        if 'S12_01_4' in women_data.columns:
            women_data = women_data.rename(columns={'S12_01_4': 'age_years'})
        if 'S12_05' in women_data.columns:
            women_data = women_data.rename(columns={'S12_05': 'education_level'})
        if 'S12_03' in women_data.columns:
            women_data = women_data.rename(columns={'S12_03': 'pregnant'})
        if 'S12_11' in women_data.columns:
            women_data = women_data.rename(columns={'S12_11': 'breastfeeding'})
        if 'S12_15' in women_data.columns:
            women_data = women_data.rename(columns={'S12_15': 'received_ifa'})
        if 'S12_12' in women_data.columns:
            women_data = women_data.rename(columns={'S12_12': 'bmi'})
        if 'MDDWLess5' in women_data.columns:
            women_data['dietary_diversity'] = women_data['MDDWLess5'].map({'<5 food groups': 3, '5 food groups or more': 6})
        if 'S12_14_1' in women_data.columns:
            women_data = women_data.rename(columns={'S12_14_1': 'anemic'})

        return hh_data, child_data, women_data

    except Exception as e:
        st.warning(f"⚠️ Could not load data files: {e}")
        st.info("Using sample data for demonstration")
        return create_sample_data()


def create_sample_data():
    """Create comprehensive sample data for all three datasets"""
    np.random.seed(42)
    n_households = 5000
    
    # Household data
    hh_data = pd.DataFrame({
        '___index': [f'HH_{i:04d}' for i in range(n_households)],
        'province': np.random.choice(['Kigali', 'North', 'South', 'East', 'West'], n_households),
        'district': np.random.choice(['Gasabo', 'Nyarugenge', 'Kicukiro', 'Musanze', 'Huye'], n_households),
        'urban_rural': np.random.choice(['Urban', 'Rural'], n_households, p=[0.3, 0.7]),
        'wealth_index': np.random.choice(['Poorest', 'Poor', 'Middle', 'Rich', 'Richest'], n_households),
        'food_consumption_score': np.random.randint(10, 80, n_households),
        'dietary_diversity_score': np.random.randint(0, 10, n_households),
        'coping_strategy_index': np.random.randint(0, 40, n_households),
        'has_improved_water': np.random.choice([0, 1], n_households, p=[0.3, 0.7]),
        'has_improved_sanitation': np.random.choice([0, 1], n_households, p=[0.4, 0.6]),
        'hh_size': np.random.randint(1, 10, n_households),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_households),
        'food_insecure': np.random.choice([0, 1], n_households, p=[0.7, 0.3])
    })
    
    # Child data (multiple children per household)
    child_data = []
    for hh_id in hh_data['___index']:
        n_children = np.random.randint(0, 3)
        for i in range(n_children):
            child_data.append({
                '___index': hh_id,
                'child_id': f"{hh_id}_C{i}",
                'age_months': np.random.randint(6, 60),
                'sex': np.random.choice(['Male', 'Female']),
                'height_cm': np.random.normal(85, 10),
                'weight_kg': np.random.normal(12, 3),
                'stunting_zscore': np.random.normal(-1, 1.5),
                'wasting_zscore': np.random.normal(-0.5, 1),
                'underweight_zscore': np.random.normal(-0.8, 1.2),
                'received_vitamin_a': np.random.choice([0, 1], p=[0.4, 0.6]),
                'dietary_diversity': np.random.randint(0, 8),
                'had_diarrhea': np.random.choice([0, 1], p=[0.7, 0.3]),
                'had_fever': np.random.choice([0, 1], p=[0.6, 0.4])
            })
    child_data = pd.DataFrame(child_data)
    
    # Women data
    women_data = []
    for hh_id in hh_data['___index']:
        n_women = np.random.randint(0, 2)
        for i in range(n_women):
            women_data.append({
                '___index': hh_id,
                'woman_id': f"{hh_id}_W{i}",
                'age_years': np.random.randint(15, 50),
                'education_level': np.random.choice(['None', 'Primary', 'Secondary', 'Higher']),
                'pregnant': np.random.choice([0, 1], p=[0.8, 0.2]),
                'breastfeeding': np.random.choice([0, 1], p=[0.6, 0.4]),
                'received_ifa': np.random.choice([0, 1], p=[0.5, 0.5]),
                'bmi': np.random.normal(22, 3),
                'dietary_diversity': np.random.randint(0, 10),
                'anemic': np.random.choice([0, 1], p=[0.7, 0.3])
            })
    women_data = pd.DataFrame(women_data)
    
    return hh_data, child_data, women_data


def merge_datasets(hh_data, child_data, women_data):
    """Merge all three datasets"""
    # Merge child data with household data
    merged_data = child_data.merge(
        hh_data, 
        on='___index', 
        how='left',
        suffixes=('_child', '_hh')
    )
    
    # Merge with women data
    merged_data = merged_data.merge(
        women_data,
        on='___index',
        how='left',
        suffixes=('', '_women')
    )
    
    # Create malnutrition risk indicators
    if 'stunting' in merged_data.columns:
        merged_data['stunting_risk'] = (merged_data['stunting'] != 'Normal').astype(int)
    if 'wasting' in merged_data.columns:
        merged_data['wasting_risk'] = (merged_data['wasting'] != 'Normal').astype(int)
    if 'underweight' in merged_data.columns:
        merged_data['underweight_risk'] = (merged_data['underweight'] != 'Normal').astype(int)
    
    # Create any malnutrition indicator
    risk_columns = [col for col in ['stunting_risk', 'wasting_risk', 'underweight_risk'] if col in merged_data.columns]
    if risk_columns:
        merged_data['any_malnutrition'] = merged_data[risk_columns].max(axis=1)
    
    return merged_data


def create_sidebar():
    """Create the sidebar navigation"""
    with st.sidebar:
        st.markdown("""
        <div class="dashboard-header" style="flex-direction: column; align-items: flex-start; padding: 15px; margin-bottom: 20px;">
            <div class="logo">IMENA Group</div>
            <div class="user-info">
                <div>
                    <div style="font-weight: bold;">DATA SCIENTIST</div>
                    <div style="font-size: 0.8rem; color: #aaa;">Ending Hidden Hunger</div>
                </div>
                <div class="user-avatar">
                    <span style="color: white; font-weight: bold;">A-M</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data loading status
        st.markdown("### MicronutriSense is a data-driven initiative focused on ending hidden hunger in Rwanda.\
                We use geospatial mapping and predictive analytics to identify micronutrient deficiency hotspots.")
        if not st.session_state.data_loaded:
            st.warning("")
            with st.spinner("Please wait..."):
                hh_data, child_data, women_data = load_all_data()
                merged_data = merge_datasets(hh_data, child_data, women_data)
                
                st.session_state.hh_data = hh_data
                st.session_state.child_data = child_data
                st.session_state.women_data = women_data
                st.session_state.merged_data = merged_data
                st.session_state.data_loaded = True
            
            #st.success("")
        else:
            #st.success("")
            st.markdown("### Main Navigation")
        # # Show data summary
        # if st.session_state.data_loaded:
        #     col1, col2 = st.columns(2)
        #     with col1:
        #         st.metric("Households", len(st.session_state.hh_data))
        #         st.metric("Children", len(st.session_state.child_data))
        #     with col2:
        #         st.metric("Women", len(st.session_state.women_data))
        #         st.metric("Merged Records", len(st.session_state.merged_data))
        
            # st.markdown("---")
        
        # Page selection
        pages = {
            "🏠 Dashboard Overview",
            "🗺️ Micronutrient Hotspots",
            "🔮 Risk Prediction",
            "📊 Root Cause Analysis",
            "👶 Child Nutrition",
            "👩 Women's Health",
            "📋 Policy Recommendations"
        }        
        selected_page = st.radio("", pages, index=0)
        
        
        st.markdown("### About us")
        st.info("""
        **Ending Hidden Hunger Dashboard**
        
        Analyzing micronutrient deficiencies and malnutrition determinants
        """)
        
        # Add data refresh button
        if st.button("🔄 Refresh Data"):
            st.session_state.data_loaded = False
            st.rerun()
        
        return selected_page


# Import page modules
from dashboard_overview2 import dashboard_overview
from micronutrient_hotspots_page import micronutrient_hotspots
from aaapp import risk_prediction
from root_cause_analysis2 import root_cause_analysis
from child_nutrition2 import child_nutrition
from womens_health2 import womens_health
from policy_recommendations2 import policy_recommendations


# Main application
def main():
    # Create sidebar with navigation and load data
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