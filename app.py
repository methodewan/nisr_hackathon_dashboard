import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG MUST BE FIRST ====================
st.set_page_config(
    page_title="Ending Hidden Hunger Dashboard",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INITIALIZATION ====================
# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.hh_data = None
    st.session_state.child_data = None
    st.session_state.women_data = None
    st.session_state.merged_data = None


@st.cache_data
def load_all_data():
    """Load all three datasets"""
    try:
        hh_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH data.csv")
        child_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv")
        women_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv")
        
        return hh_data, child_data, women_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def merge_datasets(hh_data, child_data, women_data):
    """Merge datasets"""
    try:
        if '___index' in hh_data.columns and '___index' in child_data.columns:
            merged_data = child_data.merge(
                hh_data[['___index', 'wealth_index', 'urban_rural', 'province', 'district']], 
                on='___index', how='left'
            )
        else:
            merged_data = child_data.copy()
        return merged_data
    except Exception as e:
        st.error(f"Error merging datasets: {e}")
        return child_data


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
    
    # Data loading
    st.sidebar.markdown("### 📥 Data Management")
    
    if not st.session_state.data_loaded:
        with st.spinner("🔄 Auto-loading data..."):
            hh_data, child_data, women_data = load_all_data()
            
            if hh_data is not None and child_data is not None and women_data is not None:
                st.session_state.hh_data = hh_data
                st.session_state.child_data = child_data
                st.session_state.women_data = women_data
                st.session_state.merged_data = merge_datasets(hh_data, child_data, women_data)
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data auto-loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to auto-load data")
    else:
        st.sidebar.success("✅ Data is loaded")
        
        st.sidebar.markdown("#### 📊 Data Summary")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.session_state.child_data is not None:
                st.sidebar.metric("Children (6-59m)", f"{len(st.session_state.child_data):,}")
            if st.session_state.women_data is not None:
                st.sidebar.metric("Women (15-49y)", f"{len(st.session_state.women_data):,}")
        
        with col2:
            if st.session_state.hh_data is not None:
                st.sidebar.metric("Households", f"{len(st.session_state.hh_data):,}")
    
    st.sidebar.markdown("---")
    
    # Navigation
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
    
    selected_page = st.sidebar.radio("Select Page:", pages)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About:**
    - Dashboard Version: 1.0
    - Last Updated: November 2025
    - Focus: Ending Hidden Hunger
    """)
    
    return selected_page


# Import page modules
from dashboard_overview import dashboard_overview
from micronutrient_hotspots_page import micronutrient_hotspots
from risk_prediction1 import risk_prediction
from root_cause_analysis import root_cause_analysis
from child_nutrition import child_nutrition
from womens_health import womens_health
from policy_recommendations import policy_recommendations


def main():
    """Main application"""
    page = create_sidebar()
    
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


if __name__ == "__main__":
    main()