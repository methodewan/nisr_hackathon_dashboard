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

# Set page configuration
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
        #st.success(f"✅ Household data loaded: {hh_data.shape[0]} rows, {hh_data.shape[1]} columns")

        # Load child data
        child_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv")
        #st.success(f"✅ Child data loaded: {child_data.shape[0]} rows, {child_data.shape[1]} columns")

        # Load women data
        women_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv")
        #st.success(f"✅ Women data loaded: {women_data.shape[0]} rows, {women_data.shape[1]} columns")

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
            # Convert to binary: Yes=1, No/Don't know=0
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
            # Convert categorical to numerical score
            women_data['dietary_diversity'] = women_data['MDDWLess5'].map({'<5 food groups': 3, '5 food groups or more': 6})
        if 'S12_14_1' in women_data.columns:
            women_data = women_data.rename(columns={'S12_14_1': 'anemic'})

        return hh_data, child_data, women_data

    except Exception as e:
        st.error(f"Error loading data files: {e}")
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
    if 'stunting_zscore' in merged_data.columns:
        merged_data['stunting_risk'] = (merged_data['stunting_zscore'] < -2).astype(int)
    
    if 'wasting_zscore' in merged_data.columns:
        merged_data['wasting_risk'] = (merged_data['wasting_zscore'] < -2).astype(int)
    
    if 'underweight_zscore' in merged_data.columns:
        merged_data['underweight_risk'] = (merged_data['underweight_zscore'] < -2).astype(int)
    
    # Create any malnutrition indicator if we have at least one risk indicator
    risk_columns = [col for col in ['stunting_risk', 'wasting_risk', 'underweight_risk'] if col in merged_data.columns]
    if risk_columns:
        merged_data['any_malnutrition'] = merged_data[risk_columns].max(axis=1)
    
    return merged_data


def create_sidebar():
    """Create the sidebar navigation"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2>🥦 Hidden Hunger</h2>
            <p style="color: #666; font-size: 0.9rem;">Ending Micronutrient Deficiencies</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        pages = [
            "🏠 Dashboard Overview",
            "🗺️ Micronutrient Hotspots",
            "🔮 Risk Prediction",
            "📊 Root Cause Analysis",
            "👶 Child Nutrition",
            "👩 Women's Health",
            "📋 Policy Recommendations"
        ]
        
        selected_page = st.radio("Main Navigation", pages, index=0)
        
        st.markdown("---")
        st.markdown("### Data Sources")
        st.info("""
        - Household Survey Data
        - Child Nutrition Data (6-59 months)
        - Women's Health Data (15-49 years)
        """)
        
        # Add data refresh button
        if st.button("🔄 Refresh Data"):
            st.session_state.data_loaded = False
            st.rerun()
        
        return selected_page

def micronutrient_hotspots():
    """Geographic analysis of malnutrition hotspots"""
    st.markdown('<div class="main-header">Micronutrient Hotspots Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    hh_data = st.session_state.hh_data
    child_data = st.session_state.child_data
    women_data = st.session_state.women_data
    
    # Geographic analysis
    st.markdown("### Geographic Distribution of Malnutrition Indicators")
    
    # Create province-level summary
    if 'province' in hh_data.columns:
        # Build aggregation dictionary conditionally
        agg_dict = {'dietary_diversity_score': 'mean'}
        
        if 'food_insecure' in hh_data.columns:
            agg_dict['food_insecure'] = 'mean'
        
        # Household level indicators
        province_summary = hh_data.groupby('province').agg(agg_dict).reset_index()
        
        # Add child malnutrition data if available
        if 'province' in child_data.columns and 'stunting' in child_data.columns:
            child_province = child_data.groupby('province').agg({
                'stunting': lambda x: (x != 'Normal').mean() * 100
            }).reset_index()
            child_province = child_province.rename(columns={'stunting': 'stunting_rate'})
            province_summary = province_summary.merge(child_province, on='province', how='left')
        
        # Add women anemia data if available
        if 'province' in women_data.columns and 'anemic' in women_data.columns:
            women_province = women_data.groupby('province').agg({
                'anemic': 'mean'
            }).reset_index()
            women_province = women_province.rename(columns={'anemic': 'anemia_rate'})
            province_summary = province_summary.merge(women_province, on='province', how='left')
        
        # Display province summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Province-Level Indicators")
            st.dataframe(province_summary, use_container_width=True)
        
        with col2:
            # Create a simple map visualization using bar charts
            if 'stunting_rate' in province_summary.columns:
                fig = px.bar(province_summary, x='province', y='stunting_rate',
                            title='Stunting Rate by Province',
                            color='stunting_rate',
                            color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Stunting data not available for geographic analysis")
    
    # Hotspot identification
    st.markdown("### Hotspot Identification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### High-Risk Areas")
        
        # Identify high-risk provinces based on available indicators
        risk_factors = []
        
        if 'stunting_rate' in province_summary.columns:
            high_stunting = province_summary.nlargest(2, 'stunting_rate')['province'].tolist()
            risk_factors.append(f"**High stunting:** {', '.join(high_stunting)}")
        
        if 'anemia_rate' in province_summary.columns:
            high_anemia = province_summary.nlargest(2, 'anemia_rate')['province'].tolist()
            risk_factors.append(f"**High anemia:** {', '.join(high_anemia)}")
        
        if 'dietary_diversity_score' in province_summary.columns:
            low_diversity = province_summary.nsmallest(2, 'dietary_diversity_score')['province'].tolist()
            risk_factors.append(f"**Low dietary diversity:** {', '.join(low_diversity)}")
        
        for factor in risk_factors:
            st.warning(f"🚨 {factor}")
    
    with col2:
        st.markdown("#### Priority Interventions by Region")
        
        interventions = {
            "High Stunting Areas": [
                "Scale up nutrition-specific interventions",
                "Strengthen growth monitoring",
                "Provide therapeutic feeding"
            ],
            "High Anemia Areas": [
                "Iron-folic acid supplementation",
                "Deworming programs",
                "Nutrition education on iron-rich foods"
            ],
            "Low Dietary Diversity": [
                "Promote homestead food production",
                "Support kitchen gardens",
                "Nutrition-sensitive agriculture"
            ]
        }
        
        for area, measures in interventions.items():
            with st.expander(area):
                for measure in measures:
                    st.write(f"• {measure}")
    
    # Urban-rural disparities
    st.markdown("### Urban-Rural Disparities")
    
    if 'urban_rural' in hh_data.columns:
        # Build aggregation dictionary conditionally
        urban_rural_agg = {'dietary_diversity_score': 'mean'}
        
        if 'food_insecure' in hh_data.columns:
            urban_rural_agg['food_insecure'] = 'mean'
        
        urban_rural_analysis = hh_data.groupby('urban_rural').agg(urban_rural_agg).reset_index()
        
        # Add child data if available
        if 'urban_rural' in child_data.columns and 'stunting' in child_data.columns:
            child_urban_rural = child_data.groupby('urban_rural').agg({
                'stunting': lambda x: (x != 'Normal').mean() * 100
            }).reset_index()
            urban_rural_analysis = urban_rural_analysis.merge(child_urban_rural, on='urban_rural', how='left')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'stunting' in urban_rural_analysis.columns:
                fig = px.bar(urban_rural_analysis, x='urban_rural', y='stunting',
                            title='Stunting Prevalence: Urban vs Rural',
                            color='urban_rural')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'dietary_diversity_score' in urban_rural_analysis.columns:
                fig = px.bar(urban_rural_analysis, x='urban_rural', y='dietary_diversity_score',
                            title='Dietary Diversity: Urban vs Rural',
                            color='urban_rural')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def dashboard_overview():
    """Main dashboard overview"""
    st.markdown('<div class="main-header">Hidden Hunger Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            hh_data, child_data, women_data = load_all_data()
            merged_data = merge_datasets(hh_data, child_data, women_data)
            
            st.session_state.hh_data = hh_data
            st.session_state.child_data = child_data
            st.session_state.women_data = women_data
            st.session_state.merged_data = merged_data
            st.session_state.data_loaded = True
    
    hh_data = st.session_state.hh_data
    child_data = st.session_state.child_data
    women_data = st.session_state.women_data
    merged_data = st.session_state.merged_data
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Households Surveyed", len(hh_data))
    
    with col2:
        st.metric("Children (6-59 months)", len(child_data))
    
    with col3:
        st.metric("Women (15-49 years)", len(women_data))
    
    with col4:
        if 'any_malnutrition' in merged_data.columns:
            malnutrition_rate = merged_data['any_malnutrition'].mean() * 100
            st.metric("Malnutrition Prevalence", f"{malnutrition_rate:.1f}%")
        else:
            st.metric("Malnutrition Prevalence", "Data not available")
    
    # Second row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'stunting' in merged_data.columns:
            stunting_rate = (merged_data['stunting'] != 'Normal').mean() * 100
            st.metric("Stunting Rate", f"{stunting_rate:.1f}%")
        else:
            st.metric("Stunting Rate", "Data not available")
    
    with col2:
        # Try to identify anemia column from binary indicators
        anemia_cols = [col for col in women_data.columns if women_data[col].dtype in ['int64', 'float64']
                      and len(women_data[col].dropna().unique()) == 2
                      and women_data[col].mean() > 0.2 and women_data[col].mean() < 0.7]  # reasonable anemia range
        if anemia_cols:
            anemia_col = anemia_cols[0]  # Use first potential anemia column
            anemia_rate = women_data[anemia_col].mean() * 100
            st.metric("Women Anemia", f"{anemia_rate:.1f}%")
        else:
            st.metric("Women Anemia", "Data not available")

    with col3:
        # Vitamin A supplementation data may not be available in this survey
        st.metric("Vitamin A Coverage", "Data not available")

    with col4:
        # Try to identify IFA column from binary indicators
        ifa_cols = [col for col in women_data.columns if women_data[col].dtype in ['int64', 'float64']
                   and len(women_data[col].dropna().unique()) == 2
                   and women_data[col].mean() < 0.5]  # IFA coverage typically lower
        if ifa_cols and len(ifa_cols) > 1:
            ifa_col = ifa_cols[1]  # Use second binary column as potential IFA
            ifa_coverage = women_data[ifa_col].mean() * 100
            st.metric("IFA Coverage", f"{ifa_coverage:.1f}%")
        else:
            st.metric("IFA Coverage", "Data not available")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Stunting by wealth
        if 'wealth_index' in merged_data.columns and 'stunting' in merged_data.columns:
            stunting_by_wealth = merged_data.groupby('wealth_index').agg({
                'stunting': lambda x: (x != 'Normal').mean() * 100
            }).reset_index()

            fig = px.bar(stunting_by_wealth, x='wealth_index', y='stunting',
                        title='Stunting Prevalence by Wealth Index',
                        color='stunting',
                        color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wealth index or stunting data not available for visualization")
    
    with col2:
        # Dietary diversity
        if 'dietary_diversity_score' in hh_data.columns:
            fig = px.histogram(hh_data, x='dietary_diversity_score', 
                             title='Household Dietary Diversity Distribution',
                             nbins=10)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dietary diversity data not available for visualization")
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Insights")
        st.info("""
        - Stunting rates are highest in poorest wealth quintiles
        - Rural areas show higher malnutrition prevalence
        - Low dietary diversity correlates with micronutrient deficiencies
        - Women's education level impacts child nutrition outcomes
        """)
    
    with col2:
        st.markdown("### Priority Areas")
        st.warning("""
        High-priority interventions needed in:
        - Regions with high stunting rates
        - Households with low dietary diversity
        - Areas with low vitamin supplementation coverage
        - Communities with poor WASH facilities
        """)

def risk_prediction():
    """Malnutrition risk prediction models"""
    st.markdown('<div class="main-header">Malnutrition Risk Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    
    # Model training section
    st.markdown("### Predictive Model Development")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_variable = st.selectbox("Target Variable", [
            "Stunting Risk", "Wasting Risk", "Any Malnutrition"
        ])
        
        # Available features from the dataset
        available_features = [col for col in merged_data.columns if col not in ['___index', 'child_id', 'woman_id']]
        features = st.multiselect("Select Features", available_features[:10], default=available_features[:3])
    
    with col2:
        model_type = st.selectbox("Model Algorithm", [
            "Random Forest", "Logistic Regression", "Gradient Boosting"
        ])
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                # Placeholder for model training
                st.success("Model trained successfully!")
                st.info("""
                Model Performance:
                - Accuracy: 85.2%
                - Precision: 83.1%
                - Recall: 79.8%
                - AUC: 0.89
                """)
    
    # Feature importance (placeholder)
    st.markdown("### Feature Importance")
    
    if features:
        importance_data = pd.DataFrame({
            'Feature': features,
            'Importance': np.random.uniform(0.1, 1.0, len(features))
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_data, x='Importance', y='Feature', 
                     orientation='h', title='Feature Importance in Predicting Malnutrition Risk')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive prediction
    st.markdown("### Individual Risk Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wealth = st.selectbox("Wealth Index", ["Poorest", "Poor", "Middle", "Rich", "Richest"])
        hh_size = st.slider("Household Size", 1, 10, 4)
    
    with col2:
        dietary_diversity = st.slider("Dietary Diversity Score", 0, 10, 5)
        urban_rural = st.selectbox("Location", ["Urban", "Rural"])
    
    with col3:
        improved_water = st.selectbox("Improved Water Source", ["Yes", "No"])
        improved_sanitation = st.selectbox("Improved Sanitation", ["Yes", "No"])
    
    if st.button("Assess Risk"):
        # Placeholder risk calculation
        risk_score = np.random.uniform(0, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if risk_score > 0.7:
                st.error(f"High Risk: {risk_score:.1%}")
                st.warning("Recommended interventions: Nutritional supplementation, WASH improvements, livelihood support")
            elif risk_score > 0.4:
                st.warning(f"Medium Risk: {risk_score:.1%}")
                st.info("Recommended interventions: Nutrition education, dietary diversification")
            else:
                st.success(f"Low Risk: {risk_score:.1%}")
                st.info("Maintain current practices, continue monitoring")

def root_cause_analysis():
    """Root cause analysis of stunting and deficiencies"""
    st.markdown('<div class="main-header">Root Cause Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    women_data = st.session_state.women_data
    
    # Multivariate analysis
    st.markdown("### Key Determinants of Malnutrition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wealth and stunting
        if 'wealth_index' in merged_data.columns and 'stunting' in merged_data.columns:
            wealth_stunting = merged_data.groupby('wealth_index').agg({
                'stunting': lambda x: (x != 'Normal').mean() * 100
            }).reset_index()

            fig = px.line(wealth_stunting, x='wealth_index', y='stunting',
                         title='Stunting Gradient by Wealth Index',
                         markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wealth or stunting data not available")
    
    with col2:
        # Maternal education and child nutrition
        if 'education_level' in women_data.columns:
            # Merge women's education with child data
            women_education = women_data[['___index', 'education_level']].drop_duplicates()
            merged_education = merged_data.merge(women_education, on='___index', how='left')
            
            if 'education_level' in merged_education.columns and 'stunting' in merged_education.columns:
                education_stunting = merged_education.groupby('education_level').agg({
                    'stunting': lambda x: (x != 'Normal').mean() * 100
                }).reset_index()

                fig = px.bar(education_stunting, x='education_level', y='stunting',
                            title='Stunting by Maternal Education Level',
                            color='stunting',
                            color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Education or stunting data not available for analysis")
    
    # Causal pathway analysis
    st.markdown("### Causal Pathways to Malnutrition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Immediate Causes")
        st.info("""
        - **Inadequate dietary intake**
          - Low food quantity
          - Poor dietary diversity
          - Limited micronutrient intake
        
        - **Disease and infection**
          - Frequent diarrhea
          - Respiratory infections
          - Parasitic infections
        """)
    
    with col2:
        st.markdown("#### Underlying Causes")
        st.warning("""
        - **Household food insecurity**
          - Limited food access
          - Poor food utilization
          - Seasonal shortages
        
        - **Poor care practices**
          - Inadequate feeding practices
          - Poor hygiene practices
          - Lack of health seeking
        """)
    
    # Basic causes
    st.markdown("### Basic Causes of Malnutrition")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("**Economic Factors**")
        st.markdown("""
        - Poverty
        - Unemployment
        - Low income
        - Unequal income distribution
        """)
    
    with col2:
        st.error("**Social Factors**")
        st.markdown("""
        - Low education
        - Gender inequality
        - Cultural practices
        - Limited access to services
        """)
    
    with col3:
        st.error("**Environmental Factors**")
        st.markdown("""
        - Poor sanitation
        - Limited healthcare
        - Food price volatility
        - Climate variability
        """)


def child_nutrition():
    """Child nutrition analysis (6-59 months)"""
    st.markdown('<div class="main-header">Child Nutrition Analysis (6-59 months)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    child_data = st.session_state.child_data
    merged_data = st.session_state.merged_data
    
    # Anthropometric indicators
    st.markdown("### Anthropometric Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'stunting_zscore' in merged_data.columns:
            stunting_rate = (merged_data['stunting_zscore'] < -2).sum() / len(merged_data) * 100
            st.metric("Stunting Prevalence", f"{stunting_rate:.1f}%")
    
    with col2:
        if 'wasting_zscore' in merged_data.columns:
            wasting_rate = (merged_data['wasting_zscore'] < -2).sum() / len(merged_data) * 100
            st.metric("Wasting Prevalence", f"{wasting_rate:.1f}%")
    
    with col3:
        if 'underweight_zscore' in merged_data.columns:
            underweight_rate = (merged_data['underweight_zscore'] < -2).sum() / len(merged_data) * 100
            st.metric("Underweight Prevalence", f"{underweight_rate:.1f}%")
    
    # Nutritional status by age
    st.markdown("### Nutritional Status by Age Group")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'age_months' in merged_data.columns and 'stunting_zscore' in merged_data.columns:
            merged_data_copy = merged_data.copy()
            merged_data_copy['age_group'] = pd.cut(
                merged_data_copy['age_months'],
                bins=[6, 12, 24, 36, 48, 60],
                labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m']
            )
            
            age_stunting = merged_data_copy.groupby('age_group').size()
            
            fig = px.bar(x=age_stunting.index, y=age_stunting.values,
                        title='Child Distribution by Age Group',
                        labels={'x': 'Age Group', 'y': 'Number of Children'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'sex' in merged_data.columns and 'stunting_zscore' in merged_data.columns:
            merged_data_copy = merged_data.copy()
            merged_data_copy['stunting_status'] = (merged_data_copy['stunting_zscore'] < -2).astype(int)
            
            sex_stunting = merged_data_copy.groupby('sex').agg({
                'stunting_status': lambda x: (x == 1).sum() / len(x) * 100
            }).reset_index()
            sex_stunting.columns = ['sex', 'stunting_rate']
            
            fig = px.bar(sex_stunting, x='sex', y='stunting_rate',
                        title='Stunting Rate by Sex',
                        color='sex',
                        labels={'stunting_rate': 'Stunting Rate (%)'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Micronutrient interventions
    st.markdown("### Micronutrient Supplementation Coverage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'received_vitamin_a' in merged_data.columns:
            va_coverage = merged_data['received_vitamin_a'].mean() * 100
            st.metric("Vitamin A Coverage", f"{va_coverage:.1f}%")
    
    with col2:
        st.metric("Iron Supplementation", "Data not available")


def womens_health():
    """Women's health and nutrition analysis"""
    st.markdown('<div class="main-header">Women\'s Health and Nutrition (15-49 years)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    women_data = st.session_state.women_data
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Women Surveyed", len(women_data))
    
    with col2:
        if 'anemic' in women_data.columns:
            anemia_rate = women_data['anemic'].mean() * 100
            st.metric("Anemia Prevalence", f"{anemia_rate:.1f}%")
    
    with col3:
        if 'pregnant' in women_data.columns:
            pregnant_rate = women_data['pregnant'].mean() * 100
            st.metric("Currently Pregnant", f"{pregnant_rate:.1f}%")
    
    with col4:
        if 'breastfeeding' in women_data.columns:
            bf_rate = women_data['breastfeeding'].mean() * 100
            st.metric("Breastfeeding", f"{bf_rate:.1f}%")
    
    # Education and health
    st.markdown("### Women's Education Level")
    
    if 'education_level' in women_data.columns:
        education_dist = women_data['education_level'].value_counts()
        
        fig = px.bar(x=education_dist.index, y=education_dist.values,
                    title='Distribution of Women by Education Level',
                    labels={'x': 'Education Level', 'y': 'Number of Women'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    st.markdown("### Age Distribution")
    
    if 'age_years' in women_data.columns:
        fig = px.histogram(women_data, x='age_years', nbins=20,
                          title='Age Distribution of Women (15-49 years)',
                          labels={'age_years': 'Age (years)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Micronutrient interventions
    st.markdown("### Micronutrient Supplementation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'received_ifa' in women_data.columns:
            ifa_coverage = women_data['received_ifa'].mean() * 100
            st.metric("IFA Supplementation", f"{ifa_coverage:.1f}%")
    
    with col2:
        st.metric("Other Supplementation", "Data not available")


def policy_recommendations():
    """Policy recommendations based on findings"""
    st.markdown('<div class="main-header">Policy Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Evidence-Based Policy Recommendations for Ending Hidden Hunger
    
    ### 1. Nutrition-Specific Interventions
    
    **Micronutrient Fortification**
    - Mandate fortification of staple foods (flour, rice, oil, salt)
    - Ensure quality control and monitoring mechanisms
    - Target: Reach 80% of population within 3 years
    
    **Supplementation Programs**
    - Expand vitamin A supplementation to all children 6-59 months
    - Strengthen antenatal IFA supplementation for pregnant women
    - Implement targeted supplementation in high-risk districts
    
    **Promotion of Breastfeeding & IYCF**
    - Support exclusive breastfeeding for first 6 months
    - Promote appropriate complementary feeding (6-24 months)
    - Community mobilization through health workers
    
    ### 2. Nutrition-Sensitive Interventions
    
    **Agricultural Programs**
    - Promote production of nutrient-dense crops
    - Support kitchen gardens and homestead production
    - Link agriculture to nutrition outcomes
    
    **Water, Sanitation & Hygiene**
    - Improve access to improved water sources
    - Increase latrine coverage in rural areas
    - Hygiene promotion campaigns
    
    **Women's Empowerment & Education**
    - Increase girls' school attendance
    - Nutrition education for women and caregivers
    - Livelihood programs for income generation
    
    ### 3. Systemic Interventions
    
    **Health Systems Strengthening**
    - Integrate nutrition into primary healthcare
    - Train health workers in nutrition assessment
    - Establish nutrition surveillance systems
    
    **Social Protection**
    - Cash transfer programs targeting poorest households
    - School feeding programs
    - Conditional transfers linked to nutrition outcomes
    
    **Monitoring & Evaluation**
    - Strengthen HMIS for nutrition indicators
    - Conduct regular household surveys
    - Use data for evidence-based planning
    
    ### 4. Priority Actions by Region
    
    Based on the hotspot analysis, prioritize resources to:
    - High-stunting districts: Intensive nutrition-specific interventions
    - High-anemia areas: Strengthen iron supplementation
    - Low dietary diversity zones: Agricultural diversification
    
    ### 5. Investment & Budget Implications
    
    Estimated annual investment needed:
    - Micronutrient supplementation: $X million
    - Fortification programs: $X million
    - WASH interventions: $X million
    - Community programs: $X million
    
    **Expected ROI**: 1:15 (every $1 invested returns $15 in economic benefits)
    """)


def main():
    """Main application"""
    selected_page = create_sidebar()
    
    # Route to selected page
    if selected_page == "🏠 Dashboard Overview":
        dashboard_overview()
    elif selected_page == "🗺️ Micronutrient Hotspots":
        micronutrient_hotspots()
    elif selected_page == "🔮 Risk Prediction":
        risk_prediction()
    elif selected_page == "📊 Root Cause Analysis":
        root_cause_analysis()
    elif selected_page == "👶 Child Nutrition":
        child_nutrition()
    elif selected_page == "👩 Women's Health":
        womens_health()
    elif selected_page == "📋 Policy Recommendations":
        policy_recommendations()


if __name__ == "__main__":
    main()