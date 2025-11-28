import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Hidden Hunger Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class HiddenHungerAnalyzer:
    def __init__(self):
        self.hh_data = None
        self.child_data = None
        self.women_data = None
        self.merged_data = None
        
    def load_data(self):
        """Load and preprocess the datasets"""
        try:
            # Load datasets - CHANGED: using read_csv instead of read_stata for CSV files
            self.hh_data = pd.read_csv('Microdata1111/Microdata/csvfile/CFSVA2024_HH data.csv')
            self.child_data = pd.read_csv('Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv')
            self.women_data = pd.read_csv('Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv')
            
            # Basic preprocessing
            self._preprocess_data()
            self._merge_datasets()
            
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def _preprocess_data(self):
        """Preprocess and clean the data"""
        # Household data preprocessing
        # Check for food consumption score with different possible column names
        food_score_cols = [col for col in self.hh_data.columns if 'food' in col.lower() and 'score' in col.lower()]
        if food_score_cols:
            food_col = food_score_cols[0]
            self.hh_data['food_insecurity_level'] = pd.cut(
                self.hh_data[food_col],
                bins=[0, 21, 35, 100],
                labels=['Poor', 'Borderline', 'Acceptable']
            )
        
        # Child data preprocessing
        # Check for height/age z-score with different possible column names
        haz_cols = [col for col in self.child_data.columns if any(x in col.lower() for x in ['haz', 'height_age', 'zscore'])]
        if haz_cols:
            haz_col = haz_cols[0]
            self.child_data['stunting_status'] = pd.cut(
                self.child_data[haz_col],
                bins=[-float('inf'), -3, -2, float('inf')],
                labels=['Severe Stunting', 'Moderate Stunting', 'Normal']
            )
        
        # Women data preprocessing
        # Check for dietary diversity with different possible column names
        dd_cols = [col for col in self.women_data.columns if 'diet' in col.lower() and 'divers' in col.lower()]
        if dd_cols:
            dd_col = dd_cols[0]
            self.women_data['diet_diversity_category'] = pd.cut(
                self.women_data[dd_col],
                bins=[0, 3, 5, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
    
    def _merge_datasets(self):
        """Merge datasets for comprehensive analysis"""
        try:
            # Find common ID columns (more flexible approach)
            hh_id_cols = [col for col in self.hh_data.columns if 'id' in col.lower() or 'hh' in col.lower()]
            child_id_cols = [col for col in self.child_data.columns if 'id' in col.lower() or 'hh' in col.lower()]
            women_id_cols = [col for col in self.women_data.columns if 'id' in col.lower() or 'hh' in col.lower()]
            
            if hh_id_cols and child_id_cols:
                hh_id = hh_id_cols[0]
                child_id = child_id_cols[0]
                
                self.merged_data = self.child_data.merge(
                    self.hh_data, 
                    left_on=child_id, 
                    right_on=hh_id,
                    how='left', 
                    suffixes=('_child', '_hh')
                )
                
                if women_id_cols and self.merged_data is not None:
                    women_id = women_id_cols[0]
                    self.merged_data = self.merged_data.merge(
                        self.women_data,
                        left_on=child_id,
                        right_on=women_id,
                        how='left',
                        suffixes=('', '_women')
                    )
        except Exception as e:
            st.warning(f"Could not merge all datasets: {e}")

def create_hotspot_map(analyzer):
    """Create geospatial hotspot maps for malnutrition"""
    st.markdown('<div class="section-header">🌍 Malnutrition Hotspot Mapping</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create sample hotspot data (replace with actual geographic data)
        if analyzer.hh_data is not None:
            # If region data exists, use it
            region_cols = [col for col in analyzer.hh_data.columns if any(x in col.lower() for x in ['region', 'district', 'area', 'zone'])]
            if region_cols:
                region_col = region_cols[0]
                numeric_cols = analyzer.hh_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    region_stats = analyzer.hh_data.groupby(region_col).agg({
                        numeric_cols[0]: 'mean'
                    }).reset_index()
                    
                    fig = px.bar(
                        region_stats,
                        x=region_col,
                        y=numeric_cols[0],
                        title=f"Average {numeric_cols[0]} by {region_col.title()}",
                        color=numeric_cols[0]
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No geographic data found. Please ensure your dataset contains region/area information for mapping.")
    
    with col2:
        if analyzer.child_data is not None and 'stunting_status' in analyzer.child_data.columns:
            stunting_summary = analyzer.child_data['stunting_status'].value_counts(normalize=True)
            fig = px.pie(
                values=stunting_summary.values,
                names=stunting_summary.index,
                title="Stunting Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

def build_predictive_models(analyzer):
    """Build predictive models for malnutrition risk"""
    st.markdown('<div class="section-header">🔮 Predictive Models for Malnutrition Risk</div>', unsafe_allow_html=True)
    
    if analyzer.merged_data is not None:
        # Prepare features for prediction
        feature_columns = []
        potential_features = [
            'household_size', 'food_consumption_score', 'dietary_diversity',
            'income_level', 'water_source', 'sanitation_facility',
            'mother_education', 'asset_index'
        ]
        
        # Select available features
        available_features = []
        for feature in potential_features:
            # Check for exact match or similar column names
            matching_cols = [col for col in analyzer.merged_data.columns if feature in col.lower()]
            if matching_cols:
                available_features.append(matching_cols[0])
        
        if available_features and 'stunting_status' in analyzer.merged_data.columns:
            # Prepare data
            model_data = analyzer.merged_data[available_features + ['stunting_status']].dropna()
            model_data['stunting_binary'] = model_data['stunting_status'].isin(['Severe Stunting', 'Moderate Stunting'])
            
            X = pd.get_dummies(model_data[available_features], drop_first=True)
            y = model_data['stunting_binary']
            
            if len(X) > 0 and len(y) > 0:
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    accuracy = accuracy_score(y_test, model.predict(X_test))
                    st.metric("Model Accuracy", f"{accuracy:.2%}")
                
                with col2:
                    feature_importance = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    st.write("Top Predictive Features:")
                    for i, row in feature_importance.head(5).iterrows():
                        st.write(f"• {row['feature']}: {row['importance']:.3f}")
                
                with col3:
                    st.write("Common Risk Factors:")
                    risk_factors = ["Poor sanitation", "Low dietary diversity", "Maternal education", "Household income"]
                    for factor in risk_factors:
                        st.write(f"• {factor}")
            else:
                st.warning("Insufficient data for model training")
        else:
            st.warning("Required features not available for predictive modeling")
    else:
        st.warning("Merged dataset not available for predictive modeling")

def analyze_root_causes(analyzer):
    """Analyze root causes of stunting and deficiencies"""
    st.markdown('<div class="section-header">🔍 Root Cause Analysis of Stunting</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stunting correlates analysis
        if analyzer.merged_data is not None:
            correlates = ['dietary_diversity', 'water_source', 'sanitation_facility', 'mother_education']
            available_correlates = []
            for correlate in correlates:
                matching_cols = [col for col in analyzer.merged_data.columns if correlate in col.lower()]
                if matching_cols:
                    available_correlates.append(matching_cols[0])
            
            if available_correlates and 'stunting_status' in analyzer.merged_data.columns:
                st.write("**Stunting Prevalence by Factors:**")
                
                for factor in available_correlates[:2]:  # Show first 2 factors
                    try:
                        factor_analysis = analyzer.merged_data.groupby(factor).agg({
                            'stunting_status': lambda x: (x.isin(['Severe Stunting', 'Moderate Stunting'])).mean()
                        }).reset_index()
                        
                        fig = px.bar(
                            factor_analysis,
                            x=factor,
                            y='stunting_status',
                            title=f"Stunting by {factor.replace('_', ' ').title()}",
                            labels={'stunting_status': 'Stunting Prevalence'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not analyze {factor}: {e}")
    
    with col2:
        st.write("**Key Root Causes Identified:**")
        
        root_causes = {
            "Dietary Diversity": "Limited access to diverse foods leads to micronutrient deficiencies",
            "WASH Facilities": "Poor water and sanitation contribute to infections and nutrient loss",
            "Maternal Education": "Lower education levels correlate with poorer childcare practices",
            "Household Income": "Economic constraints limit food quality and healthcare access",
            "Healthcare Access": "Limited access to prenatal and child healthcare services"
        }
        
        for cause, explanation in root_causes.items():
            with st.expander(f"📊 {cause}"):
                st.write(explanation)
                # Add relevant metrics if available
                if cause == "Dietary Diversity":
                    dd_cols = [col for col in analyzer.merged_data.columns if 'diet' in col.lower() and 'divers' in col.lower()]
                    if dd_cols:
                        avg_diversity = analyzer.merged_data[dd_cols[0]].mean()
                        st.metric("Average Dietary Diversity Score", f"{avg_diversity:.1f}")

def recommend_interventions():
    """Recommend multi-sectoral interventions"""
    st.markdown('<div class="section-header">🔄 Multi-Sectoral Intervention Recommendations</div>', unsafe_allow_html=True)
    
    sectors = {
        "Health": [
            "Scale up micronutrient supplementation programs",
            "Strengthen antenatal care and nutrition services",
            "Implement growth monitoring and promotion",
            "Treat acute malnutrition cases"
        ],
        "Agriculture": [
            "Promote biofortified crops (vitamin A maize, iron beans)",
            "Support homestead food production (vegetables, fruits)",
            "Diversify agricultural production systems",
            "Improve post-harvest handling to reduce nutrient loss"
        ],
        "Education": [
            "Integrate nutrition education in school curricula",
            "Train teachers on nutrition and hygiene practices",
            "Establish school feeding programs with diverse foods",
            "Promote school gardens for practical learning"
        ],
        "Social Protection": [
            "Targeted cash transfers for vulnerable households",
            "Nutrition-sensitive social safety nets",
            "Maternal and child nutrition programs",
            "School feeding in food-insecure areas"
        ]
    }
    
    for sector, interventions in sectors.items():
        with st.expander(f"🏥 {sector} Sector Interventions"):
            for intervention in interventions:
                st.write(f"• {intervention}")

def generate_policy_briefs():
    """Generate short policy briefs for local implementation"""
    st.markdown('<div class="section-header">📋 Policy Briefs for Local Implementation</div>', unsafe_allow_html=True)
    
    policy_areas = {
        "Immediate Actions (0-6 months)": [
            "**Emergency Supplementation**: Target pregnant women and children 6-23 months with micronutrient powders",
            "**Community Screening**: Train community health workers to identify acute malnutrition cases",
            "**WASH Promotion**: Launch hygiene behavior change campaigns in high-stunting areas"
        ],
        "Medium-term Strategies (6-24 months)": [
            "**Agriculture-Nutrition Integration**: Link farmers to school feeding programs and health centers",
            "**Social Behavior Change**: Implement mass media campaigns on infant feeding practices",
            "**Health System Strengthening**: Build capacity for nutrition assessment and counseling"
        ],
        "Long-term Structural Changes (2+ years)": [
            "**Policy Integration**: Mainstream nutrition in all relevant sectoral policies",
            "**Infrastructure Development**: Improve rural water supply and sanitation facilities",
            "**Economic Empowerment**: Create livelihood opportunities for women in nutrition-sensitive value chains"
        ]
    }
    
    for timeframe, policies in policy_areas.items():
        with st.expander(f"⏰ {timeframe}"):
            for policy in policies:
                st.write(policy)

def display_data_overview(analyzer):
    """Display overview of the loaded datasets"""
    st.markdown('<div class="section-header">📊 Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if analyzer.hh_data is not None:
            st.metric("Households Surveyed", len(analyzer.hh_data))
            st.write("**Sample Household Variables:**")
            hh_vars = list(analyzer.hh_data.columns[:8])  # Show first 8 columns
            for var in hh_vars:
                st.write(f"• {var}")
    
    with col2:
        if analyzer.child_data is not None:
            st.metric("Children (6-59 months)", len(analyzer.child_data))
            st.write("**Sample Child Variables:**")
            child_vars = list(analyzer.child_data.columns[:8])
            for var in child_vars:
                st.write(f"• {var}")
    
    with col3:
        if analyzer.women_data is not None:
            st.metric("Women (15-49 years)", len(analyzer.women_data))
            st.write("**Sample Women's Variables:**")
            women_vars = list(analyzer.women_data.columns[:8])
            for var in women_vars:
                st.write(f"• {var}")
    
    # Key statistics
    st.markdown("### Key Nutrition Indicators")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        if analyzer.child_data is not None and 'stunting_status' in analyzer.child_data.columns:
            stunting_rate = (analyzer.child_data['stunting_status'].isin(['Severe Stunting', 'Moderate Stunting'])).mean()
            st.metric("Stunting Prevalence", f"{stunting_rate:.1%}")
        else:
            st.metric("Stunting Prevalence", "Data not available")
    
    with stats_col2:
        if analyzer.hh_data is not None and 'food_insecurity_level' in analyzer.hh_data.columns:
            food_insecure = (analyzer.hh_data['food_insecurity_level'].isin(['Poor', 'Borderline'])).mean()
            st.metric("Food Insecurity Rate", f"{food_insecure:.1%}")
        else:
            st.metric("Food Insecurity Rate", "Data not available")
    
    with stats_col3:
        if analyzer.women_data is not None and 'diet_diversity_category' in analyzer.women_data.columns:
            low_diversity = (analyzer.women_data['diet_diversity_category'] == 'Low').mean()
            st.metric("Women with Low Dietary Diversity", f"{low_diversity:.1%}")
        else:
            st.metric("Dietary Diversity", "Data not available")

def main():
    # Initialize analyzer
    analyzer = HiddenHungerAnalyzer()
    
    # App title
    st.markdown('<div class="main-header">🌍 Hidden Hunger Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Addressing Micronutrient Deficiencies through Data-Driven Insights")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_section = st.sidebar.radio(
        "Select Analysis Section:",
        ["Data Overview", "Hotspot Mapping", "Predictive Models", "Root Cause Analysis", 
         "Intervention Recommendations", "Policy Briefs"]
    )
    
    # Load data with progress
    with st.spinner('Loading and processing data...'):
        if analyzer.load_data():
            st.sidebar.success("✅ Data loaded successfully!")
        else:
            st.error("Failed to load data. Please check file paths and formats.")
            return
    
    # Main content based on selection
    if app_section == "Data Overview":
        display_data_overview(analyzer)
    elif app_section == "Hotspot Mapping":
        create_hotspot_map(analyzer)
    elif app_section == "Predictive Models":
        build_predictive_models(analyzer)
    elif app_section == "Root Cause Analysis":
        analyze_root_causes(analyzer)
    elif app_section == "Intervention Recommendations":
        recommend_interventions()
    elif app_section == "Policy Briefs":
        generate_policy_briefs()

if __name__ == "__main__":
    main()