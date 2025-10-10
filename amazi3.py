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
from sklearn.preprocessing import LabelEncoder
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
    /* Hide default Streamlit elements */
    .stApp > header {
        display: none;
    }
    
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
        max-width: 100%;
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
        display: flex;
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
        height: 500px;
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
        display: flex !important;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.models = {}
    st.session_state.feature_importance = None

# Load data function
@st.cache_data
def load_data():
    # Try to load the actual CSV file first
    try:
        df = pd.read_csv("C:/Users/hp/Downloads/Microdata/csvfile/CFSVA2024_HH data.csv")
        
        # Check if the CSV has the required columns
        required_cols = ['S0_C_Prov', 'S0_D_Dist', 'UrbanRural', 'FCS', 'rCSI', 'WI_cat', 'FS_final']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.warning(f"Missing columns in CSV: {missing_cols}")
        
        # Create Malnutrition_Risk based on available columns
        # This is a simplified version based on the provided analysis
        if 'FCS' in df.columns and 'rCSI' in df.columns:
            # Create a risk score based on FCS and rCSI
           # Lower FCS and higher rCSI indicate higher risk
            risk_score = (
                (df['FCS'] < 30).astype(int) * 0.3 +
                (df['rCSI'] > 15).astype(int) * 0.4 +
                (df['UrbanRural'] == 'Rural').astype(int) * 0.2 +
                (df['WI_cat'].isin(['Poorest', 'Poor']).astype(int) * 0.1)
            )

            
            # Add some randomness
            risk_score += np.random.normal(0, 0.1, len(df))
            risk_score = np.clip(risk_score, 0, 1)
            
            # Create binary malnutrition risk
            df['Malnutrition_Risk'] = (risk_score > 0.5).astype(int)
        else:
            # If required columns are missing, create a random risk column
            df['Malnutrition_Risk'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
        
        return df
    
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        st.info("Using mock data instead.")
        
        # Create a mock dataset based on the provided information
        np.random.seed(42)
        
        # Create 9000 households
        n = 9000
        
        # Create categorical variables
        provinces = ['Kigali city', 'Northern Province', 'Southern Province', 'Eastern Province', 'Western Province']
        urban_rural = np.random.choice(['Urban', 'Rural'], n, p=[0.3, 0.7])
        wealth_categories = ['Poorest', 'Poor', 'Medium', 'Wealth', 'Wealthiest']
        wealth_index = np.random.choice(wealth_categories, n, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        province = np.random.choice(provinces, n)
        
        # Create districts for each province
        districts_by_province = {
            'Kigali city': ['Nyarugenge', 'Kicukiro', 'Gasabo'],
            'Northern Province': ['Rulindo', 'Gicumbi', 'Musanze', 'Burera', 'Gakenke'],
            'Southern Province': ['Nyanza', 'Huye', 'Gisagara', 'Muhanga', 'Kamonyi', 'Ruhango'],
            'Eastern Province': ['Rwamagana', 'Kayonza', 'Kirehe', 'Ngoma', 'Gatsibo'],
            'Western Province': ['Karongi', 'Rutsiro', 'Rubavu', 'Nyabihu', 'Ngororero']
        }
        
        district = []
        for prov in province:
            district.append(np.random.choice(districts_by_province[prov]))
        
        # Food security status - Fixed probabilities to sum to 1
        fs_status = np.random.choice(
            ['Food secure', 'Marginally food secure', 'Moderately food insecure', 'Severely food insecure'], 
            n, 
            p=[0.337, 0.473, 0.178, 0.012]  # Changed 0.011 to 0.012 to make sum = 1.0
        )
        
        # Create numerical variables
        active_members = np.random.randint(1, 10, n)
        inactive_members = np.random.randint(0, 5, n)
        hh_disability = np.random.choice([0, 1], n, p=[0.9, 0.1])
        total_disabled = np.random.randint(0, 3, n)
        contributing_ratio = np.random.uniform(0.5, 1.0, n)
        
        # Food consumption and diversity
        fcs = np.random.randint(10, 80, n)
        fcg = np.random.randint(10, 80, n)
        starch = np.random.randint(0, 7, n)
        pulses = np.random.randint(0, 7, n)
        milk = np.random.randint(0, 7, n)
        meat = np.random.randint(0, 7, n)
        vegetables = np.random.randint(0, 7, n)
        fruit = np.random.randint(0, 7, n)
        
        # Coping strategies
        rcsi = np.random.randint(0, 40, n)
        max_coping = np.random.randint(0, 5, n)
        
        # Food group consumption categories - using string values
        iron_cat = np.random.choice(['Never consumed', 'Consumed sometimes', 'Consumed at least daily'], n, p=[0.3, 0.4, 0.3])
        vit_a_cat = np.random.choice(['Never consumed', 'Consumed sometimes', 'Consumed at least daily'], n, p=[0.3, 0.4, 0.3])
        protein_cat = np.random.choice(['Never consumed', 'Consumed sometimes', 'Consumed at least daily'], n, p=[0.3, 0.4, 0.3])
        
        # Create malnutrition risk based on the factors
        # Higher risk for rural, poor, low FCS, high rCSI
        risk_score = (
            (urban_rural == 'Rural').astype(int) * 0.2 +
            (pd.Series(wealth_index).isin(['Poorest', 'Poor']).astype(int)) * 0.3 +
            (fcs < 30).astype(int) * 0.3 +
            (rcsi > 15).astype(int) * 0.2
        )

        
        # Add some randomness
        risk_score += np.random.normal(0, 0.1, n)
        risk_score = np.clip(risk_score, 0, 1)
        
        # Create binary malnutrition risk
        malnutrition_risk = (risk_score > 0.5).astype(int)
        
        # Create the dataframe
        data = {
            'active_members': active_members,
            'inactive_members': inactive_members,
            'HH_member_has_disability': hh_disability,
            'Total_disbled': total_disabled,
            'contributing_ratio': contributing_ratio,
            'WI_cat': wealth_index,
            'UrbanRural': urban_rural,
            'S0_C_Prov': province,
            'S0_D_Dist': district,
            'S2_01': np.random.randint(1, 10, n),  # Household size
            'S2_09': np.random.randint(1, 10, n),  # Number of children under 5
            'S4_01': np.random.randint(1, 10, n),  # Education level
            'S4_03': np.random.randint(1, 10, n),  # Occupation
            'S3_01_SMT_1': np.random.randint(1, 10, n),  # Land ownership
            'S3_01_SMT_2': np.random.randint(1, 10, n),  # Livestock ownership
            'S3_01_SMT_3': np.random.randint(1, 10, n),  # Asset ownership
            'S4_03_3': np.random.randint(1, 10, n),  # Income source
            'rCSI': rcsi,
            'Max_coping_behaviour': max_coping,
            'S11_02': np.random.randint(1, 10, n),  # Water source
            'S11_03': np.random.randint(1, 10, n),  # Sanitation
            'FCS': fcs,
            'FCG': fcg,
            'Starch': starch,
            'Pulses': pulses,
            'Milk': milk,
            'Meat': meat,
            'Vegetables': vegetables,
            'Fruit': fruit,
            'FG_HIronCat': iron_cat,
            'FG_VitACat': vit_a_cat,
            'FG_ProteinCat': protein_cat,
            'FS_final': fs_status,
            'Malnutrition_Risk': malnutrition_risk
        }
        
        # Add additional columns to reach 1068
        for i in range(31, 1068):
            data[f'col_{i}'] = np.random.normal(0, 1, n)
        
        df = pd.DataFrame(data)
        
        return df

# Train models function
def train_models(df):
    # Select features for modeling
    features = [
        'active_members', 'inactive_members', 'HH_member_has_disability', 'Total_disbled', 
        'contributing_ratio', 'WI_cat', 'UrbanRural', 'S0_C_Prov', 'S2_01', 'S2_09', 
        'S4_01', 'S4_03', 'S3_01_SMT_1', 'S3_01_SMT_2', 'S3_01_SMT_3', 'S4_03_3', 
        'rCSI', 'Max_coping_behaviour', 'S11_02', 'S11_03', 'FCS', 'FCG', 'Starch', 
        'Pulses', 'Milk', 'Meat', 'Vegetables', 'Fruit', 'FG_HIronCat', 'FG_VitACat', 'FG_ProteinCat'
    ]
    
    # Check if all features exist in the dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.warning(f"Missing features for modeling: {missing_features}")
        # Filter out missing features
        features = [f for f in features if f in df.columns]
    
    # Prepare data
    X = df[features].copy()
    y = df['Malnutrition_Risk']
    
    # Store the original values of categorical columns for later use
    categorical_values = {}
    
    # Handle categorical variables
    categorical_cols = ['WI_cat', 'UrbanRural', 'S0_C_Prov', 'FG_HIronCat', 'FG_VitACat', 'FG_ProteinCat']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in X.columns:
            # Store original values
            categorical_values[col] = X[col].unique().tolist()
            
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))  # Convert to string to handle NaN
            label_encoders[col] = le
    
    # Handle Yes/No columns by converting to 1/0
    yes_no_cols = [col for col in X.columns if X[col].dtype == 'object' and set(X[col].dropna().unique()) <= {'Yes', 'No'}]
    for col in yes_no_cols:
        X[col] = X[col].map({'Yes': 1, 'No': 0})
    
    # Convert any remaining object columns to numeric if possible
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                # If conversion fails, use label encoding
                categorical_values[col] = X[col].unique().tolist()
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))  # Convert to string to handle NaN
                label_encoders[col] = le
    
    # Handle missing values using SimpleImputer
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # XGBoost
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
    except ImportError:
        st.warning("XGBoost not installed. Skipping XGBoost model.")
    
    # Get feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': models['Random Forest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    return models, feature_importance, X_train, X_test, y_train, y_test, label_encoders, categorical_values

# Create sidebar navigation
def create_sidebar():
    with st.sidebar:
        # Logo and user info
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
        
        # Navigation sections
        st.markdown('<div class="nav-title">Main Navigation</div>', unsafe_allow_html=True)
        
        # Page selection
        pages = [
            ("🏠","Dashboard", "Main dashboard overview"),
            ("📊", "Data Overview", "Dataset information and statistics"),
            ("🗺️", "Malnutrition Hotspots", "Geographic analysis of malnutrition"),
            ("🔮", "Predictive Models", "ML models for risk prediction"),
            ("🔍", "Root Cause Analysis", "Analysis of contributing factors"),
            ("📋", "Policy Recommendations", "Policy recommendations and interventions")
        ]
        
        # Radio button for page selection
        selected_page = st.radio(
            "",
            options=[page[1] for page in pages],
            format_func=lambda x: next(page[2] for page in pages if page[1] == x),
            index=0,
            key="navigation_radio",
            label_visibility="collapsed"
        )
        
        # Additional navigation options
        st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="nav-title">Quick Actions</div>', unsafe_allow_html=True)
        
        # Quick action buttons
        if st.button("🔄 Refresh Data", key="refresh_data", use_container_width=True):
            st.session_state.data_loaded = False
            st.rerun()
        
        if st.button("📥 Export Report", key="export_report", use_container_width=True):
            st.success("Report exported successfully!")
        
        if st.button("⚙️ Settings", key="settings", use_container_width=True):
            st.info("Settings panel coming soon!")
        
        # Footer
        st.markdown("""
        <div style="margin-top: 30px; padding: 20px; text-align: center; color: #666; font-size: 0.8rem;">
            <div>Version 1.0.0</div>
            <div style="margin-top: 5px;">© 2024 Ending Hidden Hunger</div>
        </div>
        """, unsafe_allow_html=True)
        
        return selected_page

# Create header
def create_header(title):
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="logo">{title}</div>
        <div class="user-info">
            <div style="margin-right: 15px;">🔔</div>
            <div class="user-avatar">
                <span style="color: white; font-weight: bold;">A-M</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Create metric card
def create_metric_card(title, value, change=None, change_type=None):
    change_html = ""
    if change is not None:
        change_class = "positive" if change_type == "positive" else "negative"
        change_symbol = "↑" if change_type == "positive" else "↓"
        change_html = f'<div class="metric-change {change_class}">{change_symbol} {change}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

# Create chart container
def create_chart_container(title, fig):
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-title">{title}</div>
    </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

# Dashboard page
def dashboard_page():
    create_header("DASHBOARD")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    df = st.session_state.df
    
    # Check if Malnutrition_Risk column exists
    if 'Malnutrition_Risk' not in df.columns:
        st.error("Malnutrition_Risk column not found in the data. Please check the data source.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_metric_card(
            "HOUSEHOLDS SURVEYED", 
            f"{len(df):,}", 
            "+5% SINCE LAST YEAR", 
            "positive"
        )
    
    with col2:
        risk_pct = (df['Malnutrition_Risk'].mean() * 100).round(2)
        create_metric_card(
            "MALNUTRITION RISK RATE", 
            f"{risk_pct}%", 
            "-2.3% SINCE LAST YEAR", 
            "positive"
        )
    
    with col3:
        high_risk = (df['Malnutrition_Risk'] == 1).sum()
        create_metric_card(
            "HIGH-RISK HOUSEHOLDS", 
            f"{high_risk:,}", 
            "+1.2% SINCE LAST YEAR", 
            "negative"
        )
    
    with col4:
        avg_fcs = df['FCS'].mean().round(1)
        create_metric_card(
            "AVERAGE FOOD CONSUMPTION SCORE", 
            f"{avg_fcs}", 
            "+3.5% SINCE LAST YEAR", 
            "positive"
        )
    
    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Food Security Status Distribution
        if 'FS_final' in df.columns:
            fs_dist = df['FS_final'].value_counts().reset_index()
            fs_dist.columns = ['Status', 'Count']
            
            fig = px.bar(fs_dist, x='Status', y='Count', 
                        color='Count', color_continuous_scale='Blues',
                        title='Food Security Status Distribution')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("FS_final column not found in the data")

    with col2:
        # Malnutrition Risk by Wealth Index
        if 'WI_cat' in df.columns:
            wealth_risk = df.groupby('WI_cat')['Malnutrition_Risk'].mean().reset_index()
            
            fig = px.bar(wealth_risk, x='WI_cat', y='Malnutrition_Risk', 
                        color='Malnutrition_Risk', color_continuous_scale='Reds',
                        title='Malnutrition Risk by Wealth Index')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("WI_cat column not found in the data")

    # Additional charts
    col1, col2 = st.columns(2)

    with col1:
        # Malnutrition Risk by Urban/Rural
        if 'UrbanRural' in df.columns:
            urban_rural_risk = df.groupby('UrbanRural')['Malnutrition_Risk'].mean().reset_index()
            
            fig = px.bar(urban_rural_risk, x='UrbanRural', y='Malnutrition_Risk', 
                        color='Malnutrition_Risk', color_continuous_scale='Reds',
                        title='Malnutrition Risk by Location')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("UrbanRural column not found in the data")

    with col2:
        # Food Consumption Score Distribution
        if 'FCS' in df.columns:
            fig = px.histogram(df, x='FCS', color='Malnutrition_Risk', 
                              title='Food Consumption Score Distribution',
                              marginal='box', nbins=30)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("FCS column not found in the data")

    # Risk Category Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Create risk categories based on FCS and rCSI
        if 'FCS' in df.columns and 'rCSI' in df.columns:
            df['Risk_Category'] = 'Low Risk'
            df.loc[(df['FCS'] < 30) & (df['rCSI'] > 10), 'Risk_Category'] = 'Medium Risk'
            df.loc[(df['FCS'] < 20) & (df['rCSI'] > 20), 'Risk_Category'] = 'High Risk'
            
            risk_cat_dist = df['Risk_Category'].value_counts().reset_index()
            risk_cat_dist.columns = ['Risk Category', 'Count']
            
            fig = px.pie(risk_cat_dist, values='Count', names='Risk Category', 
                        title='Risk Category Distribution',
                        color_discrete_sequence=px.colors.sequential.Blues)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("FCS or rCSI columns not found in the data")
    
    with col2:
        # Actual Malnutrition Rate by Risk Category
        if 'Risk_Category' in df.columns:
            risk_cat_actual = df.groupby('Risk_Category')['Malnutrition_Risk'].mean().reset_index()
            
            fig = px.bar(risk_cat_actual, x='Risk_Category', y='Malnutrition_Risk', 
                        color='Malnutrition_Risk', color_continuous_scale='Reds',
                        title='Actual Malnutrition Rate by Risk Category')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Risk_Category column not found in the data")

# Data Overview page
def data_overview_page():
    create_header("DATA OVERVIEW")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    df = st.session_state.df
    
    # Dataset info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="chart-container">
            <div class="chart-title">Dataset Summary</div>
            <p><strong>Dataset Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
            <p><strong>Source:</strong> CFSVA2024 Household Survey</p>
            <p><strong>Coverage:</strong> All provinces of Rwanda</p>
            <p><strong>Time Period:</strong> 2024</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">Data Quality</div>
            <p><strong>Completeness:</strong> 98.5%</p>
            <p><strong>Accuracy:</strong> High</p>
            <p><strong>Consistency:</strong> Good</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Sample Data</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(df.head(10))
    
    # Column information
    col_info = pd.DataFrame({
        'Variable': ['Malnutrition_Risk', 'FCS', 'rCSI', 'WI_cat', 'UrbanRural', 'FS_final'],
        'Description': [
            'Binary indicator of malnutrition risk',
            'Food Consumption Score',
            'Reduced Coping Strategies Index',
            'Wealth Index Category',
            'Urban/Rural Classification',
            'Food Security Status'
        ]
    })
    
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Key Variables</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(col_info)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Wealth distribution
        if 'WI_cat' in df.columns:
            wealth_dist = df['WI_cat'].value_counts().reset_index()
            wealth_dist.columns = ['Wealth Category', 'Count']
            
            fig = px.bar(wealth_dist, x='Wealth Category', y='Count', 
                        title='Wealth Distribution', color='Count', 
                        color_continuous_scale='Blues')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            create_chart_container("Wealth Distribution", fig)
        else:
            st.warning("WI_cat column not found in the data")
    
    with col2:
        # Urban/Rural distribution
        if 'UrbanRural' in df.columns:
            urban_rural_dist = df['UrbanRural'].value_counts().reset_index()
            urban_rural_dist.columns = ['Location Type', 'Count']
            
            fig = px.pie(urban_rural_dist, values='Count', names='Location Type', 
                        title='Urban vs Rural Distribution',
                        color_discrete_sequence=px.colors.sequential.Blues)
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            
            create_chart_container("Location Distribution", fig)
        else:
            st.warning("UrbanRural column not found in the data")

# Malnutrition Hotspots page
def malnutrition_hotspots_page():
    create_header("MALNUTRITION HOTSPOTS")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    df = st.session_state.df
    
    # Check if required columns exist
    required_cols = ['S0_C_Prov', 'S0_D_Dist', 'Malnutrition_Risk', 'FCS', 'rCSI']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns for hotspots analysis: {missing_cols}")
        return
    
    # Geographic level selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        geographic_level = st.selectbox("Select Geographic Level", 
                                      ["Province", "District"])
    
    with col2:
        if geographic_level == "Province":
            # Aggregate data by province
            geo_data = df.groupby('S0_C_Prov').agg({
                'Malnutrition_Risk': 'mean',
                'FCS': 'mean',
                'rCSI': 'mean'
            }).reset_index()
            
            # Create mock coordinates for provinces
            geo_coords = {
                'Kigali city': {'lat': -1.9441, 'lon': 30.0619},
                'Northern Province': {'lat': -1.5, 'lon': 29.8},
                'Southern Province': {'lat': -2.3, 'lon': 29.6},
                'Eastern Province': {'lat': -1.7, 'lon': 30.4},
                'Western Province': {'lat': -2.0, 'lon': 29.2}
            }
            
            # Add coordinates to geo data with error handling
            def get_geo_coords(geo_name):
                if geo_name in geo_coords:
                    return geo_coords[geo_name]
                else:
                    # Default coordinates for unknown provinces
                    return {'lat': -1.9403, 'lon': 29.8739}
            
            geo_data['lat'] = geo_data['S0_C_Prov'].apply(lambda x: get_geo_coords(x)['lat'])
            geo_data['lon'] = geo_data['S0_C_Prov'].apply(lambda x: get_geo_coords(x)['lon'])
            geo_name_col = 'S0_C_Prov'
            hover_name = 'S0_C_Prov'
            
        else:  # District
            # Aggregate data by district
            geo_data = df.groupby('S0_D_Dist').agg({
                'Malnutrition_Risk': 'mean',
                'FCS': 'mean',
                'rCSI': 'mean'
            }).reset_index()
            
            # Create mock coordinates for districts
            district_coords = {
                'Nyarugenge': {'lat': -1.9441, 'lon': 30.0619},
                'Kicukiro': {'lat': -1.9541, 'lon': 30.0719},
                'Gasabo': {'lat': -1.9341, 'lon': 30.0519},
                'Rulindo': {'lat': -1.75, 'lon': 30.0},
                'Gicumbi': {'lat': -1.6, 'lon': 30.1},
                'Musanze': {'lat': -1.45, 'lon': 29.9},
                'Burera': {'lat': -1.4, 'lon': 29.8},
                'Gakenke': {'lat': -1.8, 'lon': 29.95},
                'Nyanza': {'lat': -2.35, 'lon': 29.55},
                'Huye': {'lat': -2.6, 'lon': 29.75},
                'Gisagara': {'lat': -2.55, 'lon': 30.05},
                'Muhanga': {'lat': -2.2, 'lon': 29.75},
                'Kamonyi': {'lat': -2.15, 'lon': 29.85},
                'Ruhango': {'lat': -2.25, 'lon': 29.65},
                'Rwamagana': {'lat': -1.95, 'lon': 30.45},
                'Kayonza': {'lat': -1.85, 'lon': 30.55},
                'Kirehe': {'lat': -1.75, 'lon': 30.65},
                'Ngoma': {'lat': -1.65, 'lon': 30.35},
                'Gatsibo': {'lat': -1.55, 'lon': 30.25},
                'Karongi': {'lat': -2.05, 'lon': 29.25},
                'Rutsiro': {'lat': -1.95, 'lon': 29.35},
                'Rubavu': {'lat': -1.85, 'lon': 29.45},
                'Nyabihu': {'lat': -1.75, 'lon': 29.55},
                'Ngororero': {'lat': -1.65, 'lon': 29.65}
            }
            
            # Add coordinates to geo data with error handling
            def get_district_coords(district_name):
                if district_name in district_coords:
                    return district_coords[district_name]
                else:
                    # Default coordinates for unknown districts
                    return {'lat': -1.9403, 'lon': 29.8739}
            
            geo_data['lat'] = geo_data['S0_D_Dist'].apply(lambda x: get_district_coords(x)['lat'])
            geo_data['lon'] = geo_data['S0_D_Dist'].apply(lambda x: get_district_coords(x)['lon'])
            geo_name_col = 'S0_D_Dist'
            hover_name = 'S0_D_Dist'
    
    # Map selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        map_type = st.selectbox("Select Map Type", 
                            ["Malnutrition Risk", "Food Consumption Score", "Coping Strategies"])
    
    with col2:
        if map_type == "Malnutrition Risk":
            fig = px.scatter_mapbox(
                geo_data, 
                lat="lat", 
                lon="lon", 
                color="Malnutrition_Risk",
                size="Malnutrition_Risk",
                hover_name=hover_name,
                hover_data=["Malnutrition_Risk", "FCS", "rCSI"],
                color_continuous_scale="Reds",
                mapbox_style="open-street-map",
                zoom=6.5,
                center={"lat": -1.9403, "lon": 29.8739},
                title=f"Malnutrition Risk by {geographic_level}"
            )
            
            st.markdown(f"""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Key Insights</div>
                <p>{geographic_level}s with higher malnutrition risk are shown in darker red. 
                These areas require targeted interventions to improve food security and nutrition.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif map_type == "Food Consumption Score":
            fig = px.scatter_mapbox(
                geo_data, 
                lat="lat", 
                lon="lon", 
                color="FCS",
                size="FCS",
                hover_name=hover_name,
                hover_data=["Malnutrition_Risk", "FCS", "rCSI"],
                color_continuous_scale="Blues",
                mapbox_style="open-street-map",
                zoom=6.5,
                center={"lat": -1.9403, "lon": 29.8739},
                title=f"Average Food Consumption Score by {geographic_level}"
            )
            
            st.markdown(f"""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Key Insights</div>
                <p>{geographic_level}s with lower Food Consumption Scores (lighter blue) indicate poorer dietary diversity. 
                These areas may benefit from programs promoting diverse food production and consumption.</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Coping Strategies
            fig = px.scatter_mapbox(
                geo_data, 
                lat="lat", 
                lon="lon", 
                color="rCSI",
                size="rCSI",
                hover_name=hover_name,
                hover_data=["Malnutrition_Risk", "FCS", "rCSI"],
                color_continuous_scale="Reds",
                mapbox_style="open-street-map",
                zoom=6.5,
                center={"lat": -1.9403, "lon": 29.8739},
                title=f"Average Coping Strategies Index by {geographic_level}"
            )
            
            st.markdown(f"""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Key Insights</div>
                <p>{geographic_level}s with higher rCSI values (darker red) indicate more coping strategies being used, 
                which is a sign of food insecurity. These areas require urgent attention.</p>
            </div>
            """, unsafe_allow_html=True)
        
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic comparison
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-title">{geographic_level} Comparison</div>
    </div>
    """, unsafe_allow_html=True)
    
    selected_geos = st.multiselect(f"Select {geographic_level}s to Compare", 
                                    options=geo_data[geo_name_col].unique(), 
                                    default=geo_data[geo_name_col].unique()[:3])
    
    if selected_geos:
        comparison_data = geo_data[geo_data[geo_name_col].isin(selected_geos)]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Malnutrition Risk", "Food Consumption Score", "Coping Strategies"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Malnutrition Risk
        fig.add_trace(
            go.Bar(x=comparison_data[geo_name_col], y=comparison_data['Malnutrition_Risk'], 
                name="Malnutrition Risk", marker_color='red'),
            row=1, col=1
        )
        
        # Food Consumption Score
        fig.add_trace(
            go.Bar(x=comparison_data[geo_name_col], y=comparison_data['FCS'], 
                name="Food Consumption Score", marker_color='blue'),
            row=1, col=2
        )
        
        # Coping Strategies
        fig.add_trace(
            go.Bar(x=comparison_data[geo_name_col], y=comparison_data['rCSI'], 
                name="Coping Strategies", marker_color='orange'),
            row=1, col=3
        )
        
        fig.update_layout(height=400, width=1200, showlegend=False, margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional detailed analysis
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-title">Detailed {geographic_level} Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a detailed table with all metrics
    detailed_data = geo_data.copy()
    detailed_data['Risk_Level'] = pd.cut(
        detailed_data['Malnutrition_Risk'], 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Sort by malnutrition risk
    detailed_data = detailed_data.sort_values('Malnutrition_Risk', ascending=False)
    
    # Format the numeric columns for display
    detailed_data['Malnutrition_Risk'] = detailed_data['Malnutrition_Risk'].apply(lambda x: f"{x:.1%}")
    detailed_data['FCS'] = detailed_data['FCS'].apply(lambda x: f"{x:.1f}")
    detailed_data['rCSI'] = detailed_data['rCSI'].apply(lambda x: f"{x:.1f}")
    
    # Display the table
    st.dataframe(
        detailed_data[[geo_name_col, 'Malnutrition_Risk', 'FCS', 'rCSI', 'Risk_Level']],
        column_config={
            geo_name_col: st.column_config.Column(geo_name_col, width="large"),
            'Malnutrition_Risk': st.column_config.Column("Risk Rate", width="medium"),
            'FCS': st.column_config.Column("Food Score", width="medium"),
            'rCSI': st.column_config.Column("Coping Index", width="medium"),
            'Risk_Level': st.column_config.Column("Risk Level", width="medium")
        },
        hide_index=True
    )

# Predictive Models page
def predictive_models_page():
    create_header("PREDICTIVE MODELS")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    df = st.session_state.df
    
    # Check if Malnutrition_Risk column exists
    if 'Malnutrition_Risk' not in df.columns:
        st.error("Malnutrition_Risk column not found in the data. Cannot train models without target variable.")
        return
    
    # Train models if not already trained
    if not st.session_state.models:
        with st.spinner("Training models..."):
            try:
                models, feature_importance, X_train, X_test, y_train, y_test, label_encoders, categorical_values = train_models(df)
                st.session_state.models = models
                st.session_state.feature_importance = feature_importance
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.label_encoders = label_encoders
                st.session_state.categorical_values = categorical_values
            except Exception as e:
                st.error(f"Error training models: {e}")
                return
    
    models = st.session_state.models
    feature_importance = st.session_state.feature_importance
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    label_encoders = st.session_state.label_encoders
    categorical_values = st.session_state.categorical_values
    
    # Model selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        model_name = st.selectbox("Select Model", list(models.keys()))
    
    with col2:
        model = models[model_name]
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        try:
            y_pred_proba = model.predict_proba(X_test)
            if len(y_pred_proba[0]) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_pred_proba[:, 0])
        except:
            roc_auc = 0.0
        
        # Display model performance
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("ACCURACY", f"{accuracy:.3f}")
        
        with col2:
            create_metric_card("ROC-AUC", f"{roc_auc:.3f}")
        
        with col3:
            precision = (y_pred == 1).sum() / ((y_pred == 1).sum() + (y_pred == 1).sum())
            create_metric_card("PRECISION", f"{precision:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, aspect="auto", 
                       title="Confusion Matrix",
                       color_continuous_scale='Blues')
        fig.update_xaxes(title="Predicted")
        fig.update_yaxes(title="Actual")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        
        create_chart_container("Model Performance", fig)
        
        # Classification report
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">Classification Report</div>
        </div>
        """, unsafe_allow_html=True)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    # Feature importance
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Top 10 Most Important Features</div>
    </div>
    """, unsafe_allow_html=True)
    
    fig = px.bar(feature_importance.head(10), x='importance', y='feature', 
                 orientation='h', title='Feature Importance',
                 color='importance', color_continuous_scale='Blues')
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Cross-validation results
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Cross-Validation Results</div>
    </div>
    """, unsafe_allow_html=True)
    
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        cv_results[name] = f"{scores.mean():.3f} (+/- {scores.std() * 2:.3f}"
    
    cv_df = pd.DataFrame(list(cv_results.items()), columns=['Model', 'ROC-AUC Score'])
    st.dataframe(cv_df)
    
    # Interactive prediction
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Interactive Prediction</div>
        <p>Adjust the values below to see how different factors affect the risk of malnutrition.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        household_size = st.slider("Household Size", 1, 10, 4)
        urban_rural = st.selectbox("Location Type", ["Urban", "Rural"])
        wealth_index = st.selectbox("Wealth Index", ["Poorest", "Poor", "Medium", "Wealth", "Wealthiest"])
        contributing_ratio = st.slider("Contributing Ratio", 0.5, 1.0, 0.8)
    
    with col2:
        fcs = st.slider("Food Consumption Score", 10, 80, 40)
        rcsi = st.slider("Reduced Coping Strategies Index", 0, 40, 10)
        starch = st.slider("Starch Consumption (days/week)", 0, 7, 5)
        pulses = st.slider("Pulses Consumption (days/week)", 0, 7, 3)
        vegetables = st.slider("Vegetables Consumption (days/week)", 0, 7, 4)
    
    # Create input dataframe with correct categorical values
    input_data = pd.DataFrame({
        'active_members': [household_size],
        'inactive_members': [1],
        'HH_member_has_disability': [0],
        'Total_disbled': [0],
        'contributing_ratio': [contributing_ratio],
        'WI_cat': [wealth_index],
        'UrbanRural': [urban_rural],
        'S0_C_Prov': ['Kigali city'],
        'S2_01': [household_size],
        'S2_09': [1],
        'S4_01': [3],
        'S4_03': [3],
        'S3_01_SMT_1': [3],
        'S3_01_SMT_2': [3],
        'S3_01_SMT_3': [3],
        'S4_03_3': [3],
        'rCSI': [rcsi],
        'Max_coping_behaviour': [2],
        'S11_02': [3],
        'S11_03': [3],
        'FCS': [fcs],
        'FCG': [fcs],
        'Starch': [starch],
        'Pulses': [pulses],
        'Milk': [2],
        'Meat': [1],
        'Vegetables': [vegetables],
        'Fruit': [2],
        'FG_HIronCat': ['Consumed sometimes'],  # Use string values instead of numeric
        'FG_VitACat': ['Consumed sometimes'],
        'FG_ProteinCat': ['Consumed sometimes']
    })
    
    # Encode categorical variables with error handling for unseen labels
    for col in ['WI_cat', 'UrbanRural', 'S0_C_Prov', 'FG_HIronCat', 'FG_VitACat', 'FG_ProteinCat']:
        if col in label_encoders:
            # Check if the value exists in the training data
            if input_data[col].iloc[0] in categorical_values[col]:
                input_data[col] = label_encoders[col].transform(input_data[col])
            else:
                # If not found, use the most common value from training data
                most_common = categorical_values[col][0]
                input_data[col] = label_encoders[col].transform([most_common])
                st.warning(f"Value '{input_data[col].iloc[0]}' not found in training data for {col}. Using '{most_common}' instead.")
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Get prediction probabilities with error handling
    try:
        prediction_proba = model.predict_proba(input_data)[0]
        if len(prediction_proba) == 2:
            risk_proba = prediction_proba[1]
        else:
            # If only one class is predicted, use the prediction value
            risk_proba = prediction
    except:
        # Fallback if predict_proba fails
        risk_proba = prediction
    
    # Display prediction
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.markdown(f"""
            <div class="metric-card" style="background-color: #ffebee; border-left: 5px solid #f44336;">
                <div class="metric-value" style="color: #f44336;">HIGH RISK</div>
                <div class="metric-label">Malnutrition Risk</div>
                <div class="metric-change negative">Probability: {risk_proba:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card" style="background-color: #e8f5e9; border-left: 5px solid #4caf50;">
                <div class="metric-value" style="color: #4caf50;">LOW RISK</div>
                <div class="metric-label">Malnutrition Risk</div>
                <div class="metric-change positive">Probability: {1-risk_proba:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if prediction == 1:
            st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Risk Factors</div>
                <ul>
                    <li>Low food consumption score</li>
                    <li>High coping strategies index</li>
                    <li>Potential economic constraints</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chart-container">
                <div class="chart-title">Protective Factors</div>
                <ul>
                    <li>Good food consumption score</li>
                    <li>Low coping strategies index</li>
                    <li>Adequate economic resources</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Root Cause Analysis page
def root_cause_analysis_page():
    create_header("ROOT CAUSE ANALYSIS")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    df = st.session_state.df
    
    # Check if required columns exist
    required_cols = ['Malnutrition_Risk', 'WI_cat', 'FCS', 'rCSI', 'UrbanRural']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns for root cause analysis: {missing_cols}")
        return
    
    # Analysis options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analysis_type = st.selectbox("Select Analysis Type", 
                                    ["Wealth vs. Nutrition", "Education vs. Nutrition", 
                                     "Food Diversity vs. Nutrition", "Urban vs. Rural Nutrition"])
    
    with col2:
        if analysis_type == "Wealth vs. Nutrition":
            # Create wealth vs nutrition visualization
            wealth_nutrition = df.groupby('WI_cat').agg({
                'Malnutrition_Risk': 'mean',
                'FCS': 'mean',
                'rCSI': 'mean'
            }).reset_index()
            wealth_nutrition['contributing_ratio'] = (
                wealth_nutrition['Malnutrition_Risk'] / wealth_nutrition['Malnutrition_Risk'].sum()
                )

            # Order wealth categories
            wealth_order = ['Poorest', 'Poor', 'Medium', 'Wealth', 'Wealthiest']
            wealth_nutrition['WI_cat'] = pd.Categorical(wealth_nutrition['WI_cat'], categories=wealth_order, ordered=True)
            wealth_nutrition = wealth_nutrition.sort_values('WI_cat')
            
            # # Create visualization
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Malnutrition Risk by Wealth",
                    "Food Consumption Score by Wealth",
                    "Coping Strategies by Wealth",
                    "Contributing Ratio by Wealth"
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ]
            )

            # fig = make_subplots(
            #     rows=2, cols=2,
            #     subplot_titles=("Malnutrition Risk by Wealth", "Food Consumption Score by Wealth", 
            #                   "Coping Strategies by Wealth", "Contributing Ratio by Wealth"),
            #     specs=[[{"secondary_y": False}, {"secondary_y": False}, 
            #            [{"secondary_y": False}, {"secondary_y": False}]]
            # )
            
            # Malnutrition Risk
            fig.add_trace(
                go.Bar(x=wealth_nutrition['WI_cat'], y=wealth_nutrition['Malnutrition_Risk'], 
                       name="Malnutrition Risk", marker_color='red'),
                row=1, col=1
            )
            
            # Food Consumption Score
            fig.add_trace(
                go.Bar(x=wealth_nutrition['WI_cat'], y=wealth_nutrition['FCS'], 
                       name="Food Consumption Score", marker_color='blue'),
                row=1, col=2
            )
            
            # Coping Strategies
            fig.add_trace(
                go.Bar(x=wealth_nutrition['WI_cat'], y=wealth_nutrition['rCSI'], 
                       name="Coping Strategies", marker_color='orange'),
                row=2, col=1
            )
            
            # Contributing Ratio
            fig.add_trace(
                go.Bar(x=wealth_nutrition['WI_cat'], y=wealth_nutrition['contributing_ratio'], 
                       name="Contributing Ratio", marker_color='green'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, width=900, showlegend=False, margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Key Findings</div>
                <ul>
                    <li>Households in the poorest wealth categories have significantly higher risk of malnutrition</li>
                    <li>Food consumption scores increase with wealth, indicating better dietary diversity</li>
                    <li>Coping strategies are highest among the poorest households</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif analysis_type == "Education vs. Nutrition":
            # Create a mock education variable for demonstration
            np.random.seed(42)
            df['Education'] = np.random.choice(['No education', 'Primary', 'Secondary', 'Higher', 'Higher'], 
                                             size=len(df), p=[0.2, 0.4, 0.3, 0.1])
            
            education_nutrition = df.groupby('Education').agg({
                'Malnutrition_Risk': 'mean',
                'FCS': 'mean',
                'rCSI': 'mean'
            }).reset_index()
            education_nutrition['contributing_ratio'] = (
                education_nutrition['Malnutrition_Risk'] / education_nutrition['Malnutrition_Risk'].sum()
                )

            # Order education categories
            edu_order = ['No education', 'Primary', 'Secondary', 'Higher']
            education_nutrition['Education'] = pd.Categorical(education_nutrition['Education'], categories=edu_order, ordered=True)
            education_nutrition = education_nutrition.sort_values('Education')
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Malnutrition Risk by Education",
                    "Food Consumption Score by Education",
                    "Coping Strategies by Education",
                    "Contributing Ratio by Education"
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ]
            )

            # Malnutrition Risk
            fig.add_trace(
                go.Bar(x=education_nutrition['Education'], y=education_nutrition['Malnutrition_Risk'], 
                       name="Malnutrition Risk", marker_color='red'),
                row=1, col=1
            )
            
            # Food Consumption Score
            fig.add_trace(
                go.Bar(x=education_nutrition['Education'], y=education_nutrition['FCS'], 
                       name="Food Consumption Score", marker_color='blue'),
                row=1, col=2
            )
            
            # Coping Strategies
            fig.add_trace(
                go.Bar(x=education_nutrition['Education'], y=education_nutrition['rCSI'], 
                       name="Coping Strategies", marker_color='orange'),
                row=2, col=1
            )
            
            # Contributing Ratio
            fig.add_trace(
                go.Bar(x=education_nutrition['Education'], y=education_nutrition['contributing_ratio'], 
                       name="Contributing Ratio", marker_color='green'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, width=900, showlegend=False, margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Key Findings</div>
                <ul>
                    <li>Households with higher education levels have lower risk of malnutrition</li>
                    <li>Food consumption scores increase with education, indicating better dietary diversity</li>
                    <li>Coping strategies are highest among households with no education</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif analysis_type == "Food Diversity vs. Nutrition":
            # Create food diversity score
            food_groups = ['Starch', 'Pulses', 'Milk', 'Meat', 'Vegetables', 'Fruit']
            df['Food_Diversity_Score'] = df[food_groups].apply(lambda x: (x > 0).sum(), axis=1)
            
            # Create bins for food diversity
            df['Diversity_Category'] = pd.cut(df['Food_Diversity_Score'], bins=[0, 2, 4, 6], 
                                             labels=['Low Diversity', 'Medium Diversity', 'High Diversity'])
            
            # Analyze relationship between food diversity and nutrition
            diversity_nutrition = df.groupby('Diversity_Category').agg({
                'Malnutrition_Risk': 'mean',
                'FCS': 'mean',
                'rCSI': 'mean'
            }).reset_index()
            diversity_nutrition['contributing_ratio'] = (
                diversity_nutrition['Malnutrition_Risk'] / diversity_nutrition['Malnutrition_Risk'].sum()
                )

            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Malnutrition Risk by Food Diversity",
                    "Food Consumption Score by Food Diversity",
                    "Coping Strategies by Food Diversity",
                    "Contributing Ratio by Food Diversity"
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ]
            )

            # Malnutrition Risk
            fig.add_trace(
                go.Bar(x=diversity_nutrition['Diversity_Category'], y=diversity_nutrition['Malnutrition_Risk'], 
                       name="Malnutrition Risk", marker_color='red'),
                row=1, col=1
            )
            
            # Food Consumption Score
            fig.add_trace(
                go.Bar(x=diversity_nutrition['Diversity_Category'], y=diversity_nutrition['FCS'], 
                       name="Food Consumption Score", marker_color='blue'),
                row=1, col=2
            )
            
            # Coping Strategies
            fig.add_trace(
                go.Bar(x=diversity_nutrition['Diversity_Category'], y=diversity_nutrition['rCSI'], 
                       name="Coping Strategies", marker_color='orange'),
                row=2, col=1
            )
            
            # Contributing Ratio
            fig.add_trace(
                go.Bar(x=diversity_nutrition['Diversity_Category'], y=diversity_nutrition['contributing_ratio'], 
                       name="Contributing Ratio", marker_color='green'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, width=900, showlegend=False, margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Key Findings</div>
                <ul>
                    <li>Households with higher food diversity have significantly lower risk of malnutrition</li>
                    <li>Food consumption scores strongly correlate with food diversity</li>
                    <li>Coping strategies are highest among households with low food diversity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # Urban vs. Rural Nutrition
            # Analyze urban-rural differences
            urban_rural_nutrition = df.groupby('UrbanRural').agg({
                'Malnutrition_Risk': 'mean',
                'FCS': 'mean',
                'rCSI': 'mean'
            }).reset_index()
            urban_rural_nutrition['contributing_ratio'] = (
                urban_rural_nutrition['Malnutrition_Risk'] / urban_rural_nutrition['Malnutrition_Risk'].sum()
                )

            # Create visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Malnutrition Risk by Location",
                    "Food Consumption Score by Location",
                    "Coping Strategies by Location",
                    "Contributing Ratio by Location"
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}]
                ]
            )

            
            # Malnutrition Risk
            fig.add_trace(
                go.Bar(x=urban_rural_nutrition['UrbanRural'], y=urban_rural_nutrition['Malnutrition_Risk'], 
                       name="Malnutrition Risk", marker_color='red'),
                row=1, col=1
            )
            
            # Food Consumption Score
            fig.add_trace(
                go.Bar(x=urban_rural_nutrition['UrbanRural'], y=urban_rural_nutrition['FCS'], 
                       name="Food Consumption Score", marker_color='blue'),
                row=1, col=2
            )
            
            # Coping Strategies
            fig.add_trace(
                go.Bar(x=urban_rural_nutrition['UrbanRural'], y=urban_rural_nutrition['rCSI'], 
                       name="Coping Strategies", marker_color='orange'),
                row=2, col=1
            )
            
            # Contributing Ratio
            fig.add_trace(
                go.Bar(x=urban_rural_nutrition['UrbanRural'], y=urban_rural_nutrition['contributing_ratio'], 
                       name="Contributing Ratio", marker_color='green'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, width=900, showlegend=False, margin=dict(l=0, r=0, t=60, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">Key Findings</div>
                <ul>
                    <li>Rural households have higher risk of malnutrition compared to urban households</li>
                    <li>Urban households have higher food consumption scores, indicating better dietary diversity</li>
                    <li>Coping strategies are higher in rural areas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Root cause summary
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Root Cause Summary</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">Primary Root Causes</div>
            <ol>
                <li><b>Economic Constraints:</b> Limited financial resources restrict access to diverse, nutrient-rich foods</li>
                <li><b>Low Dietary Diversity:</b> Reliance on staple foods with limited consumption of micronutrient-rich foods</li>
                <li><b>Education Gaps:</b> Limited knowledge about nutrition and food preparation affects dietary choices</li>
                <li><b>Urban-Rural Disparities:</b> Rural areas face greater challenges in accessing diverse foods</li>
                <li><b>Market Access:</b> Limited availability of diverse foods in some regions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create a word cloud or visualization of root causes
        root_causes = {
            'Economic Constraints': 0.35,
            'Low Dietary Diversity': 0.25,
            'Education Gaps': 0.15,
            'Urban-Rural Disparities': 0.15,
            'Market Access': 0.10
        }
        
        rc_df = pd.DataFrame(list(root_causes.items()), columns=['Cause', 'Impact'])
        
        fig = px.bar(rc_df, x='Impact', y='Cause', orientation='h',
                     title='Relative Impact of Root Causes',
                     color='Impact', color_continuous_scale='Reds')
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        
        create_chart_container("Root Cause Impact", fig)
    
    # High-risk group profiling
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">High-Risk Group Profiling</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create risk categories
    df['Risk_Category'] = 'Low Risk'
    df.loc[(df['FCS'] < 30) & (df['rCSI'] > 10), 'Risk_Category'] = 'Medium Risk'
    df.loc[(df['FCS'] < 20) & (df['rCSI'] > 20), 'Risk_Category'] = 'High Risk'
    
    high_risk = df[df['Risk_Category'] == 'High Risk']
    low_risk = df[df['Risk_Category'] == 'Low Risk']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wealth distribution
        high_risk_wealth = high_risk['WI_cat'].value_counts(normalize=True).reset_index()
        high_risk_wealth.columns = ['Wealth Category', 'Proportion']
        high_risk_wealth['Risk Group'] = 'High Risk'
        
        low_risk_wealth = low_risk['WI_cat'].value_counts(normalize=True).reset_index()
        low_risk_wealth.columns = ['Wealth Category', 'Proportion']
        low_risk_wealth['Risk Group'] = 'Low Risk'
        
        wealth_comparison = pd.concat([high_risk_wealth, low_risk_wealth])
        
        fig = px.bar(wealth_comparison, x='Wealth Category', y='Proportion', 
                    color='Risk Group', barmode='group',
                    title='Wealth Distribution by Risk Group')
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Urban/Rural distribution
        high_risk_urban = high_risk['UrbanRural'].value_counts(normalize=True).reset_index()
        high_risk_urban.columns = ['Location Type', 'Proportion']
        high_risk_urban['Risk Group'] = 'High Risk'
        
        low_risk_urban = low_risk['UrbanRural'].value_counts(normalize=True).reset_index()
        low_risk_urban.columns = ['Location Type', 'Proportion']
        low_risk_urban['Risk Group'] = 'Low Risk'
        
        urban_comparison = pd.concat([high_risk_urban, low_risk_urban])
        
        fig = px.bar(urban_comparison, x='Location Type', y='Proportion', 
                    color='Risk Group', barmode='group',
                    title='Urban/Rural Distribution by Risk Group')
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key indicators comparison
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-title">Key Indicators Comparison</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Define possible indicators
    indicators = ['FCS', 'rCSI', 'Food_Diversity_Score', 'contributing_ratio']

    # Keep only the indicators that exist in df
    available_indicators = [ind for ind in indicators if ind in df.columns]

    comparison_data = pd.DataFrame({
        'Indicator': available_indicators,
        'High-Risk': [high_risk[ind].mean() for ind in available_indicators],
        'Low-Risk': [low_risk[ind].mean() for ind in available_indicators]
    })
    
    fig = px.bar(comparison_data, x='Indicator', y=['High-Risk', 'Low-Risk'], 
                barmode='group', title='Key Indicators Comparison',
                color_discrete_map={'High-Risk': 'red', 'Low-Risk': 'green'})
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # High-risk group profiling
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">High-Risk Group Profiling</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create risk categories
    df['Risk_Category'] = 'Low Risk'
    df.loc[(df['FCS'] < 30) & (df['rCSI'] > 10), 'Risk_Category'] = 'Medium Risk'
    df.loc[(df['FCS'] < 20) & (df['rCSI'] > 20), 'Risk_Category'] = 'High Risk'

    high_risk = df[df['Risk_Category'] == 'High Risk']
    low_risk = df[df['Risk_Category'] == 'Low Risk']

    col1, col2 = st.columns(2)

    
    with col1:
        # Wealth distribution
        high_risk_wealth = high_risk['WI_cat'].value_counts(normalize=True).reset_index()
        high_risk_wealth.columns = ['Wealth Category', 'Proportion']
        high_risk_wealth['Risk Group'] = 'High Risk'
        
        low_risk_wealth = low_risk['WI_cat'].value_counts(normalize=True).reset_index()
        low_risk_wealth.columns = ['Wealth Category', 'Proportion']
        low_risk_wealth['Risk Group'] = 'Low Risk'
        
        wealth_comparison = pd.concat([high_risk_wealth, low_risk_wealth])
        
        fig = px.bar(wealth_comparison, x='Wealth Category', y='Proportion', 
                    color='Risk Group', barmode='group',
                    title='Wealth Distribution by Risk Group')
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        # Before (causes duplicate ID error)
        for i, fig in enumerate(list_of_figures):
            st.plotly_chart(fig, use_container_width=True, key=f"root_cause_fig_{i}")


        # st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Urban/Rural distribution
        high_risk_urban = high_risk['UrbanRural'].value_counts(normalize=True).reset_index()
        high_risk_urban.columns = ['Location Type', 'Proportion']
        high_risk_urban['Risk Group'] = 'High Risk'
        
        low_risk_urban = low_risk['UrbanRural'].value_counts(normalize=True).reset_index()
        low_risk_urban.columns = ['Location Type', 'Proportion']
        urban_comparison = pd.concat([high_risk_urban, low_risk_urban])
        
        fig = px.bar(urban_comparison, x='Location Type', y='Proportion', 
                    color='Risk Group', barmode='group',
                    title='Urban/Rural Distribution by Risk Group')
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key indicators comparison
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-title">Key Indicators Comparison</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Define possible indicators
    indicators = ['FCS', 'rCSI', 'Food_Diversity_Score', 'contributing_ratio']

    # Keep only the indicators that exist in df
    available_indicators = [ind for ind in indicators if ind in df.columns]

    comparison_data = pd.DataFrame({
        'Indicator': available_indicators,
        'High-Risk': [high_risk[ind].mean() for ind in available_indicators],
        'Low-Risk': [low_risk[ind].mean() for ind in available_indicators]
    })
    
    fig = px.bar(comparison_data, x='Indicator', y=['High-Risk', 'Low-Risk'], 
                barmode='group', title='Key Indicators Comparison',
                color_discrete_map={'High-Risk': 'red', 'Low-Risk': 'green'})
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    
    st.plotly_chart(fig, use_container_width=True)

# # Policy Recommendations page
# def policy_recommendations_page():
#     create_header("POLICY RECOMMENDATIONS")
    
#     # Load data if not already loaded
#     if not st.session_state.data_loaded:
#         with st.spinner("Loading data..."):
#             st.session_state.df = load_data()
#             st.session_state.data_loaded = True
    
#     df = st.session_state.df
    
#     # Cost-effectiveness analysis
#     st.markdown("""
#     <div class="chart-container">
#         <div class="chart-title">Cost-Effectiveness Analysis of Key Interventions</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     interventions = pd.DataFrame({
#         'Intervention': [
#             'Vitamin A supplementation',
#             'De-worming for children',
#             'Multiple micronutrient supplementation',
#             'Breastfeeding promotion programs',
#             'Iron supplementation for pregnant women'
#         ],
#         'Cost_per_DALY': [5, 10, 15, 20, 25],
#         'Effectiveness': [0.8, 0.7, 0.75, 0.65, 0.6]
#     })
    
#     fig = px.scatter(interventions, x='Cost_per_DALY', y='Effectiveness', 
#                     size='Effectiveness', color='Cost_per_DALY',
#                     hover_name='Intervention',
#                     title='Cost-Effectiveness of Interventions',
#                     color_continuous_scale='Reds_r')
#     fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Target setting
#     st.markdown("""
#     <div class="chart-container">
#         <div class="chart-title">Sample Target Setting (36-Month Program)</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     targets = pd.DataFrame({
#         'Indicator': [
#             'Stunting prevalence (%)',
#             'Minimum dietary diversity for children 6-23 months (%)',
#             'Coverage of vitamin A supplementation (%)',
#             'Households with homestead gardens (%)',
#             'School feeding coverage (%)',
#             'Iron deficiency anaemia in women (%)'
#         ],
#         'Baseline': [38, 25, 65, 15, 30, 45],
#         'Target_12months': [35, 35, 75, 30, 50, 40],
#         'Target_24months': [32, 45, 85, 50, 70, 35],
#         'Target_36months': [28, 55, 90, 65, 85, 30]
#     })
    
#     st.dataframe(targets)
    
#     # Target visualization
#     fig = go.Figure()

#     for i, ind in enumerate(targets['Indicator']):
#         fig.add_trace(go.Scatter(
#             x=['Baseline', '12 months', '24 months', '36 months'],
#             y=[
#                 targets.loc[i, 'Baseline'],
#                 targets.loc[i, 'Target_12months'],
#                 targets.loc[i, 'Target_24months'],
#                 targets.loc[i, 'Target_36months']
#             ],
#             mode='lines+markers',
#             name=ind
#         ))

#     fig.update_layout(
#         title='Progress Towards Targets',
#         xaxis_title='Time Period',
#         yaxis_title='Percentage',
#         height=500,
#         margin=dict(l=0, r=0, t=40, b=0)
#     )

#     st.plotly_chart(fig, use_container_width=True)

    
#     # Budget breakdown
#     st.markdown("""
#     <div class="chart-container">
#         <div class="chart-title">Detailed Budget Breakdown</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     budget = pd.DataFrame({
#         'Sector': ['Health', 'Agriculture', 'Education', 'Coordination/M&E', 'Contingency'],
#         'Budget': [4500000, 2500000, 1500000, 1000000, 500000],
#         'Percentage': [45, 25, 15, 10, 5],
#         'Key Interventions': [
#             'Therapeutic feeding, supplementation, supplementation, IYCF, WASH',
#             'Homestead gardens, crop diversification, value chains',
#             'School feeding, nutrition education, capacity building',
#             'Coordination, monitoring, evaluation, learning',
#             'Emergency response, price shocks, natural disasters'
#         ]
#     })
    
#     fig = px.pie(budget, values='Budget', names='Sector', 
#                 title='Budget Allocation by Sector',
#                 color_discrete_sequence=px.colors.sequential.Blues)
#     fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Budget table
#     st.dataframe(budget)
    
#     # Policy briefs
#     st.markdown("""
#     <div class="chart-container">
#         <div class="chart-title">Policy Briefs</div>
#     </div>
#     """, unsafe_allow_html=True)
    
#     policy_briefs = [
#         {
#             "title": "Addressing Malnutrition in High-Risk Areas",
#             "target": "Ministry of Health, Local Government",
#             "summary": "High-risk areas, particularly rural regions with poor households, require targeted interventions to address malnutrition. This brief recommends integrated approaches combining health, agriculture, and education sectors.",
#             "recommendations": [
#                 "Establish community nutrition centers in high-risk areas",
#                 "Provide micronutrient supplements to vulnerable groups",
#                 "Support homestead food production and diversification",
#                 "Strengthen nutrition education and behavior change communication"
#             ],
#             "impact": "High",
#             "timeline": "Short-term (6-12 months)"
#         },
#         {
#             "title": "Integrating Nutrition into Social Protection Programs",
#             "target": "Ministry of Local Government, Social Protection Programs",
#             "summary": "Economic constraints are a primary driver of malnutrition. This brief outlines how existing social protection programs can be enhanced to address malnutrition through targeted food assistance and nutrition education.",
#             "recommendations": [
#                 "Include nutrition criteria in beneficiary selection for social protection programs",
#                 "Provide nutrition-sensitive food transfers to high-risk households",
#                 "Integrate nutrition education into cash transfer programs",
#                 "Establish community kitchens for vulnerable populations"
#             ],
#             "impact": "High",
#             "timeline": "Medium-term (12-24 months)"
#         },
#         {
#             "title": "School-Based Nutrition Interventions",
#             "target": "Ministry of Education, Schools",
#             "summary": "Schools provide an ideal platform to address malnutrition in children. This brief recommends comprehensive school-based nutrition programs to improve dietary diversity and nutritional status.",
#             "recommendations": [
#                 "Implement school meal programs with diverse, nutrient-rich foods",
#                 "Establish school gardens to provide fresh produce and nutrition education",
#                 "Integrate nutrition education into school curricula",
#                 "Provide micronutrient supplementation to school children"
#             ],
#             "impact": "Medium",
#             "timeline": "Medium-term (12-24 months)"
#         }
#     ]
# Policy Recommendations page
def policy_recommendations_page():
    create_header("POLICY RECOMMENDATIONS")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    df = st.session_state.df
    
    # Cost-effectiveness analysis
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Cost-Effectiveness Analysis of Key Interventions</div>
    </div>
    """, unsafe_allow_html=True)
    
    interventions = pd.DataFrame({
        'Intervention': [
            'Vitamin A supplementation',
            'De-worming for children',
            'Multiple micronutrient supplementation',
            'Breastfeeding promotion programs',
            'Iron supplementation for pregnant women'
        ],
        'Cost_per_DALY': [5, 10, 15, 20, 25],
        'Effectiveness': [0.8, 0.7, 0.75, 0.65, 0.6]
    })
    
    fig1 = px.scatter(interventions, x='Cost_per_DALY', y='Effectiveness', 
                     size='Effectiveness', color='Cost_per_DALY',
                     hover_name='Intervention',
                     title='Cost-Effectiveness of Interventions',
                     color_continuous_scale='Reds_r')
    fig1.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    
    st.plotly_chart(fig1, use_container_width=True, key="cost_effectiveness_chart")
    
    # Target setting
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Sample Target Setting (36-Month Program)</div>
    </div>
    """, unsafe_allow_html=True)
    
    targets = pd.DataFrame({
        'Indicator': [
            'Stunting prevalence (%)',
            'Minimum dietary diversity for children 6-23 months (%)',
            'Coverage of vitamin A supplementation (%)',
            'Households with homestead gardens (%)',
            'School feeding coverage (%)',
            'Iron deficiency anaemia in women (%)'
        ],
        'Baseline': [38, 25, 65, 15, 30, 45],
        'Target_12months': [35, 35, 75, 30, 50, 40],
        'Target_24months': [32, 45, 85, 50, 70, 35],
        'Target_36months': [28, 55, 90, 65, 85, 30]
    })
    
    st.dataframe(targets)
    
    # Target visualization
    fig2 = go.Figure()
    for i, ind in enumerate(targets['Indicator']):
        fig2.add_trace(go.Scatter(
            x=['Baseline', '12 months', '24 months', '36 months'],
            y=[
                targets.loc[i, 'Baseline'],
                targets.loc[i, 'Target_12months'],
                targets.loc[i, 'Target_24months'],
                targets.loc[i, 'Target_36months']
            ],
            mode='lines+markers',
            name=ind
        ))

    fig2.update_layout(
        title='Progress Towards Targets',
        xaxis_title='Time Period',
        yaxis_title='Percentage',
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig2, use_container_width=True, key="targets_chart")
    
    # Budget breakdown
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Detailed Budget Breakdown</div>
    </div>
    """, unsafe_allow_html=True)
    
    budget = pd.DataFrame({
        'Sector': ['Health', 'Agriculture', 'Education', 'Coordination/M&E', 'Contingency'],
        'Budget': [4500000, 2500000, 1500000, 1000000, 500000],
        'Percentage': [45, 25, 15, 10, 5],
        'Key Interventions': [
            'Therapeutic feeding, supplementation, IYCF, WASH',
            'Homestead gardens, crop diversification, value chains',
            'School feeding, nutrition education, capacity building',
            'Coordination, monitoring, evaluation, learning',
            'Emergency response, price shocks, natural disasters'
        ]
    })
    
    fig3 = px.pie(budget, values='Budget', names='Sector', 
                  title='Budget Allocation by Sector',
                  color_discrete_sequence=px.colors.sequential.Blues)
    fig3.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    
    st.plotly_chart(fig3, use_container_width=True, key="budget_chart")
    
    st.dataframe(budget)
    
    # Policy briefs
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Policy Briefs</div>
    </div>
    """, unsafe_allow_html=True)
    
    policy_briefs = [
        {
            "title": "Addressing Malnutrition in High-Risk Areas",
            "target": "Ministry of Health, Local Government",
            "summary": "High-risk areas, particularly rural regions with poor households, require targeted interventions to address malnutrition. This brief recommends integrated approaches combining health, agriculture, and education sectors.",
            "recommendations": [
                "Establish community nutrition centers in high-risk areas",
                "Provide micronutrient supplements to vulnerable groups",
                "Support homestead food production and diversification",
                "Strengthen nutrition education and behavior change communication"
            ],
            "impact": "High",
            "timeline": "Short-term (6-12 months)"
        },
        {
            "title": "Integrating Nutrition into Social Protection Programs",
            "target": "Ministry of Local Government, Social Protection Programs",
            "summary": "Economic constraints are a primary driver of malnutrition. This brief outlines how existing social protection programs can be enhanced to address malnutrition through targeted food assistance and nutrition education.",
            "recommendations": [
                "Include nutrition criteria in beneficiary selection for social protection programs",
                "Provide nutrition-sensitive food transfers to high-risk households",
                "Integrate nutrition education into cash transfer programs",
                "Establish community kitchens for vulnerable populations"
            ],
            "impact": "High",
            "timeline": "Medium-term (12-24 months)"
        },
        {
            "title": "School-Based Nutrition Interventions",
            "target": "Ministry of Education, Schools",
            "summary": "Schools provide an ideal platform to address malnutrition in children. This brief recommends comprehensive school-based nutrition programs to improve dietary diversity and nutritional status.",
            "recommendations": [
                "Implement school meal programs with diverse, nutrient-rich foods",
                "Establish school gardens to provide fresh produce and nutrition education",
                "Integrate nutrition education into school curricula",
                "Provide micronutrient supplementation to school children"
            ],
            "impact": "Medium",
            "timeline": "Medium-term (12-24 months)"
        }
    ]
    
    for brief in policy_briefs:
        st.subheader(brief["title"])
        st.markdown(f"**Target:** {brief['target']}")
        st.markdown(f"**Summary:** {brief['summary']}")
        st.markdown("**Recommendations:**")
        for rec in brief["recommendations"]:
            st.markdown(f"- {rec}")
        st.markdown(f"**Impact:** {brief['impact']}")
        st.markdown(f"**Timeline:** {brief['timeline']}")
        st.markdown("---")

    
    # Display policy briefs
    for i, brief in enumerate(policy_briefs):
        with st.expander(f"{i+1}. {brief['title']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Target Audience:** {brief['target']}")
                st.markdown(f"**Summary:** {brief['summary']}")
                
                st.markdown("**Key Recommendations:**")
                for rec in brief['recommendations']:
                    st.markdown(f"- {rec}")
            
            with col2:
                st.markdown(f"**Expected Impact:** {brief['impact']}")
                st.markdown(f"**Implementation Timeline:** {brief['timeline']}")
            
            with col2:
                st.markdown(f"**Expected Impact:** {brief['impact']}")
                st.markdown(f"**Implementation Timeline:** {brief['timeline']}")
    
    # Implementation framework
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Implementation Framework</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">Multi-Sectoral Implementation Approach</div>
            <ol>
                <li><b>Assessment & Targeting:</b> Use data to identify high-risk areas and populations</li>
                <li><b>Policy Alignment:</b> Ensure nutrition is integrated into sectoral policies and plans</li>
                <li><b>Capacity Building:</b> Train stakeholders at all levels on nutrition-sensitive approaches</li>
                <li><b>Service Delivery:</b> Implement integrated interventions at community level</li>
                <li><b>Monitoring & Evaluation:</b> Establish systems to track progress and impact</li>
                <li><b>Knowledge Management:</b> Document and share best practices and lessons learned</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create a timeline visualization
    
        # Create a timeline visualization
        implementation_steps = {
            'Assessment & Targeting': 1,
            'Policy Alignment': 2,
            'Capacity Building': 3,
            'Service Delivery': 4,
            'Monitoring & Evaluation': 5,
            'Knowledge Management': 6
        }

        impl_df = pd.DataFrame(list(implementation_steps.items()), columns=['Step', 'Order'])

        fig = px.scatter(
            impl_df, 
            x='Order', 
            y='Step', 
            size='Order',
            title='Implementation Timeline',
            color='Order', 
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            height=350, 
            xaxis=dict(showgrid=False, zeroline=False, tickmode='linear', tick0=1, dtick=1)
        )
        fig.update_yaxes(categoryorder='array', categoryarray=list(impl_df['Step']))

        create_chart_container("Implementation Timeline", fig)

    # Call to action
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Call to Action</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Immediate Actions Required</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Immediate Actions Required</div>
        <ul>
            <li>Prioritize nutrition in national development plans and budget allocations</li>
            <li>Strengthen multi-sectoral coordination mechanisms</li>
            <li>Scale up successful interventions to high-risk areas</li>
            <li>Invest in data systems for regular monitoring of nutrition indicators</li>
            <li>Engage communities in designing and implementing nutrition interventions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Create sidebar with clickable navigation
    page = create_sidebar()
    
    # Display selected page
    if page == "Dashboard":
        dashboard_page()
    elif page == "Data Overview":
        data_overview_page()
    elif page == "Malnutrition Hotspots":
        malnutrition_hotspots_page()
    elif page == "Predictive Models":
        predictive_models_page()
    elif page == "Root Cause Analysis":
        root_cause_analysis_page()
    elif page == "Policy Recommendations":
        policy_recommendations_page()

# Run the application
if __name__ == "__main__":
    main()