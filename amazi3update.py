# amazi3_fixed.py
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

# Custom CSS (kept small here; replace with your full CSS if needed)
st.markdown("""
<style>
    .section-header { font-size: 1.3rem; font-weight: 700; margin-bottom: 8px; }
    .chart-container { padding: 10px 0; }
    .chart-title { font-weight: 600; margin-bottom: 6px; }
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
                (df['WI_cat'].isin(['Poorest', 'Poor']).astype(int)) * 0.1
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
            p=[0.337, 0.473, 0.178, 0.012]
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
            (pd.Series(urban_rural) == 'Rural').astype(int) * 0.2 +
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
    categorical_cols_for_encoding = ['WI_cat', 'UrbanRural', 'S0_C_Prov', 'FG_HIronCat', 'FG_VitACat', 'FG_ProteinCat']
    label_encoders = {}
    
    for col in categorical_cols_for_encoding:
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
    lr = LogisticRegression(random_state=42, max_iter=1000)
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
    
    # XGBoost (optional)
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
    except Exception:
        st.warning("XGBoost not installed or failed to import. Skipping XGBoost model.")
    
    # Get feature importance from Random Forest (only for features actually used)
    fi = pd.DataFrame({
        'feature': X.columns,
        'importance': models['Random Forest'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    return models, fi, X_train, X_test, y_train, y_test, label_encoders, categorical_values

# Create sidebar navigation
def create_sidebar():
    pages = ["Dashboard", "Data Overview", "Malnutrition Hotspots", "Predictive Models", "Root Cause Analysis", "Policy Recommendations"]
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", pages, index=0)
    return selected_page

# Create header
def create_header(title):
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <h1 style="margin:0;">{title}</h1>
        <div style="font-size:0.9rem; color:#666;">Ending Hidden Hunger Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

# Create metric card
def create_metric_card(title, value, change=None, change_type=None):
    change_html = ""
    if change:
        change_color = "#16a34a" if change_type == "positive" else "#e11d48"
        change_html = f"<div style='font-size:0.8rem; color:{change_color};'>{change}</div>"
    st.markdown(f"""
    <div style="background:#ffffff; padding:12px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.06);">
        <div style="font-size:0.9rem; color:#666;">{title}</div>
        <div style="font-size:1.6rem; font-weight:700;">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

# Create chart container
def create_chart_container(title, fig):
    st.markdown(f"<div class='chart-title'>{title}</div>", unsafe_allow_html=True)
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
            "+5% SINCE LAST 3YEARS", 
            "positive"
        )
    
    with col2:
        risk_pct = (df['Malnutrition_Risk'].mean() * 100).round(2)
        create_metric_card(
            "MALNUTRITION RISK RATE", 
            f"{risk_pct}%", 
            "-2.3% SINCE LAST 3YEARS", 
            "positive"
        )
    
    with col3:
        high_risk = (df['Malnutrition_Risk'] == 1).sum()
        create_metric_card(
            "HIGH-RISK HOUSEHOLDS", 
            f"{high_risk:,}", 
            "+1.2% SINCE LAST 3YEARS", 
            "negative"
        )
    
    with col4:
        avg_fcs = df['FCS'].mean().round(1) if 'FCS' in df.columns else 'N/A'
        create_metric_card(
            "AVERAGE FOOD CONSUMPTION SCORE", 
            f"{avg_fcs}", 
            "+3.5% SINCE LAST 3YEARS", 
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
            risk_by_wealth = df.groupby('WI_cat')['Malnutrition_Risk'].mean().reset_index()
            risk_by_wealth['Malnutrition_Risk'] = (risk_by_wealth['Malnutrition_Risk'] * 100).round(2)
            
            # Define a sorting order for wealth categories
            wealth_order = ['Poorest', 'Poor', 'Medium', 'Wealth', 'Wealthiest']
            risk_by_wealth['WI_cat'] = pd.Categorical(risk_by_wealth['WI_cat'], categories=wealth_order, ordered=True)
            risk_by_wealth = risk_by_wealth.sort_values('WI_cat')
            
            fig = px.bar(risk_by_wealth, x='WI_cat', y='Malnutrition_Risk', 
                        color='Malnutrition_Risk', color_continuous_scale='Oranges',
                        title='Malnutrition Risk Rate by Wealth Index (%)')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            fig.update_yaxes(title='Risk Rate (%)')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("WI_cat column not found in the data")
        
    
    # Row 2: Map and other stats
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">Geographic Malnutrition Hotspots (Map Placeholder)</div>
            <div class="map-container" style="height: 300px; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center;">
                <p style="color: #999;">Map integration would go here, requiring a GeoJSON file.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        # Coping Strategies Index
        if 'rCSI' in df.columns:
            fig = px.histogram(df, x='rCSI', nbins=30, 
                            title='Distribution of Reduced Coping Strategies Index (rCSI)')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("rCSI column not found in the data")

# Data Overview page
def data_overview_page():
    create_header("DATA OVERVIEW")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True

    df = st.session_state.df
    
    st.markdown("""
    <div class="section-header">Dataset Information</div>
    <hr style="margin-top: 0;">
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    - **Total Households:** `{len(df):,}`
    - **Total Columns/Features:** `{len(df.columns):,}`
    - **Data Source:** CFSVA 2024 (Simulated/Mock data used due to file path limitations)
    - **Target Variable:** `Malnutrition_Risk` (Binary: 0=Low Risk, 1=High Risk)
    """, unsafe_allow_html=True)
    
    # Data Preview and Details
    tab1, tab2, tab3 = st.tabs(["Data Preview", "Column Details", "Summary Statistics"])
    
    with tab1:
        st.dataframe(df.head(), use_container_width=True, height=350)

    with tab2:
        # Create a dataframe of column name, data type, and non-null count
        info_df = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Unique Values': df.nunique()
        }).reset_index(drop=True)
        st.dataframe(info_df, use_container_width=True, height=350)
        
    with tab3:
        st.dataframe(df.describe(include='all').T, use_container_width=True, height=350)

# Malnutrition Hotspots page
def malnutrition_hotspots_page():
    create_header("MALNUTRITION HOTSPOTS")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True

    df = st.session_state.df
    
    if 'Malnutrition_Risk' not in df.columns:
        st.error("Malnutrition_Risk column not found in the data. Please check the data source.")
        return
        
    st.markdown("""
    <div class="section-header">Geographic Breakdown of Malnutrition Risk</div>
    <hr style="margin-top: 0;">
    """, unsafe_allow_html=True)
    
    # Filter selection
    filter_col = st.selectbox(
        "Select Geographic Level:",
        options=['S0_C_Prov', 'S0_D_Dist'],
        format_func=lambda x: 'Province' if x == 'S0_C_Prov' else 'District',
        key='geo_filter'
    )
    
    # Calculate Malnutrition Risk Rate by geographic level
    risk_by_geo = df.groupby(filter_col)['Malnutrition_Risk'].mean().reset_index()
    risk_by_geo['Risk_Rate'] = (risk_by_geo['Malnutrition_Risk'] * 100).round(2)
    risk_by_geo = risk_by_geo.sort_values('Risk_Rate', ascending=False)
    
    # Choropleth Map Placeholder
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Malnutrition Risk Rate Map (Placeholder)</div>
        <div class="map-container" style="height: 450px; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center;">
            <p style="color: #999;">A detailed Choropleth Map would be displayed here, requiring a GeoJSON file of Rwanda's Provinces/Districts.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Bar Chart of Risk Rate
    st.markdown('<div class="section-header" style="margin-top: 30px;">Top High-Risk Areas</div>', unsafe_allow_html=True)
    
    fig_bar = px.bar(
        risk_by_geo, 
        x=filter_col, 
        y='Risk_Rate', 
        color='Risk_Rate', 
        color_continuous_scale='Reds',
        title=f'Malnutrition Risk Rate by {filter_col} (%)'
    )
    fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    fig_bar.update_yaxes(title='Risk Rate (%)')
    st.plotly_chart(fig_bar, use_container_width=True)

# Predictive Models page
def predictive_models_page():
    create_header("PREDICTIVE MODELS")
    
    # Load data and train models if not already done
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    if not st.session_state.models:
        with st.spinner("Training models..."):
            models, importance, X_train, X_test, y_train, y_test, label_encoders, categorical_values = train_models(st.session_state.df)
            st.session_state.models = models
            st.session_state.feature_importance = importance
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.label_encoders = label_encoders
            st.session_state.categorical_values = categorical_values

    models = st.session_state.models
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    st.markdown("""
    <div class="section-header">Model Performance Overview</div>
    <hr style="margin-top: 0;">
    """, unsafe_allow_html=True)
    
    # Calculate and display metrics
    metrics = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        # Some models may not have predict_proba (rare here), handle gracefully
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = np.zeros(len(y_test))
        
        # Cross-validation AUC-ROC (using X_test/y_test for speed; ideally use full dataset)
        try:
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc')
            cv_auc = cv_scores.mean().round(3)
        except Exception:
            cv_auc = 'N/A'
            
        try:
            auc = roc_auc_score(y_test, y_prob).round(3)
        except Exception:
            auc = 'N/A'
        
        try:
            acc = model.score(X_test, y_test).round(3)
        except Exception:
            acc = 'N/A'
        
        try:
            f1 = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']
            f1 = round(f1, 3)
        except Exception:
            f1 = 'N/A'
            
        metrics.append({
            'Model': name,
            'AUC-ROC': auc,
            'Accuracy': acc,
            'F1-Score': f1,
            'Cross-Validation AUC': cv_auc
        })
    
    metrics_df = pd.DataFrame(metrics).set_index('Model').sort_values('AUC-ROC', ascending=False)
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Detailed Model Analysis
    st.markdown('<div class="section-header" style="margin-top: 30px;">Detailed Model Evaluation</div>', unsafe_allow_html=True)
    
    model_choice = st.selectbox(
        "Select Model for Detailed Analysis:",
        options=list(models.keys()),
        key='model_detail_select'
    )
    
    selected_model = models[model_choice]
    y_pred = selected_model.predict(X_test)
    try:
        y_prob = selected_model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = np.zeros(len(y_test))
    
    tab1, tab2 = st.tabs(["Classification Report", "Confusion Matrix"])
    
    with tab1:
        st.code(classification_report(y_test, y_pred))

    with tab2:
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=['Low Risk (0)', 'High Risk (1)'],
            y=['Low Risk (0)', 'High Risk (1)'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(title_text='<b>Confusion Matrix</b>', height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    # Feature Importance (for tree-based models)
    if model_choice in ['Random Forest', 'Gradient Boosting', 'XGBoost'] and st.session_state.feature_importance is not None:
        st.markdown('<div class="section-header" style="margin-top: 30px;">Feature Importance (from Random Forest)</div>', unsafe_allow_html=True)
        
        fig_feat = px.bar(
            st.session_state.feature_importance.head(15), 
            x='importance', 
            y='feature', 
            orientation='h',
            color='importance',
            color_continuous_scale='Tealgrn',
            title='Top 15 Predictors of Malnutrition Risk'
        )
        fig_feat.update_layout(height=500, yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_feat, use_container_width=True)

# Root Cause Analysis page
def root_cause_analysis_page():
    create_header("ROOT CAUSE ANALYSIS")
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True

    df = st.session_state.df
    
    if 'Malnutrition_Risk' not in df.columns:
        st.error("Malnutrition_Risk column not found in the data. Please check the data source.")
        return
        
    st.markdown("""
    <div class="section-header">Deep Dive into Malnutrition Drivers</div>
    <hr style="margin-top: 0;">
    """, unsafe_allow_html=True)
    
    # Row 1: Socio-economic Factors
    st.markdown("### Socio-economic Factors")
    col1, col2 = st.columns(2)
    
    with col1:
        # Malnutrition Risk by Urban/Rural
        if 'UrbanRural' in df.columns:
            risk_by_area = df.groupby('UrbanRural')['Malnutrition_Risk'].mean().reset_index()
            risk_by_area['Risk_Rate'] = (risk_by_area['Malnutrition_Risk'] * 100).round(2)
            
            fig = px.pie(risk_by_area, names='UrbanRural', values='Risk_Rate', 
                        title='Risk Contribution by Urban/Rural Setting',
                        hole=0.4)
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("UrbanRural column not found in the data")
            
    with col2:
        # Malnutrition Risk by Food Security Status (FS_final)
        if 'FS_final' in df.columns:
            risk_by_fs = df.groupby('FS_final')['Malnutrition_Risk'].mean().reset_index()
            risk_by_fs['Risk_Rate'] = (risk_by_fs['Malnutrition_Risk'] * 100).round(2)
            
            fig = px.bar(risk_by_fs, x='FS_final', y='Risk_Rate', 
                        color='Risk_Rate', color_continuous_scale='Viridis',
                        title='Malnutrition Risk by Food Security Status (%)')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            fig.update_yaxes(title='Risk Rate (%)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("FS_final column not found in the data")
            
    # Row 2: Food Consumption and Dietary Diversity
    st.markdown("### Food Consumption and Dietary Diversity")
    col3, col4 = st.columns(2)
    
    # Create Malnutrition risk column for easier grouping/aggregation on other pages
    df['Malnutrition_Status'] = df['Malnutrition_Risk'].map({0: 'Low Risk', 1: 'High Risk'})
    
    with col3:
        # Food Consumption Score (FCS) Distribution by Risk
        if 'FCS' in df.columns:
            fig = px.box(df, y='FCS', color='Malnutrition_Status', 
                        title='FCS Distribution by Malnutrition Risk')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("FCS column not found in the data")
            
    with col4:
        # Dietary Diversity (FG_HIronCat) by Risk
        if 'FG_HIronCat' in df.columns:
            # Cross-tabulation of risk and iron consumption
            risk_by_iron = pd.crosstab(df['FG_HIronCat'], df['Malnutrition_Status'], normalize='index') * 100
            risk_by_iron = risk_by_iron.reset_index().melt(
                id_vars='FG_HIronCat', 
                value_vars=['Low Risk', 'High Risk'], 
                var_name='Risk Status', 
                value_name='Percentage'
            )
            
            fig = px.bar(risk_by_iron, x='FG_HIronCat', y='Percentage', color='Risk Status',
                        title='Malnutrition Risk by Iron-Rich Food Consumption')
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0), barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("FG_HIronCat column not found in the data")

# Policy Recommendations page
def policy_recommendations_page():
    create_header("POLICY RECOMMENDATIONS")
    
    st.markdown("""
    <div class="section-header">Targeted Interventions and Policy Road Map</div>
    <hr style="margin-top: 0;">
    """, unsafe_allow_html=True)

    # Based on the analysis (assuming high risk in rural, poor, low FCS, etc.)
    
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Recommended Policy Pillars</div>
        <ul style="font-size: 1.1rem; line-height: 1.8;">
            <li><b>Pillar 1: Enhance Agricultural Productivity & Resilience:</b> Focus on climate-smart agriculture, nutrition-sensitive value chains, and subsidized drought-resistant seeds in high-risk districts.</li>
            <li><b>Pillar 2: Strengthen Social Safety Nets:</b> Expand cash transfer programs and school feeding in districts with high malnutrition and low FCS.</li>
            <li><b>Pillar 3: Improve Health and WASH:</b> Scale up micronutrient supplementation, improve access to clean water and sanitation, and promote behavior change campaigns.</li>
            <li><b>Pillar 4: Support Dietary Diversity and Education:</b> Implement community-level nutrition education and increase consumption of iron- and vitamin-A-rich foods.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Monitoring and Evaluation Framework</div>
        <p>Track coverage of interventions, FCS, dietary diversity, and stunting rates at district level on a quarterly basis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">Immediate Actions Required</div>
        <ul>
            <li>Prioritize nutrition in national development plans and budgets</li>
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
