import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="CFSVA Risk Assessment Dashboard",
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
    .risk-high { 
        background-color: #ff6b6b; 
        color: white; 
        padding: 10px; 
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .risk-medium { 
        background-color: #ffd93d; 
        color: black; 
        padding: 10px; 
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .risk-low { 
        background-color: #6bcf7f; 
        color: white; 
        padding: 10px; 
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    /* Map container */
    .map-container {
        height: 1000;
        border-radius: 8px;
        overflow: hidden;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class CFSVAAnalysis:
    def __init__(self):
        self.hh_data = None
        self.child_data = None
        self.women_data = None
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        
    def load_data(self):
        """Load and merge datasets"""
        try:
            # Load datasets with optimized data types
            self.hh_data = pd.read_csv('Microdata1111/Microdata/csvfile/CFSVA2024_HH data.csv', low_memory=False)
            self.child_data = pd.read_csv('Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv', low_memory=False)
            self.women_data = pd.read_csv('Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv', low_memory=False)
            
            st.success("✅ Datasets loaded successfully!")
            
            # Show basic info about datasets
            st.sidebar.info(f"Household data: {self.hh_data.shape}")
            st.sidebar.info(f"Child data: {self.child_data.shape}")
            st.sidebar.info(f"Women data: {self.women_data.shape}")
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            # Create sample data for demonstration
            st.info("🔄 Creating sample data for demonstration...")
            self.create_sample_data()
            return True
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        
        # Sample household data
        n_households = 1000
        self.hh_data = pd.DataFrame({
            'household_id': range(1, n_households + 1),
            'district_code': np.random.choice(['D001', 'D002', 'D003', 'D004', 'D005'], n_households),
            'province_code': np.random.choice(['P01', 'P02', 'P03'], n_households),
            'household_size': np.random.randint(1, 10, n_households),
            'income_level': np.random.choice(['Low', 'Medium', 'High'], n_households, p=[0.6, 0.3, 0.1]),
            'dietary_diversity_score': np.random.randint(1, 10, n_households),
            'food_consumption_score': np.random.randint(10, 70, n_households),
            'water_source': np.random.choice(['Improved', 'Unimproved', 'Surface water'], n_households),
            'sanitation_facility': np.random.choice(['Improved', 'Unimproved', 'Open defecation'], n_households),
            'asset_index': np.random.normal(0, 1, n_households),
            'shock_exposure': np.random.randint(0, 5, n_households),
            'livelihood_type': np.random.choice(['Farming', 'Fishing', 'Pastoral', 'Urban', 'Other'], n_households),
            'food_expenditure': np.random.normal(100, 30, n_households),
            'market_access': np.random.choice(['Good', 'Fair', 'Poor'], n_households)
        })
        
        # Sample child data
        self.child_data = pd.DataFrame({
            'household_id': np.random.choice(range(1, n_households + 1), 500),
            'child_age_months': np.random.randint(6, 60, 500),
            'height_for_age': np.random.normal(-1, 2, 500),
            'weight_for_height': np.random.normal(-0.5, 1.5, 500),
            'weight_for_age': np.random.normal(-0.8, 1.8, 500),
            'birth_weight': np.random.choice(['Normal', 'Low', 'Very low'], 500, p=[0.7, 0.2, 0.1]),
            'breastfeeding_duration': np.random.randint(0, 36, 500),
            'vitamin_a_supplementation': np.random.choice([0, 1], 500, p=[0.3, 0.7])
        })
        
        # Sample women data
        self.women_data = pd.DataFrame({
            'household_id': np.random.choice(range(1, n_households + 1), 600),
            'mother_education': np.random.choice(['None', 'Primary', 'Secondary', 'Higher'], 600),
            'maternal_age': np.random.randint(15, 50, 600),
            'ifa_supplementation': np.random.choice([0, 1], 600, p=[0.4, 0.6]),
            'anc_visits': np.random.randint(0, 10, 600)
        })
    
    def preprocess_data(self):
        """Preprocess and merge datasets with memory optimization"""
        try:
            # Show available columns for debugging
            st.write("🔍 Available columns in datasets:")
            st.write(f"Household: {len(self.hh_data.columns)} columns")
            st.write(f"Child: {len(self.child_data.columns)} columns")
            st.write(f"Women: {len(self.women_data.columns)} columns")

            # Find common ID column
            common_id = '___index'  # Correct unique household identifier
            st.info(f"🔗 Merging datasets using column: '{common_id}'")

            # Select only essential columns to reduce memory usage
            # --- FIX: Rename columns before selecting them ---
            self.rename_columns()
            
            essential_cols_hh = self.select_essential_columns(self.hh_data, 'household')
            essential_cols_child = self.select_essential_columns(self.child_data, 'child')
            essential_cols_women = self.select_essential_columns(self.women_data, 'women')

            # Merge datasets with only essential columns
            merged_data = self.hh_data[essential_cols_hh + [common_id]].merge(
                self.child_data[essential_cols_child + [common_id]], 
                on=common_id,
                how='left',  # Use left join to keep all child records
                suffixes=('_hh', '_child')
            ).merge(
                self.women_data[essential_cols_women + [common_id]], 
                on=common_id,
                how='left',
                suffixes=('', '_women')
            )
            
            st.success(f"✅ Datasets merged successfully. Final shape: {merged_data.shape}")
            
            # Optimize data types to save memory
            merged_data = self.optimize_data_types(merged_data)
            
            # Handle missing values
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
            categorical_cols = merged_data.select_dtypes(include=['object']).columns
            
            if len(numeric_cols) > 0:
                merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data[numeric_cols].median())
            if len(categorical_cols) > 0:
                merged_data[categorical_cols] = merged_data[categorical_cols].fillna('Unknown')
            
            # Create target variables
            merged_data = self.create_target_variables(merged_data)
            
            return merged_data
            
        except Exception as e:
            st.error(f"❌ Error in preprocessing: {e}")
            return None

    def select_essential_columns(self, data, data_type):
        """Select only essential columns to reduce memory usage"""
        all_cols = data.columns.tolist()
        
        # Keep only columns that are likely to be useful for analysis
        if data_type == 'household':
            essential_patterns = ['FCS', 'HDDS', 'S2_01', 'S11_02', 'S11_03', 'income', 'asset', 'shock', 'livelihood', 
                                'S0_D_Dist', 'S0_C_Prov', 'S0_B_Ur', 'district', 'province', 'urban_rural']
        elif data_type == 'child':
            essential_patterns = ['HAZ', 'WHZ', 'WAZ', 'S13_01', 'age', 'height', 'weight', 'birth', 'breast']
        elif data_type == 'women':
            essential_patterns = ['S12_05', 'education', 'age', 'ifa', 'anc']
        else:
            essential_patterns = []
        
        essential_cols = []
        for col in all_cols:
            if any(pattern.lower() in col.lower() for pattern in essential_patterns):
                essential_cols.append(col)
        
        # If no essential columns found, take first 10 columns
        if not essential_cols and len(all_cols) > 10:
            essential_cols = all_cols[:10]
        elif not essential_cols:
            essential_cols = all_cols
            
        return essential_cols
    
    def rename_columns(self):
        """Rename key columns to be consistent"""
        # Define common renames for Rwanda CFSVA data
        rename_map = {
            'HDDS': 'dietary_diversity_score',
            'S0_D_Dist': 'district',
            'S0_C_Prov': 'province',
            'S0_B_Ur': 'urban_rural',
            'FCS': 'food_consumption_score',
            'HAZ': 'height_for_age_zscore',
            'WHZ': 'weight_for_height_zscore',
            'WAZ': 'weight_for_age_zscore'
        }
        
        # Apply to all dataframes
        for df in [self.hh_data, self.child_data, self.women_data]:
            if df is not None:
                df.rename(columns=rename_map, inplace=True, errors='ignore')

    def optimize_data_types(self, data):
        """Optimize data types to reduce memory usage"""
        # Convert object columns to category if they have few unique values
        for col in data.select_dtypes(include=['object']).columns:
            if data[col].nunique() / len(data) < 0.5:  # If less than 50% unique values
                data[col] = data[col].astype('category')
        
        # Downcast numeric columns
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = pd.to_numeric(data[col], downcast='integer')
            
        return data
    
    def create_target_variables(self, data):
        """Create target variables based on available columns"""
        st.header("🎯 Creating Target Variables")
        
        # STUNTING RISK
        st.subheader("Stunting Risk")
        stunting_cols = [col for col in data.columns if any(x in col.lower() for x in 
                        ['haz', 'height_for_age', 'stunting', 'height_for_age_zscore'])]
        
        if stunting_cols:
            stunting_col = stunting_cols[0]
            st.info(f"Using '{stunting_col}' for stunting risk")
            # Convert column to numeric, coercing errors to NaN
            data[stunting_col] = pd.to_numeric(data[stunting_col], errors='coerce')
            
            if data[stunting_col].dtype in [np.float64, np.int64]:
                data['stunting_risk'] = (data[stunting_col] < -2).astype(np.int8)
                st.success(f"✅ Stunting risk created from {stunting_col}")
                st.write(f"Distribution: {data['stunting_risk'].value_counts().to_dict()}")
            else:
                st.warning(f"Column '{stunting_col}' is not numeric. Creating synthetic stunting risk.")
                data['stunting_risk'] = np.random.choice([0, 1], len(data), p=[0.7, 0.3]).astype(np.int8)
        else:
            st.warning("⚠️ No stunting columns found. Creating synthetic stunting risk.")
            data['stunting_risk'] = np.random.choice([0, 1], len(data), p=[0.7, 0.3]).astype(np.int8)
        
        # DIETARY DIVERSITY RISK
        st.subheader("Dietary Diversity Risk")
        diet_cols = [col for col in data.columns if any(x in col.lower() for x in 
                    ['hdds', 'dietary_diversity', 'food_diversity', 'dds', 'dietary', 'dietary_diversity_score'])]
        
        if diet_cols:
            diet_col = diet_cols[0]
            st.info(f"Using '{diet_col}' for dietary diversity risk")
            if data[diet_col].dtype in [np.float64, np.int64]:
                data['low_dietary_diversity'] = (data[diet_col] < 4).astype(np.int8)
                st.success(f"✅ Dietary diversity risk created from {diet_col}")
                st.write(f"Distribution: {data['low_dietary_diversity'].value_counts().to_dict()}")
            else:
                st.warning(f"Column '{diet_col}' is not numeric. Creating synthetic dietary diversity risk.")
                data['low_dietary_diversity'] = np.random.choice([0, 1], len(data), p=[0.6, 0.4]).astype(np.int8)
        else:
            st.warning("⚠️ No dietary diversity columns found. Creating synthetic risk.")
            data['low_dietary_diversity'] = np.random.choice([0, 1], len(data), p=[0.6, 0.4]).astype(np.int8)
        
        # FOOD INSECURITY
        st.subheader("Food Insecurity")
        food_cols = [col for col in data.columns if any(x in col.lower() for x in 
                    ['fcs', 'food_consumption', 'food_insecurity', 'food_security', 'food_consumption_score'])]
        
        if food_cols:
            food_col = food_cols[0]
            st.info(f"Using '{food_col}' for food insecurity")
            if data[food_col].dtype in [np.float64, np.int64]:
                if 'fcs' in food_col.lower():
                    data['food_insecure'] = (data[food_col] < 35).astype(np.int8)
                else:
                    median_val = data[food_col].median()
                    data['food_insecure'] = (data[food_col] < median_val).astype(np.int8)
                st.success(f"✅ Food insecurity created from {food_col}")
                st.write(f"Distribution: {data['food_insecure'].value_counts().to_dict()}")
            else:
                st.warning(f"Column '{food_col}' is not numeric. Creating synthetic food insecurity.")
                data['food_insecure'] = np.random.choice([0, 1], len(data), p=[0.65, 0.35]).astype(np.int8)
        else:
            st.warning("⚠️ No food security columns found. Creating synthetic food insecurity.")
            data['food_insecure'] = np.random.choice([0, 1], len(data), p=[0.65, 0.35]).astype(np.int8)
        
        return data
    
    def train_models(self, data):
        """Train predictive models with memory optimization"""
        if data is None:
            st.error("❌ No data available for training")
            return
        
        st.header("🤖 Training Predictive Models")
        
        # Select only numeric features to avoid one-hot encoding issues
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target variables from features
        target_cols = ['stunting_risk', 'low_dietary_diversity', 'food_insecure']
        numeric_features = [col for col in numeric_features if col not in target_cols]
        
        # Select top 20 most important numeric features to avoid memory issues
        if len(numeric_features) > 20:
            # Use correlation with targets to select most important features
            selected_features = []
            for target in target_cols:
                if target in data.columns:
                    correlations = data[numeric_features].corrwith(data[target]).abs().sort_values(ascending=False)
                    top_features = correlations.head(10).index.tolist()
                    selected_features.extend(top_features)
            
            # Remove duplicates and keep top 20
            selected_features = list(set(selected_features))[:20]
            numeric_features = selected_features
        
        st.info(f"📊 Using {len(numeric_features)} numeric features for modeling")
        
        # Check if target variables exist
        missing_targets = [target for target in target_cols if target not in data.columns]
        if missing_targets:
            st.error(f"❌ Missing target variables: {missing_targets}")
            return
        
        models_config = {
            'stunting': {
                'features': numeric_features,
                'target': 'stunting_risk'
            },
            'dietary_diversity': {
                'features': numeric_features,
                'target': 'low_dietary_diversity'
            },
            'food_insecurity': {
                'features': numeric_features,
                'target': 'food_insecure'
            }
        }
        
        trained_models = 0
        
        for model_name, config in models_config.items():
            try:
                st.write(f"---")
                st.write(f"**Training {model_name.replace('_', ' ').title()} Model**")
                
                # Select available features
                available_features = [f for f in config['features'] if f in data.columns]
                
                if len(available_features) < 2:
                    st.warning(f"⚠️ Not enough features for {model_name} model. Skipping.")
                    continue
                
                X = data[available_features]
                y = data[config['target']]
                
                # Remove any remaining non-numeric columns
                X = X.select_dtypes(include=[np.number])
                
                # Check class balance
                class_balance = y.value_counts()
                st.write(f"Class distribution: {class_balance.to_dict()}")
                
                if len(class_balance) < 2:
                    st.warning(f"⚠️ Only one class present for {model_name}. Skipping.")
                    continue
                
                # Check if we have enough data
                if len(X) < 50:
                    st.warning(f"⚠️ Not enough data for {model_name} model. Skipping.")
                    continue
                
                # Use a subset of data if dataset is too large
                max_samples = 10000  # Limit to 10,000 samples for memory
                if len(X) > max_samples:
                    st.info(f"Using {max_samples} random samples for training (dataset too large)")
                    indices = np.random.choice(len(X), max_samples, replace=False)
                    X = X.iloc[indices]
                    y = y.iloc[indices]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model with smaller parameters for memory efficiency
                model = RandomForestClassifier(
                    n_estimators=50,  # Reduced from 100
                    max_depth=10,     # Limit depth
                    random_state=42, 
                    class_weight='balanced'
                )
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models[model_name] = model
                self.scalers[model_name] = scaler
                self.feature_names[model_name] = X.columns.tolist()
                
                st.success(f"✅ {model_name.replace('_', ' ').title()} Model trained - Accuracy: {accuracy:.3f}")
                trained_models += 1
                
                # Show feature importance for top features
                if hasattr(model, 'feature_importances_'):
                    feature_imp = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(5)
                    
                    st.write("Top 5 important features:")
                    for _, row in feature_imp.iterrows():
                        st.write(f"  - {row['feature']}: {row['importance']:.3f}")
                
            except Exception as e:
                st.error(f"❌ Error training {model_name} model: {str(e)}")
        
        if trained_models == 0:
            st.error("❌ No models were successfully trained.")
        else:
            st.success(f"🎉 Successfully trained {trained_models} out of {len(models_config)} models!")
    
    def predict_risk(self, input_data, model_type):
        """Make risk predictions"""
        if model_type in self.models:
            model = self.models[model_type]
            scaler = self.scalers[model_type]
            
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Align columns with training data
            training_features = self.feature_names[model_type]
            input_aligned = input_df[training_features]
            
            # Scale and predict
            input_scaled = scaler.transform(input_aligned)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            return prediction, probability
        return None, None

    def create_risk_map_plotly(self, data, risk_type, geographic_level="District"):
        """Create hotspot maps for Rwanda using Plotly"""
        if data is None:
            return None

        # Rwanda district coordinates with provinces
        district_coordinates = {
            'Nyarugenge': {'lat': -1.9441, 'lon': 30.0619, 'province': 'City of Kigali'},
            'Kicukiro': {'lat': -1.9541, 'lon': 30.0719, 'province': 'City of Kigali'},
            'Gasabo': {'lat': -1.9341, 'lon': 30.0519, 'province': 'City of Kigali'},
            'Rulindo': {'lat': -1.75, 'lon': 30.0, 'province': 'Northern'},
            'Gicumbi': {'lat': -1.6, 'lon': 30.1, 'province': 'Northern'},
            'Musanze': {'lat': -1.45, 'lon': 29.9, 'province': 'Northern'},
            'Burera': {'lat': -1.4, 'lon': 29.8, 'province': 'Northern'},
            'Gakenke': {'lat': -1.8, 'lon': 29.95, 'province': 'Northern'},
            'Nyanza': {'lat': -2.35, 'lon': 29.55, 'province': 'Southern'},
            'Huye': {'lat': -2.6, 'lon': 29.75, 'province': 'Southern'},
            'Gisagara': {'lat': -2.55, 'lon': 30.05, 'province': 'Southern'},
            'Muhanga': {'lat': -2.2, 'lon': 29.75, 'province': 'Southern'},
            'Kamonyi': {'lat': -2.15, 'lon': 29.85, 'province': 'Southern'},
            'Ruhango': {'lat': -2.25, 'lon': 29.65, 'province': 'Southern'},
            'Nyaruguru': {'lat': -2.45, 'lon': 29.45, 'province': 'Southern'},
            'Nyamagabe': {'lat': -2.65, 'lon': 29.35, 'province': 'Southern'},
            'Rwamagana': {'lat': -1.95, 'lon': 30.45, 'province': 'Eastern'},
            'Kayonza': {'lat': -1.85, 'lon': 30.55, 'province': 'Eastern'},
            'Kirehe': {'lat': -1.75, 'lon': 30.65, 'province': 'Eastern'},
            'Ngoma': {'lat': -1.65, 'lon': 30.35, 'province': 'Eastern'},
            'Gatsibo': {'lat': -1.55, 'lon': 30.25, 'province': 'Eastern'},
            'Nyagatare': {'lat': -1.35, 'lon': 30.15, 'province': 'Eastern'},
            'Bugesera': {'lat': -2.25, 'lon': 30.25, 'province': 'Eastern'},
            'Karongi': {'lat': -2.05, 'lon': 29.25, 'province': 'Western'},
            'Rutsiro': {'lat': -1.95, 'lon': 29.35, 'province': 'Western'},
            'Rubavu': {'lat': -1.85, 'lon': 29.45, 'province': 'Western'},
            'Nyabihu': {'lat': -1.75, 'lon': 29.55, 'province': 'Western'},
            'Ngororero': {'lat': -1.65, 'lon': 29.65, 'province': 'Western'},
            'Rusizi': {'lat': -2.55, 'lon': 28.95, 'province': 'Western'},
            'Nyamasheke': {'lat': -2.35, 'lon': 29.15, 'province': 'Western'}
        }

        # Create province coordinates from districts
        province_coordinates = {
            'City of Kigali': {'lat': -1.9441, 'lon': 30.0619},
            'Northern': {'lat': -1.6, 'lon': 29.9},
            'Southern': {'lat': -2.45, 'lon': 29.65},
            'Eastern': {'lat': -1.85, 'lon': 30.45},
            'Western': {'lat': -2.15, 'lon': 29.35}
        }

        # Detect geographic columns in the data
        district_cols = [col for col in data.columns if any(x in col.lower() for x in ['district', 'dist', 's0_d_dist'])]
        province_cols = [col for col in data.columns if any(x in col.lower() for x in ['province', 'prov', 's0_c_prov'])]
        
        # Determine the column to group by based on the selected geographic level
        if geographic_level == "District" and district_cols:
            geo_col_name = district_cols[0]
            st.info(f"📍 Using district column: '{geo_col_name}'")
            coordinate_map = district_coordinates
        elif geographic_level == "Province" and province_cols:
            geo_col_name = province_cols[0]
            st.info(f"📍 Using province column: '{geo_col_name}'")
            coordinate_map = province_coordinates
        else:
            # If no geographic columns found, show available columns and create synthetic data
            st.warning(f"⚠️ No {geographic_level.lower()} column found. Available geographic columns:")
            st.write(f"District-like: {district_cols}")
            st.write(f"Province-like: {province_cols}")
            
            # Create synthetic Rwanda geographic data
            if geographic_level == "District":
                rwanda_districts = list(district_coordinates.keys())
                data['synthetic_district'] = np.random.choice(rwanda_districts, len(data))
                geo_col_name = 'synthetic_district'
                coordinate_map = district_coordinates
            else:
                rwanda_provinces = list(province_coordinates.keys())
                data['synthetic_province'] = np.random.choice(rwanda_provinces, len(data))
                geo_col_name = 'synthetic_province'
                coordinate_map = province_coordinates

        # Use a sample if data is too large
        if len(data) > 1000:
            data_sample = data.sample(1000, random_state=42)
            st.info("Using 1,000 sample records for mapping (dataset too large)")
        else:
            data_sample = data
            
        # Aggregate data by geographic area
        try:
            geo_risk = data_sample.groupby(geo_col_name).agg({
                'stunting_risk': 'mean',
                'low_dietary_diversity': 'mean',
                'food_insecure': 'mean'
            }).reset_index()
            
            # Show available locations for debugging
            st.info(f"📍 Locations found in data: {list(geo_risk[geo_col_name].unique())[:10]}{'...' if len(geo_risk[geo_col_name].unique()) > 10 else ''}")
            
        except Exception as e:
            st.error(f"❌ Error aggregating data: {e}")
            return None
        
        # Get Rwanda coordinates for each location
        geo_risk['lat'] = geo_risk[geo_col_name].apply(
            lambda x: coordinate_map.get(str(x).strip(), {}).get('lat', -1.9403)
        )
        geo_risk['lon'] = geo_risk[geo_col_name].apply(
            lambda x: coordinate_map.get(str(x).strip(), {}).get('lon', 29.8739)
        )
        
        # Add province information for tooltips
        if geographic_level == "District":
            geo_risk['province'] = geo_risk[geo_col_name].apply(
                lambda x: coordinate_map.get(str(x).strip(), {}).get('province', 'Unknown')
            )
        
        # FIX: Correct risk column mapping
        risk_column_mapping = {
            'stunting': 'stunting_risk',
            'dietary_diversity': 'low_dietary_diversity', 
            'food_insecurity': 'food_insecure'  # Changed from 'food_insecurity' to 'food_insecure'
        }
        
        # Get the correct risk column name
        risk_column = risk_column_mapping.get(risk_type)
        
        if risk_column not in geo_risk.columns:
            st.error(f"❌ Risk column '{risk_column}' not found in data. Available columns: {list(geo_risk.columns)}")
            return None
        
        # Create hover data based on geographic level
        if geographic_level == "District":
            hover_data = {
                risk_column: ':.3f', 
                'province': True,
                'lat': False, 
                'lon': False
            }
        else:
            hover_data = {
                risk_column: ':.3f', 
                'lat': False, 
                'lon': False
            }
        
        fig = px.scatter_mapbox(
            geo_risk,
            lat="lat",
            lon="lon",
            size=risk_column,
            color=risk_column,
            hover_name=geo_col_name,
            hover_data=hover_data,
            color_continuous_scale="RdYlBu_r",
            size_max=30,
            zoom=7,
            title=f"Rwanda {risk_type.replace('_', ' ').title()} Hotspots by {geographic_level}",
            height=600
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": -1.9403, "lon": 29.8739},  # Center on Rwanda
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
        
        # Show data table
        with st.expander("📊 View Geographic Risk Data"):
            display_data = geo_risk[[geo_col_name, risk_column]].round(3)
            if geographic_level == "District" and 'province' in geo_risk.columns:
                display_data['province'] = geo_risk['province']
            st.dataframe(display_data.sort_values(risk_column, ascending=False), use_container_width=True)
        
        return fig

def main():
    st.markdown('<h1 class="main-header">🌍 Rwanda CFSVA Risk Assessment Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize analyzer
    analyzer = CFSVAAnalysis()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["📊 Data Overview", "⚡ Quick Risk Assessment", "🤖 Predictive Modeling", "🗺️ Hotspot Maps", "📈 Analytics", "ℹ️ About"]
    )
    
    # Load data
    if analyzer.load_data():
        
        if app_mode == "📊 Data Overview":
            show_data_overview(analyzer)
            
        elif app_mode == "⚡ Quick Risk Assessment":
            show_risk_assessment(analyzer)
            
        elif app_mode == "🤖 Predictive Modeling":
            show_model_training(analyzer)
            
        elif app_mode == "🗺️ Hotspot Maps":
            show_hotspot_maps(analyzer)
            
        elif app_mode == "📈 Analytics":
            show_analytics_dashboard(analyzer)
            
        elif app_mode == "ℹ️ About":
            show_about()

def show_data_overview(analyzer):
    st.header("📊 Data Overview")
    
    # Show basic info
    st.write(f"Household data shape: {analyzer.hh_data.shape}")
    st.write(f"Child data shape: {analyzer.child_data.shape}")
    st.write(f"Women data shape: {analyzer.women_data.shape}")
    
    # Show geographic columns specifically
    with st.expander("🔍 Geographic Columns Detection"):
        st.subheader("Geographic Columns Found:")
        for df_name, df in [("Household", analyzer.hh_data), ("Child", analyzer.child_data), ("Women", analyzer.women_data)]:
            district_cols = [col for col in df.columns if any(x in col.lower() for x in ['district', 'dist'])]
            province_cols = [col for col in df.columns if any(x in col.lower() for x in ['province', 'prov'])]
            st.write(f"**{df_name}**:")
            st.write(f"  - District columns: {district_cols}")
            st.write(f"  - Province columns: {province_cols}")
    
    # Show column names for debugging
    with st.expander("🔍 First 20 Columns from Each Dataset"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Household Data Columns:**")
            st.write(list(analyzer.hh_data.columns)[:20])
        with col2:
            st.write("**Child Data Columns:**")
            st.write(list(analyzer.child_data.columns)[:20])
        with col3:
            st.write("**Women Data Columns:**")
            st.write(list(analyzer.women_data.columns)[:20])
    
    tab1, tab2, tab3 = st.tabs(["🏠 Household Data", "👶 Child Data", "👩 Women Data"])
    
    with tab1:
        st.subheader("Household Dataset")
        st.dataframe(analyzer.hh_data.head(100), use_container_width=True)
        
        # Basic statistics
        st.subheader("Basic Statistics")
        st.dataframe(analyzer.hh_data.describe(), use_container_width=True)
    
    with tab2:
        st.subheader("Child Nutrition Data")
        st.dataframe(analyzer.child_data.head(100), use_container_width=True)
    
    with tab3:
        st.subheader("Women's Data")
        st.dataframe(analyzer.women_data.head(100), use_container_width=True)

def show_risk_assessment(analyzer):
    st.header("⚡ Quick Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Household Information")
        household_size = st.slider("Household Size", min_value=1, max_value=20, value=5)
        income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
        livelihood_type = st.selectbox("Livelihood Type", ["Farming", "Fishing", "Pastoral", "Urban", "Other"])
        
        st.subheader("💧 WASH Indicators")
        water_source = st.selectbox("Water Source", ["Improved", "Unimproved", "Surface water"])
        sanitation = st.selectbox("Sanitation Facility", ["Improved", "Unimproved", "Open defecation"])
    
    with col2:
        st.subheader("👶 Child Information")
        child_age = st.slider("Child Age (months)", min_value=6, max_value=59, value=24)
        birth_weight = st.selectbox("Birth Weight", ["Normal", "Low", "Very low"])
        breastfeeding = st.slider("Breastfeeding Duration (months)", min_value=0, max_value=36, value=12)
        
        st.subheader("⚠️ Shock Exposure")
        shock_exposure = st.multiselect("Recent Shocks", 
                                      ["Drought", "Flood", "Conflict", "Price shock", "Crop failure", "None"])
        
        st.subheader("👩 Maternal Factors")
        mother_education = st.selectbox("Mother's Education", ["None", "Primary", "Secondary", "Higher"])
    
    # Assessment button
    if st.button("🔍 Assess Risks", type="primary", use_container_width=True):
        display_risk_results({
            'household_size': household_size,
            'income_level': income_level,
            'livelihood_type': livelihood_type,
            'water_source': water_source,
            'sanitation_facility': sanitation,
            'child_age_months': child_age,
            'birth_weight': birth_weight,
            'breastfeeding_duration': breastfeeding,
            'mother_education': mother_education,
            'shock_exposure': len(shock_exposure)
        })

def display_risk_results(input_data):
    st.header("📋 Risk Assessment Results")
    
    # Calculate risk scores based on input
    stunting_risk = min(0.95, 0.3 + (input_data['income_level'] == 'Low') * 0.2 + 
                       (input_data['water_source'] != 'Improved') * 0.15 +
                       (input_data['mother_education'] == 'None') * 0.2)
    
    dietary_risk = min(0.95, 0.25 + (input_data['income_level'] == 'Low') * 0.3 + 
                      (input_data['shock_exposure'] > 0) * 0.2)
    
    food_insecurity_risk = min(0.95, 0.2 + (input_data['income_level'] == 'Low') * 0.3 + 
                              (input_data['shock_exposure'] > 0) * 0.25 +
                              (input_data['livelihood_type'] == 'Farming') * 0.1)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_display = f"""
        <div class="{'risk-high' if stunting_risk > 0.6 else 'risk-medium' if stunting_risk > 0.3 else 'risk-low'}">
            <h3>Stunting Risk</h3>
            <h2>{stunting_risk:.0%}</h2>
        </div>
        """
        st.markdown(risk_display, unsafe_allow_html=True)
        
        with st.expander("Key Factors"):
            factors = []
            if input_data['income_level'] == 'Low':
                factors.append("• Low income level")
            if input_data['water_source'] != 'Improved':
                factors.append("• Poor water source")
            if input_data['mother_education'] == 'None':
                factors.append("• Low maternal education")
            for factor in factors:
                st.write(factor)
    
    with col2:
        risk_display = f"""
        <div class="{'risk-high' if dietary_risk > 0.6 else 'risk-medium' if dietary_risk > 0.3 else 'risk-low'}">
            <h3>Dietary Diversity Risk</h3>
            <h2>{dietary_risk:.0%}</h2>
        </div>
        """
        st.markdown(risk_display, unsafe_allow_html=True)
        
        with st.expander("Key Factors"):
            factors = []
            if input_data['income_level'] == 'Low':
                factors.append("• Limited food budget")
            if input_data['shock_exposure'] > 0:
                factors.append("• Recent shock exposure")
            for factor in factors:
                st.write(factor)
    
    with col3:
        risk_display = f"""
        <div class="{'risk-high' if food_insecurity_risk > 0.6 else 'risk-medium' if food_insecurity_risk > 0.3 else 'risk-low'}">
            <h3>Food Insecurity Risk</h3>
            <h2>{food_insecurity_risk:.0%}</h2>
        </div>
        """
        st.markdown(risk_display, unsafe_allow_html=True)
        
        with st.expander("Key Factors"):
            factors = []
            if input_data['income_level'] == 'Low':
                factors.append("• Economic vulnerability")
            if input_data['shock_exposure'] > 0:
                factors.append("• Shock impacts")
            if input_data['livelihood_type'] == 'Farming':
                factors.append("• Climate-dependent livelihood")
            for factor in factors:
                st.write(factor)
    
    # Risk visualization
    st.subheader("📊 Risk Comparison")
    risks = ['Stunting', 'Dietary Diversity', 'Food Insecurity']
    values = [stunting_risk, dietary_risk, food_insecurity_risk]
    
    fig = go.Figure(data=[
        go.Bar(x=risks, y=values, marker_color=['#ff6b6b', '#ffd93d', '#6bcf7f'])
    ])
    fig.update_layout(
        title="Risk Level Comparison",
        yaxis_title="Risk Probability",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.subheader("🎯 Recommended Interventions")
    recommendations = []
    
    if stunting_risk > 0.5:
        recommendations.extend([
            "• Implement nutrition supplementation programs for children",
            "• Improve WASH facilities and promote hygiene practices",
            "• Provide maternal education on child feeding practices",
            "• Establish growth monitoring programs"
        ])
    
    if dietary_risk > 0.4:
        recommendations.extend([
            "• Promote kitchen gardens for diverse crop cultivation",
            "• Provide nutrition education and cooking demonstrations",
            "• Support income-generating activities for women",
            "• Implement school feeding programs"
        ])
    
    if food_insecurity_risk > 0.4:
        recommendations.extend([
            "• Distribute emergency food assistance if shocks occurred",
            "• Provide cash transfer programs for vulnerable households",
            "• Support livelihood diversification initiatives",
            "• Establish early warning systems for food crises"
        ])
    
    for i, rec in enumerate(recommendations):
        st.markdown(f"<div class='metric-card'>{rec}</div>", unsafe_allow_html=True)

def show_model_training(analyzer):
    st.header("🤖 Predictive Modeling")
    
    st.info("""
    This section trains machine learning models to predict various risks based on your data.
    The models will learn patterns from historical data to predict future risks.
    """)
    
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Preprocessing data and training models... This may take a few minutes."):
            merged_data = analyzer.preprocess_data()
            
            if merged_data is not None:
                analyzer.train_models(merged_data)
                
                if analyzer.models:
                    st.success(f"✅ {len(analyzer.models)} models trained successfully!")
                    
                    # Display model performance
                    st.subheader("📈 Model Performance")
                    
                    # Feature importance plots
                    for model_name, model in analyzer.models.items():
                        st.write(f"**{model_name.replace('_', ' ').title()} Model**")
                        
                        # Feature importance
                        if hasattr(model, 'feature_importances_'):
                            features = analyzer.feature_names[model_name]
                            importance_df = pd.DataFrame({
                                'feature': features,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=True).tail(10)
                            
                            fig = px.bar(importance_df, x='importance', y='feature', 
                                        title=f'Top 10 Feature Importance - {model_name.title()}',
                                        orientation='h')
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No models were trained. Check the data and error messages above.")
            else:
                st.error("❌ Failed to preprocess data. Cannot train models.")

def show_hotspot_maps(analyzer):
    st.header("🗺️ Rwanda Hotspot Maps")
    
    st.info("Visualize geographic distribution of risks and vulnerabilities across Rwanda.")
    
    # Preprocess data for mapping
    with st.spinner("Preparing data for mapping..."):
        merged_data = analyzer.preprocess_data()
    
    if merged_data is None:
        st.error("❌ No data available for mapping")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        # Risk type selection
        risk_type = st.selectbox(
            "Select Risk Type to Map",
            ["stunting", "dietary_diversity", "food_insecurity"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    with col2:
        # Geographic level selection
        geographic_level = st.selectbox(
            "Select Geographic Level",
            ["District", "Province"]
        )
    
    # Generate map
    fig = analyzer.create_risk_map_plotly(merged_data, risk_type, geographic_level)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not generate map. Check if data is available.")

def show_analytics_dashboard(analyzer):
    st.header("📈 Comprehensive Analytics Dashboard")
    
    with st.spinner("Loading analytics..."):
        merged_data = analyzer.preprocess_data()
    
    if merged_data is None:
        st.error("❌ No data available for analytics")
        return
    
    # Use a sample for analytics to avoid memory issues
    if len(merged_data) > 5000:
        analytics_data = merged_data.sample(5000, random_state=42)
        st.info("Using 5,000 sample records for analytics (dataset too large)")
    else:
        analytics_data = merged_data
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'stunting_risk' in analytics_data.columns:
            stunting_rate = analytics_data['stunting_risk'].mean() * 100
            st.metric("Stunting Prevalence", f"{stunting_rate:.1f}%")
    
    with col2:
        if 'low_dietary_diversity' in analytics_data.columns:
            dietary_risk = analytics_data['low_dietary_diversity'].mean() * 100
            st.metric("Low Dietary Diversity", f"{dietary_risk:.1f}%")
    
    with col3:
        if 'food_insecure' in analytics_data.columns:
            food_insecure = analytics_data['food_insecure'].mean() * 100
            st.metric("Food Insecurity", f"{food_insecure:.1f}%")
    
    with col4:
        st.metric("Total Records", f"{len(analytics_data):,}")
    
    # Simple charts that don't require much memory
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution
        if 'stunting_risk' in analytics_data.columns:
            risk_counts = analytics_data['stunting_risk'].value_counts()
            fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                        title="Stunting Risk Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Food insecurity distribution
        if 'food_insecure' in analytics_data.columns:
            food_counts = analytics_data['food_insecure'].value_counts()
            fig = px.pie(values=food_counts.values, names=food_counts.index,
                        title="Food Insecurity Distribution")
            st.plotly_chart(fig, use_container_width=True)

def show_about():
    st.header("ℹ️ About This Dashboard")
    
    st.markdown("""
    ## 🌍 Rwanda CFSVA Risk Assessment Dashboard
    
    This dashboard provides comprehensive analysis and risk assessment for the Comprehensive Food Security and Vulnerability Analysis (CFSVA) 2024 data for Rwanda.
    
    ### 🎯 Features:
    
    **📊 Data Overview**
    - Explore household, child, and women datasets
    - View basic statistics and distributions
    - Understand data structure and variables
    
    **⚡ Quick Risk Assessment**
    - Rapid risk evaluation for individual cases
    - Stunting, dietary diversity, and food insecurity risk scoring
    - Evidence-based intervention recommendations
    
    **🤖 Predictive Modeling**
    - Machine learning models for risk prediction
    - Feature importance analysis
    - Model performance metrics
    
    **🗺️ Rwanda Hotspot Maps**
    - Geographic visualization of risks across all 30 Rwanda districts
    - District and province level risk comparisons
    - Priority area identification
    
    **📈 Analytics Dashboard**
    - Comprehensive risk statistics
    - Correlation analysis
    - Trend visualization
    
    ### 🔧 Technical Features:
    - Memory-optimized data processing
    - Automatic geographic column detection
    - Real Rwanda district and province coordinates
    - Interactive maps with detailed tooltips
    """)

if __name__ == "__main__":
    main()