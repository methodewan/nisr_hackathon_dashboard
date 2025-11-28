import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a1a1a;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    .hotspot-card {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ef4444;
        margin-bottom: 0.5rem;
    }
    .medium-risk-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin-bottom: 0.5rem;
    }
    .low-risk-card {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin-bottom: 0.5rem;
    }
    .map-container {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    .control-panel {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    .indicator-info {
        background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4f46e5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_enhanced_geographic_data():
    """Load enhanced geographic data with error handling"""
    try:
        # Try to load enhanced datasets
        household_geo = pd.read_csv("enhanced_geographic_household.csv")
        child_geo = pd.read_csv("enhanced_geographic_child.csv")
        child_hotspots = pd.read_csv("hotspot_analysis_child.csv")
        district_coords = pd.read_csv("district_coordinates_master.csv")
        
        return household_geo, child_geo, child_hotspots, district_coords, True
        
    except FileNotFoundError:
        # Create sample data if enhanced data not available
        return create_sample_geographic_data()

def create_sample_geographic_data():
    """Create sample geographic data for demonstration with enhanced indicators"""
    
    # District coordinates
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
    
    # Create sample data
    np.random.seed(42)
    districts = list(district_coordinates.keys())
    
    # Enhanced sample child nutrition data with more indicators
    child_data = []
    for district in districts:
        n_records = np.random.randint(100, 200)
        coords = district_coordinates[district]
        
        for _ in range(n_records):
            # Create correlated indicators
            base_risk = np.random.uniform(0.2, 0.8)
            
            child_data.append({
                'district': district,
                'province': coords['province'],
                'lat': coords['lat'] + np.random.normal(0, 0.05),
                'lon': coords['lon'] + np.random.normal(0, 0.05),
                'stunting_rate': np.clip(base_risk + np.random.normal(0, 0.1), 0.1, 0.9),
                'wasting_rate': np.clip(base_risk * 0.7 + np.random.normal(0, 0.08), 0.05, 0.5),
                'underweight_rate': np.clip(base_risk * 0.8 + np.random.normal(0, 0.09), 0.1, 0.7),
                'vitamin_a_received': np.clip(1 - base_risk * 0.5 + np.random.normal(0, 0.1), 0.5, 1.0),
                'dietary_diversity': np.clip(1 - base_risk * 0.6 + np.random.normal(0, 0.15), 0.2, 1.0),
                'food_security': np.clip(1 - base_risk * 0.7 + np.random.normal(0, 0.12), 0.3, 1.0),
                'wealth_index': np.clip(1 - base_risk * 0.4 + np.random.normal(0, 0.2), 0.1, 1.0),
                'overall_nutrition': np.clip(1 - base_risk + np.random.normal(0, 0.1), 0.2, 0.9),
                'urban_rural': np.random.choice(['Urban', 'Rural'], p=[0.3, 0.7])
            })
    
    child_geo = pd.DataFrame(child_data)
    
    # Create hotspot analysis with all indicators
    hotspot_data = []
    for district in districts:
        district_data = child_geo[child_geo['district'] == district]
        coords = district_coordinates[district]
        
        # Calculate all indicator means
        stunting_score = district_data['stunting_rate'].mean()
        wasting_score = district_data['wasting_rate'].mean()
        underweight_score = district_data['underweight_rate'].mean()
        vitamin_a_score = 1 - district_data['vitamin_a_received'].mean()  # Invert for risk
        dietary_diversity_score = 1 - district_data['dietary_diversity'].mean()  # Invert for risk
        food_security_score = 1 - district_data['food_security'].mean()  # Invert for risk
        wealth_score = 1 - district_data['wealth_index'].mean()  # Invert for risk
        overall_nutrition_score = 1 - district_data['overall_nutrition'].mean()  # Invert for risk
        
        # Composite risk score (average of all indicators)
        composite_risk = np.mean([
            stunting_score, wasting_score, underweight_score, 
            vitamin_a_score, dietary_diversity_score, food_security_score,
            wealth_score, overall_nutrition_score
        ])
        
        # Classify severity
        if composite_risk > 0.6:
            severity = 'High'
        elif composite_risk > 0.4:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        hotspot_data.append({
            'district': district,
            'composite_risk_score': composite_risk,
            'hotspot_severity': severity,
            'stunting_rate_mean': stunting_score,
            'wasting_rate_mean': wasting_score,
            'underweight_rate_mean': underweight_score,
            'vitamin_a_received_mean': district_data['vitamin_a_received'].mean(),
            'dietary_diversity_mean': district_data['dietary_diversity'].mean(),
            'food_security_mean': district_data['food_security'].mean(),
            'wealth_index_mean': district_data['wealth_index'].mean(),
            'overall_nutrition_mean': district_data['overall_nutrition'].mean(),
            'lat_first': coords['lat'],
            'lon_first': coords['lon'],
            'province_first': coords['province'],
            'priority_rank': 0  # Will be calculated later
        })
    
    child_hotspots = pd.DataFrame(hotspot_data)
    child_hotspots['priority_rank'] = child_hotspots['composite_risk_score'].rank(ascending=False)
    
    # Create district coordinates dataframe
    district_coords = pd.DataFrame([
        {'district': k, 'lat': v['lat'], 'lon': v['lon'], 'province': v['province']}
        for k, v in district_coordinates.items()
    ])
    
    return child_geo, child_geo, child_hotspots, district_coords, False

def get_indicator_info(indicator):
    """Get information about each indicator"""
    indicator_info = {
        'stunting_rate': {
            'name': 'Stunting Rate',
            'description': 'Percentage of children with height-for-age below -2 standard deviations',
            'interpretation': 'Higher values indicate worse nutritional status',
            'color_scale': 'Reds',
            'unit': 'rate'
        },
        'wasting_rate': {
            'name': 'Wasting Rate',
            'description': 'Percentage of children with weight-for-height below -2 standard deviations',
            'interpretation': 'Higher values indicate acute malnutrition',
            'color_scale': 'Oranges',
            'unit': 'rate'
        },
        'underweight_rate': {
            'name': 'Underweight Rate',
            'description': 'Percentage of children with weight-for-age below -2 standard deviations',
            'interpretation': 'Higher values indicate overall undernutrition',
            'color_scale': 'Purples',
            'unit': 'rate'
        },
        'vitamin_a_received': {
            'name': 'Vitamin A Coverage',
            'description': 'Percentage of children who received vitamin A supplementation',
            'interpretation': 'Higher values indicate better micronutrient intervention coverage',
            'color_scale': 'Greens',
            'unit': 'coverage'
        },
        'dietary_diversity': {
            'name': 'Dietary Diversity Score',
            'description': 'Measure of food variety consumed by households',
            'interpretation': 'Higher values indicate more diverse and nutritious diets',
            'color_scale': 'Blues',
            'unit': 'score'
        },
        'food_security': {
            'name': 'Food Security Index',
            'description': 'Composite measure of household food access and availability',
            'interpretation': 'Higher values indicate better food security',
            'color_scale': 'Viridis',
            'unit': 'index'
        },
        'wealth_index': {
            'name': 'Wealth Index',
            'description': 'Relative measure of household economic status',
            'interpretation': 'Higher values indicate better economic conditions',
            'color_scale': 'Plasma',
            'unit': 'index'
        },
        'composite_risk_score': {
            'name': 'Composite Risk Score',
            'description': 'Overall malnutrition risk combining multiple indicators',
            'interpretation': 'Higher values indicate greater nutritional risk',
            'color_scale': 'RdYlBu_r',
            'unit': 'score'
        },
        'overall_nutrition': {
            'name': 'Overall Nutrition Status',
            'description': 'Composite measure of general nutritional wellbeing',
            'interpretation': 'Higher values indicate better nutritional status',
            'color_scale': 'Rainbow',
            'unit': 'status'
        }
    }
    return indicator_info.get(indicator, {
        'name': indicator.replace('_', ' ').title(),
        'description': 'Nutrition indicator',
        'interpretation': 'Higher values may indicate better or worse status',
        'color_scale': 'Reds',
        'unit': 'value'
    })

def get_available_columns(data, required_columns):
    """Get available columns from dataset"""
    available_cols = []
    for col in required_columns:
        if col in data.columns:
            available_cols.append(col)
    return available_cols

def create_enhanced_bubble_map(data, indicator_col, geographic_level):
    """Create enhanced bubble map that properly uses the selected indicator"""
    
    if geographic_level == "District":
        group_col = 'district'
        title_suffix = 'District'
        zoom = 8
        size_multiplier = 20
    else:
        group_col = 'province'
        title_suffix = 'Province'
        zoom = 7
        size_multiplier = 30
    
    # Get indicator information
    indicator_info = get_indicator_info(indicator_col)
    
    # Check if indicator exists in data
    if indicator_col not in data.columns:
        st.error(f"Indicator '{indicator_col}' not found in dataset. Available columns: {list(data.columns)}")
        return None
    
    # Check if geographic grouping column exists
    if group_col not in data.columns:
        st.error(f"Geographic column '{group_col}' not found in dataset")
        return None
    
    # Aggregate data based on geographic level
    agg_data = data.groupby(group_col).agg({
        indicator_col: 'mean'
    }).reset_index()
    
    # Add coordinates - handle different coordinate column names
    coord_cols = {}
    for coord in ['lat', 'lon', 'latitude', 'longitude', 'lat_first', 'lon_first']:
        if coord in data.columns:
            coord_data = data.groupby(group_col)[coord].mean().reset_index()
            agg_data = agg_data.merge(coord_data, on=group_col, how='left')
            if 'lat' in coord.lower():
                coord_cols['lat'] = coord
            else:
                coord_cols['lon'] = coord
    
    if not coord_cols:
        st.error("No coordinate columns found in dataset")
        return None
    
    # Remove invalid coordinates
    lat_col = coord_cols.get('lat')
    lon_col = coord_cols.get('lon')
    agg_data = agg_data[(agg_data[lat_col] != 0) & (agg_data[lon_col] != 0) & 
                       (~agg_data[lat_col].isna()) & (~agg_data[lon_col].isna())]
    
    if len(agg_data) == 0:
        st.error("No valid geographic data available for mapping")
        return None
    
    # Create bubble map with indicator-specific styling
    fig = px.scatter_mapbox(
        agg_data,
        lat=lat_col,
        lon=lon_col,
        size=indicator_col,
        color=indicator_col,
        hover_name=group_col,
        hover_data={indicator_col: ':.3f'},
        title=f'{indicator_info["name"]} by {title_suffix}',
        color_continuous_scale=indicator_info['color_scale'],
        size_max=size_multiplier,
        mapbox_style="open-street-map"
    )
    
    # Enhanced layout
    fig.update_layout(
        mapbox_center={"lat": -1.9441, "lon": 30.0619},
        mapbox_zoom=zoom,
        height=650,
        margin={"r":0,"t":40,"l":0,"b":0},
        title_font_size=16,
        title_x=0.5
    )
    
    # Add custom hover template
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                    f"{indicator_info['name']}: %{{customdata[0]:.3f}}<br>" +
                    "<extra></extra>"
    )
    
    return fig

def create_severity_classification_map(hotspot_data, geographic_level, indicator):
    """Create severity classification map that adapts to selected indicator"""
    
    if hotspot_data is None or len(hotspot_data) == 0:
        st.error("No hotspot data available")
        return None
    
    # Get indicator information
    indicator_info = get_indicator_info(indicator)
    
    # Check if indicator exists
    if indicator not in hotspot_data.columns:
        st.error(f"Indicator '{indicator}' not found in hotspot data")
        return None
    
    # Determine geographic columns
    if geographic_level == "Province":
        group_col = 'province_first'
        title_suffix = 'Province'
        zoom = 7
        size_multiplier = 40
    else:
        group_col = 'district'
        title_suffix = 'District'
        zoom = 8
        size_multiplier = 30
    
    # Check if geographic column exists
    if group_col not in hotspot_data.columns:
        st.error(f"Geographic column '{group_col}' not found")
        return None
    
    # Find coordinate columns
    lat_cols = [col for col in hotspot_data.columns if 'lat' in col.lower()]
    lon_cols = [col for col in hotspot_data.columns if 'lon' in col.lower()]
    
    if not lat_cols or not lon_cols:
        st.error("No coordinate columns found in hotspot data")
        return None
    
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Aggregate data based on geographic level
    if geographic_level == "Province":
        agg_data = hotspot_data.groupby(group_col).agg({
            indicator: 'mean',
            lat_col: 'mean',
            lon_col: 'mean'
        }).reset_index()
    else:
        agg_data = hotspot_data.copy()
    
    # Calculate severity based on selected indicator
    if indicator in ['vitamin_a_received', 'dietary_diversity', 'food_security', 'wealth_index', 'overall_nutrition']:
        # For positive indicators, lower values are worse
        agg_data['hotspot_severity'] = pd.cut(
            agg_data[indicator],
            bins=[0, 0.4, 0.7, 1.0],
            labels=['High', 'Medium', 'Low']
        )
    else:
        # For risk indicators, higher values are worse
        agg_data['hotspot_severity'] = pd.cut(
            agg_data[indicator],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
    
    # Enhanced color mapping
    color_map = {
        'High': '#dc2626',    # Red
        'Medium': '#f59e0b',  # Orange  
        'Low': '#059669'      # Green
    }
    
    fig = px.scatter_mapbox(
        agg_data,
        lat=lat_col,
        lon=lon_col,
        color='hotspot_severity',
        size=indicator,
        hover_name=group_col,
        hover_data={
            indicator: ':.3f'
        },
        title=f'🎯 {indicator_info["name"]} Severity Classification - {title_suffix} Level',
        color_discrete_map=color_map,
        size_max=size_multiplier,
        mapbox_style="open-street-map"
    )
    
    # Enhanced layout
    fig.update_layout(
        mapbox_center={"lat": -1.9441, "lon": 30.0619},
        mapbox_zoom=zoom,
        height=650,
        margin={"r":0,"t":60,"l":0,"b":0},
        title_font_size=18,
        title_x=0.5,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1,
            xanchor="right",
            x=1
        )
    )
    
    # Custom hover template
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                     "Severity: %{marker.color}<br>" +
                     f"{indicator_info['name']}: %{{customdata[0]:.3f}}<br>" +
                     "<extra></extra>"
    )
    
    return fig

def create_advanced_heat_map(data, indicator_col, geographic_level):
    """Create advanced heat map that uses selected indicator"""
    
    if indicator_col not in data.columns:
        st.error(f"Indicator '{indicator_col}' not found in dataset")
        return None
    
    # Get indicator information
    indicator_info = get_indicator_info(indicator_col)
    
    # Find coordinate columns
    lat_cols = [col for col in data.columns if 'lat' in col.lower()]
    lon_cols = [col for col in data.columns if 'lon' in col.lower()]
    
    if not lat_cols or not lon_cols:
        st.error("No coordinate columns found in dataset")
        return None
    
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Adjust sampling based on geographic level
    if geographic_level == "Province":
        sample_size = min(5000, len(data))
        radius = 15
    else:
        sample_size = min(2000, len(data))
        radius = 10
        
    sample_data = data.sample(sample_size)
    
    fig = px.density_mapbox(
        sample_data,
        lat=lat_col,
        lon=lon_col,
        z=indicator_col,
        radius=radius,
        title=f'🔥 {indicator_info["name"]} Heat Map - {geographic_level} Level',
        mapbox_style="open-street-map",
        color_continuous_scale=indicator_info['color_scale'],
        opacity=0.7
    )
    
    # Adjust zoom based on geographic level
    zoom_level = 7 if geographic_level == "Province" else 8
    
    fig.update_layout(
        mapbox_center={"lat": -1.9441, "lon": 30.0619},
        mapbox_zoom=zoom_level,
        height=650,
        margin={"r":0,"t":60,"l":0,"b":0},
        title_font_size=18,
        title_x=0.5
    )
    
    return fig

def create_intervention_priority_map(hotspot_data, geographic_level, indicator):
    """Create intervention priority map based on selected indicator"""
    
    if hotspot_data is None or len(hotspot_data) == 0:
        st.error("No hotspot data available")
        return None
    
    # Get indicator information
    indicator_info = get_indicator_info(indicator)
    
    # Check if indicator exists
    if indicator not in hotspot_data.columns:
        st.error(f"Indicator '{indicator}' not found in hotspot data")
        return None
    
    # Determine geographic columns
    if geographic_level == "Province":
        group_col = 'province_first'
        title_suffix = 'Province'
        zoom = 7
        size_multiplier = 40
    else:
        group_col = 'district'
        title_suffix = 'District'
        zoom = 8
        size_multiplier = 30
    
    # Check if geographic column exists
    if group_col not in hotspot_data.columns:
        st.error(f"Geographic column '{group_col}' not found")
        return None
    
    # Find coordinate columns
    lat_cols = [col for col in hotspot_data.columns if 'lat' in col.lower()]
    lon_cols = [col for col in hotspot_data.columns if 'lon' in col.lower()]
    
    if not lat_cols or not lon_cols:
        st.error("No coordinate columns found in hotspot data")
        return None
    
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Aggregate data based on geographic level
    if geographic_level == "Province":
        agg_data = hotspot_data.groupby(group_col).agg({
            indicator: 'mean',
            lat_col: 'mean',
            lon_col: 'mean'
        }).reset_index()
    else:
        agg_data = hotspot_data.copy()
    
    # Calculate priority rank based on selected indicator
    if indicator in ['vitamin_a_received', 'dietary_diversity', 'food_security', 'wealth_index', 'overall_nutrition']:
        # For positive indicators, lower values get higher priority
        agg_data['priority_rank'] = agg_data[indicator].rank(ascending=True)
    else:
        # For risk indicators, higher values get higher priority
        agg_data['priority_rank'] = agg_data[indicator].rank(ascending=False)
    
    # Create priority categories
    max_rank = len(agg_data)
    if geographic_level == "Province":
        bins = [0, 2, 4, max_rank]
    else:
        bins = [0, 5, 15, max_rank]
        
    agg_data['priority_category'] = pd.cut(
        agg_data['priority_rank'],
        bins=bins,
        labels=['Immediate', 'Short-term', 'Medium-term']
    )
    
    color_map = {
        'Immediate': '#dc2626',
        'Short-term': '#f59e0b', 
        'Medium-term': '#059669'
    }
    
    fig = px.scatter_mapbox(
        agg_data,
        lat=lat_col,
        lon=lon_col,
        color='priority_category',
        size=indicator,
        hover_name=group_col,
        hover_data={
            'priority_rank': True,
            indicator: ':.3f'
        },
        title=f'🚨 {indicator_info["name"]} Priority Mapping - {title_suffix} Level',
        color_discrete_map=color_map,
        size_max=size_multiplier,
        mapbox_style="open-street-map"
    )
    
    fig.update_layout(
        mapbox_center={"lat": -1.9441, "lon": 30.0619},
        mapbox_zoom=zoom,
        height=650,
        margin={"r":0,"t":60,"l":0,"b":0},
        title_font_size=18,
        title_x=0.5
    )
    
    return fig

def create_clustered_bubble_map(data, indicator_col, geographic_level):
    """Create clustered bubble map that uses selected indicator"""
    
    if indicator_col not in data.columns:
        st.error(f"Indicator '{indicator_col}' not found in dataset")
        return None
    
    if geographic_level == "District":
        group_col = 'district'
        title_suffix = 'District'
        zoom = 8
        size_multiplier = 20
    else:
        group_col = 'province'
        title_suffix = 'Province'
        zoom = 7
        size_multiplier = 30
    
    # Get indicator information
    indicator_info = get_indicator_info(indicator_col)
    
    # Check if geographic column exists
    if group_col not in data.columns:
        st.error(f"Geographic column '{group_col}' not found")
        return None
    
    # Find coordinate columns
    lat_cols = [col for col in data.columns if 'lat' in col.lower()]
    lon_cols = [col for col in data.columns if 'lon' in col.lower()]
    
    if not lat_cols or not lon_cols:
        st.error("No coordinate columns found in dataset")
        return None
    
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Aggregate data based on geographic level
    agg_data = data.groupby(group_col).agg({
        indicator_col: 'mean',
        lat_col: 'mean',
        lon_col: 'mean'
    }).reset_index()
    
    # Remove invalid coordinates
    agg_data = agg_data[(agg_data[lat_col] != 0) & (agg_data[lon_col] != 0) & 
                       (~agg_data[lat_col].isna()) & (~agg_data[lon_col].isna())]
    
    if len(agg_data) == 0:
        st.error("No valid geographic data available for mapping")
        return None
    
    # Create enhanced bubble map with clustering effect
    fig = px.scatter_mapbox(
        agg_data,
        lat=lat_col,
        lon=lon_col,
        size=indicator_col,
        color=indicator_col,
        hover_name=group_col,
        hover_data={
            indicator_col: ':.3f'
        },
        title=f'📍 {indicator_info["name"]} - {title_suffix} Level',
        color_continuous_scale=indicator_info['color_scale'],
        size_max=size_multiplier,
        mapbox_style="open-street-map",
        opacity=0.8
    )
    
    # Enhanced layout with better clustering visualization
    fig.update_layout(
        mapbox_center={"lat": -1.9441, "lon": 30.0619},
        mapbox_zoom=zoom,
        height=700,
        margin={"r":0,"t":80,"l":0,"b":0},
        title_font_size=20,
        title_x=0.5,
        showlegend=True
    )
    
    return fig

def create_comparative_analysis_map(hotspot_data, geographic_level, primary_indicator):
    """Create comparative analysis map showing multiple indicators"""
    
    if hotspot_data is None or len(hotspot_data) == 0:
        st.error("No hotspot data available")
        return None
    
    # Get primary indicator information
    primary_info = get_indicator_info(primary_indicator)
    
    # Check if primary indicator exists
    if primary_indicator not in hotspot_data.columns:
        st.error(f"Primary indicator '{primary_indicator}' not found in hotspot data")
        return None
    
    # Determine geographic columns
    if geographic_level == "Province":
        group_col = 'province_first'
        title_suffix = 'Province'
        zoom = 7
    else:
        group_col = 'district'
        title_suffix = 'District'
        zoom = 8
    
    # Check if geographic column exists
    if group_col not in hotspot_data.columns:
        st.error(f"Geographic column '{group_col}' not found")
        return None
    
    # Find coordinate columns
    lat_cols = [col for col in hotspot_data.columns if 'lat' in col.lower()]
    lon_cols = [col for col in hotspot_data.columns if 'lon' in col.lower()]
    
    if not lat_cols or not lon_cols:
        st.error("No coordinate columns found in hotspot data")
        return None
    
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Define available indicators for comparison
    comparison_indicators = [
        'stunting_rate_mean', 'wasting_rate_mean', 'underweight_rate_mean', primary_indicator
    ]
    
    # Filter to only available indicators
    available_indicators = [ind for ind in comparison_indicators if ind in hotspot_data.columns]
    
    if len(available_indicators) < 2:
        st.error("Not enough indicators available for comparative analysis")
        return None
    
    # Aggregate data based on geographic level
    if geographic_level == "Province":
        agg_data = hotspot_data.groupby(group_col).agg({
            **{ind: 'mean' for ind in available_indicators},
            lat_col: 'mean',
            lon_col: 'mean'
        }).reset_index()
    else:
        agg_data = hotspot_data.copy()
    
    # Adjust marker sizes based on geographic level
    size_multiplier = 60 if geographic_level == "Province" else 40
    
    # Create subplot for comparative analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '📊 Stunting Rate', '🔥 Wasting Rate',
            '⚖️ Underweight Rate', f'🎯 {primary_info["name"]}'
        ),
        specs=[[{"type": "scattermapbox"}, {"type": "scattermapbox"}],
               [{"type": "scattermapbox"}, {"type": "scattermapbox"}]]
    )
    
    # Plot available indicators
    plot_configs = [
        ('stunting_rate_mean', 'Reds', 'Stunting'),
        ('wasting_rate_mean', 'Oranges', 'Wasting'),
        ('underweight_rate_mean', 'Purples', 'Underweight'),
        (primary_indicator, get_indicator_info(primary_indicator)['color_scale'], primary_info["name"])
    ]
    
    for i, (indicator, color_scale, name) in enumerate(plot_configs):
        if indicator in available_indicators:
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scattermapbox(
                    lat=agg_data[lat_col],
                    lon=agg_data[lon_col],
                    mode='markers',
                    marker=dict(
                        size=agg_data[indicator] * size_multiplier + 10,
                        color=agg_data[indicator],
                        colorscale=color_scale,
                        showscale=True,
                        colorbar=dict(
                            x=0.45 if col == 1 else 0.95,
                            y=0.8 if row == 1 else 0.2,
                            len=0.15
                        )
                    ),
                    text=agg_data[group_col] + f'<br>{name}: ' + agg_data[indicator].round(3).astype(str),
                    hoverinfo='text',
                    name=name
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        mapbox1=dict(style="open-street-map", center=dict(lat=-1.9441, lon=30.0619), zoom=zoom),
        mapbox2=dict(style="open-street-map", center=dict(lat=-1.9441, lon=30.0619), zoom=zoom),
        mapbox3=dict(style="open-street-map", center=dict(lat=-1.9441, lon=30.0619), zoom=zoom),
        mapbox4=dict(style="open-street-map", center=dict(lat=-1.9441, lon=30.0619), zoom=zoom),
        height=800,
        title_text=f"🔍 Comparative Analysis: Multiple Indicators - {title_suffix} Level",
        title_x=0.5,
        title_font_size=20,
        showlegend=False
    )
    
    return fig

def micronutrient_hotspots():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header";background: black>🗺️ Micronutrient Deficiency Hotspots Mapping</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 10px; margin-bottom: 2rem;">
        <p style="color: #374151; margin: 0;text-align: left; font-size: 1.1rem;">Interactive Geographic Analysis for Targeted Micronutrient Interventions in Rwanda</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading geographic data..."):
        household_geo, child_geo, child_hotspots, district_coords, is_enhanced = load_enhanced_geographic_data()
    
    # # Data status indicator
    # if is_enhanced:
    #     st.success("✅ Enhanced geographic data loaded successfully!")
    # else:
    #     st.info("📊 Using sample data for demonstration. Run enhanced extraction for real data.")
    
    # # Show available columns for debugging
    # with st.expander("🔍 Dataset Information"):
    #     if child_geo is not None:
    #         st.write("**Child Geography Data Columns:**", list(child_geo.columns))
    #     if child_hotspots is not None:
    #         st.write("**Hotspot Data Columns:**", list(child_hotspots.columns))
    
    # Main content layout with mapping controls
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Mapping Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Dataset selection
        dataset_choice = st.selectbox(
            "📊 Dataset", 
            ["Child Nutrition (6-59 months)", "Household Analysis", "Combined View"],
            help="Select the dataset for hotspot analysis"
        )
        
        # Geographic level
        geographic_level = st.selectbox(
            "🌍 Geographic Level", 
            ["District", "Province"],
            help="Choose the administrative level for analysis"
        )
        
        # Show immediate feedback when geographic level changes
        if 'prev_geo_level' not in st.session_state:
            st.session_state.prev_geo_level = geographic_level
        
        if st.session_state.prev_geo_level != geographic_level:
            st.session_state.prev_geo_level = geographic_level
            st.rerun()
    
    with col2:
        # Get available indicators based on dataset choice
        if child_geo is not None:
            available_columns = list(child_geo.columns)
            # Filter to likely indicator columns (numeric columns that aren't coordinates or IDs)
            numeric_cols = child_geo.select_dtypes(include=[np.number]).columns.tolist()
            coord_cols = [col for col in available_columns if any(x in col.lower() for x in ['lat', 'lon', 'x', 'y'])]
            id_cols = [col for col in available_columns if any(x in col.lower() for x in ['id', 'code', 'index'])]
            
            potential_indicators = [col for col in numeric_cols if col not in coord_cols + id_cols and col not in ['', ' ']]
            
            if len(potential_indicators) > 0:
                available_indicators = potential_indicators[:10]  # Limit to first 10
            else:
                available_indicators = ['composite_risk_score']  # Fallback
        else:
            available_indicators = ['composite_risk_score']
        
        # Indicator selection
        indicator = st.selectbox(
            "📈 Micronutrient Indicator", 
            available_indicators,
            help="Select the indicator to visualize on the map"
        )
        
        # Show immediate feedback when indicator changes
        if 'prev_indicator' not in st.session_state:
            st.session_state.prev_indicator = indicator
        
        if st.session_state.prev_indicator != indicator:
            st.session_state.prev_indicator = indicator
            st.rerun()
        
        # Visualization type
        viz_type = st.selectbox(
            "🎨 Visualization Type",
            ["Interactive Bubble Map", "Severity Classification", "Advanced Heat Map", 
             "Intervention Priority", "Clustered Analysis", "Comparative Analysis"],
            help="Choose how to display the data on the map"
        )
    
    with col3:
        # Hotspot threshold
        hotspot_threshold = st.slider(
            "🎯 Hotspot Threshold (%)",
            min_value=50,
            max_value=95,
            value=75,
            help="Percentile threshold for identifying hotspots"
        )
        
        # Display options
        show_statistics = st.checkbox("📊 Show Statistics Panel", True)
        show_recommendations = st.checkbox("💡 Show Recommendations", True)
        advanced_analysis = st.checkbox("🔍 Advanced Analysis", False)
    
    # Display current indicator information
    indicator_info = get_indicator_info(indicator)
    st.markdown(f"""
    <div class="indicator-info">
        <h4 style="margin: 0; color: #4f46e5;">📊 Current Indicator: {indicator_info['name']}</h4>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;"><strong>Description:</strong> {indicator_info['description']}</p>
        <p style="margin: 0; font-size: 0.9rem;"><strong>Interpretation:</strong> {indicator_info['interpretation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display current geographic level info
    if child_hotspots is not None:
        if geographic_level == "District" and 'district' in child_hotspots.columns:
            num_units = len(child_hotspots['district'].unique())
        elif geographic_level == "Province" and 'province_first' in child_hotspots.columns:
            num_units = len(child_hotspots['province_first'].unique())
        else:
            num_units = "Unknown"
        
        st.info(f"**Current Analysis Level:** {geographic_level} level - Showing {num_units} {geographic_level.lower()}(s)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main visualization area
    if show_statistics:
        col1, col2 = st.columns([3, 1])
    else:
        col1 = st.container()
        col2 = None
    
    with col1:
        # Create main visualization - NOW WITH PROPER ERROR HANDLING
        try:
            if viz_type == "Severity Classification":
                fig = create_severity_classification_map(child_hotspots, geographic_level, indicator)
            elif viz_type == "Interactive Bubble Map":
                fig = create_enhanced_bubble_map(child_geo, indicator, geographic_level)
            elif viz_type == "Advanced Heat Map":
                fig = create_advanced_heat_map(child_geo, indicator, geographic_level)
            elif viz_type == "Intervention Priority":
                fig = create_intervention_priority_map(child_hotspots, geographic_level, indicator)
            elif viz_type == "Clustered Analysis":
                fig = create_clustered_bubble_map(child_geo, indicator, geographic_level)
            elif viz_type == "Comparative Analysis":
                fig = create_comparative_analysis_map(child_hotspots, geographic_level, indicator)
            else:
                fig = create_enhanced_bubble_map(child_geo, indicator, geographic_level)
            
            if fig:
                st.markdown('<div class="map-container">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Could not create map visualization with current data and settings")
                
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            st.info("Try selecting a different indicator or visualization type")
    
    # Statistics panel
    if show_statistics and col2 and child_geo is not None and indicator in child_geo.columns:
        with col2:
            st.markdown(f"### 📊 {indicator_info['name']} Stats")
            
            try:
                if geographic_level == "District" and 'district' in child_geo.columns:
                    # District level statistics for selected indicator
                    indicator_data = child_geo.groupby('district')[indicator].mean()
                elif geographic_level == "Province" and 'province' in child_geo.columns:
                    # Province level statistics for selected indicator
                    indicator_data = child_geo.groupby('province')[indicator].mean()
                else:
                    # Fallback to overall statistics
                    indicator_data = pd.Series([child_geo[indicator].mean()])
                
                total_units = len(indicator_data)
                mean_value = indicator_data.mean()
                std_value = indicator_data.std()
                max_value = indicator_data.max()
                min_value = indicator_data.min()
                
                # Calculate percentiles for the selected indicator
                high_threshold = indicator_data.quantile(0.75)
                low_threshold = indicator_data.quantile(0.25)
                high_risk = len(indicator_data[indicator_data >= high_threshold])
                low_risk = len(indicator_data[indicator_data <= low_threshold])
                medium_risk = total_units - high_risk - low_risk
                
                st.metric(f"🏛️ Total {geographic_level}s", total_units)
                st.metric("📊 Mean Value", f"{mean_value:.3f}")
                st.metric("📈 Std Dev", f"{std_value:.3f}")
                st.metric("🔺 Max", f"{max_value:.3f}")
                st.metric("🔻 Min", f"{min_value:.3f}")
                
                st.markdown("---")
                
                # Risk distribution
                st.markdown("**🎯 Risk Distribution**")
                st.metric("🔴 High", high_risk, delta=f"{high_risk/total_units*100:.1f}%")
                st.metric("🟡 Medium", medium_risk, delta=f"{medium_risk/total_units*100:.1f}%")
                st.metric("🟢 Low", low_risk, delta=f"{low_risk/total_units*100:.1f}%")
                
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")

if __name__ == "__main__":
    micronutrient_hotspots()