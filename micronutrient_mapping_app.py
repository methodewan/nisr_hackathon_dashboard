import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Micronutrient Hotspots Mapping",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    """Create sample geographic data for demonstration"""
    
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
        'Rwamagana': {'lat': -1.95, 'lon': 30.45, 'province': 'Estern'},
        'Kayonza': {'lat': -1.85, 'lon': 30.55, 'province': 'Estern'},
        'Kirehe': {'lat': -1.75, 'lon': 30.65, 'province': 'Estern'},
        'Ngoma': {'lat': -1.65, 'lon': 30.35, 'province': 'Estern'},
        'Gatsibo': {'lat': -1.55, 'lon': 30.25, 'province': 'Estern'},
        'Nyagatare': {'lat': -1.35, 'lon': 30.15, 'province': 'Estern'},
        'Bugesera': {'lat': -2.25, 'lon': 30.25, 'province': 'Estern'},
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
    
    # Sample child nutrition data
    child_data = []
    for district in districts:
        n_records = np.random.randint(100, 200)
        coords = district_coordinates[district]
        
        for _ in range(n_records):
            child_data.append({
                'district': district,
                'province': coords['province'],
                'lat': coords['lat'] + np.random.normal(0, 0.05),
                'lon': coords['lon'] + np.random.normal(0, 0.05),
                'stunting_rate': np.random.uniform(0.2, 0.6),
                'wasting_rate': np.random.uniform(0.05, 0.25),
                'underweight_rate': np.random.uniform(0.1, 0.3),
                'vitamin_a_received': np.random.uniform(0.7, 0.98),
                'urban_rural': np.random.choice(['Urban', 'Rural'], p=[0.3, 0.7])
            })
    
    child_geo = pd.DataFrame(child_data)
    
    # Create hotspot analysis
    hotspot_data = []
    for district in districts:
        district_data = child_geo[child_geo['district'] == district]
        coords = district_coordinates[district]
        
        # Calculate composite risk score
        stunting_score = district_data['stunting_rate'].mean()
        wasting_score = district_data['wasting_rate'].mean()
        underweight_score = district_data['underweight_rate'].mean()
        vitamin_a_score = 1 - district_data['vitamin_a_received'].mean()  # Invert for risk
        
        composite_risk = (stunting_score + wasting_score + underweight_score + vitamin_a_score) / 4
        
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

def create_enhanced_bubble_map(data, indicator_col, geographic_level):
    """Create enhanced bubble map with better styling"""
    
    if geographic_level == "District":
        group_col = 'district'
        zoom = 8
    else:
        group_col = 'province'
        zoom = 7
    
    # Aggregate data
    if group_col in data.columns and indicator_col in data.columns:
        agg_data = data.groupby(group_col).agg({
            indicator_col: 'mean',
            'lat': 'first',
            'lon': 'first'
        }).reset_index()
        
        # Remove invalid coordinates
        agg_data = agg_data[(agg_data['lat'] != 0) & (agg_data['lon'] != 0)]
        
        if len(agg_data) > 0:
            # Create bubble map with enhanced styling
            fig = px.scatter_mapbox(
                agg_data,
                lat='lat',
                lon='lon',
                size=indicator_col,
                color=indicator_col,
                hover_name=group_col,
                hover_data={indicator_col: ':.3f'},
                title=f'{indicator_col.replace("_", " ").title()} by {geographic_level}',
                color_continuous_scale='Reds',
                size_max=60,
                mapbox_style="open-street-map"
            )
            
            # Enhanced layout
            fig.update_layout(
                mapbox_center={"lat": -1.9441, "lon": 30.0619},
                mapbox_zoom=zoom,
                height=650,
                margin={"r":0,"t":60,"l":0,"b":0},
                title_font_size=16,
                title_x=0.5
            )
            
            # Add custom hover template
            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                            f"{indicator_col.replace('_', ' ').title()}: %{{customdata[0]:.3f}}<br>" +
                            "<extra></extra>"
            )
            
            return fig
    
    return None

def create_severity_classification_map(hotspot_data):
    """Create severity classification map with enhanced features"""
    
    if hotspot_data is not None and len(hotspot_data) > 0:
        # Enhanced color mapping
        color_map = {
            'High': '#dc2626',    # Red
            'Medium': '#f59e0b',  # Orange  
            'Low': '#059669'      # Green
        }
        
        fig = px.scatter_mapbox(
            hotspot_data,
            lat='lat_first',
            lon='lon_first',
            color='hotspot_severity',
            size='composite_risk_score',
            hover_name='district',
            hover_data={
                'composite_risk_score': ':.3f',
                'stunting_rate_mean': ':.1%',
                'wasting_rate_mean': ':.1%',
                'vitamin_a_received_mean': ':.1%'
            },
            title='🎯 Micronutrient Deficiency Severity Classification',
            color_discrete_map=color_map,
            size_max=70,
            mapbox_style="open-street-map"
        )
        
        # Enhanced layout
        fig.update_layout(
            mapbox_center={"lat": -1.9441, "lon": 30.0619},
            mapbox_zoom=8,
            height=650,
            margin={"r":0,"t":60,"l":0,"b":0},
            title_font_size=18,
            title_x=0.5,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Custom hover template
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>" +
                         "Severity: %{marker.color}<br>" +
                         "Risk Score: %{customdata[0]:.3f}<br>" +
                         "Stunting: %{customdata[1]:.1%}<br>" +
                         "Wasting: %{customdata[2]:.1%}<br>" +
                         "Vitamin A: %{customdata[3]:.1%}<br>" +
                         "<extra></extra>"
        )
        
        return fig
    
    return None

def create_advanced_heat_map(data, indicator_col):
    """Create advanced heat map with multiple layers"""
    
    if 'lat' in data.columns and 'lon' in data.columns and indicator_col in data.columns:
        # Sample data to avoid overcrowding
        sample_size = min(2000, len(data))
        sample_data = data.sample(sample_size)
        
        fig = px.density_mapbox(
            sample_data,
            lat='lat',
            lon='lon',
            z=indicator_col,
            radius=20,
            title=f'🔥 {indicator_col.replace("_", " ").title()} Heat Map',
            mapbox_style="open-street-map",
            opacity=0.7
        )
        
        fig.update_layout(
            mapbox_center={"lat": -1.9441, "lon": 30.0619},
            mapbox_zoom=8,
            height=650,
            margin={"r":0,"t":60,"l":0,"b":0},
            title_font_size=18,
            title_x=0.5
        )
        
        return fig
    
    return None

def create_intervention_priority_map(hotspot_data):
    """Create intervention priority map"""
    
    if hotspot_data is not None and len(hotspot_data) > 0:
        # Create priority categories
        hotspot_data['priority_category'] = pd.cut(
            hotspot_data['priority_rank'],
            bins=[0, 5, 15, 30],
            labels=['Immediate', 'Short-term', 'Medium-term']
        )
        
        color_map = {
            'Immediate': '#dc2626',
            'Short-term': '#f59e0b', 
            'Medium-term': '#059669'
        }
        
        fig = px.scatter_mapbox(
            hotspot_data,
            lat='lat_first',
            lon='lon_first',
            color='priority_category',
            size='composite_risk_score',
            hover_name='district',
            hover_data={
                'priority_rank': True,
                'composite_risk_score': ':.3f'
            },
            title='🚨 Intervention Priority Mapping',
            color_discrete_map=color_map,
            size_max=60,
            mapbox_style="open-street-map"
        )
        
        fig.update_layout(
            mapbox_center={"lat": -1.9441, "lon": 30.0619},
            mapbox_zoom=8,
            height=650,
            margin={"r":0,"t":60,"l":0,"b":0},
            title_font_size=18,
            title_x=0.5
        )
        
        return fig
    
    return None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">🗺️ Micronutrient Deficiency Hotspots Mapping</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); border-radius: 10px; margin-bottom: 2rem;">
        <p style="color: #374151; margin: 0; font-size: 1.1rem;">Interactive Geographic Analysis for Targeted Micronutrient Interventions in Rwanda</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading geographic data..."):
        household_geo, child_geo, child_hotspots, district_coords, is_enhanced = load_enhanced_geographic_data()
    
    # Data status indicator
    if is_enhanced:
        st.success("✅ Enhanced geographic data loaded successfully!")
    else:
        st.info("📊 Using sample data for demonstration. Run enhanced extraction for real data.")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Mapping Controls")
        
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
        
        # Indicator selection
        indicator_options = {
            "Child Nutrition (6-59 months)": [
                "stunting_rate", "wasting_rate", "underweight_rate", 
                "vitamin_a_received", "composite_risk_score"
            ],
            "Household Analysis": [
                "dietary_diversity", "food_security", "wealth_index"
            ],
            "Combined View": [
                "composite_risk_score", "stunting_rate", "overall_nutrition"
            ]
        }
        
        available_indicators = indicator_options.get(dataset_choice, ["stunting_rate"])
        indicator = st.selectbox(
            "📈 Micronutrient Indicator", 
            available_indicators,
            format_func=lambda x: x.replace('_', ' ').title(),
            help="Select the indicator to visualize on the map"
        )
        
        # Visualization type
        viz_type = st.selectbox(
            "🎨 Visualization Type",
            ["Severity Classification", "Interactive Bubble Map", "Advanced Heat Map", "Intervention Priority"],
            help="Choose how to display the data on the map"
        )
        
        # Hotspot threshold
        hotspot_threshold = st.slider(
            "🎯 Hotspot Threshold (%)",
            min_value=50,
            max_value=95,
            value=75,
            help="Percentile threshold for identifying hotspots"
        )
        
        st.markdown("---")
        
        # Display options
        st.markdown("### 🔧 Display Options")
        show_statistics = st.checkbox("📊 Show Statistics Panel", True)
        show_recommendations = st.checkbox("💡 Show Recommendations", True)
        show_folium_map = st.checkbox("🗺️ Show Advanced Folium Map", False)
    
    # Main content layout
    if show_statistics:
        col1, col2 = st.columns([3, 1])
    else:
        col1 = st.container()
        col2 = None
    
    with col1:
        # Create main visualization
        if viz_type == "Severity Classification":
            fig = create_severity_classification_map(child_hotspots)
        elif viz_type == "Interactive Bubble Map":
            fig = create_enhanced_bubble_map(child_geo, indicator, geographic_level)
        elif viz_type == "Advanced Heat Map":
            fig = create_advanced_heat_map(child_geo, indicator)
        else:  # Intervention Priority
            fig = create_intervention_priority_map(child_hotspots)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not create map visualization with current data")
        
        # Advanced Folium map (optional)
        if show_folium_map and child_hotspots is not None:
            st.markdown("### 🗺️ Advanced Interactive Map")
            
            # Create Folium map
            m = folium.Map(
                location=[-1.9441, 30.0619],
                zoom_start=8,
                tiles='OpenStreetMap'
            )
            
            # Add markers for hotspots
            for _, row in child_hotspots.iterrows():
                if row['hotspot_severity'] == 'High':
                    color = 'red'
                    icon = 'exclamation-sign'
                elif row['hotspot_severity'] == 'Medium':
                    color = 'orange'
                    icon = 'warning-sign'
                else:
                    color = 'green'
                    icon = 'ok-sign'
                
                folium.Marker(
                    location=[row['lat_first'], row['lon_first']],
                    popup=f"""
                    <b>{row['district']}</b><br>
                    Risk Score: {row['composite_risk_score']:.3f}<br>
                    Severity: {row['hotspot_severity']}<br>
                    Stunting: {row['stunting_rate_mean']*100:.1f}%
                    """,
                    tooltip=row['district'],
                    icon=folium.Icon(color=color, icon=icon)
                ).add_to(m)
            
            # Display map
            st_folium(m, width=700, height=500)
    
    # Statistics panel
    if show_statistics and col2:
        with col2:
            st.markdown("### 📊 Hotspot Analytics")
            
            if child_hotspots is not None:
                # Key metrics
                total_districts = len(child_hotspots)
                high_risk = len(child_hotspots[child_hotspots['hotspot_severity'] == 'High'])
                medium_risk = len(child_hotspots[child_hotspots['hotspot_severity'] == 'Medium'])
                low_risk = len(child_hotspots[child_hotspots['hotspot_severity'] == 'Low'])
                
                st.metric("🏛️ Total Districts", total_districts)
                st.metric("🔴 High Risk", high_risk, delta=f"{high_risk/total_districts*100:.1f}%")
                st.metric("🟡 Medium Risk", medium_risk, delta=f"{medium_risk/total_districts*100:.1f}%")
                st.metric("🟢 Low Risk", low_risk, delta=f"{low_risk/total_districts*100:.1f}%")
                
                st.markdown("---")
                
                # Top risk districts
                st.markdown("**🔴 Highest Risk Districts**")
                top_5 = child_hotspots.nlargest(5, 'composite_risk_score')
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    st.markdown(f"**{i}.** {row['district']}")
                    st.write(f"   Risk: {row['composite_risk_score']:.3f}")
                
                st.markdown("---")
                
                # Provincial distribution
                st.markdown("**🌍 Provincial Distribution**")
                if 'province_first' in child_hotspots.columns:
                    province_risk = child_hotspots.groupby('province_first').agg({
                        'composite_risk_score': 'mean'
                    }).round(3)
                    
                    for province, risk in province_risk['composite_risk_score'].items():
                        st.write(f"• {province}: {risk:.3f}")
    
    # Detailed analysis section
    st.markdown("### 📊 Comprehensive Hotspot Analysis")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 District Rankings", "🏛️ Provincial Summary", "📈 Risk Analysis", "🎯 Interventions"])
    
    with tab1:
        if child_hotspots is not None:
            st.markdown("#### 🏆 District Priority Rankings")
            
            # Prepare ranking data
            ranking_data = child_hotspots[[
                'district', 'composite_risk_score', 'hotspot_severity', 'priority_rank',
                'stunting_rate_mean', 'wasting_rate_mean', 'vitamin_a_received_mean'
            ]].copy()
            
            ranking_data = ranking_data.sort_values('composite_risk_score', ascending=False)
            
            # Format for display
            ranking_data['stunting_rate_mean'] = (ranking_data['stunting_rate_mean'] * 100).round(1)
            ranking_data['wasting_rate_mean'] = (ranking_data['wasting_rate_mean'] * 100).round(1)
            ranking_data['vitamin_a_received_mean'] = (ranking_data['vitamin_a_received_mean'] * 100).round(1)
            
            # Rename columns for display
            ranking_data.columns = [
                'District', 'Risk Score', 'Severity', 'Priority Rank',
                'Stunting (%)', 'Wasting (%)', 'Vitamin A (%)'
            ]
            
            # Style the dataframe
            def highlight_severity(val):
                if val == 'High':
                    return 'background-color: #fee2e2; color: #dc2626; font-weight: bold'
                elif val == 'Medium':
                    return 'background-color: #fef3c7; color: #d97706; font-weight: bold'
                else:
                    return 'background-color: #d1fae5; color: #059669; font-weight: bold'
            
            styled_df = ranking_data.style.format({
                'Risk Score': '{:.3f}',
                'Priority Rank': '{:.0f}',
                'Stunting (%)': '{:.1f}',
                'Wasting (%)': '{:.1f}',
                'Vitamin A (%)': '{:.1f}'
            }).applymap(highlight_severity, subset=['Severity'])
            
            st.dataframe(styled_df, use_container_width=True, height=600)
    
    with tab2:
        if child_hotspots is not None:
            st.markdown("#### 🏛️ Provincial Summary Analysis")
            
            # Provincial aggregation
            provincial_summary = child_hotspots.groupby('province_first').agg({
                'composite_risk_score': ['mean', 'std', 'count'],
                'stunting_rate_mean': 'mean',
                'wasting_rate_mean': 'mean',
                'vitamin_a_received_mean': 'mean'
            }).round(3)
            
            # Flatten column names
            provincial_summary.columns = [
                'Avg Risk Score', 'Risk Std Dev', 'Districts Count',
                'Avg Stunting Rate', 'Avg Wasting Rate', 'Avg Vitamin A Coverage'
            ]
            
            st.dataframe(provincial_summary, use_container_width=True)
            
            # Provincial comparison chart
            provincial_data = child_hotspots.groupby('province_first')['composite_risk_score'].mean().reset_index()
            
            fig = px.bar(
                provincial_data,
                x='province_first',
                y='composite_risk_score',
                title='Average Risk Score by Province',
                color='composite_risk_score',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, xaxis_title="Province", yaxis_title="Average Risk Score")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if child_geo is not None:
            st.markdown("#### 📈 Risk Factor Analysis")
            
            # Correlation analysis
            correlation_cols = ['stunting_rate', 'wasting_rate', 'underweight_rate', 'vitamin_a_received']
            available_corr_cols = [col for col in correlation_cols if col in child_geo.columns]
            
            if len(available_corr_cols) > 1:
                corr_matrix = child_geo[available_corr_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="📊 Correlation Matrix: Micronutrient Indicators",
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    text_auto=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Urban vs Rural analysis
            if 'urban_rural' in child_geo.columns:
                st.markdown("**🏙️ Urban vs Rural Comparison**")
                
                urban_rural_stats = child_geo.groupby('urban_rural')[available_corr_cols].mean()
                
                fig = go.Figure()
                for indicator in available_corr_cols:
                    fig.add_trace(go.Bar(
                        name=indicator.replace('_', ' ').title(),
                        x=urban_rural_stats.index,
                        y=urban_rural_stats[indicator] * 100,  # Convert to percentage
                        text=[f"{val*100:.1f}%" for val in urban_rural_stats[indicator]],
                        textposition='auto'
                    ))
                
                fig.update_layout(
                    title="Urban vs Rural Micronutrient Indicators (%)",
                    barmode='group',
                    height=400,
                    yaxis_title="Percentage (%)"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### 🎯 Targeted Intervention Strategies")
        
        if child_hotspots is not None:
            # High priority interventions
            high_risk_districts = child_hotspots[child_hotspots['hotspot_severity'] == 'High']
            medium_risk_districts = child_hotspots[child_hotspots['hotspot_severity'] == 'Medium']
            low_risk_districts = child_hotspots[child_hotspots['hotspot_severity'] == 'Low']
            
            # High severity interventions
            if len(high_risk_districts) > 0:
                st.markdown("##### 🚨 High Severity Areas (Immediate Action Required)")
                
                high_districts_list = high_risk_districts['district'].tolist()
                st.error(f"**Districts:** {', '.join(high_districts_list)}")
                
                st.markdown("""
                **Immediate Interventions (0-3 months):**
                - 🏥 Emergency micronutrient supplementation campaigns
                - 🍼 Severe acute malnutrition treatment programs
                - 💊 Mass vitamin A distribution
                - 🩺 Mobile health clinics deployment
                - 👥 Community health worker training
                """)
            
            # Medium severity interventions
            if len(medium_risk_districts) > 0:
                st.markdown("##### ⚠️ Medium Severity Areas (Preventive Focus)")
                
                medium_count = len(medium_risk_districts)
                st.warning(f"**{medium_count} districts** requiring preventive interventions")
                
                st.markdown("""
                **Preventive Interventions (3-12 months):**
                - 📚 Nutrition education programs
                - 🌱 Homestead food production support
                - 🥄 Micronutrient powder distribution
                - 🤱 Improved infant feeding practices
                - 🚿 WASH infrastructure improvements
                """)
            
            # Low severity maintenance
            if len(low_risk_districts) > 0:
                st.markdown("##### ✅ Low Severity Areas (Maintenance)")
                
                low_count = len(low_risk_districts)
                st.success(f"**{low_count} districts** with good nutrition status")
                
                st.markdown("""
                **Maintenance Activities:**
                - 📊 Regular nutrition monitoring
                - 🔄 Continued supplementation programs
                - 🌾 Sustainable food system support
                - 👥 Community volunteer programs
                """)
    
    # Summary insights
    st.markdown("### 🔍 Key Insights & Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="hotspot-card">
            <h4 style="margin: 0; color: #dc2626;">🎯 Priority Focus Areas</h4>
            <p style="margin: 0.5rem 0 0 0;">Northern and Western provinces show highest risk concentrations requiring immediate intervention.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="medium-risk-card">
            <h4 style="margin: 0; color: #d97706;">📊 Data-Driven Approach</h4>
            <p style="margin: 0.5rem 0 0 0;">Composite risk scoring enables evidence-based resource allocation and intervention targeting.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="low-risk-card">
            <h4 style="margin: 0; color: #059669;">🚀 Scalable Solutions</h4>
            <p style="margin: 0.5rem 0 0 0;">Geographic mapping enables scalable, targeted interventions for maximum impact.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with data source information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
        <p>📊 Data Source: CFSVA 2024 Survey | 🗺️ Geographic Analysis: Enhanced Mapping System | 🔄 Last Updated: 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()