import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import folium
from folium.plugins import HeatMap, MarkerCluster
import json

class GeographicMappingSystem:
    """Enhanced geographic mapping system for micronutrient hotspots"""
    
    def __init__(self):
        self.district_coordinates = {
            # Kigali City
            'Nyarugenge': {'lat': -1.9441, 'lon': 30.0619, 'province': 'City of Kigali'},
            'Kicukiro': {'lat': -1.9541, 'lon': 30.0719, 'province': 'City of Kigali'},
            'Gasabo': {'lat': -1.9341, 'lon': 30.0519, 'province': 'City of Kigali'},
            
            # Northern Province
            'Rulindo': {'lat': -1.75, 'lon': 30.0, 'province': 'Northern'},
            'Gicumbi': {'lat': -1.6, 'lon': 30.1, 'province': 'Northern'},
            'Musanze': {'lat': -1.45, 'lon': 29.9, 'province': 'Northern'},
            'Burera': {'lat': -1.4, 'lon': 29.8, 'province': 'Northern'},
            'Gakenke': {'lat': -1.8, 'lon': 29.95, 'province': 'Northern'},
            
            # Southern Province
            'Nyanza': {'lat': -2.35, 'lon': 29.55, 'province': 'Southern'},
            'Huye': {'lat': -2.6, 'lon': 29.75, 'province': 'Southern'},
            'Gisagara': {'lat': -2.55, 'lon': 30.05, 'province': 'Southern'},
            'Muhanga': {'lat': -2.2, 'lon': 29.75, 'province': 'Southern'},
            'Kamonyi': {'lat': -2.15, 'lon': 29.85, 'province': 'Southern'},
            'Ruhango': {'lat': -2.25, 'lon': 29.65, 'province': 'Southern'},
            'Nyaruguru': {'lat': -2.45, 'lon': 29.45, 'province': 'Southern'},
            'Nyamagabe': {'lat': -2.65, 'lon': 29.35, 'province': 'Southern'},
            
            # Eastern Province
            'Rwamagana': {'lat': -1.95, 'lon': 30.45, 'province': 'Estern'},
            'Kayonza': {'lat': -1.85, 'lon': 30.55, 'province': 'Estern'},
            'Kirehe': {'lat': -1.75, 'lon': 30.65, 'province': 'Estern'},
            'Ngoma': {'lat': -1.65, 'lon': 30.35, 'province': 'Estern'},
            'Gatsibo': {'lat': -1.55, 'lon': 30.25, 'province': 'Estern'},
            'Nyagatare': {'lat': -1.35, 'lon': 30.15, 'province': 'Estern'},
            'Bugesera': {'lat': -2.25, 'lon': 30.25, 'province': 'Estern'},
            
            # Western Province
            'Karongi': {'lat': -2.05, 'lon': 29.25, 'province': 'Western'},
            'Rutsiro': {'lat': -1.95, 'lon': 29.35, 'province': 'Western'},
            'Rubavu': {'lat': -1.85, 'lon': 29.45, 'province': 'Western'},
            'Nyabihu': {'lat': -1.75, 'lon': 29.55, 'province': 'Western'},
            'Ngororero': {'lat': -1.65, 'lon': 29.65, 'province': 'Western'},
            'Rusizi': {'lat': -2.55, 'lon': 28.95, 'province': 'Western'},
            'Nyamasheke': {'lat': -2.35, 'lon': 29.15, 'province': 'Western'}
        }
        
        self.province_coordinates = {
            'City of Kigali': {'lat': -1.9441, 'lon': 30.0619},
            'Northern': {'lat': -1.5, 'lon': 29.9},
            'Southern': {'lat': -2.4, 'lon': 29.6},
            'Estern': {'lat': -1.8, 'lon': 30.4},
            'Western': {'lat': -2.0, 'lon': 29.3}
        }
    
    def load_and_process_geographic_data(self):
        """Load and process geographic data with coordinates"""
        try:
            # Load existing geographic data
            geo_household = pd.read_csv("geographic_household.csv")
            geo_child = pd.read_csv("geographic_child.csv")
            geo_women = pd.read_csv("geographic_women.csv")
            
            # Add coordinates to each dataset
            for geo_data, dataset_name in [(geo_household, 'household'), 
                                         (geo_child, 'child'), 
                                         (geo_women, 'women')]:
                if 'district' in geo_data.columns:
                    geo_data['lat'] = geo_data['district'].map(
                        lambda x: self.district_coordinates.get(x, {}).get('lat', 0)
                    )
                    geo_data['lon'] = geo_data['district'].map(
                        lambda x: self.district_coordinates.get(x, {}).get('lon', 0)
                    )
                    geo_data['province_mapped'] = geo_data['district'].map(
                        lambda x: self.district_coordinates.get(x, {}).get('province', 'Unknown')
                    )
            
            return geo_household, geo_child, geo_women
            
        except Exception as e:
            st.error(f"Error loading geographic data: {e}")
            return None, None, None
    
    def calculate_micronutrient_indicators(self, data, geo_data):
        """Calculate micronutrient deficiency indicators by geographic area"""
        
        # Merge data with geographic information
        if geo_data is not None and len(data) > 0:
            merged_data = data.merge(geo_data, left_index=True, right_index=True, how='left')
        else:
            merged_data = data.copy()
            # Add sample coordinates if no geographic data
            merged_data['lat'] = np.random.uniform(-2.8, -1.0, len(merged_data))
            merged_data['lon'] = np.random.uniform(28.8, 30.9, len(merged_data))
            merged_data['district'] = np.random.choice(list(self.district_coordinates.keys()), len(merged_data))
            merged_data['province'] = merged_data['district'].map(
                lambda x: self.district_coordinates.get(x, {}).get('province', 'Unknown')
            )
        
        return merged_data
    
    def identify_hotspots(self, data, indicator_columns, threshold_percentile=75):
        """Identify micronutrient deficiency hotspots using clustering and thresholds"""
        
        hotspots = {}
        
        # District-level aggregation
        if 'district' in data.columns:
            district_stats = data.groupby('district').agg({
                **{col: ['mean', 'count'] for col in indicator_columns if col in data.columns},
                'lat': 'first',
                'lon': 'first',
                'province': 'first'
            }).reset_index()
            
            # Flatten column names
            district_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                    for col in district_stats.columns.values]
            
            # Calculate composite risk score
            risk_scores = []
            for col in indicator_columns:
                mean_col = f"{col}_mean"
                if mean_col in district_stats.columns:
                    # Normalize to 0-1 scale
                    scores = district_stats[mean_col].fillna(0)
                    normalized = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else scores
                    risk_scores.append(normalized)
            
            if risk_scores:
                district_stats['composite_risk'] = np.mean(risk_scores, axis=0)
                
                # Identify hotspots based on threshold
                threshold = np.percentile(district_stats['composite_risk'], threshold_percentile)
                district_stats['is_hotspot'] = district_stats['composite_risk'] >= threshold
                
                hotspots['district'] = district_stats
        
        # Province-level aggregation
        if 'province' in data.columns:
            province_stats = data.groupby('province').agg({
                **{col: ['mean', 'count'] for col in indicator_columns if col in data.columns},
                'lat': 'first',
                'lon': 'first'
            }).reset_index()
            
            # Flatten column names
            province_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                    for col in province_stats.columns.values]
            
            hotspots['province'] = province_stats
        
        return hotspots
    
    def create_interactive_map(self, data, indicator, map_type='bubble', geographic_level='district'):
        """Create interactive maps for micronutrient indicators"""
        
        if geographic_level == 'district' and 'district' in data.columns:
            group_col = 'district'
            lat_col, lon_col = 'lat', 'lon'
            zoom_level = 8
        else:
            group_col = 'province'
            lat_col, lon_col = 'lat', 'lon'
            zoom_level = 7
        
        # Aggregate data by geographic level
        agg_data = data.groupby(group_col).agg({
            indicator: 'mean' if indicator in data.columns else 'count',
            lat_col: 'first',
            lon_col: 'first',
            'province': 'first' if 'province' in data.columns else 'count'
        }).reset_index()
        
        # Remove rows with missing coordinates
        agg_data = agg_data[(agg_data[lat_col] != 0) & (agg_data[lon_col] != 0)]
        
        if len(agg_data) == 0:
            st.warning("No valid coordinate data available for mapping")
            return None
        
        # Create map based on type
        if map_type == 'bubble':
            fig = px.scatter_mapbox(
                agg_data, 
                lat=lat_col, 
                lon=lon_col,
                size=indicator if indicator in agg_data.columns else None,
                color=indicator if indicator in agg_data.columns else None,
                hover_name=group_col,
                hover_data={indicator: ':.2f'} if indicator in agg_data.columns else {},
                title=f'{indicator} by {geographic_level.title()}',
                color_continuous_scale='Reds',
                size_max=50,
                mapbox_style="open-street-map"
            )
        
        elif map_type == 'heatmap':
            fig = px.density_mapbox(
                agg_data,
                lat=lat_col,
                lon=lon_col,
                z=indicator if indicator in agg_data.columns else None,
                radius=20,
                title=f'{indicator} Heat Map',
                mapbox_style="open-street-map"
            )
        
        else:  # scatter
            fig = px.scatter_mapbox(
                agg_data,
                lat=lat_col,
                lon=lon_col,
                hover_name=group_col,
                title=f'{indicator} Distribution',
                mapbox_style="open-street-map"
            )
        
        # Update layout
        fig.update_layout(
            mapbox_center={"lat": -1.9441, "lon": 30.0619},
            mapbox_zoom=zoom_level,
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        return fig
    
    def create_folium_map(self, data, indicator):
        """Create a Folium map with advanced features"""
        
        # Create base map centered on Rwanda
        m = folium.Map(
            location=[-1.9441, 30.0619],
            zoom_start=8,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Prepare data for mapping
        if 'district' in data.columns and indicator in data.columns:
            district_data = data.groupby('district').agg({
                indicator: 'mean',
                'lat': 'first',
                'lon': 'first'
            }).reset_index()
            
            # Remove invalid coordinates
            district_data = district_data[(district_data['lat'] != 0) & (district_data['lon'] != 0)]
            
            # Create marker cluster
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add markers for each district
            for _, row in district_data.iterrows():
                # Color based on indicator value
                if row[indicator] > district_data[indicator].quantile(0.75):
                    color = 'red'
                    icon = 'exclamation-sign'
                elif row[indicator] > district_data[indicator].quantile(0.5):
                    color = 'orange'
                    icon = 'warning-sign'
                else:
                    color = 'green'
                    icon = 'ok-sign'
                
                folium.Marker(
                    location=[row['lat'], row['lon']],
                    popup=f"{row['district']}<br>{indicator}: {row[indicator]:.2f}",
                    tooltip=row['district'],
                    icon=folium.Icon(color=color, icon=icon)
                ).add_to(marker_cluster)
            
            # Add heatmap layer
            heat_data = [[row['lat'], row['lon'], row[indicator]] 
                        for _, row in district_data.iterrows()]
            
            HeatMap(heat_data, name='Heat Map').add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m
    
    def calculate_hotspot_severity(self, data, indicators):
        """Calculate hotspot severity using multiple indicators"""
        
        severity_data = []
        
        if 'district' in data.columns:
            for district in data['district'].unique():
                district_data = data[data['district'] == district]
                
                if len(district_data) > 0:
                    severity_score = 0
                    indicator_count = 0
                    
                    for indicator in indicators:
                        if indicator in district_data.columns:
                            # Calculate indicator severity (higher values = worse)
                            if indicator in ['stunting_rate', 'wasting_rate', 'underweight_rate', 'anemia_rate']:
                                score = district_data[indicator].mean()
                            elif indicator in ['dietary_diversity', 'vitamin_a_coverage']:
                                # For positive indicators, invert the score
                                score = 100 - district_data[indicator].mean()
                            else:
                                score = district_data[indicator].mean()
                            
                            severity_score += score
                            indicator_count += 1
                    
                    if indicator_count > 0:
                        avg_severity = severity_score / indicator_count
                        
                        # Get coordinates
                        coords = self.district_coordinates.get(district, {'lat': 0, 'lon': 0})
                        
                        severity_data.append({
                            'district': district,
                            'severity_score': avg_severity,
                            'lat': coords['lat'],
                            'lon': coords['lon'],
                            'province': coords.get('province', 'Unknown'),
                            'sample_size': len(district_data)
                        })
        
        return pd.DataFrame(severity_data)
    
    def create_comprehensive_dashboard(self, household_data, child_data, women_data):
        """Create comprehensive geographic dashboard"""
        
        st.markdown("## 🗺️ Enhanced Micronutrient Hotspots Mapping")
        
        # Load geographic data
        geo_household, geo_child, geo_women = self.load_and_process_geographic_data()
        
        # Sidebar controls
        with st.sidebar:
            st.markdown("### Map Configuration")
            
            dataset_choice = st.selectbox(
                "Dataset", 
                ["Children (6-59 months)", "Women (15-49 years)", "Households", "Combined"]
            )
            
            geographic_level = st.selectbox("Geographic Level", ["District", "Province"])
            
            indicator = st.selectbox("Micronutrient Indicator", [
                "Stunting Rate", "Wasting Rate", "Underweight Rate",
                "Anemia Rate", "Vitamin A Coverage", "Dietary Diversity Score",
                "Iron Deficiency", "Composite Risk Score"
            ])
            
            map_type = st.selectbox("Visualization Type", [
                "Interactive Bubble Map", "Heat Map", "Cluster Map", "Severity Map"
            ])
            
            show_statistics = st.checkbox("Show Statistics Panel", True)
        
        # Main content area
        col1, col2 = st.columns([3, 1] if show_statistics else [1])
        
        with col1:
            # Select appropriate dataset
            if dataset_choice == "Children (6-59 months)":
                data = self.calculate_micronutrient_indicators(child_data, geo_child)
                primary_indicators = ['stunting_rate', 'wasting_rate', 'vitamin_a_coverage']
            elif dataset_choice == "Women (15-49 years)":
                data = self.calculate_micronutrient_indicators(women_data, geo_women)
                primary_indicators = ['anemia_rate', 'dietary_diversity', 'iron_deficiency']
            elif dataset_choice == "Households":
                data = self.calculate_micronutrient_indicators(household_data, geo_household)
                primary_indicators = ['food_security', 'dietary_diversity', 'wealth_index']
            else:  # Combined
                # Merge all datasets
                combined_data = self.merge_all_datasets(household_data, child_data, women_data)
                data = self.calculate_micronutrient_indicators(combined_data, geo_household)
                primary_indicators = ['composite_risk', 'stunting_rate', 'anemia_rate']
            
            # Create and display map
            if len(data) > 0:
                if map_type == "Interactive Bubble Map":
                    fig = self.create_interactive_map(data, indicator.lower().replace(' ', '_'), 'bubble', geographic_level.lower())
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif map_type == "Heat Map":
                    fig = self.create_interactive_map(data, indicator.lower().replace(' ', '_'), 'heatmap', geographic_level.lower())
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif map_type == "Severity Map":
                    severity_data = self.calculate_hotspot_severity(data, primary_indicators)
                    if len(severity_data) > 0:
                        fig = px.scatter_mapbox(
                            severity_data,
                            lat='lat',
                            lon='lon',
                            size='severity_score',
                            color='severity_score',
                            hover_name='district',
                            hover_data={'severity_score': ':.2f', 'sample_size': True},
                            title='Micronutrient Deficiency Severity Map',
                            color_continuous_scale='Reds',
                            size_max=60,
                            mapbox_style="open-street-map"
                        )
                        fig.update_layout(
                            mapbox_center={"lat": -1.9441, "lon": 30.0619},
                            mapbox_zoom=8,
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                else:  # Cluster Map
                    # Use K-means clustering to identify hotspot clusters
                    if 'lat' in data.columns and 'lon' in data.columns:
                        coords = data[['lat', 'lon']].dropna()
                        if len(coords) > 5:
                            kmeans = KMeans(n_clusters=5, random_state=42)
                            data_with_coords = data.dropna(subset=['lat', 'lon'])
                            clusters = kmeans.fit_predict(data_with_coords[['lat', 'lon']])
                            data_with_coords = data_with_coords.copy()
                            data_with_coords['cluster'] = clusters
                            
                            fig = px.scatter_mapbox(
                                data_with_coords,
                                lat='lat',
                                lon='lon',
                                color='cluster',
                                hover_name='district' if 'district' in data_with_coords.columns else None,
                                title='Micronutrient Deficiency Clusters',
                                mapbox_style="open-street-map"
                            )
                            fig.update_layout(
                                mapbox_center={"lat": -1.9441, "lon": 30.0619},
                                mapbox_zoom=8,
                                height=600
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for mapping")
        
        # Statistics panel
        if show_statistics:
            with col2:
                st.markdown("### 📊 Hotspot Statistics")
                
                if len(data) > 0:
                    # Top 5 highest risk districts
                    if 'district' in data.columns:
                        district_risk = data.groupby('district').agg({
                            indicator.lower().replace(' ', '_'): 'mean' if indicator.lower().replace(' ', '_') in data.columns else 'count'
                        }).reset_index()
                        
                        if len(district_risk) > 0:
                            top_districts = district_risk.nlargest(5, indicator.lower().replace(' ', '_'))
                            
                            st.markdown("**🔴 Highest Risk Districts**")
                            for _, row in top_districts.iterrows():
                                value = row[indicator.lower().replace(' ', '_')] if indicator.lower().replace(' ', '_') in row else 0
                                st.write(f"• {row['district']}: {value:.1f}")
                    
                    # Summary statistics
                    st.markdown("**📈 Summary Statistics**")
                    if indicator.lower().replace(' ', '_') in data.columns:
                        col_data = data[indicator.lower().replace(' ', '_')]
                        st.write(f"• Mean: {col_data.mean():.2f}")
                        st.write(f"• Median: {col_data.median():.2f}")
                        st.write(f"• Std Dev: {col_data.std():.2f}")
                        st.write(f"• Min: {col_data.min():.2f}")
                        st.write(f"• Max: {col_data.max():.2f}")
                    
                    # Geographic coverage
                    st.markdown("**🌍 Geographic Coverage**")
                    if 'province' in data.columns:
                        provinces = data['province'].nunique()
                        st.write(f"• Provinces: {provinces}")
                    if 'district' in data.columns:
                        districts = data['district'].nunique()
                        st.write(f"• Districts: {districts}")
                    
                    st.write(f"• Total Records: {len(data):,}")
        
        # Hotspot identification results
        st.markdown("### 🎯 Identified Hotspots")
        
        if len(data) > 0:
            hotspots = self.identify_hotspots(data, primary_indicators)
            
            if 'district' in hotspots and len(hotspots['district']) > 0:
                hotspot_districts = hotspots['district'][hotspots['district']['is_hotspot']]
                
                if len(hotspot_districts) > 0:
                    st.markdown("**Priority Districts for Intervention:**")
                    
                    # Create columns for hotspot display
                    cols = st.columns(3)
                    for i, (_, district) in enumerate(hotspot_districts.iterrows()):
                        with cols[i % 3]:
                            st.error(f"🚨 {district['district']}")
                            st.write(f"Risk Score: {district['composite_risk']:.2f}")
                            st.write(f"Province: {district['province_first']}")
                else:
                    st.success("No critical hotspots identified based on current thresholds")
            
            # Recommendations based on hotspots
            st.markdown("### 💡 Targeted Recommendations")
            
            recommendations = {
                "High Priority": [
                    "Immediate micronutrient supplementation programs",
                    "Enhanced nutrition education campaigns",
                    "Improved access to diverse foods",
                    "Strengthened health service delivery"
                ],
                "Medium Priority": [
                    "Community-based nutrition programs",
                    "Agricultural diversification support",
                    "WASH infrastructure improvements",
                    "Maternal health service enhancement"
                ],
                "Long-term": [
                    "Policy framework development",
                    "Sustainable food system transformation",
                    "Economic empowerment programs",
                    "Research and monitoring systems"
                ]
            }
            
            for priority, actions in recommendations.items():
                with st.expander(f"{priority} Interventions"):
                    for action in actions:
                        st.write(f"• {action}")
    
    def merge_all_datasets(self, household_data, child_data, women_data):
        """Merge all datasets for comprehensive analysis"""
        
        # Start with household data as base
        merged = household_data.copy()
        
        # Add child nutrition indicators
        if len(child_data) > 0:
            child_summary = child_data.groupby('___index').agg({
                'stunting_zscore': 'mean',
                'wasting_zscore': 'mean',
                'underweight_zscore': 'mean',
                'received_vitamin_a': 'mean',
                'dietary_diversity': 'mean'
            }).reset_index()
            
            merged = merged.merge(child_summary, on='___index', how='left', suffixes=('', '_child'))
        
        # Add women's health indicators
        if len(women_data) > 0:
            women_summary = women_data.groupby('___index').agg({
                'anemic': 'mean',
                'bmi': 'mean',
                'dietary_diversity': 'mean',
                'received_ifa': 'mean'
            }).reset_index()
            
            merged = merged.merge(women_summary, on='___index', how='left', suffixes=('', '_women'))
        
        return merged

def create_enhanced_mapping_page():
    """Create the enhanced mapping page for the Streamlit app"""
    
    # Initialize the mapping system
    mapping_system = GeographicMappingSystem()
    
    # Load data (assuming it's already loaded in session state)
    if 'data_loaded' in st.session_state and st.session_state.data_loaded:
        household_data = st.session_state.hh_data
        child_data = st.session_state.child_data
        women_data = st.session_state.women_data
        
        # Create the comprehensive dashboard
        mapping_system.create_comprehensive_dashboard(household_data, child_data, women_data)
    
    else:
        st.warning("Please load data first from the Dashboard Overview page")
        
        # Provide sample data option
        if st.button("Load Sample Data for Demo"):
            with st.spinner("Loading sample data..."):
                # Create sample data
                np.random.seed(42)
                n_records = 1000
                
                sample_data = pd.DataFrame({
                    '___index': [f'HH_{i:04d}' for i in range(n_records)],
                    'district': np.random.choice(list(mapping_system.district_coordinates.keys()), n_records),
                    'stunting_rate': np.random.uniform(10, 60, n_records),
                    'wasting_rate': np.random.uniform(5, 25, n_records),
                    'anemia_rate': np.random.uniform(20, 70, n_records),
                    'dietary_diversity': np.random.uniform(2, 8, n_records),
                    'vitamin_a_coverage': np.random.uniform(30, 90, n_records)
                })
                
                # Add coordinates
                sample_data['lat'] = sample_data['district'].map(
                    lambda x: mapping_system.district_coordinates.get(x, {}).get('lat', 0)
                )
                sample_data['lon'] = sample_data['district'].map(
                    lambda x: mapping_system.district_coordinates.get(x, {}).get('lon', 0)
                )
                sample_data['province'] = sample_data['district'].map(
                    lambda x: mapping_system.district_coordinates.get(x, {}).get('province', 'Unknown')
                )
                
                # Store in session state
                st.session_state.sample_data = sample_data
                st.success("Sample data loaded successfully!")
                st.rerun()

# Export function for integration
def get_enhanced_mapping_function():
    """Return the enhanced mapping function for integration into main app"""
    return create_enhanced_mapping_page

if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Enhanced Geographic Mapping", layout="wide")
    create_enhanced_mapping_page()