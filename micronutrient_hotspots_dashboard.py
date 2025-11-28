import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import json

def load_enhanced_geographic_data():
    """Load enhanced geographic data with coordinates and micronutrient indicators"""
    try:
        # Load enhanced datasets
        household_geo = pd.read_csv("enhanced_geographic_household.csv")
        child_geo = pd.read_csv("enhanced_geographic_child.csv")
        
        # Load hotspot analysis
        child_hotspots = pd.read_csv("hotspot_analysis_child.csv")
        
        # Load district coordinates master
        district_coords = pd.read_csv("district_coordinates_master.csv")
        
        return household_geo, child_geo, child_hotspots, district_coords
        
    except Exception as e:
        st.error(f"Error loading enhanced geographic data: {e}")
        return None, None, None, None

def create_micronutrient_hotspots_page():
    """Create comprehensive micronutrient hotspots mapping page"""
    
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0;">🗺️ Micronutrient Deficiency Hotspots</h1>
        <p style="color: white; opacity: 0.9; margin: 0.5rem 0 0 0;">Interactive Geographic Analysis for Targeted Interventions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    household_geo, child_geo, child_hotspots, district_coords = load_enhanced_geographic_data()
    
    if child_geo is None:
        st.error("Could not load geographic data. Please run the enhanced extraction first.")
        
        if st.button("🔄 Run Enhanced Geographic Extraction"):
            with st.spinner("Extracting enhanced geographic data..."):
                import subprocess
                result = subprocess.run(["python", "extract_geographic_enhanced.py"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Enhanced geographic data extracted successfully!")
                    st.rerun()
                else:
                    st.error(f"Error running extraction: {result.stderr}")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Map Configuration")
        
        # Dataset selection
        dataset_choice = st.selectbox(
            "📊 Dataset", 
            ["Child Nutrition (6-59 months)", "Household Data", "Combined Analysis"],
            help="Select the dataset for hotspot analysis"
        )
        
        # Geographic level
        geographic_level = st.selectbox(
            "🌍 Geographic Level", 
            ["District", "Province"],
            help="Choose the administrative level for analysis"
        )
        
        # Indicator selection
        if dataset_choice == "Child Nutrition (6-59 months)":
            available_indicators = [
                "Stunting Rate", "Wasting Rate", "Underweight Rate", 
                "Vitamin A Coverage", "Composite Risk Score"
            ]
        elif dataset_choice == "Household Data":
            available_indicators = [
                "Food Consumption Score", "Dietary Diversity", "Wealth Index"
            ]
        else:
            available_indicators = [
                "Composite Risk Score", "Stunting Rate", "Food Security"
            ]
        
        indicator = st.selectbox(
            "📈 Micronutrient Indicator", 
            available_indicators,
            help="Select the indicator to visualize on the map"
        )
        
        # Visualization type
        viz_type = st.selectbox(
            "🎨 Visualization Type",
            ["Interactive Bubble Map", "Heat Map", "Severity Classification", "Cluster Analysis"],
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
        show_district_labels = st.checkbox("🏷️ Show District Labels", False)

    # Main content area
    if show_statistics:
        col1, col2 = st.columns([3, 1])
    else:
        col1 = st.container()
        col2 = None
    
    with col1:
        # Create the main map visualization
        if dataset_choice == "Child Nutrition (6-59 months)" and child_geo is not None:
            data_for_mapping = child_geo.copy()
            
            # Map indicator to column name
            indicator_mapping = {
                "Stunting Rate": "stunting_rate",
                "Wasting Rate": "wasting_rate", 
                "Underweight Rate": "underweight_rate",
                "Vitamin A Coverage": "vitamin_a_received",
                "Composite Risk Score": "composite_risk"
            }
            
            indicator_col = indicator_mapping.get(indicator, "stunting_rate")
            
            # Create visualization based on type
            if viz_type == "Interactive Bubble Map":
                fig = create_bubble_map(data_for_mapping, indicator_col, geographic_level)
                
            elif viz_type == "Heat Map":
                fig = create_heat_map(data_for_mapping, indicator_col)
                
            elif viz_type == "Severity Classification":
                fig = create_severity_map(child_hotspots)
                
            else:  # Cluster Analysis
                fig = create_cluster_map(data_for_mapping)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Hotspot identification section
        st.markdown("### 🎯 Identified Micronutrient Hotspots")
        
        if child_hotspots is not None:
            # Filter hotspots based on threshold
            threshold_value = np.percentile(child_hotspots['composite_risk_score'], hotspot_threshold)
            high_risk_districts = child_hotspots[child_hotspots['composite_risk_score'] >= threshold_value]
            
            if len(high_risk_districts) > 0:
                # Display hotspots in columns
                cols = st.columns(3)
                for i, (_, district) in enumerate(high_risk_districts.iterrows()):
                    with cols[i % 3]:
                        severity_color = "🔴" if district['hotspot_severity'] == 'High' else "🟡"
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid #ef4444; margin-bottom: 1rem;">
                            <h4 style="margin: 0; color: #1f2937;">{severity_color} {district['district']}</h4>
                            <p style="margin: 0.5rem 0; color: #6b7280;">Risk Score: {district['composite_risk_score']:.3f}</p>
                            <p style="margin: 0; color: #6b7280;">Severity: {district['hotspot_severity']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical hotspots identified at current threshold level")
        
        # Detailed analysis section
        st.markdown("### 📊 Detailed Geographic Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📍 District Rankings", "🏛️ Provincial Summary", "📈 Trends", "🎯 Interventions"])
        
        with tab1:
            if child_hotspots is not None:
                # District rankings table
                rankings_data = child_hotspots[['district', 'composite_risk_score', 'hotspot_severity', 'priority_rank']].copy()
                rankings_data = rankings_data.sort_values('composite_risk_score', ascending=False)
                
                st.markdown("**Top 15 Priority Districts for Micronutrient Interventions**")
                
                # Style the dataframe
                styled_df = rankings_data.head(15).style.format({
                    'composite_risk_score': '{:.3f}',
                    'priority_rank': '{:.0f}'
                }).background_gradient(subset=['composite_risk_score'], cmap='Reds')
                
                st.dataframe(styled_df, use_container_width=True)
        
        with tab2:
            if child_geo is not None:
                # Provincial summary
                provincial_summary = child_geo.groupby('province').agg({
                    'stunting_rate': 'mean',
                    'wasting_rate': 'mean', 
                    'underweight_rate': 'mean',
                    'vitamin_a_received': 'mean',
                    'district': 'count'
                }).round(3)
                provincial_summary.columns = ['Stunting Rate', 'Wasting Rate', 'Underweight Rate', 'Vitamin A Coverage', 'Sample Size']
                
                st.markdown("**Provincial Micronutrient Indicators Summary**")
                st.dataframe(provincial_summary, use_container_width=True)
                
                # Provincial comparison chart
                fig = px.bar(
                    provincial_summary.reset_index(),
                    x='province',
                    y='Stunting Rate',
                    title='Stunting Rates by Province',
                    color='Stunting Rate',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Trends and correlations
            if child_geo is not None:
                st.markdown("**Correlation Analysis: Micronutrient Indicators**")
                
                # Calculate correlations
                correlation_cols = ['stunting_rate', 'wasting_rate', 'underweight_rate', 'vitamin_a_received']
                available_corr_cols = [col for col in correlation_cols if col in child_geo.columns]
                
                if len(available_corr_cols) > 1:
                    corr_matrix = child_geo[available_corr_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Matrix: Micronutrient Indicators",
                        color_continuous_scale='RdBu_r',
                        aspect="auto"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Urban vs Rural comparison
                if 'urban_rural' in child_geo.columns:
                    urban_rural_comparison = child_geo.groupby('urban_rural')[available_corr_cols].mean()
                    
                    fig = go.Figure()
                    for indicator in available_corr_cols:
                        fig.add_trace(go.Bar(
                            name=indicator.replace('_', ' ').title(),
                            x=urban_rural_comparison.index,
                            y=urban_rural_comparison[indicator]
                        ))
                    
                    fig.update_layout(
                        title="Urban vs Rural Micronutrient Indicators",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Intervention recommendations
            st.markdown("**Targeted Intervention Strategies by Hotspot Severity**")
            
            if child_hotspots is not None:
                # High severity interventions
                high_severity = child_hotspots[child_hotspots['hotspot_severity'] == 'High']
                if len(high_severity) > 0:
                    st.markdown("#### 🔴 High Severity Areas (Immediate Action Required)")
                    
                    for _, district in high_severity.iterrows():
                        with st.expander(f"📍 {district['district']} District"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Immediate Interventions (0-3 months):**")
                                st.markdown("""
                                - Emergency micronutrient supplementation
                                - Severe acute malnutrition treatment
                                - Vitamin A mass distribution campaign
                                - Iron-folic acid supplementation for women
                                - Community nutrition screening
                                """)
                            
                            with col2:
                                st.markdown("**Medium-term Actions (3-12 months):**")
                                st.markdown("""
                                - Establish community nutrition centers
                                - Train community health workers
                                - Implement growth monitoring programs
                                - Develop local food production initiatives
                                - Strengthen health service delivery
                                """)
                
                # Medium severity interventions
                medium_severity = child_hotspots[child_hotspots['hotspot_severity'] == 'Medium']
                if len(medium_severity) > 0:
                    st.markdown("#### 🟡 Medium Severity Areas (Preventive Focus)")
                    
                    st.markdown("""
                    **Recommended Interventions:**
                    - Nutrition education programs
                    - Dietary diversification support
                    - Homestead food production
                    - Micronutrient powder distribution
                    - Improved infant and young child feeding practices
                    """)
                    
                    # Show medium severity districts
                    medium_districts = medium_severity['district'].tolist()
                    st.info(f"**Districts:** {', '.join(medium_districts[:10])}" + 
                           (f" and {len(medium_districts)-10} more" if len(medium_districts) > 10 else ""))

    # Statistics panel
    if show_statistics and col2:
        with col2:
            st.markdown("### 📊 Hotspot Statistics")
            
            if child_hotspots is not None:
                # Summary metrics
                total_districts = len(child_hotspots)
                high_risk = len(child_hotspots[child_hotspots['hotspot_severity'] == 'High'])
                medium_risk = len(child_hotspots[child_hotspots['hotspot_severity'] == 'Medium'])
                low_risk = len(child_hotspots[child_hotspots['hotspot_severity'] == 'Low'])
                
                st.metric("Total Districts", total_districts)
                st.metric("High Risk", high_risk, delta=f"{high_risk/total_districts*100:.1f}%")
                st.metric("Medium Risk", medium_risk, delta=f"{medium_risk/total_districts*100:.1f}%")
                st.metric("Low Risk", low_risk, delta=f"{low_risk/total_districts*100:.1f}%")
                
                st.markdown("---")
                
                # Top 5 highest risk districts
                st.markdown("**🔴 Highest Risk Districts**")
                top_5 = child_hotspots.nlargest(5, 'composite_risk_score')
                for _, row in top_5.iterrows():
                    st.write(f"• **{row['district']}**: {row['composite_risk_score']:.3f}")
                
                st.markdown("---")
                
                # Key statistics
                st.markdown("**📈 Key Statistics**")
                avg_stunting = child_hotspots['stunting_rate_mean'].mean() * 100
                avg_wasting = child_hotspots['wasting_rate_mean'].mean() * 100
                avg_vitamin_a = child_hotspots['vitamin_a_received_mean'].mean() * 100
                
                st.write(f"• Avg Stunting: {avg_stunting:.1f}%")
                st.write(f"• Avg Wasting: {avg_wasting:.1f}%")
                st.write(f"• Avg Vitamin A: {avg_vitamin_a:.1f}%")
                
                st.markdown("---")
                
                # Geographic distribution
                st.markdown("**🌍 Geographic Distribution**")
                if 'province_first' in child_hotspots.columns:
                    province_counts = child_hotspots['province_first'].value_counts()
                    for province, count in province_counts.items():
                        st.write(f"• {province}: {count} districts")

def create_bubble_map(data, indicator_col, geographic_level):
    """Create interactive bubble map"""
    
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
            
            fig.update_layout(
                mapbox_center={"lat": -1.9441, "lon": 30.0619},
                mapbox_zoom=zoom,
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            
            return fig
    
    return None

def create_heat_map(data, indicator_col):
    """Create heat map visualization"""
    
    if 'lat' in data.columns and 'lon' in data.columns and indicator_col in data.columns:
        # Sample data for heat map (to avoid overcrowding)
        sample_data = data.sample(min(1000, len(data)))
        
        fig = px.density_mapbox(
            sample_data,
            lat='lat',
            lon='lon',
            z=indicator_col,
            radius=15,
            title=f'{indicator_col.replace("_", " ").title()} Heat Map',
            mapbox_style="open-street-map"
        )
        
        fig.update_layout(
            mapbox_center={"lat": -1.9441, "lon": 30.0619},
            mapbox_zoom=8,
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        return fig
    
    return None

def create_severity_map(hotspot_data):
    """Create severity classification map"""
    
    if hotspot_data is not None and len(hotspot_data) > 0:
        # Color mapping for severity
        color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
        hotspot_data['color'] = hotspot_data['hotspot_severity'].map(color_map)
        
        fig = px.scatter_mapbox(
            hotspot_data,
            lat='lat_first',
            lon='lon_first',
            color='hotspot_severity',
            size='composite_risk_score',
            hover_name='district',
            hover_data={
                'composite_risk_score': ':.3f',
                'stunting_rate_mean': ':.3f',
                'wasting_rate_mean': ':.3f'
            },
            title='Micronutrient Deficiency Severity Classification',
            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
            size_max=50,
            mapbox_style="open-street-map"
        )
        
        fig.update_layout(
            mapbox_center={"lat": -1.9441, "lon": 30.0619},
            mapbox_zoom=8,
            height=600,
            margin={"r":0,"t":50,"l":0,"b":0}
        )
        
        return fig
    
    return None

def create_cluster_map(data):
    """Create cluster analysis map"""
    
    if 'lat' in data.columns and 'lon' in data.columns:
        from sklearn.cluster import KMeans
        
        # Prepare data for clustering
        coords = data[['lat', 'lon']].dropna()
        
        if len(coords) > 10:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            clusters = kmeans.fit_predict(coords)
            
            coords_with_clusters = coords.copy()
            coords_with_clusters['cluster'] = clusters
            coords_with_clusters['cluster'] = coords_with_clusters['cluster'].astype(str)
            
            fig = px.scatter_mapbox(
                coords_with_clusters,
                lat='lat',
                lon='lon',
                color='cluster',
                title='Micronutrient Deficiency Geographic Clusters',
                mapbox_style="open-street-map"
            )
            
            fig.update_layout(
                mapbox_center={"lat": -1.9441, "lon": 30.0619},
                mapbox_zoom=8,
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            
            return fig
    
    return None

def create_folium_hotspots_map(hotspot_data):
    """Create advanced Folium map for hotspots"""
    
    # Create base map
    m = folium.Map(
        location=[-1.9441, 30.0619],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers
    folium.TileLayer('Stamen Terrain').add_to(m)
    folium.TileLayer('CartoDB positron').add_to(m)
    
    if hotspot_data is not None and len(hotspot_data) > 0:
        # Add markers for each district
        for _, row in hotspot_data.iterrows():
            # Determine marker properties based on severity
            if row['hotspot_severity'] == 'High':
                color = 'red'
                icon = 'exclamation-sign'
                popup_color = '#ef4444'
            elif row['hotspot_severity'] == 'Medium':
                color = 'orange'
                icon = 'warning-sign'
                popup_color = '#f59e0b'
            else:
                color = 'green'
                icon = 'ok-sign'
                popup_color = '#10b981'
            
            # Create detailed popup
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 200px;">
                <h4 style="color: {popup_color}; margin: 0;">{row['district']}</h4>
                <hr style="margin: 5px 0;">
                <p><strong>Risk Score:</strong> {row['composite_risk_score']:.3f}</p>
                <p><strong>Severity:</strong> {row['hotspot_severity']}</p>
                <p><strong>Stunting Rate:</strong> {row['stunting_rate_mean']*100:.1f}%</p>
                <p><strong>Wasting Rate:</strong> {row['wasting_rate_mean']*100:.1f}%</p>
                <p><strong>Vitamin A Coverage:</strong> {row['vitamin_a_received_mean']*100:.1f}%</p>
            </div>
            """
            
            folium.Marker(
                location=[row['lat_first'], row['lon_first']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{row['district']} - {row['hotspot_severity']} Risk",
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
        
        # Add heat map layer
        heat_data = [[row['lat_first'], row['lon_first'], row['composite_risk_score']] 
                    for _, row in hotspot_data.iterrows()]
        
        HeatMap(heat_data, name='Risk Heat Map', radius=20).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def show_intervention_recommendations():
    """Show detailed intervention recommendations"""
    
    st.markdown("### 💡 Evidence-Based Intervention Recommendations")
    
    # Load hotspot data for recommendations
    try:
        child_hotspots = pd.read_csv("hotspot_analysis_child.csv")
        
        # Priority interventions by severity
        interventions = {
            "High": {
                "title": "🚨 Emergency Interventions",
                "color": "#ef4444",
                "actions": [
                    "Immediate micronutrient supplementation (Vitamin A, Iron, Zinc)",
                    "Treatment of severe acute malnutrition cases",
                    "Emergency food assistance with nutrient-dense foods",
                    "Mobile health clinics for remote areas",
                    "Community-based management of acute malnutrition"
                ]
            },
            "Medium": {
                "title": "⚠️ Preventive Interventions", 
                "color": "#f59e0b",
                "actions": [
                    "Nutrition education and behavior change programs",
                    "Homestead food production and kitchen gardens",
                    "Micronutrient powder (MNP) distribution",
                    "Improved infant and young child feeding practices",
                    "Water, sanitation, and hygiene improvements"
                ]
            },
            "Low": {
                "title": "✅ Maintenance Interventions",
                "color": "#10b981", 
                "actions": [
                    "Regular nutrition monitoring and surveillance",
                    "Continued micronutrient supplementation",
                    "Sustainable food system strengthening",
                    "Community nutrition volunteer programs",
                    "Integration with existing health services"
                ]
            }
        }
        
        for severity, info in interventions.items():
            districts_count = len(child_hotspots[child_hotspots['hotspot_severity'] == severity])
            
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {info['color']}; margin-bottom: 1rem;">
                <h4 style="color: {info['color']}; margin: 0 0 1rem 0;">{info['title']} ({districts_count} districts)</h4>
            """, unsafe_allow_html=True)
            
            for action in info['actions']:
                st.markdown(f"• {action}")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.warning("Could not load hotspot data for recommendations")

# Main function to integrate with the existing app
def micronutrient_hotspots_enhanced():
    """Enhanced micronutrient hotspots page for integration"""
    
    # Check if enhanced data exists
    try:
        pd.read_csv("enhanced_geographic_child.csv")
        create_micronutrient_hotspots_page()
    except FileNotFoundError:
        st.warning("Enhanced geographic data not found. Please run the extraction first.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Enhanced Geographic Extraction", type="primary"):
                with st.spinner("Extracting enhanced geographic data..."):
                    import subprocess
                    result = subprocess.run(["python", "extract_geographic_enhanced.py"], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ Enhanced geographic data extracted successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Error running extraction: {result.stderr}")
        
        with col2:
            st.info("""
            **What the enhanced extraction provides:**
            - Precise district coordinates for all 30 districts
            - Micronutrient deficiency indicators
            - Hotspot severity classification
            - Composite risk scoring
            - Priority ranking for interventions
            """)

if __name__ == "__main__":
    st.set_page_config(page_title="Micronutrient Hotspots", layout="wide")
    micronutrient_hotspots_enhanced()