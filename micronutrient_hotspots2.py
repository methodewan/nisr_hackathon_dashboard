import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')


def micronutrient_hotspots():
    """Map malnutrition hotspots using geospatial data"""
    st.markdown('<div class="main-header">🗺️ Micronutrient Hotspots Mapping</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    hh_data = st.session_state.hh_data
    child_data = st.session_state.child_data
    women_data = st.session_state.women_data
    
    # Show available columns
    with st.expander("📋 Available Columns", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Household data:**")
            st.write(hh_data.columns.tolist())
        with col2:
            st.write("**Child data:**")
            st.write(child_data.columns.tolist())
        with col3:
            st.write("**Women data:**")
            st.write(women_data.columns.tolist())
    
    # Create geographic analysis
    st.markdown("### 📊 Geographic Distribution of Malnutrition Indicators")
    
    # Province-level aggregation
    if 'province' in hh_data.columns:
        st.markdown("#### Province-Level Analysis")
        
        # Create province-level statistics dynamically
        province_stats_list = []
        
        # Household level indicators - only aggregate available columns
        available_hh_cols = [col for col in ['dietary_diversity_score', 'food_consumption_score', 'coping_strategy_index', 'hh_size'] 
                            if col in hh_data.columns]
        
        if available_hh_cols and 'province' in hh_data.columns:
            try:
                agg_dict = {col: 'mean' for col in available_hh_cols}
                province_hh = hh_data.groupby('province').agg(agg_dict).reset_index()
                province_stats_list.append(province_hh)
                st.success(f"✅ Household indicators: {', '.join(available_hh_cols)}")
            except Exception as e:
                st.warning(f"Could not aggregate household data: {e}")
        
        # Child level indicators
        available_child_cols = [col for col in ['stunting_zscore', 'wasting_zscore', 'underweight_zscore', 'age_months'] 
                               if col in child_data.columns]
        
        if available_child_cols and 'province' in child_data.columns:
            try:
                agg_dict = {col: 'mean' for col in available_child_cols}
                province_child = child_data.groupby('province').agg(agg_dict).reset_index()
                province_stats_list.append(province_child)
                st.success(f"✅ Child indicators: {', '.join(available_child_cols)}")
            except Exception as e:
                st.warning(f"Could not aggregate child data: {e}")
        
        # Women level indicators
        available_women_cols = [col for col in ['anemic', 'age_years', 'dietary_diversity', 'bmi'] 
                               if col in women_data.columns]
        
        if available_women_cols and 'province' in women_data.columns:
            try:
                agg_dict = {col: 'mean' for col in available_women_cols}
                province_women = women_data.groupby('province').agg(agg_dict).reset_index()
                province_stats_list.append(province_women)
                st.success(f"✅ Women indicators: {', '.join(available_women_cols)}")
            except Exception as e:
                st.warning(f"Could not aggregate women data: {e}")
        
        # Merge all province statistics
        if province_stats_list:
            province_analysis = province_stats_list[0]
            for stats in province_stats_list[1:]:
                province_analysis = province_analysis.merge(stats, on='province', how='outer')
            
            # Create hotspot score from available columns
            score_columns = [col for col in ['stunting_zscore', 'wasting_zscore', 'anemic'] 
                           if col in province_analysis.columns]
            
            if score_columns:
                # Normalize columns to 0-100 scale (higher = worse)
                for col in score_columns:
                    if col in province_analysis.columns:
                        min_val = province_analysis[col].min()
                        max_val = province_analysis[col].max()
                        if max_val > min_val:
                            # For negative z-scores, we want higher values to indicate worse conditions
                            if 'zscore' in col:
                                province_analysis[f'{col}_normalized'] = ((province_analysis[col] - min_val) / (max_val - min_val)) * 100
                            else:
                                province_analysis[f'{col}_normalized'] = province_analysis[col] * 100
                        else:
                            province_analysis[f'{col}_normalized'] = 50
                
                # Calculate overall hotspot score (average of normalized scores)
                normalized_cols = [f'{col}_normalized' for col in score_columns if f'{col}_normalized' in province_analysis.columns]
                if normalized_cols:
                    province_analysis['hotspot_score'] = province_analysis[normalized_cols].mean(axis=1)
                else:
                    province_analysis['hotspot_score'] = 50
            else:
                province_analysis['hotspot_score'] = 50
            
            # Display province-level table
            st.markdown("#### Province Statistics Table")
            display_cols = ['province', 'hotspot_score'] + [col for col in province_analysis.columns 
                                                          if col not in ['province', 'hotspot_score'] 
                                                          and not col.endswith('_normalized')]
            st.dataframe(province_analysis[display_cols].round(2), use_container_width=True)
            
            # Create hotspot map
            st.markdown("#### 🗺️ Malnutrition Hotspots Map")
            
            # Get province coordinates
            province_coords = get_province_coordinates()
            
            # Create map data
            map_data = []
            for province in province_analysis['province'].unique():
                province_data = province_analysis[province_analysis['province'] == province]
                if not province_data.empty:
                    hotspot_score = province_data['hotspot_score'].iloc[0]
                    # Clean province name for matching
                    clean_province = str(province).strip().title()
                    coords = province_coords.get(clean_province, {'lat': -1.95, 'lon': 30.0})
                    
                    # Get additional metrics for tooltip
                    tooltip_data = {}
                    for col in ['stunting_zscore', 'wasting_zscore', 'anemic', 'dietary_diversity_score']:
                        if col in province_data.columns:
                            tooltip_data[col] = round(province_data[col].iloc[0], 2)
                    
                    map_data.append({
                        'province': clean_province,
                        'lat': coords['lat'],
                        'lon': coords['lon'],
                        'hotspot_score': hotspot_score,
                        **tooltip_data
                    })
            
            if map_data:
                map_df = pd.DataFrame(map_data)
                
                # Create custom hover text
                map_df['hover_text'] = map_df.apply(
                    lambda x: f"<b>{x['province']}</b><br>"
                             f"Hotspot Score: {x['hotspot_score']:.1f}<br>"
                             f"{'Stunting: ' + str(x.get('stunting_zscore', 'N/A')) + '<br>' if 'stunting_zscore' in x else ''}"
                             f"{'Wasting: ' + str(x.get('wasting_zscore', 'N/A')) + '<br>' if 'wasting_zscore' in x else ''}"
                             f"{'Anemia: ' + str(x.get('anemic', 'N/A')) + '<br>' if 'anemic' in x else ''}"
                             f"{'Diet Diversity: ' + str(x.get('dietary_diversity_score', 'N/A')) if 'dietary_diversity_score' in x else ''}",
                    axis=1
                )
                
                # Create the map
                fig = px.scatter_mapbox(
                    map_df,
                    lat="lat",
                    lon="lon",
                    size="hotspot_score",
                    color="hotspot_score",
                    hover_name="province",
                    hover_data={
                        "hotspot_score": ":.1f",
                        "lat": False,
                        "lon": False
                    },
                    custom_data=["hover_text"],
                    color_continuous_scale="RdYlGn_r",  # Red for high risk, green for low risk
                    size_max=40,
                    zoom=7,
                    title="Malnutrition Hotspots by Province (Higher Score = Higher Risk)",
                    labels={"hotspot_score": "Risk Score"}
                )
                
                # Update hover template
                fig.update_traces(
                    hovertemplate="%{customdata[0]}<extra></extra>"
                )
                
                fig.update_layout(
                    mapbox_style="open-street-map",
                    height=600,
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    coloraxis_colorbar=dict(
                        title="Risk Score",
                        thickness=20,
                        len=0.75
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Hotspot analysis
                st.markdown("### 🎯 Hotspot Priority Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### 🔴 Critical Hotspots")
                    st.markdown("**(Score > 70)**")
                    high_risk = map_df[map_df['hotspot_score'] > 70].sort_values('hotspot_score', ascending=False)
                    if not high_risk.empty:
                        for _, region in high_risk.iterrows():
                            with st.container():
                                st.error(f"**{region['province']}**  \nScore: {region['hotspot_score']:.1f}")
                    else:
                        st.info("✅ No critical hotspots")
                
                with col2:
                    st.markdown("#### 🟡 Medium Priority")
                    st.markdown("**(Score 40-70)**")
                    medium_risk = map_df[(map_df['hotspot_score'] >= 40) & (map_df['hotspot_score'] <= 70)].sort_values('hotspot_score', ascending=False)
                    if not medium_risk.empty:
                        for _, region in medium_risk.iterrows():
                            with st.container():
                                st.warning(f"**{region['province']}**  \nScore: {region['hotspot_score']:.1f}")
                    else:
                        st.info("✅ No medium-risk areas")
                
                with col3:
                    st.markdown("#### 🟢 Low Priority")
                    st.markdown("**(Score < 40)**")
                    low_risk = map_df[map_df['hotspot_score'] < 40].sort_values('hotspot_score', ascending=False)
                    if not low_risk.empty:
                        for _, region in low_risk.iterrows():
                            with st.container():
                                st.success(f"**{region['province']}**  \nScore: {region['hotspot_score']:.1f}")
                    else:
                        st.info("✅ No low-risk areas")
                
                # Intervention recommendations
                st.markdown("---")
                st.markdown("### 💡 Intervention Strategy by Risk Level")
                
                tab1, tab2, tab3 = st.tabs(["🔴 Critical Hotspots", "🟡 Medium Priority", "🟢 Low Priority"])
                
                with tab1:
                    st.error("""
                    **Critical Hotspots (Score > 70) - Immediate Response Required**
                    
                    🚨 **Emergency Actions (0-3 months):**
                    - Deploy mobile nutrition clinics
                    - Emergency food and micronutrient distribution
                    - Intensive screening for acute malnutrition
                    - WASH emergency interventions
                    
                    📋 **Stabilization Phase (3-6 months):**
                    - Establish therapeutic feeding centers
                    - Train community health workers
                    - Implement cash transfer programs
                    - Strengthen health facility capacity
                    """)
                
                with tab2:
                    st.warning("""
                    **Medium-Risk Areas (Score 40-70) - Targeted Interventions**
                    
                    🎯 **Priority Actions (0-6 months):**
                    - Community nutrition education programs
                    - Dietary diversification initiatives
                    - Quarterly growth monitoring
                    - Maternal and child health services
                    
                    📈 **Capacity Building (6-12 months):**
                    - Agricultural extension for nutrient-dense crops
                    - School feeding programs
                    - Women's empowerment programs
                    - Community-based management
                    """)
                
                with tab3:
                    st.success("""
                    **Low-Risk Areas (Score < 40) - Preventive Maintenance**
                    
                    🛡️ **Sustaining Actions:**
                    - Continue routine health services
                    - Nutrition education reinforcement
                    - Regular monitoring and surveillance
                    - Community capacity building
                    
                    🌱 **Long-term Strategies:**
                    - Sustainable agriculture practices
                    - Economic development programs
                    - Education and awareness campaigns
                    - Infrastructure development
                    """)
        
        else:
            st.warning("No province-level data available for analysis")
    else:
        st.error("❌ 'province' column not found in household data. Cannot create geographic analysis.")
        st.info("Please ensure your data contains geographic information (province/district columns)")
    
    # District-level analysis (if available)
    if 'district' in hh_data.columns:
        st.markdown("---")
        st.markdown("### 📍 District-Level Analysis")
        
        # Get top districts by household count
        district_counts = hh_data['district'].dropna().value_counts()
        districts = district_counts.head(10).index.tolist()
        
        if districts:
            district_data = []
            
            for district in districts:
                # Count records per district
                district_count = len(hh_data[hh_data['district'] == district])
                
                # Calculate available metrics
                metrics = {'district': district, 'households': district_count}
                
                # Add any available numeric columns with error handling
                numeric_cols = ['dietary_diversity_score', 'hh_size', 'food_consumption_score']
                for col in numeric_cols:
                    if col in hh_data.columns:
                        try:
                            district_subset = hh_data[hh_data['district'] == district]
                            if not district_subset.empty:
                                avg_val = district_subset[col].mean()
                                metrics[col] = round(avg_val, 2)
                        except Exception:
                            metrics[col] = None
                
                district_data.append(metrics)
            
            if district_data:
                district_df = pd.DataFrame(district_data)
                
                # Sort by household count
                district_df = district_df.sort_values('households', ascending=False)
                
                st.markdown("#### Top Districts by Household Count")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Districts", len(district_counts))
                with col2:
                    st.metric("Largest District", district_df.iloc[0]['district'])
                with col3:
                    st.metric("Max Households", int(district_df['households'].max()))
                
                st.dataframe(district_df, use_container_width=True)
                
                # Create district chart
                fig = px.bar(
                    district_df.head(8),  # Show top 8 districts
                    x='households',
                    y='district',
                    orientation='h',
                    title='Top Districts by Number of Households',
                    color='households',
                    color_continuous_scale='Blues',
                    labels={'households': 'Number of Households', 'district': 'District'}
                )
                fig.update_layout(
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No district data available for analysis")
    else:
        st.info("ℹ️ District-level data not available. Province-level analysis shown above.")


def get_province_coordinates():
    """Get approximate coordinates for Rwandan provinces"""
    return {
        # 'Kigali': {'lat': -1.9441, 'lon': 30.0619},
        # 'North': {'lat': -1.2864, 'lon': 29.5646},
        # 'South': {'lat': -2.5636, 'lon': 29.7471},
        # 'East': {'lat': -1.9480, 'lon': 30.9093},
        # 'West': {'lat': -2.0392, 'lon': 29.3589},
        # Add common variations
        'Kigali City': {'lat': -1.9441, 'lon': 30.0619},
        'Northern': {'lat': -1.2864, 'lon': 29.5646},
        'Southern': {'lat': -2.5636, 'lon': 29.7471},
        'Eastern': {'lat': -1.9480, 'lon': 30.9093},
        'Western': {'lat': -2.0392, 'lon': 29.3589}
    }