# geographic_analysis.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def geographic_analysis():
    """Geographic analysis page using alternative visualization methods"""
    
    st.markdown('<div class="main-header">🗺️ Geographic Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please wait for data to load or check if data files are available.")
        return
    
    merged_data = st.session_state.merged_data
    hh_data = st.session_state.hh_data
    
    st.markdown("""
    <div class="section-header">Regional Distribution of Nutrition Indicators</div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Option 1: Interactive choropleth using Plotly (if you have geographic coordinates)
        if 'province' in merged_data.columns:
            # Create a summary by province
            province_summary = merged_data.groupby('province').agg({
                'stunting_zscore': 'mean',
                'wasting_zscore': 'mean',
                'received_vitamin_a': 'mean'
            }).reset_index()
            
            # Display province-level data in a bar chart
            indicator = st.selectbox(
                "Select Nutrition Indicator",
                ["Stunting Z-score", "Wasting Z-score", "Vitamin A Coverage"],
                key="geo_indicator"
            )
            
            if indicator == "Stunting Z-score":
                fig = px.bar(province_summary, 
                            x='province', y='stunting_zscore',
                            title=f"Average Stunting Z-score by Province",
                            color='stunting_zscore',
                            color_continuous_scale='RdYlBu_r')
            elif indicator == "Wasting Z-score":
                fig = px.bar(province_summary, 
                            x='province', y='wasting_zscore',
                            title=f"Average Wasting Z-score by Province",
                            color='wasting_zscore',
                            color_continuous_scale='RdYlBu_r')
            else:
                fig = px.bar(province_summary, 
                            x='province', y='received_vitamin_a',
                            title=f"Vitamin A Coverage by Province",
                            color='received_vitamin_a',
                            color_continuous_scale='Blues')
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Regional Summary")
        
        if 'province' in merged_data.columns:
            # Display key metrics
            num_provinces = merged_data['province'].nunique()
            avg_stunting = merged_data['stunting_zscore'].mean()
            avg_wasting = merged_data['wasting_zscore'].mean()
            vit_a_coverage = merged_data['received_vitamin_a'].mean()
            
            st.metric("Number of Provinces", num_provinces)
            st.metric("Average Stunting Z-score", f"{avg_stunting:.2f}")
            st.metric("Average Wasting Z-score", f"{avg_wasting:.2f}")
            st.metric("Vitamin A Coverage", f"{vit_a_coverage*100:.1f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional geographic visualizations
    st.markdown("""
    <div class="section-header">Urban-Rural Distribution</div>
    """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if 'urban_rural' in merged_data.columns:
            urban_rural_summary = merged_data.groupby('urban_rural').agg({
                'stunting_zscore': 'mean',
                'wasting_zscore': 'mean',
                'received_vitamin_a': 'mean'
            }).reset_index()
            
            fig = px.pie(urban_rural_summary, 
                        values='stunting_zscore', 
                        names='urban_rural',
                        title='Distribution of Stunting by Urban-Rural Classification')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        if 'urban_rural' in merged_data.columns and 'wealth_index' in merged_data.columns:
            # Cross-tabulation of urban-rural vs wealth
            cross_tab = pd.crosstab(merged_data['urban_rural'], 
                                  merged_data['wealth_index'], 
                                  normalize='index') * 100
            
            fig = px.imshow(cross_tab,
                           title='Wealth Distribution by Urban-Rural Classification (%)',
                           color_continuous_scale='Blues',
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data table for detailed view
    st.markdown("""
    <div class="section-header">Detailed Regional Data</div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    if 'province' in merged_data.columns:
        # Create a detailed summary table
        detailed_summary = merged_data.groupby('province').agg({
            'stunting_zscore': ['mean', 'std', 'count'],
            'wasting_zscore': ['mean', 'std'],
            'received_vitamin_a': 'mean',
            'urban_rural': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).round(3)
        
        # Flatten column names
        detailed_summary.columns = ['_'.join(col).strip() for col in detailed_summary.columns.values]
        detailed_summary = detailed_summary.reset_index()
        
        st.dataframe(detailed_summary, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    geographic_analysis()