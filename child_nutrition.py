import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def child_nutrition():
    """Child nutrition analysis"""
    st.markdown('<div class="main-header">Child Nutrition Analysis (6-59 months)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    child_data = st.session_state.child_data
    merged_data = st.session_state.merged_data
    
    # Child nutrition indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'stunting' in child_data.columns:
            stunting_rate = (child_data['stunting'] != 'Normal').mean() * 100
            st.metric("Stunting Rate", f"{stunting_rate:.1f}%")
        else:
            st.metric("Stunting Rate", "Data not available")
    
    with col2:
        if 'wasting_zscore' in child_data.columns:
            wasting_rate = (child_data['wasting_zscore'] < -2).mean() * 100
            st.metric("Wasting Rate", f"{wasting_rate:.1f}%")
        else:
            st.metric("Wasting Rate", "Data not available")
    
    with col3:
        if 'underweight_zscore' in child_data.columns:
            underweight_rate = (child_data['underweight_zscore'] < -2).mean() * 100
            st.metric("Underweight Rate", f"{underweight_rate:.1f}%")
        else:
            st.metric("Underweight Rate", "Data not available")
    
    with col4:
        if 'received_vitamin_a' in child_data.columns:
            vit_a_coverage = child_data['received_vitamin_a'].mean() * 100
            st.metric("Vitamin A Coverage", f"{vit_a_coverage:.1f}%")
        else:
            st.metric("Vitamin A Coverage", "Data not available")
    
    # Age-specific analysis
    st.markdown("### Age-Specific Nutrition Patterns")
    
    if 'age_months' in child_data.columns and 'stunting' in child_data.columns:
        child_data_copy = child_data.copy()  # Avoid modifying original data
        child_data_copy['age_group'] = pd.cut(child_data_copy['age_months'],
                                       bins=[6, 12, 24, 36, 48, 60],
                                       labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m'])

        age_stunting = child_data_copy.groupby('age_group').agg({
            'stunting': lambda x: (x != 'Normal').mean() * 100
        }).reset_index()

        fig = px.line(age_stunting, x='age_group', y='stunting',
                     title='Stunting Prevalence by Age Group',
                     markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Age or stunting data not available for age-specific analysis")
    
    # Dietary diversity analysis
    st.markdown("### Dietary Diversity and Nutrition Outcomes")
    
    if 'dietary_diversity' in child_data.columns and 'stunting' in child_data.columns:
        diversity_stunting = child_data.groupby('dietary_diversity').agg({
            'stunting': lambda x: (x != 'Normal').mean() * 100
        }).reset_index()

        fig = px.scatter(diversity_stunting, x='dietary_diversity', y='stunting',
                        title='Stunting vs Dietary Diversity',
                        trendline='lowess')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dietary diversity or stunting data not available for analysis")