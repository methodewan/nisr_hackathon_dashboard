import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
            st