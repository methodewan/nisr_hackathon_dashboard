import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def womens_health():
    """Women's health and nutrition analysis"""
    st.markdown('<div class="main-header">Women\'s Health & Nutrition (15-49 years)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    women_data = st.session_state.women_data
    
    # Women's health indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'anemic' in women_data.columns:
            anemia_rate = women_data['anemic'].mean() * 100
            st.metric("Anemia Rate", f"{anemia_rate:.1f}%")
        else:
            st.metric("Anemia Rate", "Data not available")
    
    with col2:
        if 'received_ifa' in women_data.columns:
            ifa_coverage = women_data['received_ifa'].mean() * 100
            st.metric("IFA Coverage", f"{ifa_coverage:.1f}%")
        else:
            st.metric("IFA Coverage", "Data not available")
    
    with col3:
        if 'bmi' in women_data.columns:
            low_bmi = (women_data['bmi'] < 18.5).mean() * 100
            st.metric("Low BMI (<18.5)", f"{low_bmi:.1f}%")
        else:
            st.metric("Low BMI", "Data not available")
    
    with col4:
        if 'dietary_diversity' in women_data.columns:
            avg_diversity = women_data['dietary_diversity'].mean()
            st.metric("Avg Dietary Diversity", f"{avg_diversity:.1f}")
        else:
            st.metric("Dietary Diversity", "Data not available")
    
    # Education and nutrition
    st.markdown("### Education and Nutritional Status")
    
    if 'education_level' in women_data.columns and 'bmi' in women_data.columns and 'anemic' in women_data.columns:
        education_nutrition = women_data.groupby('education_level').agg({
            'bmi': 'mean',
            'anemic': 'mean'
        }).reset_index()
        
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Average BMI by Education', 'Anemia Rate by Education'))
        
        fig.add_trace(
            go.Bar(x=education_nutrition['education_level'], y=education_nutrition['bmi'],
                   name='BMI'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=education_nutrition['education_level'], y=education_nutrition['anemic'] * 100,
                   name='Anemia Rate'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Education, BMI, or anemia data not available for analysis")
    
    # Pregnancy and nutrition
    st.markdown("### Maternal Nutrition")
    
    if 'pregnant' in women_data.columns:
        pregnant_women = women_data[women_data['pregnant'] == 1]
        
        if len(pregnant_women) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Pregnant Women", len(pregnant_women))
                if 'received_ifa' in pregnant_women.columns:
                    ifa_pregnant = pregnant_women['received_ifa'].mean() * 100
                    st.metric("IFA in Pregnancy", f"{ifa_pregnant:.1f}%")
                else:
                    st.metric("IFA in Pregnancy", "Data not available")
            
            with col2:
                if 'anemic' in pregnant_women.columns:
                    anemia_pregnant = pregnant_women['anemic'].mean() * 100
                    st.metric("Anemia in Pregnancy", f"{anemia_pregnant:.1f}%")
                else:
                    st.metric("Anemia in Pregnancy", "Data not available")
                
                if 'bmi' in pregnant_women.columns:
                    low_bmi_pregnant = (pregnant_women['bmi'] < 18.5).mean() * 100
                    st.metric("Low BMI in Pregnancy", f"{low_bmi_pregnant:.1f}%")
                else:
                    st.metric("Low BMI in Pregnancy", "Data not available")
        else:
            st.info("No pregnant women data available")