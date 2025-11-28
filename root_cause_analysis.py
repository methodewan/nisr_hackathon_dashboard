import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def root_cause_analysis():
    """Root cause analysis of stunting and deficiencies"""
    st.markdown('<div class="main-header">Root Cause Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    women_data = st.session_state.women_data
    
    # Multivariate analysis
    st.markdown("### Key Determinants of Malnutrition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wealth and stunting
        if 'wealth_index' in merged_data.columns and 'stunting' in merged_data.columns:
            wealth_stunting = merged_data.groupby('wealth_index').agg({
                'stunting': lambda x: (x != 'Normal').mean() * 100
            }).reset_index()

            fig = px.line(wealth_stunting, x='wealth_index', y='stunting',
                         title='Stunting Gradient by Wealth Index',
                         markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wealth or stunting data not available")
    
    with col2:
        # Maternal education and child nutrition
        if 'education_level' in women_data.columns:
            # Merge women's education with child data
            women_education = women_data[['___index', 'education_level']].drop_duplicates()
            merged_education = merged_data.merge(women_education, on='___index', how='left')
            
            if 'education_level' in merged_education.columns and 'stunting' in merged_education.columns:
                education_stunting = merged_education.groupby('education_level').agg({
                    'stunting': lambda x: (x != 'Normal').mean() * 100
                }).reset_index()

                fig = px.bar(education_stunting, x='education_level', y='stunting',
                            title='Stunting by Maternal Education Level',
                            color='stunting',
                            color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Education or stunting data not available for analysis")
    
    # Causal pathway analysis
    st.markdown("### Causal Pathways to Malnutrition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Immediate Causes")
        st.info("""
        - **Inadequate dietary intake**
          - Low food quantity
          - Poor dietary diversity
          - Limited micronutrient intake
        
        - **Disease and infection**
          - Frequent diarrhea
          - Respiratory infections
          - Parasitic infections
        """)
    
    with col2:
        st.markdown("#### Underlying Causes")
        st.warning("""
        - **Household food insecurity**
          - Limited food access
          - Poor food utilization
          - Seasonal shortages
        
        - **Poor care practices**
          - Inadequate feeding practices
          - Poor hygiene practices
          - Lack of health seeking
        """)
    
    # Basic causes
    st.markdown("#### Basic Causes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("**Economic Factors**")
        st.write("- Poverty")
        st.write("- Unemployment")
        st.write("- Low income")
    
    with col2:
        st.error("**Social Factors**")
        st.write("- Low education")
        st.write("- Gender inequality")
        st.write("- Cultural practices")
    
    with col3:
        st.error("**Environmental Factors**")
        st.write("- Poor sanitation")
        st.write("- Limited healthcare")
        st.write("- Food price volatility")