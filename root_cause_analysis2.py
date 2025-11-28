import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def root_cause_analysis():
    """Root cause analysis of stunting and deficiencies"""
    st.markdown('<div class="main-header">📊 Root Cause Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    women_data = st.session_state.women_data
    hh_data = st.session_state.hh_data
    child_data = st.session_state.child_data
    
    # Show available columns for debugging
    with st.expander("🔍 View Available Data Columns", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Household Data:**")
            st.write(hh_data.columns.tolist())
        with col2:
            st.write("**Child Data:**")
            st.write(child_data.columns.tolist())
        with col3:
            st.write("**Women Data:**")
            st.write(women_data.columns.tolist())
    
    # Multivariate analysis
    st.markdown("### 📈 Key Determinants of Malnutrition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wealth and stunting - IMPROVED VERSION
        wealth_stunting_available = check_wealth_stunting_data(merged_data, child_data, hh_data)
        
        if wealth_stunting_available:
            wealth_stunting_data = prepare_wealth_stunting_data(merged_data, child_data, hh_data)
            
            if len(wealth_stunting_data) > 0:
                wealth_stunting = wealth_stunting_data.groupby('wealth_index').agg({
                    'stunting_indicator': 'mean'
                }).reset_index()

                fig = px.line(wealth_stunting, x='wealth_index', y='stunting_indicator',
                             title='Stunting Gradient by Wealth Index',
                             markers=True,
                             labels={'stunting_indicator': 'Stunting Rate (%)', 'wealth_index': 'Wealth Index'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for wealth-stunting analysis")
        else:
            st.info("Wealth or stunting data not available")
    
    with col2:
        # Maternal education and child nutrition - FIXED VERSION
        education_stunting_available = check_education_stunting_data(merged_data, women_data, child_data)
        
        if education_stunting_available:
            education_stunting_data = prepare_education_stunting_data(merged_data, women_data, child_data)
            
            if len(education_stunting_data) > 0:
                education_stunting = education_stunting_data.groupby('education_level').agg({
                    'stunting_indicator': 'mean'
                }).reset_index()

                fig = px.bar(education_stunting, x='education_level', y='stunting_indicator',
                            title='Stunting by Maternal Education Level',
                            color='stunting_indicator',
                            color_continuous_scale='Reds',
                            labels={'stunting_indicator': 'Stunting Rate (%)', 'education_level': 'Education Level'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available for education-stunting analysis")
        else:
            st.info("Education or stunting data not available for analysis")
    
    # Dietary patterns analysis
    st.markdown("### 🍎 Dietary Patterns and Nutrient Intake")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'dietary_diversity_score' in hh_data.columns and 'wealth_index' in hh_data.columns:
            # Clean the data
            diversity_data = hh_data[['wealth_index', 'dietary_diversity_score']].copy()
            diversity_data = diversity_data.dropna()
            
            if len(diversity_data) > 0:
                # Convert dietary diversity to numeric
                diversity_data['dietary_diversity_score'] = pd.to_numeric(diversity_data['dietary_diversity_score'], errors='coerce')
                diversity_data = diversity_data.dropna()
                
                if len(diversity_data) > 0:
                    diversity_wealth = diversity_data.groupby('wealth_index')['dietary_diversity_score'].mean().reset_index()
                    
                    fig = px.bar(diversity_wealth, x='wealth_index', y='dietary_diversity_score',
                                title='Dietary Diversity by Wealth Index',
                                color='dietary_diversity_score',
                                color_continuous_scale='Blues',
                                labels={'dietary_diversity_score': 'Dietary Diversity Score', 'wealth_index': 'Wealth Index'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'urban_rural' in hh_data.columns and 'dietary_diversity_score' in hh_data.columns:
            # Clean the data
            urban_data = hh_data[['urban_rural', 'dietary_diversity_score']].copy()
            urban_data = urban_data.dropna()
            
            if len(urban_data) > 0:
                # Convert dietary diversity to numeric
                urban_data['dietary_diversity_score'] = pd.to_numeric(urban_data['dietary_diversity_score'], errors='coerce')
                urban_data = urban_data.dropna()
                
                if len(urban_data) > 0:
                    urban_rural_diversity = urban_data.groupby('urban_rural')['dietary_diversity_score'].mean().reset_index()
                    
                    fig = px.pie(urban_rural_diversity, values='dietary_diversity_score', names='urban_rural',
                                title='Dietary Diversity: Urban vs Rural')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Causal pathway analysis
    st.markdown("### 🔗 Causal Pathways to Malnutrition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Immediate Causes")
        st.info("""
        **Inadequate dietary intake:**
        - Low food quantity
        - Poor dietary diversity  
        - Limited micronutrient intake
        - Infrequent meals
        
        **Disease and infection:**
        - Frequent diarrhea
        - Respiratory infections
        - Parasitic infections
        - Recurrent fevers
        """)
    
    with col2:
        st.markdown("#### 🏠 Underlying Causes")
        st.warning("""
        **Household food insecurity:**
        - Limited food access
        - Poor food utilization
        - Seasonal shortages
        - Price volatility
        
        **Poor care practices:**
        - Inadequate feeding practices
        - Poor hygiene practices
        - Lack of health seeking
        - Insufficient childcare
        """)
    
    # Basic causes
    st.markdown("#### 🌍 Basic Causes")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.error("**💰 Economic Factors**")
        st.write("• Poverty and low income")
        st.write("• Unemployment/underemployment")
        st.write("• Limited assets")
        st.write("• High dependency ratios")
    
    with col2:
        st.error("**👥 Social Factors**")
        st.write("• Low education levels")
        st.write("• Gender inequality")
        st.write("• Cultural practices")
        st.write("• Early marriage/pregnancy")
    
    with col3:
        st.error("**🌳 Environmental Factors**")
        st.write("• Poor sanitation")
        st.write("• Limited healthcare access")
        st.write("• Food price volatility")
        st.write("• Climate vulnerability")
    
    # Show data availability summary
    st.markdown("### 📋 Data Availability Summary")
    
    availability_data = []
    
    # Check wealth-stunting data
    wealth_stunting_avail = check_wealth_stunting_data(merged_data, child_data, hh_data)
    availability_data.append({"Analysis": "Wealth vs Stunting", "Available": wealth_stunting_avail})
    
    # Check education-stunting data
    education_stunting_avail = check_education_stunting_data(merged_data, women_data, child_data)
    availability_data.append({"Analysis": "Education vs Stunting", "Available": education_stunting_avail})
    
    # Check dietary diversity data
    diet_diversity_avail = 'dietary_diversity_score' in hh_data.columns
    availability_data.append({"Analysis": "Dietary Diversity", "Available": diet_diversity_avail})
    
    # Check urban-rural data
    urban_rural_avail = 'urban_rural' in hh_data.columns
    availability_data.append({"Analysis": "Urban-Rural Analysis", "Available": urban_rural_avail})
    
    availability_df = pd.DataFrame(availability_data)
    
    # Create availability visualization
    fig = px.bar(availability_df, x='Analysis', y='Available',
                 title='Data Availability for Root Cause Analysis',
                 color='Available',
                 color_continuous_scale=['red', 'green'])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data troubleshooting guide
    with st.expander("🔧 Data Troubleshooting Guide"):
        st.markdown("""
        **Common Issues and Solutions:**
        
        **1. Education Data Missing:**
        - Check if women's data contains 'education_level' column
        - Verify that women's data is properly loaded
        - Ensure household IDs match between datasets
        
        **2. Stunting Data Missing:**
        - Check if child data contains stunting indicators
        - Look for columns like 'stunting', 'stunting_zscore', 'height_for_age'
        - Verify data merging between household and child data
        
        **3. Wealth Data Missing:**
        - Check for 'wealth_index' in household data
        - Look for alternative wealth indicators like 'income_level'
        - Verify data quality and completeness
        
        **4. Data Merging Issues:**
        - Ensure all datasets have common household IDs ('___index')
        - Check for consistent column names across datasets
        - Verify that data loading completed successfully
        """)
        
        # Show specific column availability
        st.markdown("**Current Column Availability:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Education Columns:**")
            education_cols = [col for col in women_data.columns if 'education' in col.lower() or 'educ' in col.lower()]
            st.write(education_cols if education_cols else "None found")
        
        with col2:
            st.write("**Stunting Columns:**")
            stunting_cols = []
            for df in [child_data, merged_data]:
                stunting_cols.extend([col for col in df.columns if 'stunt' in col.lower() or 'height' in col.lower()])
            st.write(stunting_cols if stunting_cols else "None found")
        
        with col3:
            st.write("**Wealth Columns:**")
            wealth_cols = []
            for df in [hh_data, merged_data]:
                wealth_cols.extend([col for col in df.columns if 'wealth' in col.lower() or 'income' in col.lower() or 'wi_' in col.lower()])
            st.write(wealth_cols if wealth_cols else "None found")

def check_wealth_stunting_data(merged_data, child_data, hh_data):
    """Check if wealth and stunting data is available"""
    # Check for wealth index
    wealth_columns = []
    for df in [merged_data, hh_data]:
        wealth_columns.extend([col for col in df.columns if 'wealth' in col.lower() or 'wi_' in col.lower()])
    
    # Check for stunting indicators
    stunting_columns = []
    for df in [merged_data, child_data]:
        stunting_columns.extend([col for col in df.columns if any(keyword in col.lower() for keyword in 
                              ['stunt', 'height_for_age', 'haz', 'height_age'])])
    
    return len(wealth_columns) > 0 and len(stunting_columns) > 0

def prepare_wealth_stunting_data(merged_data, child_data, hh_data):
    """Prepare wealth and stunting data for analysis"""
    # Find wealth column
    wealth_col = None
    for df in [merged_data, hh_data]:
        potential_cols = [col for col in df.columns if 'wealth' in col.lower() or 'wi_' in col.lower()]
        if potential_cols:
            wealth_col = potential_cols[0]
            wealth_data = df[['___index', wealth_col]].copy() if '___index' in df.columns else df[[wealth_col]].copy()
            break
    
    # Find stunting column
    stunting_col = None
    for df in [merged_data, child_data]:
        potential_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                          ['stunt', 'height_for_age', 'haz', 'height_age'])]
        if potential_cols:
            stunting_col = potential_cols[0]
            stunting_data = df[['___index', stunting_col]].copy() if '___index' in df.columns else df[[stunting_col]].copy()
            break
    
    if not wealth_col or not stunting_col:
        return pd.DataFrame()
    
    # Merge data if we have household IDs
    if '___index' in wealth_data.columns and '___index' in stunting_data.columns:
        combined_data = wealth_data.merge(stunting_data, on='___index', how='inner')
    else:
        # If no common ID, try to combine directly (assuming same order - not ideal but works for demo)
        combined_data = pd.concat([wealth_data, stunting_data], axis=1)
    
    # Clean the data
    combined_data = combined_data.dropna()
    
    if len(combined_data) == 0:
        return pd.DataFrame()
    
    # Convert stunting to binary indicator if needed
    if combined_data[stunting_col].dtype == 'object':
        combined_data['stunting_indicator'] = (combined_data[stunting_col] != 'Normal').astype(int) * 100
    else:
        # For z-scores, consider <-2 as stunted
        combined_data['stunting_indicator'] = (combined_data[stunting_col] < -2).astype(int) * 100
    
    # Prepare final data
    final_data = combined_data.rename(columns={wealth_col: 'wealth_index'})
    final_data = final_data[['wealth_index', 'stunting_indicator']].dropna()
    
    return final_data

def check_education_stunting_data(merged_data, women_data, child_data):
    """Check if education and stunting data is available"""
    # Check for education level in women's data
    education_columns = [col for col in women_data.columns if 'education' in col.lower() or 'educ' in col.lower()]
    
    # Check for stunting indicators
    stunting_columns = []
    for df in [merged_data, child_data]:
        stunting_columns.extend([col for col in df.columns if any(keyword in col.lower() for keyword in 
                              ['stunt', 'height_for_age', 'haz', 'height_age'])])
    
    return len(education_columns) > 0 and len(stunting_columns) > 0

def prepare_education_stunting_data(merged_data, women_data, child_data):
    """Prepare education and stunting data for analysis"""
    # Find education column in women's data
    education_col = None
    potential_cols = [col for col in women_data.columns if 'education' in col.lower() or 'educ' in col.lower()]
    if potential_cols:
        education_col = potential_cols[0]
        education_data = women_data[['___index', education_col]].copy()
    else:
        return pd.DataFrame()
    
    # Find stunting column
    stunting_col = None
    for df in [merged_data, child_data]:
        potential_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                          ['stunt', 'height_for_age', 'haz', 'height_age'])]
        if potential_cols:
            stunting_col = potential_cols[0]
            stunting_data = df[['___index', stunting_col]].copy()
            break
    
    if not stunting_col:
        return pd.DataFrame()
    
    # Merge education and stunting data using household ID
    combined_data = education_data.merge(stunting_data, on='___index', how='inner')
    
    # Clean the data
    combined_data = combined_data.dropna()
    
    if len(combined_data) == 0:
        return pd.DataFrame()
    
    # Convert stunting to binary indicator if needed
    if combined_data[stunting_col].dtype == 'object':
        combined_data['stunting_indicator'] = (combined_data[stunting_col] != 'Normal').astype(int) * 100
    else:
        # For z-scores, consider <-2 as stunted
        combined_data['stunting_indicator'] = (combined_data[stunting_col] < -2).astype(int) * 100
    
    # Prepare final data
    final_data = combined_data.rename(columns={education_col: 'education_level'})
    final_data = final_data[['education_level', 'stunting_indicator']].dropna()
    
    return final_data

# Helper function to create demo data if real data is missing
def create_demo_education_stunting_data():
    """Create demonstration data for education-stunting analysis"""
    education_levels = ['None', 'Primary', 'Secondary', 'Higher']
    demo_data = []
    
    for education in education_levels:
        # Simulate decreasing stunting rates with higher education
        if education == 'None':
            stunting_rate = 45
        elif education == 'Primary':
            stunting_rate = 35
        elif education == 'Secondary':
            stunting_rate = 25
        else:  # Higher
            stunting_rate = 15
            
        demo_data.append({
            'education_level': education,
            'stunting_indicator': stunting_rate
        })
    
    return pd.DataFrame(demo_data)