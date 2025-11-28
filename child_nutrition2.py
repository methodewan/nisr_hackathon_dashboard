import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def child_nutrition():
    """Child nutrition analysis"""
    st.markdown('<div class="main-header">👶 Child Nutrition Analysis (6-59 months)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    child_data = st.session_state.child_data
    merged_data = st.session_state.merged_data
    
    # Show available columns for debugging
    with st.expander("🔍 View Available Data Columns", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Child Data Columns:**")
            st.write(child_data.columns.tolist())
        with col2:
            st.write("**Sample Data (first 5 rows):**")
            st.dataframe(child_data.head(), use_container_width=True)
    
    # Child nutrition indicators
    st.markdown("### 📊 Child Nutrition Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'stunting' in child_data.columns:
            # Handle different stunting formats (categorical or binary)
            if child_data['stunting'].dtype == 'object':
                stunting_rate = (child_data['stunting'] != 'Normal').mean() * 100
            else:
                stunting_rate = child_data['stunting'].mean() * 100
            st.metric("Stunting Rate", f"{stunting_rate:.1f}%")
        else:
            st.metric("Stunting Rate", "Data not available")
    
    with col2:
        # Check for wasting indicators with different possible column names
        wasting_columns = [col for col in child_data.columns if 'wast' in col.lower() or 'wasting' in col.lower()]
        if wasting_columns:
            wasting_col = wasting_columns[0]
            if child_data[wasting_col].dtype == 'object':
                wasting_rate = (child_data[wasting_col] != 'Normal').mean() * 100
            else:
                # Assume z-score format
                wasting_rate = (child_data[wasting_col] < -2).mean() * 100
            st.metric("Wasting Rate", f"{wasting_rate:.1f}%")
        else:
            st.metric("Wasting Rate", "Data not available")
    
    with col3:
        # Check for underweight indicators with different possible column names
        underweight_columns = [col for col in child_data.columns if 'underweight' in col.lower() or 'weight' in col.lower()]
        if underweight_columns:
            underweight_col = underweight_columns[0]
            if child_data[underweight_col].dtype == 'object':
                underweight_rate = (child_data[underweight_col] != 'Normal').mean() * 100
            else:
                # Assume z-score format
                underweight_rate = (child_data[underweight_col] < -2).mean() * 100
            st.metric("Underweight Rate", f"{underweight_rate:.1f}%")
        else:
            st.metric("Underweight Rate", "Data not available")
    
    with col4:
        if 'received_vitamin_a' in child_data.columns:
            vit_a_coverage = child_data['received_vitamin_a'].mean() * 100
            st.metric("Vitamin A Coverage", f"{vit_a_coverage:.1f}%")
        else:
            # Check for other vitamin A related columns
            vit_a_columns = [col for col in child_data.columns if 'vitamin' in col.lower() or 'vita' in col.lower()]
            if vit_a_columns:
                vit_a_col = vit_a_columns[0]
                vit_a_coverage = child_data[vit_a_col].mean() * 100
                st.metric("Vitamin A Coverage", f"{vit_a_coverage:.1f}%")
            else:
                st.metric("Vitamin A Coverage", "Data not available")
    
    # Age-specific analysis
    st.markdown("### 📈 Age-Specific Nutrition Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'age_months' in child_data.columns and 'stunting' in child_data.columns:
            child_data_copy = child_data.copy()
            
            # Clean and convert age_months to numeric
            child_data_copy['age_months'] = pd.to_numeric(child_data_copy['age_months'], errors='coerce')
            child_data_copy = child_data_copy.dropna(subset=['age_months'])
            
            # Create age groups
            child_data_copy['age_group'] = pd.cut(child_data_copy['age_months'],
                                           bins=[6, 12, 24, 36, 48, 60],
                                           labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m'])

            age_stunting = child_data_copy.groupby('age_group').agg({
                'stunting': lambda x: (x != 'Normal').mean() * 100 if x.dtype == 'object' else x.mean() * 100
            }).reset_index()

            fig = px.line(age_stunting, x='age_group', y='stunting',
                         title='Stunting Prevalence by Age Group',
                         markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Age or stunting data not available for age-specific analysis")
    
    with col2:
        # Use any available wasting indicator for age analysis
        wasting_cols = [col for col in child_data.columns if 'wast' in col.lower()]
        if 'age_months' in child_data.columns and wasting_cols:
            wasting_col = wasting_cols[0]
            child_data_copy = child_data.copy()
            
            # Clean age data
            child_data_copy['age_months'] = pd.to_numeric(child_data_copy['age_months'], errors='coerce')
            child_data_copy = child_data_copy.dropna(subset=['age_months'])
            
            child_data_copy['age_group'] = pd.cut(child_data_copy['age_months'],
                                           bins=[6, 12, 24, 36, 48, 60],
                                           labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m'])

            age_wasting = child_data_copy.groupby('age_group').agg({
                wasting_col: lambda x: (x != 'Normal').mean() * 100 if x.dtype == 'object' else (x < -2).mean() * 100
            }).reset_index()

            fig = px.line(age_wasting, x='age_group', y=wasting_col,
                         title='Wasting Prevalence by Age Group',
                         markers=True)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Age or wasting data not available for age-specific analysis")
    
    # Dietary diversity analysis - FIXED VERSION
    st.markdown("### 🍎 Dietary Diversity and Nutrition Outcomes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'dietary_diversity' in child_data.columns and 'stunting' in child_data.columns:
            # Create a clean copy for analysis
            diversity_data = child_data[['dietary_diversity', 'stunting']].copy()
            
            # Convert dietary_diversity to numeric, handling errors
            diversity_data['dietary_diversity'] = pd.to_numeric(diversity_data['dietary_diversity'], errors='coerce')
            
            # Remove rows with invalid dietary diversity values
            diversity_data = diversity_data.dropna(subset=['dietary_diversity'])
            
            if len(diversity_data) > 0:
                # Group by dietary diversity and calculate stunting rates
                diversity_stunting = diversity_data.groupby('dietary_diversity').agg({
                    'stunting': lambda x: (x != 'Normal').mean() * 100 if x.dtype == 'object' else x.mean() * 100
                }).reset_index()

                # Create scatter plot with trendline
                fig = px.scatter(diversity_stunting, x='dietary_diversity', y='stunting',
                                title='Stunting vs Dietary Diversity',
                                trendline='lowess',
                                labels={'dietary_diversity': 'Dietary Diversity Score', 
                                       'stunting': 'Stunting Rate (%)'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid dietary diversity data available for analysis")
        else:
            st.info("Dietary diversity or stunting data not available for analysis")
    
    with col2:
        # Use any available wasting indicator for dietary diversity analysis
        wasting_cols = [col for col in child_data.columns if 'wast' in col.lower()]
        if 'dietary_diversity' in child_data.columns and wasting_cols:
            wasting_col = wasting_cols[0]
            # Create a clean copy for analysis
            diversity_data = child_data[['dietary_diversity', wasting_col]].copy()
            
            # Convert dietary_diversity to numeric, handling errors
            diversity_data['dietary_diversity'] = pd.to_numeric(diversity_data['dietary_diversity'], errors='coerce')
            
            # Remove rows with invalid dietary diversity values
            diversity_data = diversity_data.dropna(subset=['dietary_diversity'])
            
            if len(diversity_data) > 0:
                # Group by dietary diversity and calculate wasting rates
                diversity_wasting = diversity_data.groupby('dietary_diversity').agg({
                    wasting_col: lambda x: (x != 'Normal').mean() * 100 if x.dtype == 'object' else (x < -2).mean() * 100
                }).reset_index()

                fig = px.scatter(diversity_wasting, x='dietary_diversity', y=wasting_col,
                                title='Wasting vs Dietary Diversity',
                                trendline='lowess',
                                labels={'dietary_diversity': 'Dietary Diversity Score', 
                                       wasting_col: 'Wasting Rate (%)'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid dietary diversity data available for analysis")
        else:
            st.info("Dietary diversity or wasting data not available for analysis")
    
    # Sex-based analysis - FIXED VERSION
    st.markdown("### 👦👧 Sex-Based Nutrition Differences")
    
    # Safely get available columns for sex analysis
    available_columns = ['sex']
    if 'stunting' in child_data.columns:
        available_columns.append('stunting')
    
    # Add any wasting-related columns
    wasting_cols = [col for col in child_data.columns if 'wast' in col.lower()]
    if wasting_cols:
        available_columns.append(wasting_cols[0])
    
    # Add any underweight-related columns
    underweight_cols = [col for col in child_data.columns if 'underweight' in col.lower() or 'weight' in col.lower()]
    if underweight_cols:
        available_columns.append(underweight_cols[0])
    
    if len(available_columns) >= 2:  # Need at least sex and one nutrition indicator
        # Clean sex data
        sex_data = child_data[available_columns].copy()
        sex_data = sex_data.dropna(subset=['sex'])
        
        # Standardize sex values
        sex_data['sex'] = sex_data['sex'].str.strip().str.title()
        sex_data = sex_data[sex_data['sex'].isin(['Male', 'Female'])]
        
        if len(sex_data) > 0:
            # Calculate rates for each available indicator
            nutrition_indicators = [col for col in available_columns if col != 'sex']
            sex_nutrition_data = []
            
            for indicator in nutrition_indicators:
                indicator_data = sex_data.groupby('sex').agg({
                    indicator: lambda x: (x != 'Normal').mean() * 100 if x.dtype == 'object' else (x < -2).mean() * 100
                }).reset_index()
                indicator_data['Indicator'] = indicator
                indicator_data.rename(columns={indicator: 'Rate'}, inplace=True)
                sex_nutrition_data.append(indicator_data)
            
            if sex_nutrition_data:
                combined_data = pd.concat(sex_nutrition_data, ignore_index=True)
                
                fig = px.bar(combined_data, x='sex', y='Rate', color='Indicator',
                            barmode='group', title='Nutrition Indicators by Sex',
                            labels={'sex': 'Sex', 'Rate': 'Prevalence (%)', 'Indicator': 'Nutrition Indicator'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Supplementation coverage
    st.markdown("### 💊 Supplementation Coverage and Impact")
    
    if 'received_vitamin_a' in child_data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Vitamin A coverage by age group
            if 'age_months' in child_data.columns:
                child_data_copy = child_data.copy()
                
                # Clean age data
                child_data_copy['age_months'] = pd.to_numeric(child_data_copy['age_months'], errors='coerce')
                child_data_copy = child_data_copy.dropna(subset=['age_months'])
                
                child_data_copy['age_group'] = pd.cut(child_data_copy['age_months'],
                                               bins=[6, 12, 24, 36, 48, 60],
                                               labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m'])
                
                # Clean vitamin A data
                child_data_copy = child_data_copy.dropna(subset=['received_vitamin_a'])
                
                vit_a_coverage = child_data_copy.groupby('age_group')['received_vitamin_a'].mean() * 100
                vit_a_coverage = vit_a_coverage.reset_index()
                
                fig = px.bar(vit_a_coverage, x='age_group', y='received_vitamin_a',
                            title='Vitamin A Coverage by Age Group',
                            labels={'received_vitamin_a': 'Coverage (%)', 'age_group': 'Age Group'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Impact of supplementation on stunting
            if 'stunting' in child_data.columns:
                supplementation_data = child_data[['received_vitamin_a', 'stunting']].copy()
                supplementation_data = supplementation_data.dropna()
                
                supplementation_impact = supplementation_data.groupby('received_vitamin_a').agg({
                    'stunting': lambda x: (x != 'Normal').mean() * 100 if x.dtype == 'object' else x.mean() * 100
                }).reset_index()
                
                # Convert to meaningful labels
                supplementation_impact['received_vitamin_a'] = supplementation_impact['received_vitamin_a'].map({
                    0: 'Not Received', 
                    1: 'Received'
                })
                
                fig = px.bar(supplementation_impact, x='received_vitamin_a', y='stunting',
                            title='Stunting Rates by Vitamin A Supplementation',
                            labels={'received_vitamin_a': 'Vitamin A Supplementation', 'stunting': 'Stunting Rate (%)'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Nutritional Status Distribution
    st.markdown("### 📋 Nutritional Status Distribution")
    
    # Create a summary of nutritional status
    nutrition_summary = []
    
    if 'stunting' in child_data.columns:
        if child_data['stunting'].dtype == 'object':
            stunting_dist = child_data['stunting'].value_counts(normalize=True) * 100
        else:
            # For binary data, create categories
            stunting_dist = pd.Series({
                'Stunted': child_data['stunting'].mean() * 100,
                'Normal': (1 - child_data['stunting'].mean()) * 100
            })
        for status, percentage in stunting_dist.items():
            nutrition_summary.append({'Indicator': 'Stunting', 'Status': status, 'Percentage': percentage})
    
    # Check for wasting data with different column names
    wasting_cols = [col for col in child_data.columns if 'wast' in col.lower()]
    if wasting_cols:
        wasting_col = wasting_cols[0]
        if child_data[wasting_col].dtype == 'object':
            wasting_dist = child_data[wasting_col].value_counts(normalize=True) * 100
        else:
            # Create categories from z-scores
            wasting_categories = pd.cut(child_data[wasting_col], 
                                      bins=[-float('inf'), -3, -2, float('inf')],
                                      labels=['Severe', 'Moderate', 'Normal'])
            wasting_dist = wasting_categories.value_counts(normalize=True) * 100
        for status, percentage in wasting_dist.items():
            nutrition_summary.append({'Indicator': 'Wasting', 'Status': status, 'Percentage': percentage})
    
    if nutrition_summary:
        summary_df = pd.DataFrame(nutrition_summary)
        fig = px.bar(summary_df, x='Indicator', y='Percentage', color='Status',
                    title='Distribution of Nutritional Status',
                    barmode='stack')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Factors Analysis
    st.markdown("### ⚠️ Key Risk Factors Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Common Risk Factors")
        risk_factors = [
            "Low dietary diversity (<4 food groups)",
            "Poor household wealth status",
            "Lack of improved water source",
            "Inadequate sanitation facilities",
            "Low maternal education",
            "Young child age (<24 months)",
            "Frequent illness episodes",
            "Inadequate breastfeeding"
        ]
        
        for factor in risk_factors:
            st.write(f"• {factor}")
    
    with col2:
        st.markdown("#### Protective Factors")
        protective_factors = [
            "Diverse diet (≥6 food groups)",
            "Good household economic status",
            "Access to clean water",
            "Improved sanitation",
            "Higher maternal education",
            "Regular health check-ups",
            "Vitamin A supplementation",
            "Appropriate breastfeeding"
        ]
        
        for factor in protective_factors:
            st.write(f"• {factor}")
    
    # Data Quality Information
    with st.expander("📊 Data Quality Information"):
        st.markdown("""
        **Data Processing Notes:**
        - Invalid or non-numeric values are automatically filtered out
        - Age data is cleaned and converted to numeric format
        - Missing values are excluded from analysis
        - Sex categories are standardized (Male/Female)
        - Multiple column name variations are handled automatically
        """)
        
        # Show data quality metrics
        quality_info = []
        
        if 'dietary_diversity' in child_data.columns:
            total_dietary = len(child_data['dietary_diversity'])
            numeric_dietary = pd.to_numeric(child_data['dietary_diversity'], errors='coerce').notna().sum()
            quality_info.append(f"**Dietary Diversity:** {numeric_dietary}/{total_dietary} valid entries ({numeric_dietary/total_dietary:.1%})")
        
        if 'age_months' in child_data.columns:
            total_age = len(child_data['age_months'])
            numeric_age = pd.to_numeric(child_data['age_months'], errors='coerce').notna().sum()
            quality_info.append(f"**Age Data:** {numeric_age}/{total_age} valid entries ({numeric_age/total_age:.1%})")
        
        if 'sex' in child_data.columns:
            total_sex = len(child_data['sex'])
            valid_sex = child_data['sex'].notna().sum()
            quality_info.append(f"**Sex Data:** {valid_sex}/{total_sex} valid entries ({valid_sex/total_sex:.1%})")
        
        for info in quality_info:
            st.write(info)

# Helper function to clean and prepare data
def clean_nutrition_data(child_data):
    """Clean and prepare child nutrition data for analysis"""
    cleaned_data = child_data.copy()
    
    # Convert numeric columns
    numeric_columns = ['age_months', 'dietary_diversity']
    # Add any columns that might contain z-scores
    z_score_columns = [col for col in cleaned_data.columns if 'zscore' in col.lower() or 'score' in col.lower()]
    numeric_columns.extend(z_score_columns)
    
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    # Clean categorical columns
    if 'sex' in cleaned_data.columns:
        cleaned_data['sex'] = cleaned_data['sex'].str.strip().str.title()
    
    if 'stunting' in cleaned_data.columns:
        cleaned_data['stunting'] = cleaned_data['stunting'].str.strip()
    
    return cleaned_data

# Helper function to find related columns
def find_related_columns(child_data, keyword):
    """Find columns related to a specific keyword"""
    return [col for col in child_data.columns if keyword.lower() in col.lower()]