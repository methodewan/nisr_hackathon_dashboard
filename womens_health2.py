import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def womens_health():
    """Women's health and nutrition analysis"""
    st.markdown('<div class="main-header">👩 Women\'s Health & Nutrition (15-49 years)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    women_data = st.session_state.women_data
    
    # Show available columns for debugging
    with st.expander("🔍 View Available Data Columns", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Women Data Columns:**")
            st.write(women_data.columns.tolist())
        with col2:
            st.write("**Sample Data (first 5 rows):**")
            st.dataframe(women_data.head(), use_container_width=True)
    
    # Clean and preprocess the data
    women_data_clean = clean_women_data(women_data)
    
    # Women's health indicators
    st.markdown("### 📊 Women's Health Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Anemia rate - handle different column names and formats
        anemia_columns = [col for col in women_data_clean.columns if 'anemi' in col.lower() or 'anaemi' in col.lower()]
        if anemia_columns:
            anemia_col = anemia_columns[0]
            if women_data_clean[anemia_col].dtype == 'object':
                # Convert Yes/No to numeric
                anemia_binary = women_data_clean[anemia_col].map({'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0}).fillna(0)
                anemia_rate = anemia_binary.mean() * 100
            else:
                anemia_rate = women_data_clean[anemia_col].mean() * 100
            st.metric("Anemia Rate", f"{anemia_rate:.1f}%")
        else:
            st.metric("Anemia Rate", "Data not available")
    
    with col2:
        # IFA coverage - handle different column names and formats
        ifa_columns = [col for col in women_data_clean.columns if 'ifa' in col.lower() or 'iron' in col.lower() or 'folic' in col.lower()]
        if ifa_columns:
            ifa_col = ifa_columns[0]
            if women_data_clean[ifa_col].dtype == 'object':
                # Handle string values like 'Yes'/'No'
                ifa_binary = convert_yes_no_to_binary(women_data_clean[ifa_col])
                ifa_coverage = ifa_binary.mean() * 100
            else:
                ifa_coverage = women_data_clean[ifa_col].mean() * 100
            st.metric("IFA Coverage", f"{ifa_coverage:.1f}%")
        else:
            st.metric("IFA Coverage", "Data not available")
    
    with col3:
        # BMI indicators
        bmi_columns = [col for col in women_data_clean.columns if 'bmi' in col.lower()]
        if bmi_columns:
            bmi_col = bmi_columns[0]
            # Convert to numeric if needed
            bmi_numeric = pd.to_numeric(women_data_clean[bmi_col], errors='coerce')
            low_bmi = (bmi_numeric < 18.5).mean() * 100
            st.metric("Low BMI (<18.5)", f"{low_bmi:.1f}%")
        else:
            st.metric("Low BMI", "Data not available")
    
    with col4:
        # Dietary diversity
        diet_columns = [col for col in women_data_clean.columns if 'diet' in col.lower() or 'food' in col.lower()]
        if diet_columns:
            diet_col = diet_columns[0]
            diet_numeric = pd.to_numeric(women_data_clean[diet_col], errors='coerce')
            avg_diversity = diet_numeric.mean()
            st.metric("Avg Dietary Diversity", f"{avg_diversity:.1f}")
        else:
            st.metric("Dietary Diversity", "Data not available")
    
    # Education and nutrition
    st.markdown("### 🎓 Education and Nutritional Status")
    
    if 'education_level' in women_data_clean.columns:
        # Check what nutrition indicators we have
        nutrition_indicators = []
        if anemia_columns:
            nutrition_indicators.append(anemia_columns[0])
        if bmi_columns:
            nutrition_indicators.append(bmi_columns[0])
        
        if nutrition_indicators:
            # Create analysis for each available indicator
            for indicator in nutrition_indicators:
                education_nutrition = women_data_clean.groupby('education_level').agg({
                    indicator: 'mean'
                }).reset_index()
                
                # Convert to percentages for binary indicators
                if women_data_clean[indicator].dtype == 'object':
                    education_nutrition[indicator] = education_nutrition[indicator] * 100
                    title_suffix = "Rate (%)"
                else:
                    title_suffix = "Score"
                
                fig = px.bar(education_nutrition, x='education_level', y=indicator,
                            title=f'{indicator.replace("_", " ").title()} by Education Level',
                            labels={'education_level': 'Education Level', indicator: title_suffix})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No nutrition indicators available for education analysis")
    else:
        st.info("Education level data not available for analysis")
    
    # Age-specific analysis
    st.markdown("### 📈 Age-Specific Health Patterns")
    
    if 'age_years' in women_data_clean.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            age_numeric = pd.to_numeric(women_data_clean['age_years'], errors='coerce')
            age_clean = age_numeric.dropna()
            
            if len(age_clean) > 0:
                fig = px.histogram(age_clean, x='age_years', nbins=15,
                                 title='Age Distribution of Women',
                                 labels={'age_years': 'Age (years)'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anemia by age group
            if anemia_columns:
                anemia_col = anemia_columns[0]
                age_anemia_data = women_data_clean[['age_years', anemia_col]].copy()
                age_anemia_data['age_years'] = pd.to_numeric(age_anemia_data['age_years'], errors='coerce')
                age_anemia_data = age_anemia_data.dropna()
                
                if len(age_anemia_data) > 0:
                    age_anemia_data['age_group'] = pd.cut(age_anemia_data['age_years'],
                                                   bins=[15, 20, 25, 30, 35, 40, 45, 50],
                                                   labels=['15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49'])
                    
                    # Convert anemia to binary if needed
                    if age_anemia_data[anemia_col].dtype == 'object':
                        age_anemia_data[anemia_col] = convert_yes_no_to_binary(age_anemia_data[anemia_col])
                    
                    anemia_age = age_anemia_data.groupby('age_group')[anemia_col].mean() * 100
                    anemia_age = anemia_age.reset_index()
                    
                    fig = px.line(anemia_age, x='age_group', y=anemia_col,
                                 title='Anemia Prevalence by Age Group',
                                 markers=True)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Pregnancy and nutrition
    st.markdown("### 🤰 Maternal Nutrition")
    
    # Find pregnancy-related columns
    pregnancy_columns = [col for col in women_data_clean.columns if 'preg' in col.lower() or 'pregnant' in col.lower()]
    
    if pregnancy_columns:
        pregnant_col = pregnancy_columns[0]
        
        # Convert pregnancy data to binary
        pregnant_binary = convert_yes_no_to_binary(women_data_clean[pregnant_col])
        pregnant_women = women_data_clean[pregnant_binary == 1]
        
        if len(pregnant_women) > 0:
            st.success(f"📊 Found {len(pregnant_women)} pregnant women in the dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Pregnant Women", len(pregnant_women))
            
            with col2:
                # IFA in pregnancy
                if ifa_columns:
                    ifa_col = ifa_columns[0]
                    ifa_pregnant = convert_yes_no_to_binary(pregnant_women[ifa_col]).mean() * 100
                    st.metric("IFA in Pregnancy", f"{ifa_pregnant:.1f}%")
                else:
                    st.metric("IFA in Pregnancy", "Data not available")
            
            with col3:
                # Anemia in pregnancy
                if anemia_columns:
                    anemia_col = anemia_columns[0]
                    anemia_pregnant = convert_yes_no_to_binary(pregnant_women[anemia_col]).mean() * 100
                    st.metric("Anemia in Pregnancy", f"{anemia_pregnant:.1f}%")
                else:
                    st.metric("Anemia in Pregnancy", "Data not available")
            
            with col4:
                # Low BMI in pregnancy
                if bmi_columns:
                    bmi_col = bmi_columns[0]
                    bmi_pregnant = pd.to_numeric(pregnant_women[bmi_col], errors='coerce')
                    low_bmi_pregnant = (bmi_pregnant < 18.5).mean() * 100
                    st.metric("Low BMI in Pregnancy", f"{low_bmi_pregnant:.1f}%")
                else:
                    st.metric("Low BMI in Pregnancy", "Data not available")
            
            # Pregnancy nutrition details
            st.markdown("#### Nutritional Status of Pregnant Women")
            
            if len(pregnant_women) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create summary of key indicators for pregnant women
                    preg_indicators = []
                    if ifa_columns:
                        ifa_rate = convert_yes_no_to_binary(pregnant_women[ifa_columns[0]]).mean() * 100
                        preg_indicators.append({'Indicator': 'IFA Coverage', 'Rate': ifa_rate})
                    
                    if anemia_columns:
                        anemia_rate = convert_yes_no_to_binary(pregnant_women[anemia_columns[0]]).mean() * 100
                        preg_indicators.append({'Indicator': 'Anemia', 'Rate': anemia_rate})
                    
                    if preg_indicators:
                        preg_df = pd.DataFrame(preg_indicators)
                        fig = px.bar(preg_df, x='Indicator', y='Rate',
                                    title='Health Indicators for Pregnant Women',
                                    color='Indicator')
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Age distribution of pregnant women
                    preg_ages = pd.to_numeric(pregnant_women['age_years'], errors='coerce').dropna()
                    if len(preg_ages) > 0:
                        fig = px.histogram(preg_ages, title='Age Distribution of Pregnant Women',
                                         labels={'value': 'Age'})
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pregnant women data available in the current sample")
    else:
        st.info("Pregnancy data not available")
    
    # Breastfeeding analysis
    st.markdown("### 🤱 Breastfeeding Practices")
    
    # Find breastfeeding-related columns
    breastfeeding_columns = [col for col in women_data_clean.columns if 'breast' in col.lower() or 'bf' in col.lower()]
    
    if breastfeeding_columns:
        breastfeeding_col = breastfeeding_columns[0]
        breastfeeding_binary = convert_yes_no_to_binary(women_data_clean[breastfeeding_col])
        breastfeeding_women = women_data_clean[breastfeeding_binary == 1]
        
        if len(breastfeeding_women) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Breastfeeding Women", len(breastfeeding_women))
                
                if anemia_columns:
                    anemia_bf = convert_yes_no_to_binary(breastfeeding_women[anemia_columns[0]]).mean() * 100
                    st.metric("Anemia in Breastfeeding", f"{anemia_bf:.1f}%")
            
            with col2:
                if ifa_columns:
                    ifa_bf = convert_yes_no_to_binary(breastfeeding_women[ifa_columns[0]]).mean() * 100
                    st.metric("IFA in Breastfeeding", f"{ifa_bf:.1f}%")
                
                if bmi_columns:
                    bmi_bf = pd.to_numeric(breastfeeding_women[bmi_columns[0]], errors='coerce')
                    low_bmi_bf = (bmi_bf < 18.5).mean() * 100
                    st.metric("Low BMI in Breastfeeding", f"{low_bmi_bf:.1f}%")
    
    # Dietary patterns
    st.markdown("### 🍎 Women's Dietary Patterns")
    
    if diet_columns:
        diet_col = diet_columns[0]
        diet_numeric = pd.to_numeric(women_data_clean[diet_col], errors='coerce').dropna()
        
        if len(diet_numeric) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Dietary diversity distribution
                fig = px.histogram(diet_numeric, x=diet_col, nbins=10,
                                 title='Distribution of Dietary Diversity Scores')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Dietary diversity by anemia status
                if anemia_columns:
                    anemia_col = anemia_columns[0]
                    diet_anemia_data = women_data_clean[[diet_col, anemia_col]].copy()
                    diet_anemia_data[diet_col] = pd.to_numeric(diet_anemia_data[diet_col], errors='coerce')
                    diet_anemia_data = diet_anemia_data.dropna()
                    
                    if len(diet_anemia_data) > 0:
                        # Convert anemia to categorical
                        diet_anemia_data[anemia_col] = convert_yes_no_to_binary(diet_anemia_data[anemia_col])
                        diet_anemia_data[anemia_col] = diet_anemia_data[anemia_col].map({1: 'Anemic', 0: 'Not Anemic'})
                        
                        diet_anemia_avg = diet_anemia_data.groupby(anemia_col)[diet_col].mean().reset_index()
                        
                        fig = px.bar(diet_anemia_avg, x=anemia_col, y=diet_col,
                                    title='Dietary Diversity by Anemia Status')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
    
    # Health Risk Assessment
    st.markdown("### ⚠️ Women's Health Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### High-Risk Indicators")
        high_risk_factors = [
            "Anemia (Hb < 12 g/dL)",
            "Low BMI (<18.5 kg/m²)",
            "Adolescent pregnancy (<20 years)",
            "Short birth spacing (<2 years)",
            "Multiple pregnancies",
            "Chronic malnutrition",
            "Lack of antenatal care",
            "Poor dietary diversity"
        ]
        
        for factor in high_risk_factors:
            st.write(f"• {factor}")
    
    with col2:
        st.markdown("#### Protective Factors")
        protective_factors = [
            "Adequate iron-folic acid supplementation",
            "Balanced diet with diverse foods",
            "Regular health check-ups",
            "Adequate birth spacing",
            "Education and empowerment",
            "Economic stability",
            "Access to healthcare",
            "Good sanitation practices"
        ]
        
        for factor in protective_factors:
            st.write(f"• {factor}")
    
    # Data Quality Information
    with st.expander("📊 Data Quality Information"):
        st.markdown("""
        **Data Processing Notes:**
        - Yes/No values are automatically converted to binary (1/0)
        - Invalid numeric values are filtered out
        - Missing values are excluded from analysis
        - Multiple column name variations are handled
        - Data types are automatically detected and converted
        """)
        
        # Show data quality metrics
        quality_info = []
        
        if anemia_columns:
            total_anemia = len(women_data_clean[anemia_columns[0]])
            valid_anemia = women_data_clean[anemia_columns[0]].notna().sum()
            quality_info.append(f"**Anemia Data:** {valid_anemia}/{total_anemia} valid entries")
        
        if ifa_columns:
            total_ifa = len(women_data_clean[ifa_columns[0]])
            valid_ifa = women_data_clean[ifa_columns[0]].notna().sum()
            quality_info.append(f"**IFA Data:** {valid_ifa}/{total_ifa} valid entries")
        
        if 'age_years' in women_data_clean.columns:
            total_age = len(women_data_clean['age_years'])
            numeric_age = pd.to_numeric(women_data_clean['age_years'], errors='coerce').notna().sum()
            quality_info.append(f"**Age Data:** {numeric_age}/{total_age} valid entries")
        
        for info in quality_info:
            st.write(info)

def clean_women_data(women_data):
    """Clean and preprocess women's health data"""
    cleaned_data = women_data.copy()
    
    # Convert common binary columns from Yes/No to 1/0
    binary_columns = [col for col in cleaned_data.columns if any(keyword in col.lower() for keyword in 
                     ['anemi', 'preg', 'breast', 'ifa', 'iron', 'folic', 'supplement'])]
    
    for col in binary_columns:
        if col in cleaned_data.columns and cleaned_data[col].dtype == 'object':
            cleaned_data[col] = convert_yes_no_to_binary(cleaned_data[col])
    
    # Convert numeric columns
    numeric_columns = ['age_years', 'bmi']
    for col in numeric_columns:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    # Clean education level if it exists
    if 'education_level' in cleaned_data.columns:
        cleaned_data['education_level'] = cleaned_data['education_level'].str.strip().str.title()
    
    return cleaned_data

def convert_yes_no_to_binary(series):
    """Convert Yes/No series to binary (1/0)"""
    return series.map({'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0)

def find_related_columns(women_data, keyword):
    """Find columns related to a specific keyword"""
    return [col for col in women_data.columns if keyword.lower() in col.lower()]