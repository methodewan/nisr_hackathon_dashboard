import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def dashboard_overview():
    """Main dashboard overview with Pantone color scheme"""
    st.markdown('<div class="main-header">🏠 Hidden Hunger Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            hh_data, child_data, women_data = load_all_data()
            merged_data = merge_datasets(hh_data, child_data, women_data)
            
            st.session_state.hh_data = hh_data
            st.session_state.child_data = child_data
            st.session_state.women_data = women_data
            st.session_state.merged_data = merged_data
            st.session_state.data_loaded = True
    
    hh_data = st.session_state.hh_data
    child_data = st.session_state.child_data
    women_data = st.session_state.women_data
    merged_data = st.session_state.merged_data
    
    # Debug data issues
    debug_data_issues(merged_data)
    
    # Define Pantone color scheme
    PANTONE_2728C = "#2E3192"  # Deep Blue
    PANTONE_CYAN = "#00FFFF"   # Bright Cyan
    PANTONE_362C = "#3A913F"   # Forest Green
    PANTONE_3965C = "#EED500"  # Golden Yellow
    
    # Color palette for charts
    COLOR_PALETTE = [PANTONE_2728C, PANTONE_CYAN, PANTONE_362C, PANTONE_3965C]
    SEQUENTIAL_BLUES = [PANTONE_2728C, "#4A4DA8", "#676ABF", "#8588D5", "#A2A5EC"]
    
    # Key metrics - Using Pantone colors
    st.markdown("### 📊 Key Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: {PANTONE_2728C}15; padding: 1rem; border-radius: 10px; border-left: 4px solid {PANTONE_2728C};">
            <h3 style="color: {PANTONE_2728C}; margin: 0; font-size: 1.8rem;">{len(hh_data):,}</h3>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Households Surveyed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: {PANTONE_CYAN}15; padding: 1rem; border-radius: 10px; border-left: 4px solid {PANTONE_CYAN};">
            <h3 style="color: {PANTONE_CYAN}; margin: 0; font-size: 1.8rem;">{len(child_data):,}</h3>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Children (6-59 months)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background-color: {PANTONE_362C}15; padding: 1rem; border-radius: 10px; border-left: 4px solid {PANTONE_362C};">
            <h3 style="color: {PANTONE_362C}; margin: 0; font-size: 1.8rem;">{len(women_data):,}</h3>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Women (15-49 years)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if 'any_malnutrition' in merged_data.columns:
            malnutrition_rate = merged_data['any_malnutrition'].mean() * 100
            st.markdown(f"""
            <div style="background-color: {PANTONE_3965C}15; padding: 1rem; border-radius: 10px; border-left: 4px solid {PANTONE_3965C};">
                <h3 style="color: {PANTONE_3965C}; margin: 0; font-size: 1.8rem;">{malnutrition_rate:.1f}%</h3>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Malnutrition Prevalence</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.metric("Malnutrition Prevalence", "Data not available")
    
    # Nutrition indicators - Second row with Pantone colors
    st.markdown("### 🎯 Nutrition Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'stunting' in merged_data.columns:
            stunting_data = merged_data[merged_data['stunting'] != 'Unknown']
            if len(stunting_data) > 0:
                stunting_rate = (stunting_data['stunting'] != 'Normal').mean() * 100
                st.markdown(f"""
                <div style="background-color: {PANTONE_2728C}10; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: {PANTONE_2728C}; margin: 0;">{stunting_rate:.1f}%</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">Stunting Rate</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: #666; margin: 0;">N/A</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">Stunting Rate</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: #666; margin: 0;">N/A</h4>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">Stunting Rate</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        anemia_cols = [col for col in women_data.columns if 'anemi' in col.lower() or 'anemi' in col.lower()]
        if not anemia_cols:
            # Look for binary columns that might represent anemia
            anemia_cols = [col for col in women_data.columns if women_data[col].dtype in ['int64', 'float64']
                          and len(women_data[col].dropna().unique()) == 2
                          and women_data[col].mean() > 0.2 and women_data[col].mean() < 0.7]
        
        if anemia_cols:
            anemia_col = anemia_cols[0]
            # Ensure the column is numeric
            women_data[anemia_col] = pd.to_numeric(women_data[anemia_col], errors='coerce')
            anemia_data = women_data[anemia_col].dropna()
            if len(anemia_data) > 0:
                anemia_rate = anemia_data.mean() * 100
                st.markdown(f"""
                <div style="background-color: {PANTONE_CYAN}10; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: {PANTONE_CYAN}; margin: 0;">{anemia_rate:.1f}%</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">Women Anemia</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: #666; margin: 0;">N/A</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">Women Anemia</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: #666; margin: 0;">N/A</h4>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">Women Anemia</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        vit_a_cols = [col for col in child_data.columns if 'vitamin' in col.lower() or 'vita' in col.lower()]
        if vit_a_cols:
            vit_a_col = vit_a_cols[0]
            # Convert to numeric and handle errors
            child_data[vit_a_col] = pd.to_numeric(child_data[vit_a_col], errors='coerce')
            vit_a_data = child_data[vit_a_col].dropna()
            if len(vit_a_data) > 0:
                vit_a_coverage = vit_a_data.mean() * 100
                st.markdown(f"""
                <div style="background-color: {PANTONE_362C}10; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: {PANTONE_362C}; margin: 0;">{vit_a_coverage:.1f}%</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">Vitamin A Coverage</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: #666; margin: 0;">N/A</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">Vitamin A Coverage</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: #666; margin: 0;">N/A</h4>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">Vitamin A Coverage</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        ifa_cols = [col for col in women_data.columns if 'ifa' in col.lower() or 'iron' in col.lower()]
        if not ifa_cols:
            # Look for other supplementation columns
            ifa_cols = [col for col in women_data.columns if 'supplement' in col.lower() or 'received' in col.lower()]
        
        if ifa_cols:
            ifa_col = ifa_cols[0]
            # Convert string values to numeric
            women_data[ifa_col] = convert_to_numeric(women_data[ifa_col])
            ifa_data = women_data[ifa_col].dropna()
            if len(ifa_data) > 0:
                ifa_coverage = ifa_data.mean() * 100
                st.markdown(f"""
                <div style="background-color: {PANTONE_3965C}10; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: {PANTONE_3965C}; margin: 0;">{ifa_coverage:.1f}%</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">IFA Coverage</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="color: #666; margin: 0;">N/A</h4>
                    <p style="color: #666; margin: 0; font-size: 0.8rem;">IFA Coverage</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                <h4 style="color: #666; margin: 0;">N/A</h4>
                <p style="color: #666; margin: 0; font-size: 0.8rem;">IFA Coverage</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Main charts - Using Pantone color scheme
    st.markdown("### 📈 Key Trends & Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Wealth and nutrition gradient with Pantone colors
        wealth_cols = [col for col in merged_data.columns if 'wealth' in col.lower() or 'wi_' in col.lower()]
        if wealth_cols and 'stunting' in merged_data.columns:
            wealth_col = wealth_cols[0]
            wealth_stunting_data = merged_data[['stunting', wealth_col]].dropna()
            wealth_stunting_data = wealth_stunting_data[wealth_stunting_data['stunting'] != 'Unknown']
            
            if len(wealth_stunting_data) > 0:
                wealth_stunting = wealth_stunting_data.groupby(wealth_col).agg({
                    'stunting': lambda x: (x != 'Normal').mean() * 100
                }).reset_index()
                
                # Try to order wealth categories
                wealth_order = ['Poorest', 'Poor', 'Middle', 'Rich', 'Richest', 'Wealthiest', 'Lowest', 'Highest']
                existing_categories = [cat for cat in wealth_order if cat in wealth_stunting[wealth_col].values]
                
                if existing_categories:
                    wealth_stunting[wealth_col] = pd.Categorical(
                        wealth_stunting[wealth_col], 
                        categories=existing_categories, 
                        ordered=True
                    )
                    wealth_stunting = wealth_stunting.sort_values(wealth_col)

                fig = px.bar(
                    wealth_stunting, 
                    x=wealth_col, 
                    y='stunting',
                    title='📊 Stunting Prevalence by Wealth Quintile',
                    labels={
                        wealth_col: 'Wealth Quintile',
                        'stunting': 'Stunting Rate (%)'
                    },
                    color='stunting',
                    color_continuous_scale=SEQUENTIAL_BLUES,
                    text_auto='.1f'
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Wealth Quintile",
                    yaxis_title="Stunting Rate (%)",
                    yaxis_range=[0, max(wealth_stunting['stunting']) * 1.1] if len(wealth_stunting) > 0 else [0, 100],
                    font=dict(size=12),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                fig.update_traces(
                    textposition='outside',
                    marker_line_width=0,
                    marker_color=PANTONE_2728C
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No stunting data available for wealth analysis")
        else:
            st.info("Wealth index or stunting data not available for visualization")
    
    with col2:
        # Dietary diversity distribution with Pantone colors
        diet_cols = [col for col in hh_data.columns if 'diet' in col.lower() or 'fcs' in col.lower()]
        if diet_cols:
            diet_col = diet_cols[0]
            # Ensure numeric data
            hh_data[diet_col] = pd.to_numeric(hh_data[diet_col], errors='coerce')
            diet_data = hh_data[diet_col].dropna()
            
            if len(diet_data) > 0:
                fig = px.histogram(
                    diet_data, 
                    x=diet_col,
                    title='🥗 Household Dietary Diversity Distribution',
                    labels={
                        diet_col: 'Dietary Diversity Score',
                        'count': 'Number of Households'
                    },
                    nbins=10,
                    color_discrete_sequence=[PANTONE_362C]
                )
                
                # Add reference lines with Pantone colors
                fig.add_vline(x=4, line_dash="dash", line_color=PANTONE_3965C, 
                            annotation_text="Low Diversity", annotation_position="top")
                fig.add_vline(x=6, line_dash="dash", line_color=PANTONE_CYAN, 
                            annotation_text="Medium Diversity", annotation_position="top")
                fig.add_vline(x=8, line_dash="dash", line_color=PANTONE_362C, 
                            annotation_text="High Diversity", annotation_position="top")
                
                fig.update_layout(
                    height=400,
                    xaxis_title="Dietary Diversity Score",
                    yaxis_title="Number of Households",
                    showlegend=False,
                    font=dict(size=12),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No dietary diversity data available")
        else:
            st.info("Dietary diversity data not available for visualization")
    
    # Geographic and demographic patterns
    st.markdown("### 🗺️ Geographic & Demographic Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Urban-rural disparities with Pantone colors - FIXED VERSION
        urban_rural_col = None
        stunting_col = None

        # Check for possible column names for urban_rural
        urban_rural_candidates = ['urban_rural', 'UrbanRural', 'residence', 'urban_rural_child', 'urban_rural_hh', 'urban_rural_women']
        for col in urban_rural_candidates:
            if col in merged_data.columns:
                urban_rural_col = col
                break

        # Check for stunting data
        if 'stunting' in merged_data.columns:
            stunting_col = 'stunting'

        if urban_rural_col and stunting_col:
            # Clean urban_rural data
            urban_rural_data = merged_data[[urban_rural_col, stunting_col]].dropna()
            urban_rural_data = urban_rural_data[urban_rural_data[stunting_col] != 'Unknown']
            
            if len(urban_rural_data) > 0:
                urban_rural_data[urban_rural_col] = urban_rural_data[urban_rural_col].astype(str)
                
                # Standardize urban/rural values
                urban_rural_data[urban_rural_col] = urban_rural_data[urban_rural_col].str.replace(
                    '1', 'Urban'
                ).str.replace(
                    '2', 'Rural'
                ).str.replace(
                    'urban', 'Urban'
                ).str.replace(
                    'rural', 'Rural'
                ).str.title()
                
                urban_rural_stunting = urban_rural_data.groupby(urban_rural_col).agg({
                    stunting_col: lambda x: (x != 'Normal').mean() * 100
                }).reset_index()
                
                fig = px.bar(
                    urban_rural_stunting,
                    x=urban_rural_col,
                    y=stunting_col,
                    title='🏙️ vs 🏡 Stunting: Urban-Rural Disparities',
                    labels={
                        urban_rural_col: 'Residence',
                        stunting_col: 'Stunting Rate (%)'
                    },
                    color=urban_rural_col,
                    color_discrete_map={'Urban': PANTONE_CYAN, 'Rural': PANTONE_2728C},
                    text_auto='.1f'
                )
                
                fig.update_layout(
                    height=350,
                    showlegend=False,
                    xaxis_title="Residence Type",
                    yaxis_title="Stunting Rate (%)",
                    font=dict(size=12),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid stunting data available for urban-rural analysis")
        else:
            # Show what columns are available for debugging
            available_cols = [col for col in merged_data.columns if 'urban' in col.lower() or 'rural' in col.lower() or 'stunt' in col.lower()]
            st.info(f"Urban-rural or stunting data not available. Available relevant columns: {available_cols}")
            
            # Alternative visualization
            if urban_rural_col:
                # Show just urban-rural distribution
                urban_rural_dist = merged_data[urban_rural_col].value_counts()
                if len(urban_rural_dist) > 0:
                    fig = px.pie(
                        values=urban_rural_dist.values,
                        names=urban_rural_dist.index,
                        title='🏙️ Urban-Rural Distribution',
                        color_discrete_sequence=[PANTONE_CYAN, PANTONE_2728C]
                    )
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No urban-rural distribution data available")
            else:
                st.info("No urban-rural data available for visualization")
    
    with col2:
        # Age distribution with Pantone colors
        age_cols = [col for col in child_data.columns if 'age' in col.lower() and 'month' in col.lower()]
        if age_cols:
            age_col = age_cols[0]
            # Ensure numeric data
            child_data[age_col] = pd.to_numeric(child_data[age_col], errors='coerce')
            age_data = child_data[age_col].dropna()
            
            if len(age_data) > 0:
                age_groups = pd.cut(age_data, 
                                  bins=[0, 12, 24, 36, 48, 60],
                                  labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m'])
                
                age_dist = age_groups.value_counts().sort_index()
                
                # Create custom colors for pie chart using Pantone palette
                pie_colors = [PANTONE_2728C, PANTONE_CYAN, PANTONE_362C, PANTONE_3965C, "#4A4DA8"]
                
                fig = px.pie(
                    values=age_dist.values,
                    names=age_dist.index,
                    title='👶 Child Age Distribution',
                    color_discrete_sequence=pie_colors
                )
                
                fig.update_layout(
                    height=350,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No age data available for children")
        else:
            st.info("Age data not available for children")
    
    # Additional insights with Pantone-colored sections
    st.markdown("### 💡 Key Insights & Priorities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: {PANTONE_2728C}08; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {PANTONE_2728C};">
            <h4 style="color: {PANTONE_2728C}; margin-top: 0;">🔍 Key Findings</h4>
            <p style="color: #555; margin-bottom: 0.5rem;"><strong>Patterns Observed:</strong></p>
            <ul style="color: #555; margin-bottom: 0;">
                <li>Stunting rates show clear wealth gradient</li>
                <li>Rural areas have higher malnutrition burden</li>
                <li>Dietary diversity is a key constraint</li>
                <li>Maternal education impacts child nutrition</li>
                <li>Supplementation coverage needs improvement</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: {PANTONE_3965C}08; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {PANTONE_3965C};">
            <h4 style="color: {PANTONE_3965C}; margin-top: 0;">🎯 Priority Actions</h4>
            <p style="color: #555; margin-bottom: 0.5rem;"><strong>Immediate Focus Areas:</strong></p>
            <ul style="color: #555; margin-bottom: 0;">
                <li>High-stunting geographic clusters</li>
                <li>Households with low dietary diversity</li>
                <li>Areas with low supplementation coverage</li>
                <li>Communities with poor WASH facilities</li>
                <li>Vulnerable wealth quintiles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Data Preview Section ---
    with st.expander("📂 Dataset Previews", expanded=False):
        st.markdown("Successfully loaded previews of the core datasets. Full files are large, so these are chunked previews.")
        
        tab1, tab2, tab3 = st.tabs(["Households", "Children (6-59months)", "Women (15-49years)"])

        with tab1:
            st.markdown("#### Household Dataset – `CFSVA2024_HH.csv`")
            st.markdown("""
            Contains household-level variables such as food security indicators, dietary diversity, income, assets, WASH variables, shock exposure, livelihood types, and geographic codes.
            
            **This dataset is key for:**
            - ✔ Hotspot mapping
            - ✔ Building predictive models
            - ✔ Root-cause analysis
            """)
            st.dataframe(hh_data.head())

        with tab2:
            st.markdown("#### Children 6–59 Months Dataset – `CFSVA2024_HH_CHILD_6_59_MONTHS.csv`")
            st.markdown("""
            Contains child-level nutrition indicators, including anthropometry (stunting, wasting, underweight), dietary diversity, illness history, Vitamin A intake, and micronutrient deficiency proxies.
            
            **This dataset is essential for:**
            - ✔ Stunting analysis
            - ✔ Child malnutrition risk prediction
            - ✔ Linking household determinants to nutrition outcomes
            """)
            st.dataframe(child_data.head())

        with tab3:
            st.markdown("#### Women 15–49 Years Dataset – `CFSVA2024_HH_WOMEN_15_49_YEARS.csv`")
            st.markdown("""
            Includes variables such as maternal nutrition, IFA supplementation, education levels, food consumption, and dietary diversity.
            
            **This enables:**
            - ✔ Maternal nutrition analysis
            - ✔ Multi-generational malnutrition pathways
            - ✔ Root cause insights (education, diet, anemia risk)
            """)
            st.dataframe(women_data.head())

def convert_to_numeric(series):
    """Convert string series to numeric, handling 'Yes'/'No' and other common patterns"""
    # If already numeric, return as is
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Create a copy to avoid modifying original
    series_copy = series.copy()
    
    # Handle common string patterns
    # Replace common yes/no patterns
    series_copy = series_copy.replace({
        'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0,
        'yes': 1, 'no': 0, '1': 1, '0': 0,
        'True': 1, 'False': 0, 'TRUE': 1, 'FALSE': 0
    })
    
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(series_copy, errors='coerce')

def debug_data_issues(merged_data):
    """Debug function to identify data issues"""
    st.sidebar.markdown("### 🔧 Data Debug Info")
    
    # Check for urban-rural columns
    urban_cols = [col for col in merged_data.columns if 'urban' in col.lower()]
    rural_cols = [col for col in merged_data.columns if 'rural' in col.lower()]
    stunting_cols = [col for col in merged_data.columns if 'stunt' in col.lower()]
    
    st.sidebar.write(f"Urban columns: {urban_cols}")
    st.sidebar.write(f"Rural columns: {rural_cols}")
    st.sidebar.write(f"Stunting columns: {stunting_cols}")
    
    # Check sample values
    if urban_cols:
        sample_urban = merged_data[urban_cols[0]].dropna().unique()[:5]
        st.sidebar.write(f"Sample urban values: {sample_urban}")
    
    if stunting_cols:
        sample_stunting = merged_data[stunting_cols[0]].dropna().unique()[:5]
        st.sidebar.write(f"Sample stunting values: {sample_stunting}")

def load_all_data():
    """Load all three datasets and merge them"""
    try:
        # Load household data
        hh_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH data.csv")
        
        # Load child data
        child_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv")
        
        # Load women data
        women_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv")
        
        # Column renaming and processing
        if 'WI_cat' in hh_data.columns:
            hh_data = hh_data.rename(columns={'WI_cat': 'wealth_index'})
            hh_data['wealth_index'] = hh_data['wealth_index'].replace({
                'Wealth': 'Rich',
                'Wealthiest': 'Richest'
            })
        if 'FCS' in hh_data.columns:
            hh_data = hh_data.rename(columns={'FCS': 'dietary_diversity_score'})
        if 'UrbanRural' in hh_data.columns:
            hh_data = hh_data.rename(columns={'UrbanRural': 'urban_rural'})
        if 'S0_C_Prov' in hh_data.columns:
            hh_data = hh_data.rename(columns={'S0_C_Prov': 'province'})
        if 'S0_D_Dist' in hh_data.columns:
            hh_data = hh_data.rename(columns={'S0_D_Dist': 'district'})

        # For child data
        if 'S0_C_Prov' in child_data.columns:
            child_data = child_data.rename(columns={'S0_C_Prov': 'province'})
        if 'S0_D_Dist' in child_data.columns:
            child_data = child_data.rename(columns={'S0_D_Dist': 'district'})
        if 'UrbanRural' in child_data.columns:
            child_data = child_data.rename(columns={'UrbanRural': 'urban_rural'})
        if 'S13_01' in child_data.columns:
            child_data = child_data.rename(columns={'S13_01': 'age_months'})
        if 'S13_04_01' in child_data.columns:
            child_data = child_data.rename(columns={'S13_04_01': 'height_cm'})
        if 'S13_05' in child_data.columns:
            child_data = child_data.rename(columns={'S13_05': 'weight_kg'})
        if 'S13_01_3' in child_data.columns:
            child_data = child_data.rename(columns={'S13_01_3': 'sex'})
        if 'S13_07' in child_data.columns:
            child_data = child_data.rename(columns={'S13_07': 'received_vitamin_a'})
            # Use the new conversion function
            child_data['received_vitamin_a'] = convert_to_numeric(child_data['received_vitamin_a'])
        if 'S13_20' in child_data.columns:
            child_data = child_data.rename(columns={'S13_20': 'dietary_diversity'})

        # For women data
        if 'S12_01_4' in women_data.columns:
            women_data = women_data.rename(columns={'S12_01_4': 'age_years'})
        if 'S12_05' in women_data.columns:
            women_data = women_data.rename(columns={'S12_05': 'education_level'})
        if 'S12_03' in women_data.columns:
            women_data = women_data.rename(columns={'S12_03': 'pregnant'})
            women_data['pregnant'] = convert_to_numeric(women_data['pregnant'])
        if 'S12_11' in women_data.columns:
            women_data = women_data.rename(columns={'S12_11': 'breastfeeding'})
            women_data['breastfeeding'] = convert_to_numeric(women_data['breastfeeding'])
        if 'S12_15' in women_data.columns:
            women_data = women_data.rename(columns={'S12_15': 'received_ifa'})
            women_data['received_ifa'] = convert_to_numeric(women_data['received_ifa'])
        if 'S12_12' in women_data.columns:
            women_data = women_data.rename(columns={'S12_12': 'bmi'})
        if 'MDDWLess5' in women_data.columns:
            women_data['dietary_diversity'] = women_data['MDDWLess5'].map({'<5 food groups': 3, '5 food groups or more': 6})
        if 'S12_14_1' in women_data.columns:
            women_data = women_data.rename(columns={'S12_14_1': 'anemic'})
            women_data['anemic'] = convert_to_numeric(women_data['anemic'])

        return hh_data, child_data, women_data

    except Exception as e:
        st.error(f"Error loading data files: {e}")
        st.info("Using sample data for demonstration")
        return create_sample_data()

def merge_datasets(hh_data, child_data, women_data):
    """Merge all three datasets with improved stunting calculation"""
    # Merge child data with household data
    merged_data = child_data.merge(
        hh_data, 
        on='___index', 
        how='left',
        suffixes=('_child', '_hh')
    )
    
    # Merge with women data
    merged_data = merged_data.merge(
        women_data,
        on='___index',
        how='left',
        suffixes=('', '_women')
    )
    
    # Create stunting classification if we have the data
    if 'stunting_zscore' in merged_data.columns:
        def classify_stunting(zscore):
            if pd.isna(zscore):
                return 'Unknown'
            elif zscore < -3:
                return 'Severe'
            elif zscore < -2:
                return 'Moderate'
            else:
                return 'Normal'
        
        merged_data['stunting'] = merged_data['stunting_zscore'].apply(classify_stunting)
    
    # Create other malnutrition risk indicators
    malnutrition_indicators = []
    
    if 'stunting' in merged_data.columns:
        merged_data['stunting_risk'] = (merged_data['stunting'] != 'Normal').astype(int)
        malnutrition_indicators.append('stunting_risk')
    
    if 'wasting' in merged_data.columns:
        merged_data['wasting_risk'] = (merged_data['wasting'] != 'Normal').astype(int)
        malnutrition_indicators.append('wasting_risk')
    elif 'wasting_zscore' in merged_data.columns:
        # Create wasting from zscore
        def classify_wasting(zscore):
            if pd.isna(zscore):
                return 'Unknown'
            elif zscore < -3:
                return 'Severe'
            elif zscore < -2:
                return 'Moderate'
            else:
                return 'Normal'
        merged_data['wasting'] = merged_data['wasting_zscore'].apply(classify_wasting)
        merged_data['wasting_risk'] = (merged_data['wasting'] != 'Normal').astify(int)
        malnutrition_indicators.append('wasting_risk')
    
    # Create any malnutrition indicator
    if malnutrition_indicators:
        merged_data['any_malnutrition'] = merged_data[malnutrition_indicators].max(axis=1)
    
    return merged_data

def create_sample_data():
    """Create comprehensive sample data for all three datasets"""
    np.random.seed(42)
    n_households = 5000
    
    # Household data
    hh_data = pd.DataFrame({
        '___index': [f'HH_{i:04d}' for i in range(n_households)],
        'province': np.random.choice(['Kigali', 'North', 'South', 'East', 'West'], n_households),
        'district': np.random.choice(['Gasabo', 'Nyarugenge', 'Kicukiro', 'Musanze', 'Huye'], n_households),
        'urban_rural': np.random.choice(['Urban', 'Rural'], n_households, p=[0.3, 0.7]),
        'wealth_index': np.random.choice(['Poorest', 'Poor', 'Middle', 'Rich', 'Richest'], n_households),
        'food_consumption_score': np.random.randint(10, 80, n_households),
        'dietary_diversity_score': np.random.randint(0, 10, n_households),
        'coping_strategy_index': np.random.randint(0, 40, n_households),
        'has_improved_water': np.random.choice([0, 1], n_households, p=[0.3, 0.7]),
        'has_improved_sanitation': np.random.choice([0, 1], n_households, p=[0.4, 0.6]),
        'hh_size': np.random.randint(1, 10, n_households),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_households),
        'food_insecure': np.random.choice([0, 1], n_households, p=[0.7, 0.3])
    })
    
    # Child data (multiple children per household)
    child_data = []
    for hh_id in hh_data['___index']:
        n_children = np.random.randint(0, 3)  # 0-2 children per household
        for i in range(n_children):
            child_data.append({
                '___index': hh_id,
                'child_id': f"{hh_id}_C{i}",
                'age_months': np.random.randint(6, 60),
                'sex': np.random.choice(['Male', 'Female']),
                'height_cm': np.random.normal(85, 10),
                'weight_kg': np.random.normal(12, 3),
                'stunting_zscore': np.random.normal(-1, 1.5),
                'wasting_zscore': np.random.normal(-0.5, 1),
                'underweight_zscore': np.random.normal(-0.8, 1.2),
                'received_vitamin_a': np.random.choice([0, 1], p=[0.4, 0.6]),
                'dietary_diversity': np.random.randint(0, 8),
                'had_diarrhea': np.random.choice([0, 1], p=[0.7, 0.3]),
                'had_fever': np.random.choice([0, 1], p=[0.6, 0.4])
            })
    child_data = pd.DataFrame(child_data)
    
    # Women data
    women_data = []
    for hh_id in hh_data['___index']:
        n_women = np.random.randint(0, 2)  # 0-1 women per household
        for i in range(n_women):
            women_data.append({
                '___index': hh_id,
                'woman_id': f"{hh_id}_W{i}",
                'age_years': np.random.randint(15, 50),
                'education_level': np.random.choice(['None', 'Primary', 'Secondary', 'Higher']),
                'pregnant': np.random.choice([0, 1], p=[0.8, 0.2]),
                'breastfeeding': np.random.choice([0, 1], p=[0.6, 0.4]),
                'received_ifa': np.random.choice([0, 1], p=[0.5, 0.5]),
                'bmi': np.random.normal(22, 3),
                'dietary_diversity': np.random.randint(0, 10),
                'anemic': np.random.choice([0, 1], p=[0.7, 0.3])
            })
    women_data = pd.DataFrame(women_data)
    
    return hh_data, child_data, women_data