import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------------------------------
# CUSTOM CSS FOR SNAF-RWANDA STYLE DASHBOARD
# ---------------------------------------------------
def load_custom_css():
    st.markdown("""
        <style>
            /* GLOBAL FONT + BACKGROUND */
            body, .main, .block-container {
                background-color: #f0f0f1 !important;
                font-family: "Inter", sans-serif;
            }

            /* SIDEBAR */
            section[data-testid="stSidebar"] {
                background-color: #0c2740 !important;
                padding: 20px;
            }
            section[data-testid="stSidebar"] * {
                color: #fff !important;
                font-size: 15px;
            }

            /* MAIN HEADER */
            .main-header {
                font-size: 32px;
                font-weight: 700;
                padding: 10px 0 10px 0;
                color: #50575e;
            }

            /* METRIC CARDS */
            .metric-card {
                padding: 15px;
                border-radius: 10px;
                background-color: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                text-align: left;
                border: 1px solid #eee;
            }
            
            .metric-title {
                font-size: 15px;
                color: #6c7a89;
            }
            
            .metric-value {
                font-size: 32px;
                font-weight: 700;
                margin-top: 6px;
                color: #333;
            }

            /* COLORED METRICS */
            .card-blue { background-color:#e8f1ff !important; }
            .card-green { background-color:#e7ffef !important; }
            .card-purple { background-color:#f4eaff !important; }
            .card-yellow { background-color:#fff9dd !important; }

            /* ROUNDED PANELS */
            .panel {
                padding: 25px;
                background-color: white;
                border-radius: 18px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.05);
                margin-bottom: 25px;
            }
        </style>
    """, unsafe_allow_html=True)


def find_column(df, keywords):
    """Find column by keywords (case-insensitive partial match)"""
    if df is None:
        return None
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                return col
    return None


def find_numeric_binary_column(df, keywords, min_ratio=0.1, max_ratio=0.9):
    """Find binary numeric column with given keywords"""
    if df is None:
        return None
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                if df[col].dtype in ['int64', 'float64']:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 2:
                        ratio = df[col].mean()
                        if min_ratio < ratio < max_ratio:
                            return col
    return None


def dashboard_overview():
    """Main dashboard overview with dynamic column detection"""
    load_custom_css()

    st.markdown('<div class="main-header">🥦 Hidden Hunger Dashboard</div>', unsafe_allow_html=True)

    # Data loading
    if not st.session_state.data_loaded:
        st.warning("Data not loaded. Please check sidebar.")
        return

    hh_data = st.session_state.hh_data
    child_data = st.session_state.child_data
    women_data = st.session_state.women_data
    merged_data = st.session_state.merged_data

    # Debug: Show available columns
    with st.expander("📋 Column Inspector (for debugging)", expanded=False):
        st.write("**Child Data Columns:**")
        st.write(child_data.columns.tolist()[:20] if child_data is not None else [])
        
        st.write("**HH Data Columns:**")
        st.write(hh_data.columns.tolist()[:20] if hh_data is not None else [])
        
        st.write("**Women Data Columns:**")
        st.write(women_data.columns.tolist()[:20] if women_data is not None else [])

    # ---------------------------------------------------
    # METRICS ROW 1
    # ---------------------------------------------------
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.markdown('<div class="metric-card card-blue">', unsafe_allow_html=True)
        st.markdown("**Households Surveyed**")
        st.metric("", f"{len(hh_data):,}" if hh_data is not None else "0")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="metric-card card-green">', unsafe_allow_html=True)
        st.markdown("**Children (6-59m)**")
        st.metric("", f"{len(child_data):,}" if child_data is not None else "0")
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="metric-card card-purple">', unsafe_allow_html=True)
        st.markdown("**Women (15-49y)**")
        st.metric("", f"{len(women_data):,}" if women_data is not None else "0")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------
    # METRICS ROW 2 - Dynamic columns
    # ---------------------------------------------------
    
    with c4:
        st.markdown('<div class="metric-card card-green">', unsafe_allow_html=True)
        st.markdown("**Stunting Rate**")
        
        # Try to find stunting column
        stunting_col = find_column(merged_data, ['stunting', 'stunt', 'HAZ', 'height_for_age'])
        
        if stunting_col:
            try:
                # Handle different data types
                if merged_data[stunting_col].dtype == 'object':
                    # Text data - count non-normal
                    stunting_rate = (merged_data[stunting_col] != 'Normal').sum() / len(merged_data) * 100
                else:
                    # Numeric data - check if binary or continuous
                    if len(merged_data[stunting_col].unique()) <= 2:
                        stunting_rate = merged_data[stunting_col].mean() * 100
                    else:
                        # Use -2 threshold for Z-scores
                        stunting_rate = (merged_data[stunting_col] < -2).sum() / len(merged_data) * 100
                
                st.metric("", f"{stunting_rate:.1f}%")
            except Exception as e:
                st.metric("", "Error")
                st.caption(f"Column: {stunting_col}")
        else:
            st.metric("", "N/A")
            st.caption("No stunting column found")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        st.markdown('<div class="metric-card card-green">', unsafe_allow_html=True)
        st.markdown("**Women Anemia**")
        
        anemia_col = find_numeric_binary_column(women_data, ['anemic', 'anemia', 'HGB', 'hemoglobin'])
        
        if anemia_col:
            try:
                anemia_rate = women_data[anemia_col].mean() * 100
                st.metric("", f"{anemia_rate:.1f}%")
            except:
                st.metric("", "N/A")
        else:
            st.metric("", "N/A")
            st.caption("No anemia column found")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="metric-card card-green">', unsafe_allow_html=True)
        st.markdown("**IFA Coverage**")
        
        ifa_col = find_numeric_binary_column(women_data, ['ifa', 'iron', 'folic', 'supplementation'], 
                                            min_ratio=0.1, max_ratio=0.9)
        
        if ifa_col:
            try:
                ifa_rate = women_data[ifa_col].mean() * 100
                st.metric("", f"{ifa_rate:.1f}%")
            except:
                st.metric("", "N/A")
        else:
            st.metric("", "N/A")
            st.caption("No IFA column found")
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------
    # CHARTS ROW
    # ---------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 📊 Stunting by Wealth Index")
        
        wealth_col = find_column(merged_data, ['wealth', 'wealth_index'])
        stunting_col = find_column(merged_data, ['stunting', 'stunt', 'HAZ'])
        
        if wealth_col and stunting_col:
            try:
                if merged_data[stunting_col].dtype == 'object':
                    stunting_by_wealth = merged_data.groupby(wealth_col).apply(
                        lambda x: (x[stunting_col] != 'Normal').sum() / len(x) * 100
                    ).reset_index()
                    stunting_by_wealth.columns = [wealth_col, 'stunting_rate']
                else:
                    if len(merged_data[stunting_col].unique()) <= 2:
                        stunting_by_wealth = merged_data.groupby(wealth_col)[stunting_col].mean().reset_index()
                        stunting_by_wealth.columns = [wealth_col, 'stunting_rate']
                        stunting_by_wealth['stunting_rate'] *= 100
                    else:
                        stunting_by_wealth = merged_data.groupby(wealth_col).apply(
                            lambda x: (x[stunting_col] < -2).sum() / len(x) * 100
                        ).reset_index()
                        stunting_by_wealth.columns = [wealth_col, 'stunting_rate']
                
                fig = px.bar(
                    stunting_by_wealth,
                    x=wealth_col,
                    y='stunting_rate',
                    title='Stunting Prevalence by Wealth Index',
                    color='stunting_rate',
                    color_continuous_scale='Reds',
                    labels={wealth_col: 'Wealth Index', 'stunting_rate': 'Rate (%)'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting: {str(e)}")
        else:
            st.info(f"Missing columns - Wealth: {wealth_col}, Stunting: {stunting_col}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🥘 Dietary Diversity Distribution")
        
        diversity_col = find_column(hh_data, ['dietary', 'diversity', 'diet', 'DDS'])
        
        if diversity_col:
            try:
                fig = px.histogram(
                    hh_data,
                    x=diversity_col,
                    title='Household Dietary Diversity Distribution',
                    nbins=12,
                    labels={diversity_col: 'Dietary Diversity Score'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting: {str(e)}")
        else:
            st.info("Dietary diversity data not available")
        
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ---------------------------------------------------
    # INSIGHTS ROW
    # ---------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 💡 Key Insights")
        st.info("""
        ✓ Stunting rates highest in poorest wealth groups  
        ✓ Rural areas show higher malnutrition prevalence  
        ✓ Low dietary diversity linked to higher deficiencies  
        ✓ Women's education improves nutrition outcomes  
        ✓ WASH access critical for child health
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### 🎯 Priority Areas")
        st.warning("""
        ⚠️ Interventions needed:  
        • High-stunting districts and wealth groups  
        • Low dietary diversity households  
        • Limited supplementation coverage  
        • Poor WASH and sanitation communities  
        • Areas with limited healthcare access
        """)
        st.markdown('</div>', unsafe_allow_html=True)