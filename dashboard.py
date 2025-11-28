import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# Configure the page
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-change {
        font-size: 0.8rem;
        color: #00aa00;
        font-weight: 600;
    }
    .metric-change.negative {
        color: #ff4b4b;
    }
    .section-header {
        font-size: 1.3rem;
        color: #1f77b4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# LOGOTYPE")
    st.markdown("---")
    st.markdown("## HOME")
    st.markdown("**HOME > DASHBOARD**")
    
    # Navigation
    st.markdown("---")
    st.markdown("### Navigation")
    menu_options = ["🏠 Dashboard Overview",
        "🗺️ Micronutrient Hotspots",
        "🔮 Risk Prediction",
        "📊 Root Cause Analysis",
        "👶 Child Nutrition",
        "👩 Women's Health",
        "📋 Policy Recommendations"]
    for option in menu_options:
        st.button(option, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Health Care")
    st.info("Weather Updates: Sunny, 72°F")

# Main content area
st.markdown('<div class="main-header">Analytics Dashboard</div>', unsafe_allow_html=True)

# First Row: Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">TOTAL TRAFFIC</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">325,456</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">+5% SINCE LAST MONTH</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">NEW USERS</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">3,006</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">+4.54% SINCE LAST MONTH</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">60%</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">+2.54% SINCE LAST MONTH</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">SALES</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">852</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">+6.54% SINCE LAST MONTH</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Second Row: Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-header">PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    
    # Performance Chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create sample data for performance chart
    categories = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    values = [45, 52, 48, 58, 55, 60, 65, 62, 68, 72, 75, 78]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=categories,
        y=values,
        mode='lines+markers',
        name='Performance',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Performance Over Time",
        xaxis_title="Months",
        yaxis_title="Performance %",
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">USER ANALYTICS</div>', unsafe_allow_html=True)
    
    # User Analytics Chart
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create sample data for user analytics
    user_categories = ['New Users', 'Returning', 'Active', 'Inactive']
    user_values = [3006, 2450, 1890, 856]
    
    fig = px.pie(
        values=user_values,
        names=user_categories,
        title="User Distribution"
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Third Row: Additional Metrics
st.markdown('<div class="section-header">DETAILED METRICS</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">PERSONAGE</div>', unsafe_allow_html=True)
    
    # Create a simple bar chart for personage
    fig = go.Figure(data=[
        go.Bar(
            x=['Q1', 'Q2', 'Q3', 'Q4'],
            y=[1200, 1900, 3000, 2500],
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
    ])
    
    fig.update_layout(
        title="Quarterly Personage",
        height=250,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">TOTAL ORDERS</div>', unsafe_allow_html=True)
    
    # Create a line chart for total orders
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    orders = [850, 920, 780, 1100, 1050, 1200]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=orders,
        mode='lines+markers',
        name='Orders',
        line=dict(color='#2ca02c', width=3)
    ))
    
    fig.update_layout(
        title="Monthly Orders",
        height=250,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">UPDATES</div>', unsafe_allow_html=True)
    
    # Create update metrics
    updates_data = {
        'Type': ['System', 'Security', 'Features', 'Maintenance'],
        'Count': [45, 23, 67, 12]
    }
    
    fig = px.bar(
        updates_data,
        x='Type',
        y='Count',
        color='Type',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    )
    
    fig.update_layout(
        title="Recent Updates",
        height=250,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2024 Analytics Dashboard. All rights reserved.")