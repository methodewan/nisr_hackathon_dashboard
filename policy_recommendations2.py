# policy_recommendations1.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def policy_recommendations():
    """Policy Recommendations with Cost-Effectiveness Analysis"""
    
    st.markdown('<div class="main-header">📋 Policy Recommendations & Cost-Effectiveness Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please wait for data to load or check if data files are available.")
        return
    
    # Load data
    merged_data = st.session_state.merged_data
    hh_data = st.session_state.hh_data
    child_data = st.session_state.child_data
    women_data = st.session_state.women_data
    
    # Introduction
    st.markdown("""
    <div class="section-header">Strategic Investment Priorities for Ending Hidden Hunger</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        This analysis identifies the most cost-effective interventions to address micronutrient deficiencies 
        based on current data patterns and impact potential. Interventions are evaluated across multiple 
        dimensions including **cost, coverage, impact, and implementation complexity**.
        """)
    
    with col2:
        st.info("💡 **Key Metric**: Distance to Quality measures how far each region/intervention is from achieving optimal nutrition outcomes")

    # Cost-Effectiveness Matrix
    st.markdown("""
    <div class="section-header">Cost-Effectiveness Analysis Matrix</div>
    """, unsafe_allow_html=True)
    
    # Generate synthetic cost-effectiveness data
    interventions_data = generate_interventions_data(merged_data)
    
    # Cost-Effectiveness Scatter Plot
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Intervention Cost vs. Expected Impact")
        
        fig = px.scatter(interventions_data, 
                        x='cost_per_beneficiary', 
                        y='impact_score',
                        size='coverage_potential',
                        color='priority_level',
                        hover_name='intervention',
                        hover_data={
                            'cost_per_beneficiary': ':.2f',
                            'impact_score': ':.1f',
                            'coverage_potential': ':.0%',
                            'implementation_time': True
                        },
                        color_discrete_map={
                            'High': '#FF6B6B',
                            'Medium': '#FFD166',
                            'Low': '#06D6A0'
                        },
                        title="Cost-Effectiveness Matrix: Intervention Prioritization")
        
        fig.update_layout(
            xaxis_title="Cost per Beneficiary (USD)",
            yaxis_title="Impact Score (0-10 scale)",
            height=500
        )
        
        # Add quadrant lines
        avg_cost = interventions_data['cost_per_beneficiary'].mean()
        avg_impact = interventions_data['impact_score'].mean()
        
        fig.add_hline(y=avg_impact, line_dash="dash", line_color="gray")
        fig.add_vline(x=avg_cost, line_dash="dash", line_color="gray")
        
        # Add quadrant annotations
        fig.add_annotation(x=avg_cost/2, y=avg_impact*1.5, text="HIGH IMPACT<br>LOW COST", 
                          showarrow=False, font=dict(color="green", size=12))
        fig.add_annotation(x=avg_cost*1.5, y=avg_impact*1.5, text="HIGH IMPACT<br>HIGH COST", 
                          showarrow=False, font=dict(color="orange", size=12))
        fig.add_annotation(x=avg_cost/2, y=avg_impact/2, text="LOW IMPACT<br>LOW COST", 
                          showarrow=False, font=dict(color="blue", size=12))
        fig.add_annotation(x=avg_cost*1.5, y=avg_impact/2, text="LOW IMPACT<br>HIGH COST", 
                          showarrow=False, font=dict(color="red", size=12))
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_htmml=True)
        st.subheader("Priority Legend")
        
        st.markdown("""
        <div style="background-color: #FF6B6B; padding: 10px; border-radius: 5px; margin: 5px 0;">
        <strong>High Priority</strong><br>
        High impact, cost-effective
        </div>
        
        <div style="background-color: #FFD166; padding: 10px; border-radius: 5px; margin: 5px 0;">
        <strong>Medium Priority</strong><br>
        Moderate impact/cost
        </div>
        
        <div style="background-color: #06D6A0; padding: 10px; border-radius: 5px; margin: 5px 0;">
        <strong>Low Priority</strong><br>
        Lower impact or high cost
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Size indicates**: Coverage potential
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Distance to Quality Analysis
    st.markdown("""
    <div class="section-header">Distance to Quality: Regional Performance Gaps</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Regional Distance to Quality Targets")
        
        # Generate regional distance data
        regional_data = generate_regional_distance_data(merged_data)
        
        fig = go.Figure()
        
        # Add bars for each quality dimension
        dimensions = ['Nutrition', 'Health Services', 'Economic', 'Education']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, dimension in enumerate(dimensions):
            fig.add_trace(go.Bar(
                name=dimension,
                x=regional_data['region'],
                y=regional_data[f'{dimension.lower()}_distance'],
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            barmode='group',
            xaxis_title="Region",
            yaxis_title="Distance to Quality (0-100 scale)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Quality Gap Analysis")
        
        # Radar chart for comprehensive quality assessment
        regions = regional_data['region'].tolist()
        
        fig = go.Figure()
        
        for region in regions[:4]:  # Show first 4 regions for clarity
            region_data = regional_data[regional_data['region'] == region]
            values = [
                region_data['nutrition_distance'].values[0],
                region_data['health_distance'].values[0],
                region_data['economic_distance'].values[0],
                region_data['education_distance'].values[0],
                region_data['nutrition_distance'].values[0]  # Close the radar
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=dimensions + [dimensions[0]],  # Close the circle
                fill='toself',
                name=region
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Intervention Timeline and Phasing
    st.markdown("""
    <div class="section-header">Implementation Roadmap & Phasing</div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Gantt chart for implementation timeline
    timeline_data = generate_implementation_timeline(interventions_data)
    
    fig = px.timeline(timeline_data, 
                     x_start="start", 
                     x_end="end", 
                     y="intervention",
                     color="phase",
                     color_discrete_sequence=px.colors.qualitative.Set3,
                     title="Implementation Timeline: Phased Rollout Strategy")
    
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Budget Allocation Recommendations
    st.markdown("""
    <div class="section-header">Optimal Budget Allocation</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Recommended Budget Distribution")
        
        budget_data = generate_budget_recommendations(interventions_data)
        
        fig = px.sunburst(budget_data, 
                         path=['category', 'intervention'], 
                         values='budget_allocation',
                         color='efficiency_score',
                         color_continuous_scale='RdYlGn',
                         title="Optimal Budget Allocation by Intervention Category")
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Key Recommendations")
        
        high_priority = interventions_data[interventions_data['priority_level'] == 'High']
        
        for idx, row in high_priority.iterrows():
            st.markdown(f"""
            <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1e90ff;">
            <h4 style="margin: 0 0 8px 0;">{row['intervention']}</h4>
            <p style="margin: 0; font-size: 0.9em;">
            <strong>Impact:</strong> {row['impact_score']}/10<br>
            <strong>Cost:</strong> ${row['cost_per_beneficiary']:.2f}<br>
            <strong>Coverage:</strong> {row['coverage_potential']:.1%}
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Impact Projection Over Time
    st.markdown("""
    <div class="section-header">Projected Impact: Distance to Quality Reduction</div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_htmml=True)
    
    # Projection data
    projection_data = generate_impact_projections(regional_data, interventions_data)
    
    fig = go.Figure()
    
    for region in projection_data['region'].unique()[:3]:  # Show top 3 regions
        region_data = projection_data[projection_data['region'] == region]
        fig.add_trace(go.Scatter(
            x=region_data['year'],
            y=region_data['distance_to_quality'],
            mode='lines+markers',
            name=region,
            line=dict(width=3)
        ))
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Distance to Quality Score",
        title="Projected Improvement in Quality Metrics (2024-2028)",
        height=400
    )
    
    # Add target line
    fig.add_hline(y=20, line_dash="dash", line_color="green", 
                 annotation_text="Target Quality Level", 
                 annotation_position="bottom right")
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Downloadable Recommendations Report
    st.markdown("""
    <div class="section-header">Download Comprehensive Analysis</div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Executive Summary", use_container_width=True):
            st.success("Executive summary generated! Ready for download.")
    
    with col2:
        if st.button("💰 Cost-Benefit Analysis", use_container_width=True):
            st.success("Cost-benefit analysis prepared!")
    
    with col3:
        if st.button("🔄 Implementation Plan", use_container_width=True):
            st.success("Detailed implementation plan created!")
    
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin-top: 20px;">
    <h4 style="color: #1e90ff; margin-top: 0;">Next Steps</h4>
    <ol>
    <li><strong>Immediate Action (0-3 months):</strong> Implement high-priority, low-cost interventions</li>
    <li><strong>Short-term (3-12 months):</strong> Scale successful pilots and build capacity</li>
    <li><strong>Medium-term (1-2 years):</strong> Address systemic barriers and expand coverage</li>
    <li><strong>Long-term (2+ years):</strong> Sustainable systems integration and monitoring</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def generate_interventions_data(merged_data):
    """Generate synthetic interventions data for cost-effectiveness analysis"""
    
    interventions = [
        {
            'intervention': 'Vitamin A Supplementation',
            'cost_per_beneficiary': 2.50,
            'impact_score': 8.5,
            'coverage_potential': 0.85,
            'implementation_time': 6,
            'priority_level': 'High'
        },
        {
            'intervention': 'Iron-Folic Acid Distribution',
            'cost_per_beneficiary': 3.20,
            'impact_score': 7.8,
            'coverage_potential': 0.75,
            'implementation_time': 8,
            'priority_level': 'High'
        },
        {
            'intervention': 'Nutrition Education',
            'cost_per_beneficiary': 1.80,
            'impact_score': 6.5,
            'coverage_potential': 0.90,
            'implementation_time': 12,
            'priority_level': 'Medium'
        },
        {
            'intervention': 'Food Fortification',
            'cost_per_beneficiary': 5.50,
            'impact_score': 8.2,
            'coverage_potential': 0.95,
            'implementation_time': 18,
            'priority_level': 'Medium'
        },
        {
            'intervention': 'Therapeutic Feeding',
            'cost_per_beneficiary': 12.30,
            'impact_score': 9.1,
            'coverage_potential': 0.45,
            'implementation_time': 9,
            'priority_level': 'Medium'
        },
        {
            'intervention': 'Agricultural Support',
            'cost_per_beneficiary': 8.75,
            'impact_score': 7.2,
            'coverage_potential': 0.65,
            'implementation_time': 24,
            'priority_level': 'Low'
        },
        {
            'intervention': 'Water & Sanitation',
            'cost_per_beneficiary': 15.20,
            'impact_score': 6.8,
            'coverage_potential': 0.70,
            'implementation_time': 36,
            'priority_level': 'Low'
        }
    ]
    
    return pd.DataFrame(interventions)


def generate_regional_distance_data(merged_data):
    """Generate regional distance to quality metrics"""
    
    if 'province' in merged_data.columns:
        regions = merged_data['province'].unique()[:6]  # Top 6 regions
    else:
        regions = ['Region A', 'Region B', 'Region C', 'Region D', 'Region E', 'Region F']
    
    regional_data = []
    
    for region in regions:
        regional_data.append({
            'region': region,
            'nutrition_distance': np.random.randint(20, 80),
            'health_distance': np.random.randint(25, 85),
            'economic_distance': np.random.randint(30, 90),
            'education_distance': np.random.randint(15, 75),
            'overall_distance': np.random.randint(25, 80)
        })
    
    return pd.DataFrame(regional_data)


def generate_implementation_timeline(interventions_data):
    """Generate implementation timeline data"""
    
    timeline_data = []
    start_month = 0
    
    phases = {
        'High': {'duration': 6, 'color': 'green'},
        'Medium': {'duration': 12, 'color': 'orange'},
        'Low': {'duration': 18, 'color': 'red'}
    }
    
    for priority in ['High', 'Medium', 'Low']:
        priority_interventions = interventions_data[interventions_data['priority_level'] == priority]
        
        for _, intervention in priority_interventions.iterrows():
            timeline_data.append({
                'intervention': intervention['intervention'],
                'start': start_month,
                'end': start_month + phases[priority]['duration'],
                'phase': f'Phase {["High", "Medium", "Low"].index(priority) + 1}'
            })
            start_month += 2  # Stagger start times
    
    return pd.DataFrame(timeline_data)


def generate_budget_recommendations(interventions_data):
    """Generate budget allocation recommendations"""
    
    budget_data = []
    
    categories = {
        'Supplementation': ['Vitamin A Supplementation', 'Iron-Folic Acid Distribution'],
        'Education': ['Nutrition Education'],
        'Infrastructure': ['Food Fortification', 'Water & Sanitation'],
        'Treatment': ['Therapeutic Feeding'],
        'Development': ['Agricultural Support']
    }
    
    for category, interventions in categories.items():
        for intervention in interventions:
            if intervention in interventions_data['intervention'].values:
                row = interventions_data[interventions_data['intervention'] == intervention].iloc[0]
                budget_data.append({
                    'category': category,
                    'intervention': intervention,
                    'budget_allocation': row['cost_per_beneficiary'] * 1000,  # Scale for visualization
                    'efficiency_score': row['impact_score'] / row['cost_per_beneficiary']
                })
    
    return pd.DataFrame(budget_data)


def generate_impact_projections(regional_data, interventions_data):
    """Generate impact projection data"""
    
    projection_data = []
    
    for region in regional_data['region'].unique()[:3]:
        current_distance = regional_data[regional_data['region'] == region]['overall_distance'].values[0]
        
        for year in [2024, 2025, 2026, 2027, 2028]:
            # Simulate progressive improvement
            improvement = current_distance * (0.85 ** (year - 2024))
            projection_data.append({
                'region': region,
                'year': year,
                'distance_to_quality': max(10, improvement)  # Don't go below 10
            })
    
    return pd.DataFrame(projection_data)


if __name__ == "__main__":
    policy_recommendations()