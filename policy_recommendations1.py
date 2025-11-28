import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import itertools

def policy_recommendations():
    """Data-driven policy recommendations with impact prediction"""
    st.markdown('<div class="main-header">Policy Recommendations & Interventions</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    
    # ML-powered policy optimization
    st.markdown("### AI-Powered Policy Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        policy_goal = st.selectbox("Primary Policy Goal", [
            "Reduce Stunting", "Decrease Anemia", "Improve Dietary Diversity",
            "Enhance Supplement Coverage", "Reduce Wealth Disparities"
        ])
    
    with col2:
        budget_constraint = st.selectbox("Budget Level", [
            "Low Budget", "Medium Budget", "High Budget", "Unconstrained"
        ])
    
    if st.button("🚀 Generate Optimized Policy Package"):
        with st.spinner("Optimizing policy recommendations..."):
            policy_package = generate_optimized_policies(merged_data, policy_goal, budget_constraint)
            display_policy_recommendations(policy_package)

def generate_optimized_policies(data, goal, budget):
    """Generate optimized policy recommendations using ML"""
    try:
        # Simulate policy impact prediction
        policies = [
            {
                "name": "School Feeding Program",
                "category": "Education",
                "cost": budget_level_to_cost(budget, "low"),
                "impact_stunting": 0.15,
                "impact_anemia": 0.10,
                "impact_diversity": 0.25,
                "coverage": 0.7,
                "timeline": "6-12 months"
            },
            {
                "name": "Iron-Folic Acid Supplementation",
                "category": "Health",
                "cost": budget_level_to_cost(budget, "very_low"),
                "impact_stunting": 0.05,
                "impact_anemia": 0.35,
                "impact_diversity": 0.02,
                "coverage": 0.8,
                "timeline": "3-6 months"
            },
            {
                "name": "Nutrition Education Campaign",
                "category": "Multi-sectoral",
                "cost": budget_level_to_cost(budget, "low"),
                "impact_stunting": 0.10,
                "impact_anemia": 0.08,
                "impact_diversity": 0.20,
                "coverage": 0.6,
                "timeline": "6-18 months"
            },
            {
                "name": "Agricultural Diversification",
                "category": "Agriculture",
                "cost": budget_level_to_cost(budget, "high"),
                "impact_stunting": 0.20,
                "impact_anemia": 0.15,
                "impact_diversity": 0.40,
                "coverage": 0.5,
                "timeline": "12-24 months"
            },
            {
                "name": "WASH Infrastructure",
                "category": "Infrastructure",
                "cost": budget_level_to_cost(budget, "very_high"),
                "impact_stunting": 0.25,
                "impact_anemia": 0.12,
                "impact_diversity": 0.05,
                "coverage": 0.4,
                "timeline": "18-36 months"
            }
        ]
        
        # Select policies based on goal and budget
        selected_policies = optimize_policy_selection(policies, goal, budget)
        
        # Calculate expected impacts
        total_impact = calculate_total_impact(selected_policies, goal)
        total_cost = sum(p['cost'] for p in selected_policies)
        
        return {
            'selected_policies': selected_policies,
            'total_impact': total_impact,
            'total_cost': total_cost,
            'goal': goal,
            'budget': budget
        }
    
    except Exception as e:
        st.error(f"Policy optimization failed: {e}")
        return None

def budget_level_to_cost(budget_level, policy_cost_level):
    """Convert budget level to actual cost values"""
    cost_matrix = {
        "Low Budget": {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5},
        "Medium Budget": {"very_low": 2, "low": 3, "medium": 4, "high": 5, "very_high": 6},
        "High Budget": {"very_low": 3, "low": 4, "medium": 5, "high": 6, "very_high": 7},
        "Unconstrained": {"very_low": 4, "low": 5, "medium": 6, "high": 7, "very_high": 8}
    }
    
    cost_level_values = {"very_low": 1, "low": 2, "medium": 3, "high": 4, "very_high": 5}
    
    return cost_matrix.get(budget_level, cost_matrix["Medium Budget"])[policy_cost_level]

def optimize_policy_selection(policies, goal, budget):
    """Optimize policy selection based on goals and constraints"""
    # Simple optimization: select policies with highest impact for the goal within budget
    budget_limits = {
        "Low Budget": 8,
        "Medium Budget": 12,
        "High Budget": 16,
        "Unconstrained": 100
    }
    
    budget_limit = budget_limits[budget]
    
    # Map goal to impact metric
    goal_impact_map = {
        "Reduce Stunting": "impact_stunting",
        "Decrease Anemia": "impact_anemia",
        "Improve Dietary Diversity": "impact_diversity"
    }
    
    impact_metric = goal_impact_map.get(goal, "impact_stunting")
    
    # Sort policies by impact per cost
    for policy in policies:
        policy['efficiency'] = policy[impact_metric] / policy['cost']
    
    sorted_policies = sorted(policies, key=lambda x: x['efficiency'], reverse=True)
    
    # Select policies within budget
    selected = []
    total_cost = 0
    
    for policy in sorted_policies:
        if total_cost + policy['cost'] <= budget_limit:
            selected.append(policy)
            total_cost += policy['cost']
    
    return selected

def calculate_total_impact(policies, goal):
    """Calculate total expected impact of selected policies"""
    goal_impact_map = {
        "Reduce Stunting": "impact_stunting",
        "Decrease Anemia": "impact_anemia",
        "Improve Dietary Diversity": "impact_diversity"
    }
    
    impact_metric = goal_impact_map.get(goal, "impact_stunting")
    
    total_impact = 0
    for policy in policies:
        total_impact += policy[impact_metric] * policy['coverage']
    
    return min(1.0, total_impact)  # Cap at 100%

def display_policy_recommendations(package):
    """Display optimized policy recommendations"""
    if not package:
        return
    
    st.success("✅ Optimized policy package generated!")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Interventions", len(package['selected_policies']))
    
    with col2:
        st.metric("Total Cost Score", package['total_cost'])
    
    with col3:
        expected_impact = package['total_impact'] * 100
        st.metric("Expected Impact", f"{expected_impact:.1f}%")
    
    # Policy details
    st.markdown("### Recommended Intervention Package")
    
    for i, policy in enumerate(package['selected_policies'], 1):
        with st.expander(f"{i}. {policy['name']} ({policy['category']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Impact Scores:**")
                st.write(f"- Stunting reduction: {policy['impact_stunting']*100:.1f}%")
                st.write(f"- Anemia reduction: {policy['impact_anemia']*100:.1f}%")
                st.write(f"- Dietary diversity improvement: {policy['impact_diversity']*100:.1f}%")
                st.write(f"- Coverage: {policy['coverage']*100:.1f}%")
            
            with col2:
                st.metric("Cost Score", policy['cost'])
                st.metric("Timeline", policy['timeline'])
    
    # Impact visualization
    st.markdown("### Expected Impact Distribution")
    
    impact_data = []
    for policy in package['selected_policies']:
        impact_data.append({
            'Policy': policy['name'],
            'Stunting Impact': policy['impact_stunting'] * policy['coverage'] * 100,
            'Anemia Impact': policy['impact_anemia'] * policy['coverage'] * 100,
            'Diversity Impact': policy['impact_diversity'] * policy['coverage'] * 100
        })
    
    impact_df = pd.DataFrame(impact_data)
    
    fig = px.bar(impact_df, x='Policy', y=['Stunting Impact', 'Anemia Impact', 'Diversity Impact'],
                title='Expected Impact by Policy',
                barmode='group')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Implementation roadmap
    st.markdown("### Implementation Roadmap")
    
    timeline_data = []
    for policy in package['selected_policies']:
        timeline_data.append({
            'Policy': policy['name'],
            'Start': 0,
            'Finish': policy_timeline_to_months(policy['timeline']),
            'Category': policy['category']
        })
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.timeline(timeline_df, x_start="Start", x_end="Finish", y="Policy", color="Category",
                     title="Policy Implementation Timeline")
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cost-effectiveness analysis
    st.markdown("### Cost-Effectiveness Analysis")
    
    cost_effect_data = []
    for policy in package['selected_policies']:
        cost_effect_data.append({
            'Policy': policy['name'],
            'Cost': policy['cost'],
            'Effectiveness': policy['impact_stunting'] * policy['coverage'] * 100,
            'Efficiency': policy['efficiency']
        })
    
    cost_effect_df = pd.DataFrame(cost_effect_data)
    
    fig = px.scatter(cost_effect_df, x='Cost', y='Effectiveness', size='Efficiency',
                    text='Policy', title='Cost vs Effectiveness Analysis',
                    labels={'Cost': 'Cost Score', 'Effectiveness': 'Impact (%)'})
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

def policy_timeline_to_months(timeline):
    """Convert timeline description to months"""
    if '3-6' in timeline:
        return 6
    elif '6-12' in timeline:
        return 12
    elif '6-18' in timeline:
        return 18
    elif '12-24' in timeline:
        return 24
    elif '18-36' in timeline:
        return 36
    else:
        return 12

# Original policy framework (kept for compatibility)
def display_original_policy_framework():
    """Display the original policy framework"""
    st.markdown("### Multi-Sectoral Intervention Framework")
    
    tabs = st.tabs(["Health", "Agriculture", "Education", "Social Protection"])
    
    with tabs[0]:
        st.markdown("#### Health Sector Interventions")
        st.info("""
        **Immediate Actions (0-6 months):**
        - Scale up vitamin A supplementation for children 6-59 months
        - Strengthen iron-folic acid supplementation for pregnant women
        - Implement growth monitoring and promotion programs
        - Treat severe acute malnutrition cases
        """)
    
    with tabs[1]:
        st.markdown("#### Agriculture Sector Interventions")
        st.info("""
        **Immediate Actions (0-6 months):**
        - Promote homestead food production (vegetables, fruits)
        - Support kitchen gardens for diverse food production
        - Distribute nutrient-rich seeds and planting materials
        """)
    
    with tabs[2]:
        st.markdown("#### Education Sector Interventions")
        st.info("""
        **Immediate Actions (0-6 months):**
        - Integrate nutrition education in school curricula
        - Train teachers on nutrition and hygiene practices
        - Establish school feeding programs in food-insecure areas
        """)
    
    with tabs[3]:
        st.markdown("#### Social Protection Interventions")
        st.info("""
        **Immediate Actions (0-6 months):**
        - Target cash transfers to households with malnourished children
        - Provide nutrition-sensitive social safety nets
        - Support pregnant and lactating women with nutrition packages
        """)

# Call the original framework at the end of the main function
def policy_recommendations_complete():
    """Complete policy recommendations with both ML and traditional content"""
    policy_recommendations()
    st.markdown("---")
    display_original_policy_framework()