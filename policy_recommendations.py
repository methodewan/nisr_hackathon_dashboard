import streamlit as st

def policy_recommendations():
    """Policy recommendations and interventions"""
    st.markdown('<div class="main-header">Policy Recommendations & Interventions</div>', unsafe_allow_html=True)
    
    # Multi-sectoral interventions
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
        
        **Medium-term Strategies (6-24 months):**
        - Integrate nutrition services into primary healthcare
        - Strengthen antenatal care nutrition components
        - Build capacity of health workers on nutrition
        - Establish community-based management of acute malnutrition
        """)
    
    with tabs[1]:
        st.markdown("#### Agriculture Sector Interventions")
        st.info("""
        **Immediate Actions (0-6 months):**
        - Promote homestead food production (vegetables, fruits)
        - Support kitchen gardens for diverse food production
        - Distribute nutrient-rich seeds and planting materials
        
        **Medium-term Strategies (6-24 months):**
        - Promote biofortified crops (vitamin A maize, iron beans)
        - Support small-scale livestock rearing for animal-source foods
        - Develop value chains for nutrient-dense foods
        - Integrate nutrition education into agricultural extension
        """)
    
    with tabs[2]:
        st.markdown("#### Education Sector Interventions")
        st.info("""
        **Immediate Actions (0-6 months):**
        - Integrate nutrition education in school curricula
        - Train teachers on nutrition and hygiene practices
        - Establish school feeding programs in food-insecure areas
        
        **Medium-term Strategies (6-24 months):**
        - Promote school gardens for practical learning
        - Implement nutrition-focused parent education programs
        - Develop school-based nutrition monitoring systems
        - Strengthen WASH facilities in schools
        """)
    
    with tabs[3]:
        st.markdown("#### Social Protection Interventions")
        st.info("""
        **Immediate Actions (0-6 months):**
        - Target cash transfers to households with malnourished children
        - Provide nutrition-sensitive social safety nets
        - Support pregnant and lactating women with nutrition packages
        
        **Medium-term Strategies (6-24 months):**
        - Integrate nutrition criteria into social protection programs
        - Develop nutrition-sensitive livelihood programs
        - Establish community kitchens for vulnerable groups
        - Link social protection with nutrition education
        """)
    
    # Policy briefs
    st.markdown("### Policy Briefs for Local Implementation")
    
    policy_briefs = [
        {
            "title": "Addressing Micronutrient Deficiencies in High-Risk Areas",
            "target": "Local Government, Health Districts",
            "key_recommendations": [
                "Establish community nutrition centers in high-stunting areas",
                "Provide micronutrient powders to children 6-23 months",
                "Strengthen vitamin A supplementation campaigns",
                "Implement targeted IFA supplementation for adolescent girls"
            ],
            "timeline": "6-12 months",
            "budget": "Medium"
        },
        {
            "title": "Integrating Nutrition into Agriculture Programs",
            "target": "Ministry of Agriculture, Local Farmers",
            "key_recommendations": [
                "Promote biofortified crop varieties",
                "Support homestead food production systems",
                "Develop nutrition-sensitive value chains",
                "Train farmers on nutrient-dense food production"
            ],
            "timeline": "12-24 months",
            "budget": "High"
        },
        {
            "title": "School-Based Nutrition Interventions",
            "target": "Ministry of Education, Schools",
            "key_recommendations": [
                "Implement school meal programs with diverse foods",
                "Establish school gardens for practical learning",
                "Integrate nutrition education in curricula",
                "Provide deworming and micronutrient supplementation"
            ],
            "timeline": "6-18 months",
            "budget": "Medium"
        }
    ]
    
    for i, brief in enumerate(policy_briefs):
        with st.expander(f"{i+1}. {brief['title']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Target Audience:** {brief['target']}")
                st.markdown("**Key Recommendations:**")
                for rec in brief['key_recommendations']:
                    st.markdown(f"- {rec}")
            
            with col2:
                st.markdown(f"**Timeline:** {brief['timeline']}")
                st.markdown(f"**Budget:** {brief['budget']}")
    
    # Implementation framework
    st.markdown("### Implementation Framework")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Monitoring & Evaluation")
        st.info("""
        - **Process Indicators:**
          - Coverage of supplementation programs
          - Number of households with kitchen gardens
          - School feeding program coverage
        
        - **Outcome Indicators:**
          - Stunting prevalence
          - Anemia rates
          - Dietary diversity scores
        
        - **Impact Indicators:**
          - Child mortality rates
          - Cognitive development scores
          - Economic productivity
        """)
    
    with col2:
        st.markdown("#### Stakeholder Engagement")
        st.info("""
        - **Government:** Policy alignment, resource allocation
        - **NGOs:** Program implementation, capacity building
        - **Communities:** Participation, ownership, feedback
        - **Private Sector:** Supply chains, innovation
        - **Donors:** Funding, technical assistance
        """)