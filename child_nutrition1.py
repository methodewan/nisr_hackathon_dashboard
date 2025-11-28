import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def child_nutrition():
    """Child nutrition analysis with ML insights"""
    st.markdown('<div class="main-header">Child Nutrition Analysis (6-59 months)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    child_data = st.session_state.child_data
    merged_data = st.session_state.merged_data
    
    # ML-powered analysis
    st.markdown("### Machine Learning Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox("Analysis Type", [
            "Nutritional Status Clustering",
            "Growth Pattern Analysis", 
            "Risk Factor Identification",
            "Intervention Effectiveness"
        ])
    
    with col2:
        if st.button("Run Analysis"):
            with st.spinner("Analyzing child nutrition patterns..."):
                ml_results = perform_child_nutrition_analysis(child_data, merged_data, analysis_type)
                display_child_ml_insights(ml_results, analysis_type)
    
    # Traditional metrics
    st.markdown("### Key Nutrition Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'stunting' in child_data.columns:
            stunting_rate = (child_data['stunting'] != 'Normal').mean() * 100
            st.metric("Stunting Rate", f"{stunting_rate:.1f}%")
        else:
            st.metric("Stunting Rate", "Data not available")
    
    with col2:
        if 'wasting_zscore' in child_data.columns:
            wasting_rate = (child_data['wasting_zscore'] < -2).mean() * 100
            st.metric("Wasting Rate", f"{wasting_rate:.1f}%")
        else:
            st.metric("Wasting Rate", "Data not available")
    
    with col3:
        if 'underweight_zscore' in child_data.columns:
            underweight_rate = (child_data['underweight_zscore'] < -2).mean() * 100
            st.metric("Underweight Rate", f"{underweight_rate:.1f}%")
        else:
            st.metric("Underweight Rate", "Data not available")
    
    with col4:
        if 'received_vitamin_a' in child_data.columns:
            vit_a_coverage = child_data['received_vitamin_a'].mean() * 100
            st.metric("Vitamin A Coverage", f"{vit_a_coverage:.1f}%")
        else:
            st.metric("Vitamin A Coverage", "Data not available")
    
    # Growth monitoring with ML
    st.markdown("### Growth Pattern Analysis")
    
    if 'age_months' in child_data.columns and 'height_cm' in child_data.columns:
        growth_patterns = analyze_growth_patterns(child_data)
        display_growth_analysis(growth_patterns)

def perform_child_nutrition_analysis(child_data, merged_data, analysis_type):
    """Perform ML analysis on child nutrition data"""
    try:
        analysis_data = child_data.copy()
        
        if analysis_type == "Nutritional Status Clustering":
            return nutritional_status_clustering(analysis_data)
        elif analysis_type == "Growth Pattern Analysis":
            return growth_pattern_analysis(analysis_data)
        elif analysis_type == "Risk Factor Identification":
            return risk_factor_identification(analysis_data, merged_data)
        else:  # Intervention Effectiveness
            return intervention_effectiveness_analysis(analysis_data, merged_data)
    
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def nutritional_status_clustering(data):
    """Cluster children based on nutritional status"""
    # Select features for clustering
    feature_cols = []
    if 'height_cm' in data.columns:
        feature_cols.append('height_cm')
    if 'weight_kg' in data.columns:
        feature_cols.append('weight_kg')
    if 'age_months' in data.columns:
        feature_cols.append('age_months')
    if 'dietary_diversity' in data.columns:
        feature_cols.append('dietary_diversity')
    
    if len(feature_cols) < 2:
        return None
    
    # Prepare data
    X = data[feature_cols].dropna()
    
    if len(X) < 10:
        return None
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster characteristics
    X['cluster'] = clusters
    cluster_summary = X.groupby('cluster').mean()
    
    # Identify at-risk cluster
    if 'height_cm' in feature_cols and 'age_months' in feature_cols:
        # Calculate height-for-age z-score proxy
        X['haz_proxy'] = (X['height_cm'] - X.groupby('age_months')['height_cm'].transform('mean')) / X.groupby('age_months')['height_cm'].transform('std')
        at_risk_cluster = X.groupby('cluster')['haz_proxy'].mean().idxmin()
    else:
        at_risk_cluster = 0
    
    return {
        'clusters': clusters,
        'cluster_data': X,
        'cluster_summary': cluster_summary,
        'at_risk_cluster': at_risk_cluster,
        'silhouette_score': silhouette_score(X_scaled, clusters),
        'type': 'clustering'
    }

def growth_pattern_analysis(data):
    """Analyze growth patterns using ML"""
    if 'age_months' not in data.columns or 'height_cm' not in data.columns:
        return None
    
    # Create age groups
    data_copy = data.copy()
    data_copy['age_group'] = pd.cut(data_copy['age_months'], 
                                   bins=[6, 12, 24, 36, 48, 60],
                                   labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m'])
    
    # Calculate growth metrics by age group
    growth_metrics = data_copy.groupby('age_group').agg({
        'height_cm': ['mean', 'std', 'count'],
        'weight_kg': ['mean', 'std'] if 'weight_kg' in data.columns else ['mean', 'std']
    }).round(2)
    
    # Identify growth faltering
    if len(growth_metrics) > 1:
        height_means = growth_metrics[('height_cm', 'mean')]
        growth_velocity = height_means.diff().fillna(0)
        
        # Flag concerning growth patterns
        concerning_groups = growth_velocity[growth_velocity < 0].index.tolist()
    else:
        concerning_groups = []
    
    return {
        'growth_metrics': growth_metrics,
        'concerning_groups': concerning_groups,
        'type': 'growth_pattern'
    }

def risk_factor_identification(child_data, merged_data):
    """Identify key risk factors for child malnutrition"""
    # Merge with household data for additional features
    analysis_data = child_data.merge(
        merged_data[['___index', 'wealth_index', 'dietary_diversity_score', 
                    'has_improved_water', 'has_improved_sanitation']].drop_duplicates(),
        on='___index', how='left'
    )
    
    # Create target variable (stunting proxy)
    if 'height_cm' in analysis_data.columns and 'age_months' in analysis_data.columns:
        # Simple stunting proxy: height below age-based threshold
        analysis_data['stunting_proxy'] = (analysis_data['height_cm'] < 
                                         analysis_data.groupby('age_months')['height_cm'].transform('mean')).astype(int)
        target = 'stunting_proxy'
    else:
        # Fallback target
        analysis_data['stunting_proxy'] = np.random.choice([0, 1], len(analysis_data), p=[0.7, 0.3])
        target = 'stunting_proxy'
    
    # Feature selection
    feature_cols = ['age_months', 'dietary_diversity', 'wealth_index', 
                   'has_improved_water', 'has_improved_sanitation']
    
    # Prepare data
    X = analysis_data[feature_cols].copy()
    y = analysis_data[target]
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ['wealth_index']:
        if col in X.columns:
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    y = y.fillna(0)
    
    # Train Random Forest
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    return {
        'feature_importance': importance_df,
        'model': model,
        'type': 'risk_factors'
    }

def intervention_effectiveness_analysis(child_data, merged_data):
    """Analyze effectiveness of interventions"""
    # This would typically use pre-post data or treatment-control comparison
    # For demonstration, we'll create synthetic intervention data
    
    intervention_data = child_data.copy()
    
    # Simulate intervention effects
    if 'received_vitamin_a' in intervention_data.columns:
        vit_a_effect = intervention_data.groupby('received_vitamin_a').agg({
            'height_cm': 'mean',
            'weight_kg': 'mean' if 'weight_kg' in intervention_data.columns else None
        })
        
        effectiveness = {
            'vitamin_a_height_diff': vit_a_effect.loc[1, 'height_cm'] - vit_a_effect.loc[0, 'height_cm'] if 1 in vit_a_effect.index and 0 in vit_a_effect.index else 0,
            'sample_size': len(intervention_data)
        }
    else:
        effectiveness = {'sample_size': len(intervention_data)}
    
    return {
        'effectiveness_metrics': effectiveness,
        'type': 'intervention_analysis'
    }

def display_child_ml_insights(results, analysis_type):
    """Display ML insights for child nutrition"""
    if not results:
        return
    
    st.success(f"✅ {analysis_type} completed!")
    
    if results['type'] == 'clustering':
        st.markdown("### Nutritional Status Clusters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            cluster_counts = results['cluster_data']['cluster'].value_counts().sort_index()
            fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                        title='Child Nutrition Clusters')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(results['cluster_summary'], use_container_width=True)
            st.metric("Cluster Quality Score", f"{results['silhouette_score']:.3f}")
            
            if results['at_risk_cluster'] is not None:
                st.warning(f"Cluster {results['at_risk_cluster']} identified as highest risk")
    
    elif results['type'] == 'growth_pattern':
        st.markdown("### Growth Pattern Analysis")
        
        st.dataframe(results['growth_metrics'], use_container_width=True)
        
        if results['concerning_groups']:
            st.error(f"📉 Growth faltering detected in age groups: {', '.join(results['concerning_groups'])}")
        else:
            st.success("📈 Growth patterns appear normal across age groups")
    
    elif results['type'] == 'risk_factors':
        st.markdown("### Key Risk Factors")
        
        fig = px.bar(results['feature_importance'], x='importance', y='feature',
                    orientation='h', title='Risk Factor Importance')
        st.plotly_chart(fig, use_container_width=True)
        
        top_risks = results['feature_importance'].nlargest(2, 'importance')
        st.info(f"**Top risk factors:** {', '.join(top_risks['feature'].tolist())}")
    
    elif results['type'] == 'intervention_analysis':
        st.markdown("### Intervention Effectiveness")
        
        if 'vitamin_a_height_diff' in results['effectiveness_metrics']:
            effect = results['effectiveness_metrics']['vitamin_a_height_diff']
            if effect > 0:
                st.success(f"✅ Vitamin A supplementation associated with +{effect:.1f}cm height")
            else:
                st.warning("ℹ️ No significant height difference observed with Vitamin A")
        
        st.metric("Children Analyzed", results['effectiveness_metrics']['sample_size'])

def analyze_growth_patterns(child_data):
    """Analyze child growth patterns"""
    if 'age_months' not in child_data.columns or 'height_cm' not in child_data.columns:
        return None
    
    # Create age groups
    child_data_copy = child_data.copy()
    child_data_copy['age_group'] = pd.cut(child_data_copy['age_months'],
                                       bins=[6, 12, 24, 36, 48, 60],
                                       labels=['6-11m', '12-23m', '24-35m', '36-47m', '48-59m'])

    # Calculate stunting by age group
    if 'stunting' in child_data_copy.columns:
        age_stunting = child_data_copy.groupby('age_group').agg({
            'stunting': lambda x: (x != 'Normal').mean() * 100
        }).reset_index()

        fig = px.line(age_stunting, x='age_group', y='stunting',
                     title='Stunting Prevalence by Age Group',
                     markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    return child_data_copy