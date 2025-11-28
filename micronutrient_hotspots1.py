import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import folium
from streamlit_folium import folium_static

def micronutrient_hotspots():
    """Geographic analysis of malnutrition hotspots using clustering"""
    st.markdown('<div class="main-header">Micronutrient Hotspots Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    
    # ML-based hotspot detection
    st.markdown("### Machine Learning Hotspot Detection")
    
    # Cluster configuration
    col1, col2 = st.columns(2)
    
    with col1:
        clustering_method = st.selectbox("Clustering Algorithm", 
                                       ["K-Means", "DBSCAN", "Gaussian Mixture"])
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
    
    with col2:
        features = st.multiselect("Features for Clustering",
                                ['stunting_risk', 'wasting_risk', 'dietary_diversity', 
                                 'wealth_index', 'urban_rural'],
                                default=['stunting_risk', 'dietary_diversity'])
    
    if st.button("Detect Hotspots"):
        with st.spinner("Analyzing geographic patterns..."):
            hotspots = detect_hotspots(merged_data, features, clustering_method, n_clusters)
            
            if hotspots:
                display_hotspot_results(hotspots, merged_data)

def detect_hotspots(data, features, method="K-Means", n_clusters=4):
    """Detect malnutrition hotspots using clustering algorithms"""
    try:
        # Prepare data for clustering
        cluster_data = data.copy()
        
        # Encode categorical variables
        if 'wealth_index' in features:
            wealth_mapping = {'Poorest': 0, 'Poor': 1, 'Middle': 2, 'Rich': 3, 'Richest': 4}
            cluster_data['wealth_index'] = cluster_data['wealth_index'].map(wealth_mapping)
        
        if 'urban_rural' in features:
            cluster_data['urban_rural'] = cluster_data['urban_rural'].map({'Urban': 1, 'Rural': 0})
        
        # Select features
        X = cluster_data[features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply clustering
        if method == "K-Means":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "DBSCAN":
            model = DBSCAN(eps=0.5, min_samples=5)
        else:
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        
        clusters = model.fit_predict(X_scaled)
        
        # Calculate cluster characteristics
        cluster_data['cluster'] = clusters
        cluster_summary = cluster_data.groupby('cluster')[features].mean()
        
        # Identify high-risk clusters
        if 'stunting_risk' in features:
            high_risk_clusters = cluster_summary[cluster_summary['stunting_risk'] > 
                                               cluster_summary['stunting_risk'].mean()].index.tolist()
        else:
            high_risk_clusters = []
        
        # Calculate silhouette score for K-Means
        if method == "K-Means":
            silhouette_avg = silhouette_score(X_scaled, clusters)
        else:
            silhouette_avg = None
        
        return {
            'clusters': clusters,
            'cluster_data': cluster_data,
            'cluster_summary': cluster_summary,
            'high_risk_clusters': high_risk_clusters,
            'silhouette_score': silhouette_avg,
            'features': features
        }
    
    except Exception as e:
        st.error(f"Hotspot detection failed: {e}")
        return None

def display_hotspot_results(hotspots, original_data):
    """Display hotspot analysis results"""
    
    # Cluster visualization
    st.markdown("### Cluster Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster distribution
        cluster_counts = pd.Series(hotspots['clusters']).value_counts().sort_index()
        fig = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                    title='Cluster Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster characteristics
        st.dataframe(hotspots['cluster_summary'], use_container_width=True)
    
    # High-risk clusters
    if hotspots['high_risk_clusters']:
        st.warning(f"🚨 High-risk clusters detected: {hotspots['high_risk_clusters']}")
        
        # Show high-risk cluster characteristics
        high_risk_data = hotspots['cluster_data'][
            hotspots['cluster_data']['cluster'].isin(hotspots['high_risk_clusters'])
        ]
        
        st.markdown("#### High-Risk Cluster Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Households in High-Risk", len(high_risk_data))
        
        with col2:
            if 'stunting_risk' in high_risk_data.columns:
                stunting_rate = high_risk_data['stunting_risk'].mean() * 100
                st.metric("Average Stunting Rate", f"{stunting_rate:.1f}%")
        
        with col3:
            if hotspots['silhouette_score']:
                st.metric("Cluster Quality", f"{hotspots['silhouette_score']:.3f}")
    
    # Geographic visualization (if coordinates available)
    st.markdown("### Geographic Hotspot Map")
    
    # Create synthetic coordinates for demonstration
    np.random.seed(42)
    hotspots['cluster_data']['lat'] = np.random.uniform(-1.5, -2.5, len(hotspots['cluster_data']))
    hotspots['cluster_data']['lon'] = np.random.uniform(29.0, 31.0, len(hotspots['cluster_data']))
    
    # Create cluster map
    fig = px.scatter_mapbox(hotspots['cluster_data'], 
                          lat="lat", lon="lon", 
                          color="cluster",
                          hover_data=hotspots['features'],
                          title="Malnutrition Hotspots",
                          mapbox_style="open-street-map",
                          zoom=8)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Intervention recommendations
    st.markdown("### Targeted Interventions by Cluster")
    
    for cluster_id in hotspots['high_risk_clusters']:
        with st.expander(f"Cluster {cluster_id} - Priority Interventions"):
            cluster_profile = hotspots['cluster_summary'].loc[cluster_id]
            
            st.markdown("**Cluster Characteristics:**")
            for feature in hotspots['features']:
                if feature in cluster_profile:
                    st.write(f"- {feature}: {cluster_profile[feature]:.3f}")
            
            st.markdown("**Recommended Actions:**")
            st.info("""
            - Intensive nutrition supplementation
            - Targeted food assistance
            - WASH infrastructure improvement
            - Livelihood support programs
            - Regular growth monitoring
            """)