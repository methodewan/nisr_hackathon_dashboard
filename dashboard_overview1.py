import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_all_data():
    """Load all three datasets"""
    try:
        hh_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH data.csv")
        child_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_CHILD_6_59_MONTHS.csv")
        women_data = pd.read_csv("Microdata1111/Microdata/csvfile/CFSVA2024_HH_WOMEN_15_49_YEARS.csv")
        return hh_data, child_data, women_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


def merge_datasets(hh_data, child_data, women_data):
    """Merge datasets"""
    try:
        if '___index' in hh_data.columns and '___index' in child_data.columns:
            merged_data = child_data.merge(
                hh_data[['___index', 'wealth_index', 'urban_rural', 'province', 'district']], 
                on='___index', how='left'
            )
        else:
            merged_data = child_data.copy()
        return merged_data
    except Exception as e:
        st.error(f"Error merging datasets: {e}")
        return child_data


def dashboard_overview():
    """Main dashboard overview with ML insights"""
    st.markdown('<div class="main-header">🥦 Hidden Hunger Dashboard</div>', unsafe_allow_html=True)
    
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
    
    # Display available columns
    with st.expander("📋 Available Data", expanded=False):
        st.write("**Child Data Columns:**", child_data.columns.tolist()[:15])
        st.write("**HH Data Columns:**", hh_data.columns.tolist()[:15])
    
    # Key metrics
    st.markdown("### 📊 Survey Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🏠 Households", f"{len(hh_data):,}")
    
    with col2:
        st.metric("👶 Children (6-59m)", f"{len(child_data):,}")
    
    with col3:
        st.metric("👩 Women (15-49y)", f"{len(women_data):,}")
    
    with col4:
        total_respondents = len(hh_data) + len(child_data) + len(women_data)
        st.metric("👥 Total Respondents", f"{total_respondents:,}")
    
    # Nutritional Status
    st.markdown("### 📈 Nutritional Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Stunting")
        if 'stunting_zscore' in merged_data.columns:
            stunting_data = merged_data['stunting_zscore'].dropna()
            if len(stunting_data) > 0:
                stunted = (stunting_data < -2).sum()
                stunting_rate = stunted / len(stunting_data) * 100
                st.metric("Stunted Children", f"{stunted:,}", delta=f"{stunting_rate:.1f}%")
                
                # Distribution
                fig = px.histogram(stunting_data, nbins=30, title="Stunting Z-Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Stunting data not available")
    
    with col2:
        st.markdown("#### Wasting")
        if 'wasting_zscore' in merged_data.columns:
            wasting_data = merged_data['wasting_zscore'].dropna()
            if len(wasting_data) > 0:
                wasted = (wasting_data < -2).sum()
                wasting_rate = wasted / len(wasting_data) * 100
                st.metric("Wasted Children", f"{wasted:,}", delta=f"{wasting_rate:.1f}%")
                
                fig = px.histogram(wasting_data, nbins=30, title="Wasting Z-Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wasting data not available")
    
    with col3:
        st.markdown("#### Anemia")
        if 'anemic' in merged_data.columns:
            anemic_data = merged_data['anemic'].dropna()
            if len(anemic_data) > 0:
                anemic_count = (anemic_data == 1).sum()
                anemic_rate = anemic_count / len(anemic_data) * 100
                st.metric("Anemic Children", f"{anemic_count:,}", delta=f"{anemic_rate:.1f}%")
                
                fig = px.pie(values=[anemic_count, len(anemic_data) - anemic_count],
                            names=['Anemic', 'Non-Anemic'], title="Anemia Status")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Anemia data not available")
    
    # Wealth Index Analysis
    st.markdown("---")
    st.markdown("### 💰 Socioeconomic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Wealth Index Distribution")
        if 'wealth_index' in merged_data.columns:
            wealth_dist = merged_data['wealth_index'].value_counts()
            fig = px.bar(x=wealth_dist.index, y=wealth_dist.values,
                        title="Household Wealth Distribution",
                        labels={'x': 'Wealth Index', 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Wealth index data not available")
    
    with col2:
        st.markdown("#### Urban-Rural Distribution")
        if 'urban_rural' in merged_data.columns:
            location_dist = merged_data['urban_rural'].value_counts()
            fig = px.pie(values=location_dist.values, names=location_dist.index,
                        title="Urban vs Rural")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Location data not available")
    
    # Wealth-Stunting Relationship
    st.markdown("---")
    st.markdown("### 🔗 Key Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Stunting by Wealth Index")
        if 'wealth_index' in merged_data.columns and 'stunting_zscore' in merged_data.columns:
            stunting_by_wealth = merged_data.groupby('wealth_index').apply(
                lambda x: ((x['stunting_zscore'] < -2).sum() / len(x) * 100) if len(x) > 0 else 0
            )
            fig = px.bar(x=stunting_by_wealth.index, y=stunting_by_wealth.values,
                        title="Stunting Rate by Wealth",
                        labels={'x': 'Wealth Index', 'y': 'Stunting Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need wealth and stunting data")
    
    with col2:
        st.markdown("#### Stunting by Location")
        if 'urban_rural' in merged_data.columns and 'stunting_zscore' in merged_data.columns:
            stunting_by_location = merged_data.groupby('urban_rural').apply(
                lambda x: ((x['stunting_zscore'] < -2).sum() / len(x) * 100) if len(x) > 0 else 0
            )
            fig = px.bar(x=stunting_by_location.index, y=stunting_by_location.values,
                        title="Stunting Rate by Location",
                        labels={'x': 'Location', 'y': 'Stunting Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need location and stunting data")
    
    # ML Insights
    st.markdown("---")
    st.markdown("### 🤖 ML-Powered Insights")
    
    ml_insights = train_dashboard_ml_model(merged_data)
    
    if ml_insights:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(ml_insights['feature_importance'], use_container_width=True)
        
        with col2:
            st.markdown("#### Model Performance")
            st.success(f"**Model Accuracy:** {ml_insights['accuracy']:.1%}")
            
            st.markdown("**Top Risk Factors:**")
            for i, feature in enumerate(ml_insights['top_features'][:5], 1):
                st.write(f"{i}. {feature}")
    
    # Data Quality
    st.markdown("---")
    st.markdown("### 📊 Data Quality Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Child Data")
        completeness = (1 - child_data.isnull().sum().sum() / (len(child_data) * len(child_data.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col2:
        st.markdown("#### HH Data")
        completeness = (1 - hh_data.isnull().sum().sum() / (len(hh_data) * len(hh_data.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col3:
        st.markdown("#### Women Data")
        completeness = (1 - women_data.isnull().sum().sum() / (len(women_data) * len(women_data.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")


def train_dashboard_ml_model(data):
    """Train ML model for dashboard insights"""
    try:
        # Detect available numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out ID columns
        exclude = ['___index', 'child_id', 'woman_id']
        feature_cols = [col for col in numeric_cols if col not in exclude][:10]
        
        if len(feature_cols) < 3:
            st.warning("Not enough features for ML model")
            return None
        
        # Create target from stunting if available
        if 'stunting_zscore' in data.columns:
            y = (data['stunting_zscore'] < -2).fillna(0).astype(int)
        elif 'wasting_zscore' in data.columns:
            y = (data['wasting_zscore'] < -2).fillna(0).astype(int)
        else:
            # Create random target for demo
            y = np.random.choice([0, 1], len(data), p=[0.7, 0.3])
        
        # Prepare features
        X = data[feature_cols].copy()
        X = X.fillna(X.median())
        
        # Remove rows with NaN in target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            st.warning("Not enough samples for ML model")
            return None
        
        if len(y.unique()) < 2:
            st.warning("Target variable needs both classes")
            return None
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Metrics
        accuracy = model.score(X_test, y_test)
        
        # Feature importance
        feature_importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig = px.bar(
            feature_importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Feature Importance',
            color='importance',
            color_continuous_scale='Viridis'
        )
        
        return {
            'accuracy': accuracy,
            'feature_importance': fig,
            'top_features': feature_importance_df['feature'].tolist()
        }
    
    except Exception as e:
        st.error(f"ML model error: {e}")
        return None