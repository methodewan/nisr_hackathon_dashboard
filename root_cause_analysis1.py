import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')


def root_cause_analysis():
    """Advanced root cause analysis using ML techniques"""
    st.markdown('<div class="main-header">📊 Root Cause Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    
    # Show available columns
    with st.expander("📋 Available Columns", expanded=False):
        st.write("**Data Shape:**", merged_data.shape)
        st.write("**Columns:**")
        cols_by_type = {
            'Numeric': merged_data.select_dtypes(include=[np.number]).columns.tolist(),
            'Categorical': merged_data.select_dtypes(include=['object']).columns.tolist()
        }
        for col_type, cols in cols_by_type.items():
            st.write(f"**{col_type}:** {', '.join(cols[:10])}")
    
    # Causal analysis methods
    st.markdown("### 🔍 Machine Learning-Based Causal Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_method = st.selectbox("Analysis Method:", [
            "Feature Importance Analysis", 
            "Partial Dependence Plots",
            "Statistical Correlation Analysis",
            "Causal Tree Analysis"
        ])
    
    with col2:
        target_variable = st.selectbox("Target Outcome:", [
            "Stunting Risk", "Wasting Risk", "Any Malnutrition", "Dietary Diversity"
        ])
    
    if st.button("🔍 Analyze Root Causes", type="primary", use_container_width=True):
        with st.spinner("Performing advanced causal analysis..."):
            results = perform_root_cause_analysis(merged_data, target_variable, analysis_method)
            if results:
                display_causal_insights(results, analysis_method)


def perform_root_cause_analysis(data, target_var, method):
    """Perform ML-based root cause analysis"""
    try:
        analysis_data = data.copy()
        
        # Auto-detect available feature columns from actual data
        possible_features = [
            'wealth_index', 'dietary_diversity_score', 'hh_size',
            'has_improved_water', 'has_improved_sanitation', 'urban_rural',
            'age_months', 'sex', 'province', 'district',
            'received_vitamin_a', 'received_ifa', 'anemic'
        ]
        
        # Get features that actually exist in the data
        feature_cols = [col for col in possible_features if col in analysis_data.columns]
        
        if not feature_cols:
            st.error("❌ No feature columns found in your data")
            st.write("Available columns:", analysis_data.columns.tolist())
            return None
        
        st.write(f"✅ Using {len(feature_cols)} features: {feature_cols}")
        
        # Create target variable from REAL data
        y = None
        target_msg = ""
        
        if target_var == "Stunting Risk":
            if 'stunting_zscore' in analysis_data.columns:
                valid_data = analysis_data['stunting_zscore'].dropna()
                if len(valid_data) > 0:
                    threshold = valid_data.quantile(0.25)
                    y = (analysis_data['stunting_zscore'] < threshold).fillna(0).astype(int)
                    target_msg = f"Created from stunting_zscore (threshold: {threshold:.2f})"
            
            if y is None and 'height_for_age' in analysis_data.columns:
                valid_data = analysis_data['height_for_age'].dropna()
                if len(valid_data) > 0:
                    threshold = valid_data.median()
                    y = (analysis_data['height_for_age'] < threshold).fillna(0).astype(int)
                    target_msg = f"Created from height_for_age (threshold: {threshold:.2f})"
        
        elif target_var == "Wasting Risk":
            if 'wasting_zscore' in analysis_data.columns:
                valid_data = analysis_data['wasting_zscore'].dropna()
                if len(valid_data) > 0:
                    threshold = valid_data.quantile(0.25)
                    y = (analysis_data['wasting_zscore'] < threshold).fillna(0).astype(int)
                    target_msg = f"Created from wasting_zscore (threshold: {threshold:.2f})"
        
        elif target_var == "Any Malnutrition":
            flags = []
            if 'stunting_zscore' in analysis_data.columns:
                valid = analysis_data['stunting_zscore'].dropna()
                if len(valid) > 0:
                    threshold = valid.quantile(0.25)
                    flags.append((analysis_data['stunting_zscore'] < threshold).fillna(0).astype(int))
            
            if 'wasting_zscore' in analysis_data.columns:
                valid = analysis_data['wasting_zscore'].dropna()
                if len(valid) > 0:
                    threshold = valid.quantile(0.25)
                    flags.append((analysis_data['wasting_zscore'] < threshold).fillna(0).astype(int))
            
            if 'anemic' in analysis_data.columns:
                flags.append(analysis_data['anemic'].fillna(0).astype(int))
            
            if flags:
                combined = pd.concat(flags, axis=1).sum(axis=1)
                y = (combined > 0).astype(int)
                target_msg = f"Created from {len(flags)} malnutrition indicators"
        
        elif target_var == "Dietary Diversity":
            if 'dietary_diversity_score' in analysis_data.columns:
                y = analysis_data['dietary_diversity_score'].fillna(0).astype(int)
                target_msg = "Using dietary_diversity_score"
            else:
                st.error("Dietary diversity score not found")
                return None
        
        if y is None:
            st.error(f"❌ Could not create target variable for {target_var}")
            return None
        
        st.info(f"✅ Target: {target_msg}")
        st.write(f"Target distribution: {dict(y.value_counts())}")
        
        # Prepare features
        X = analysis_data[feature_cols].copy()
        
        # Encode categorical variables
        categorical_cols = ['wealth_index', 'urban_rural', 'sex', 'province', 'district']
        
        for col in categorical_cols:
            if col in X.columns:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
                except:
                    pass
        
        # Handle missing values
        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna(0).astype(int)
        
        # Remove rows with any remaining NaN
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 20:
            st.error(f"Not enough valid samples: {len(X)}")
            return None
        
        st.write(f"✅ Analysis on {len(X)} valid samples")
        
        # Perform analysis
        if method == "Feature Importance Analysis":
            return feature_importance_analysis(X, y, feature_cols)
        elif method == "Partial Dependence Plots":
            return partial_dependence_analysis(X, y, feature_cols)
        elif method == "Statistical Correlation Analysis":
            return correlation_analysis(X, y, feature_cols)
        else:  # Causal Tree Analysis
            return causal_tree_analysis(X, y, feature_cols)
    
    except Exception as e:
        st.error(f"❌ Root cause analysis failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def feature_importance_analysis(X, y, feature_cols):
    """Analyze feature importance using Random Forest"""
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Permutation importance
        perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        
        results = {
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_,
                'permutation_importance': perm_importance.importances_mean
            }).sort_values('importance', ascending=False),
            'model': model,
            'type': 'feature_importance'
        }
        
        return results
    except Exception as e:
        st.error(f"Feature importance analysis failed: {e}")
        return None


def partial_dependence_analysis(X, y, feature_cols):
    """Generate partial dependence plots"""
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Create PDPs for top 3 features
        pdp_data = {}
        
        for feature in feature_cols[:3]:
            unique_vals = np.linspace(X[feature].min(), X[feature].max(), 10)
            pdp_vals = []
            
            for val in unique_vals:
                X_temp = X.copy()
                X_temp[feature] = val
                pdp_vals.append(model.predict(X_temp).mean())
            
            pdp_data[feature] = {
                'values': unique_vals.tolist(),
                'predictions': pdp_vals
            }
        
        return {
            'pdp_data': pdp_data,
            'type': 'partial_dependence'
        }
    except Exception as e:
        st.error(f"PDP analysis failed: {e}")
        return None


def correlation_analysis(X, y, feature_cols):
    """Perform statistical correlation analysis"""
    try:
        correlations = []
        p_values = []
        
        for feature in feature_cols:
            if len(X[feature].unique()) > 1:
                corr, p_val = stats.pearsonr(X[feature], y)
                correlations.append(corr)
                p_values.append(p_val)
            else:
                correlations.append(0)
                p_values.append(1)
        
        return {
            'correlations': pd.DataFrame({
                'feature': feature_cols,
                'correlation': correlations,
                'p_value': p_values
            }).sort_values('correlation', key=abs, ascending=False),
            'type': 'correlation'
        }
    except Exception as e:
        st.error(f"Correlation analysis failed: {e}")
        return None


def causal_tree_analysis(X, y, feature_cols):
    """Causal tree analysis"""
    try:
        from sklearn.tree import DecisionTreeRegressor, export_text
        
        model = DecisionTreeRegressor(max_depth=4, random_state=42)
        model.fit(X, y)
        
        tree_rules = export_text(model, feature_names=feature_cols)
        
        return {
            'tree_rules': tree_rules,
            'feature_importance': pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False),
            'type': 'causal_tree'
        }
    except Exception as e:
        st.error(f"Causal tree analysis failed: {e}")
        return None


def display_causal_insights(results, method):
    """Display causal analysis insights"""
    if not results:
        return
    
    st.success(f"✅ {method} completed!")
    st.markdown("---")
    
    if results['type'] == 'feature_importance':
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Feature Importance (Random Forest)")
            fig = px.bar(
                results['feature_importance'], 
                x='importance', 
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis',
                title='Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Permutation Importance")
            fig = px.bar(
                results['feature_importance'],
                x='permutation_importance',
                y='feature',
                orientation='h',
                color='permutation_importance',
                color_continuous_scale='Blues',
                title='Permutation Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔑 Key Drivers Identified")
        top_features = results['feature_importance'].head(5)
        for idx, (_, row) in enumerate(top_features.iterrows(), 1):
            st.write(f"**{idx}. {row['feature']}**: {row['importance']:.3f}")
    
    elif results['type'] == 'partial_dependence':
        st.markdown("### Partial Dependence Analysis")
        
        for feature, data in results['pdp_data'].items():
            fig = px.line(
                x=data['values'],
                y=data['predictions'],
                title=f'Partial Dependence: {feature}',
                markers=True
            )
            fig.update_xaxes(title_text=feature)
            fig.update_yaxes(title_text='Predicted Outcome')
            st.plotly_chart(fig, use_container_width=True)
    
    elif results['type'] == 'correlation':
        st.markdown("### Statistical Correlations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                results['correlations'],
                x='correlation',
                y='feature',
                orientation='h',
                color='correlation',
                color_continuous_scale='RdBu_r',
                title='Feature Correlations with Outcome'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Correlation Table:**")
            st.dataframe(results['correlations'].round(4), use_container_width=True)
        
        # Significance
        significant = results['correlations'][results['correlations']['p_value'] < 0.05]
        st.markdown(f"### Statistically Significant Features (p < 0.05): {len(significant)}")
        if len(significant) > 0:
            st.dataframe(significant.round(4), use_container_width=True)
    
    elif results['type'] == 'causal_tree':
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Decision Rules")
            st.code(results['tree_rules'], language='text')
        
        with col2:
            fig = px.bar(
                results['feature_importance'],
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='Viridis',
                title='Causal Tree Feature Importance'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Policy implications
    st.markdown("---")
    st.markdown("### 💡 Policy Implications")
    
    feature_names = results.get('feature_importance', {}).get('feature', []).tolist() if hasattr(results.get('feature_importance', {}), 'get') else []
    
    if 'wealth_index' in feature_names:
        st.warning("""
        🔴 **Wealth Index is a key driver**
        
        Recommended interventions:
        - Targeted social protection programs for poorest households
        - Livelihood and income-generating activities
        - Conditional cash transfers
        - Asset-building programs
        """)
    
    if 'dietary_diversity_score' in feature_names:
        st.info("""
        🟡 **Dietary Diversity is crucial**
        
        Recommended interventions:
        - Nutrition education and counseling
        - Kitchen garden and home production initiatives
        - Food diversification campaigns
        - Promotion of nutrient-dense foods
        """)
    
    if 'has_improved_water' in feature_names or 'has_improved_sanitation' in feature_names:
        st.error("""
        🔴 **WASH (Water & Sanitation) is critical**
        
        Recommended interventions:
        - Infrastructure development for water and sanitation
        - Community hygiene promotion
        - Behavior change communication
        - Water quality monitoring
        """)
    
    if 'anemic' in feature_names:
        st.warning("""
        🟡 **Anemia is a significant factor**
        
        Recommended interventions:
        - Iron and folate supplementation programs
        - Nutrition-sensitive agriculture
        - Dietary iron intake promotion
        - Health service strengthening
        """)