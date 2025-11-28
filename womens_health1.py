import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

def womens_health():
    """Women's health analysis with ML predictions"""
    st.markdown('<div class="main-header">Women\'s Health & Nutrition (15-49 years)</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    women_data = st.session_state.women_data
    
    # ML-powered women's health analysis
    st.markdown("### Predictive Health Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        health_outcome = st.selectbox("Health Outcome to Predict", [
            "Anemia Risk", "Low BMI Risk", "Nutritional Deficiency", 
            "Supplement Need"
        ])
    
    with col2:
        if st.button("Generate Predictions"):
            with st.spinner("Analyzing women's health patterns..."):
                predictions = predict_womens_health_risks(women_data, health_outcome)
                display_health_predictions(predictions, health_outcome)
    
    # Health indicators
    st.markdown("### Key Health Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'anemic' in women_data.columns:
            anemia_rate = women_data['anemic'].mean() * 100
            st.metric("Anemia Rate", f"{anemia_rate:.1f}%")
        else:
            st.metric("Anemia Rate", "Data not available")
    
    with col2:
        if 'received_ifa' in women_data.columns:
            ifa_coverage = women_data['received_ifa'].mean() * 100
            st.metric("IFA Coverage", f"{ifa_coverage:.1f}%")
        else:
            st.metric("IFA Coverage", "Data not available")
    
    with col3:
        if 'bmi' in women_data.columns:
            low_bmi = (women_data['bmi'] < 18.5).mean() * 100
            st.metric("Low BMI (<18.5)", f"{low_bmi:.1f}%")
        else:
            st.metric("Low BMI", "Data not available")
    
    with col4:
        if 'dietary_diversity' in women_data.columns:
            avg_diversity = women_data['dietary_diversity'].mean()
            st.metric("Avg Dietary Diversity", f"{avg_diversity:.1f}")
        else:
            st.metric("Dietary Diversity", "Data not available")
    
    # Risk stratification
    st.markdown("### Population Risk Stratification")
    perform_risk_stratification(women_data)

def predict_womens_health_risks(data, outcome):
    """Predict women's health risks using ML"""
    try:
        analysis_data = data.copy()
        
        # Prepare features
        feature_cols = ['age_years', 'education_level', 'pregnant', 'breastfeeding']
        
        # Create target variable based on outcome
        if outcome == "Anemia Risk":
            if 'anemic' in analysis_data.columns:
                y = analysis_data['anemic']
            else:
                # Synthetic data for demonstration
                analysis_data['anemic'] = np.random.choice([0, 1], len(analysis_data), p=[0.7, 0.3])
                y = analysis_data['anemic']
        
        elif outcome == "Low BMI Risk":
            if 'bmi' in analysis_data.columns:
                y = (analysis_data['bmi'] < 18.5).astype(int)
            else:
                analysis_data['low_bmi_risk'] = np.random.choice([0, 1], len(analysis_data), p=[0.8, 0.2])
                y = analysis_data['low_bmi_risk']
        
        elif outcome == "Supplement Need":
            # Complex logic for supplement need
            if 'anemic' in analysis_data.columns and 'pregnant' in analysis_data.columns:
                y = ((analysis_data['anemic'] == 1) | (analysis_data['pregnant'] == 1)).astype(int)
            else:
                y = np.random.choice([0, 1], len(analysis_data), p=[0.5, 0.5])
        
        else:  # Nutritional Deficiency
            if 'dietary_diversity' in analysis_data.columns:
                y = (analysis_data['dietary_diversity'] < 5).astype(int)
            else:
                y = np.random.choice([0, 1], len(analysis_data), p=[0.6, 0.4])
        
        # Prepare features
        X = analysis_data[feature_cols].copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        for col in ['education_level']:
            if col in X.columns:
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        return {
            'model': model,
            'feature_importance': feature_importance,
            'accuracy': model.score(X_test, y_test),
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'prediction_proba': y_pred_proba,
            'outcome': outcome
        }
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def display_health_predictions(results, outcome):
    """Display health prediction results"""
    if not results:
        return
    
    st.success(f"✅ {outcome} predictions generated!")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", f"{results['accuracy']:.1%}")
    
    with col2:
        st.metric("AUC Score", f"{results['auc_score']:.3f}")
    
    with col3:
        high_risk_rate = results['predictions'].mean() * 100
        st.metric("High-Risk Population", f"{high_risk_rate:.1f}%")
    
    # Feature importance
    st.markdown("### Key Predictors")
    fig = px.bar(results['feature_importance'], x='importance', y='feature',
                orientation='h', title=f'Feature Importance for {outcome}')
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk distribution
    st.markdown("### Risk Distribution")
    risk_proba = results['prediction_proba']
    
    fig = px.histogram(x=risk_proba, nbins=20, 
                      title='Predicted Risk Probability Distribution',
                      labels={'x': 'Risk Probability', 'y': 'Count'})
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                 annotation_text="Risk Threshold")
    st.plotly_chart(fig, use_container_width=True)

def perform_risk_stratification(data):
    """Perform risk stratification of women population"""
    try:
        # Create risk scores based on available data
        risk_data = data.copy()
        
        # Calculate composite risk score
        risk_score = 0
        
        # Age risk (very young or older mothers)
        if 'age_years' in risk_data.columns:
            risk_data['age_risk'] = np.where(
                (risk_data['age_years'] < 20) | (risk_data['age_years'] > 35), 1, 0
            )
            risk_score += risk_data['age_risk']
        
        # Anemia risk
        if 'anemic' in risk_data.columns:
            risk_score += risk_data['anemic']
        
        # Low BMI risk
        if 'bmi' in risk_data.columns:
            risk_data['bmi_risk'] = (risk_data['bmi'] < 18.5).astype(int)
            risk_score += risk_data['bmi_risk']
        
        # Pregnancy risk
        if 'pregnant' in risk_data.columns:
            risk_score += risk_data['pregnant']
        
        # Low education risk (proxy)
        if 'education_level' in risk_data.columns:
            risk_data['education_risk'] = risk_data['education_level'].isin(['None', 'Primary']).astype(int)
            risk_score += risk_data['education_risk']
        
        # Categorize risk levels
        risk_data['risk_category'] = pd.cut(risk_score, 
                                          bins=[-1, 1, 2, 10],
                                          labels=['Low', 'Medium', 'High'])
        
        # Display risk stratification
        risk_counts = risk_data['risk_category'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                        title='Women Population Risk Stratification')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Risk Category Definitions")
            st.info("""
            **Low Risk (0-1 factors):**
            - Generally healthy
            - Continue routine care
            
            **Medium Risk (2 factors):**
            - Requires monitoring
            - Basic interventions
            
            **High Risk (3+ factors):**
            - Priority for interventions
            - Intensive support needed
            """)
        
        # High-risk profile
        high_risk = risk_data[risk_data['risk_category'] == 'High']
        if len(high_risk) > 0:
            st.warning(f"🚨 {len(high_risk)} women identified as high risk")
            
            # High-risk characteristics
            st.markdown("#### High-Risk Population Profile")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'age_years' in high_risk.columns:
                    avg_age = high_risk['age_years'].mean()
                    st.metric("Average Age", f"{avg_age:.1f} years")
            
            with col2:
                if 'pregnant' in high_risk.columns:
                    pregnant_pct = high_risk['pregnant'].mean() * 100
                    st.metric("Pregnant", f"{pregnant_pct:.1f}%")
            
            with col3:
                if 'anemic' in high_risk.columns:
                    anemic_pct = high_risk['anemic'].mean() * 100
                    st.metric("Anemic", f"{anemic_pct:.1f}%")
    
    except Exception as e:
        st.error(f"Risk stratification failed: {e}")

# Education and nutrition analysis from original function
def education_nutrition_analysis(women_data):
    """Analyze relationship between education and nutrition"""
    if 'education_level' in women_data.columns and 'bmi' in women_data.columns and 'anemic' in women_data.columns:
        education_nutrition = women_data.groupby('education_level').agg({
            'bmi': 'mean',
            'anemic': 'mean'
        }).reset_index()
        
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=('Average BMI by Education', 'Anemia Rate by Education'))
        
        fig.add_trace(
            go.Bar(x=education_nutrition['education_level'], y=education_nutrition['bmi'],
                   name='BMI'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=education_nutrition['education_level'], y=education_nutrition['anemic'] * 100,
                   name='Anemia Rate'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)