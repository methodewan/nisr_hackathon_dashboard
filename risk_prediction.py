import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


def get_feature_importance(model, X, feature_names, model_type):
    """Extract feature importance from model"""
    try:
        if model_type in ["Random Forest", "Gradient Boosting"]:
            importance = model.feature_importances_
        else:
            # For Logistic Regression, handle coefficient shape
            coef = model.coef_
            if coef.shape[0] > 1:
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef[0])
        
        # Ensure feature_names and importance have same length
        if len(feature_names) != len(importance):
            feature_names = feature_names[:len(importance)]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    except Exception as e:
        st.error(f"Error calculating feature importance: {e}")
        return None


def prepare_training_data(merged_data):
    """Prepare data for model training"""
    try:
        # Create copy
        data = merged_data.copy()
        
        # Create target variable - Multiple malnutrition indicators
        # Try different target variables in order of preference
        target_created = False
        
        # Option 1: Stunting Risk (Z-score < -2)
        if 'stunting_zscore' in data.columns and not target_created:
            stunting_data = data['stunting_zscore'].dropna()
            if len(stunting_data) > 0 and stunting_data.std() > 0:
                data['malnutrition_risk'] = (data['stunting_zscore'] < -2).astype(int)
                if len(data['malnutrition_risk'].unique()) == 2:
                    target_created = True
                    st.info("✅ Using Stunting Risk as target variable")
        
        # Option 2: Wasting Risk (Z-score < -2)
        if 'wasting_zscore' in data.columns and not target_created:
            wasting_data = data['wasting_zscore'].dropna()
            if len(wasting_data) > 0 and wasting_data.std() > 0:
                data['malnutrition_risk'] = (data['wasting_zscore'] < -2).astype(int)
                if len(data['malnutrition_risk'].unique()) == 2:
                    target_created = True
                    st.info("✅ Using Wasting Risk as target variable")
        
        # Option 3: Any Micronutrient Deficiency
        if not target_created:
            # Check for anemia, vitamin A deficiency, etc.
            deficiency_indicators = []
            
            if 'anemic' in data.columns:
                deficiency_indicators.append(data['anemic'].fillna(0).astype(int))
            if 'received_vitamin_a' in data.columns:
                deficiency_indicators.append((1 - data['received_vitamin_a'].fillna(0).astype(int)))
            if 'received_ifa' in data.columns:
                deficiency_indicators.append((1 - data['received_ifa'].fillna(0).astype(int)))
            
            if deficiency_indicators:
                data['malnutrition_risk'] = (pd.concat(deficiency_indicators, axis=1).sum(axis=1) > 0).astype(int)
                if len(data['malnutrition_risk'].unique()) == 2:
                    target_created = True
                    st.info("✅ Using Micronutrient Deficiency as target variable")
        
        # Option 4: Dietary Diversity Risk (Low dietary diversity)
        if 'dietary_diversity_score' in data.columns and not target_created:
            diversity_data = data['dietary_diversity_score'].dropna()
            if len(diversity_data) > 0:
                threshold = diversity_data.median()
                data['malnutrition_risk'] = (data['dietary_diversity_score'] < threshold).astype(int)
                if len(data['malnutrition_risk'].unique()) == 2:
                    target_created = True
                    st.info("✅ Using Low Dietary Diversity as target variable")
        
        if not target_created:
            st.error("Cannot create target variable with both classes. Check your data!")
            return None, None, None, None, None
        
        # Display target variable distribution
        risk_dist = data['malnutrition_risk'].value_counts()
        st.write(f"**Target Variable Distribution:**")
        st.write(f"- Low Risk (0): {risk_dist.get(0, 0)} ({risk_dist.get(0, 0)/len(data)*100:.1f}%)")
        st.write(f"- High Risk (1): {risk_dist.get(1, 0)} ({risk_dist.get(1, 0)/len(data)*100:.1f}%)")
        
        # Select numeric and categorical features
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and ID columns
        exclude_cols = ['___index', 'child_id', 'woman_id', 'stunting_zscore', 'malnutrition_risk', 
                       'wasting_zscore', 'underweight_zscore', 'stunting_risk']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        categorical_features = [col for col in categorical_features if col not in exclude_cols]
        
        # Check if we have any features
        if not numeric_features and not categorical_features:
            st.error("No suitable features found for modeling")
            return None, None, None, None, None
        
        # Prepare numeric data with imputation
        X_numeric = pd.DataFrame()
        if numeric_features:
            X_numeric = data[numeric_features].copy()
            # Impute numeric features with median
            imputer = SimpleImputer(strategy='median')
            X_numeric = pd.DataFrame(
                imputer.fit_transform(X_numeric),
                columns=numeric_features
            )
        
        # Encode categorical variables
        X_categorical = pd.DataFrame()
        label_encoders = {}
        
        for col in categorical_features:
            if col in data.columns:
                le = LabelEncoder()
                valid_data = data[col].astype(str).fillna('Unknown')
                X_categorical[col] = le.fit_transform(valid_data)
                label_encoders[col] = le
        
        # Combine features
        if X_numeric.empty and X_categorical.empty:
            st.error("No features could be prepared")
            return None, None, None, None, None
        
        X = pd.concat([X_numeric, X_categorical], axis=1)
        y = data['malnutrition_risk']
        
        # Remove rows with NaN in target
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Double-check for any remaining NaN values
        if X.isnull().any().any():
            st.warning("⚠️ Still found NaN values, dropping rows with NaN...")
            valid_mask = ~X.isnull().any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
        
        # Final check for both classes
        if len(y.unique()) < 2:
            st.error(f"⚠️ Target variable has only {len(y.unique())} class(es). Need both 0 and 1.")
            return None, None, None, None, None
        
        st.success(f"✅ Data prepared: {len(X)} samples, {len(numeric_features) + len(categorical_features)} features")
        
        return X, y, numeric_features, categorical_features, label_encoders
    
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None


def train_single_model(X, y, model_type):
    """Train a single model"""
    try:
        # Check minimum samples
        if len(X) < 20:
            st.error("Not enough samples for training (minimum 20 required)")
            return None
        
        # Ensure no NaN values
        X = X.fillna(X.mean(numeric_only=True))
        
        # Remove any remaining NaN
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 20:
            st.error("Not enough valid samples after cleaning")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        elif model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        else:  # Gradient Boosting
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        train_accuracy = (y_pred_train == y_train).mean()
        test_accuracy = (y_pred_test == y_test).mean()
        
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0.0
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        except:
            cv_scores = np.array([test_accuracy])
        
        return {
            'model': model,
            'scaler': scaler,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'y_pred_proba': y_pred_proba,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'auc_score': auc_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    except Exception as e:
        st.error(f"Error training {model_type}: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def train_all_models(X, y):
    """Train all three models"""
    models_dict = {}
    model_types = ["Random Forest", "Logistic Regression", "Gradient Boosting"]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, model_type in enumerate(model_types):
        status_text.text(f"Training {model_type}... ({idx+1}/3)")
        results = train_single_model(X, y, model_type)
        
        if results:
            models_dict[model_type] = results
        
        progress_bar.progress((idx + 1) / len(model_types))
    
    if models_dict:
        status_text.text("✅ All models trained successfully!")
    else:
        status_text.text("❌ Failed to train models")
    
    return models_dict


def display_model_metrics(models_dict):
    """Display metrics for all models"""
    st.markdown("### 📊 Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, results in models_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Train Accuracy': results['train_accuracy'],
            'Test Accuracy': results['test_accuracy'],
            'AUC Score': results['auc_score'],
            'CV Mean': results['cv_mean'],
            'CV Std': results['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_test_acc = comparison_df.loc[comparison_df['Test Accuracy'].idxmax()]
        st.metric("Best Test Accuracy", f"{best_test_acc['Test Accuracy']:.1%}", 
                 delta=best_test_acc['Model'])
    
    with col2:
        best_auc = comparison_df.loc[comparison_df['AUC Score'].idxmax()]
        st.metric("Best AUC Score", f"{best_auc['AUC Score']:.3f}", 
                 delta=best_auc['Model'])
    
    with col3:
        best_cv = comparison_df.loc[comparison_df['CV Mean'].idxmax()]
        st.metric("Best CV Score", f"{best_cv['CV Mean']:.1%}", 
                 delta=best_cv['Model'])
    
    with col4:
        st.metric("Models Trained", len(models_dict))
    
    # Detailed comparison table
    st.markdown("#### Detailed Metrics")
    st.dataframe(comparison_df.round(4), use_container_width=True)
    
    # Visualization - Metrics Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig = go.Figure(data=[
            go.Bar(name='Train', x=comparison_df['Model'], y=comparison_df['Train Accuracy']),
            go.Bar(name='Test', x=comparison_df['Model'], y=comparison_df['Test Accuracy'])
        ])
        fig.update_layout(title='Accuracy Comparison', height=400, barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # AUC comparison
        fig = px.bar(comparison_df, x='Model', y='AUC Score', 
                    color='AUC Score', color_continuous_scale='Viridis',
                    title='AUC Score Comparison', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    return comparison_df


def display_model_details(models_dict, all_features, label_encoders):
    """Display detailed metrics for each model"""
    st.markdown("### 🔍 Detailed Model Analysis")
    
    selected_model = st.selectbox("Select Model to Analyze", list(models_dict.keys()))
    
    if selected_model:
        results = models_dict[selected_model]
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Classification Report", "Confusion Matrix", "ROC Curve", "Feature Importance"])
        
        with tab1:
            st.markdown(f"#### {selected_model} - Classification Report")
            report = classification_report(results['y_test'], results['y_pred_test'], 
                                         output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
        
        with tab2:
            st.markdown(f"#### {selected_model} - Confusion Matrix")
            cm = confusion_matrix(results['y_test'], results['y_pred_test'])
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['No Risk', 'At Risk'],
                y=['No Risk', 'At Risk'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues'
            ))
            fig.update_layout(title=f'{selected_model} - Confusion Matrix', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown(f"#### {selected_model} - ROC Curve")
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            fig = px.area(x=fpr, y=tpr, 
                         labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
                         title=f'{selected_model} - ROC Curve (AUC = {results["auc_score"]:.3f})')
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, 
                         line=dict(dash='dash', color='red'))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown(f"#### {selected_model} - Feature Importance")
            importance_df = get_feature_importance(results['model'], results['X_train'], 
                                                  all_features, selected_model)
            
            if importance_df is not None:
                # Top 15 features
                top_features = importance_df.head(15)
                
                fig = px.bar(top_features, x='Importance', y='Feature', 
                            orientation='h', color='Importance',
                            color_continuous_scale='Viridis',
                            title=f'{selected_model} - Top 15 Features')
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Full table
                with st.expander("📋 View All Features"):
                    st.dataframe(importance_df.reset_index(drop=True), use_container_width=True)


def risk_prediction():
    """Malnutrition risk prediction models"""
    st.markdown('<div class="main-header">🔮 Malnutrition Risk Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    
    # ==================== MODEL TRAINING SECTION ====================
    st.markdown("### 🤖 Predictive Model Development")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**Train all three models to compare their performance**")
    
    with col2:
        if st.button("🚀 Train All Models", type="primary", use_container_width=True):
            with st.spinner("🔄 Training all models..."):
                # Prepare data
                X, y, numeric_features, categorical_features, label_encoders = prepare_training_data(merged_data)
                
                if X is not None:
                    all_features = numeric_features + categorical_features
                    
                    # Train all models
                    models_dict = train_all_models(X, y)
                    
                    if models_dict:
                        # Store in session state
                        st.session_state.models_dict = models_dict
                        st.session_state.all_features = all_features
                        st.session_state.label_encoders = label_encoders
                        st.session_state.comparison_df = None
                        
                        st.balloons()
    
    with col3:
        if hasattr(st.session_state, 'models_dict') and st.session_state.models_dict:
            if st.button("📊 Compare", use_container_width=True):
                st.session_state.show_comparison = True
    
    # ==================== MODEL PERFORMANCE METRICS ====================
    if hasattr(st.session_state, 'models_dict') and st.session_state.models_dict:
        comparison_df = display_model_metrics(st.session_state.models_dict)
        st.session_state.comparison_df = comparison_df
        
        # ==================== DETAILED ANALYSIS ====================
        st.markdown("---")
        display_model_details(st.session_state.models_dict, st.session_state.all_features, 
                            st.session_state.label_encoders)
        
        # ==================== MODEL SELECTION ====================
        st.markdown("---")
        st.markdown("### 🎯 Best Model Selection")
        
        best_model_name = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
        best_results = st.session_state.models_dict[best_model_name]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.success(f"**Recommended Model:** {best_model_name}")
            st.metric("Test Accuracy", f"{best_results['test_accuracy']:.1%}")
            st.metric("AUC Score", f"{best_results['auc_score']:.3f}")
        
        with col2:
            st.info(f"""
            The **{best_model_name}** model achieves the best performance on the test set.
            
            **Why?**
            - Highest test accuracy: {best_results['test_accuracy']:.1%}
            - Highest AUC score: {best_results['auc_score']:.3f}
            - Consistent cross-validation: {best_results['cv_mean']:.1%} ± {best_results['cv_std']:.1%}
            
            **Use this model for:**
            - Production predictions
            - Individual risk assessments
            - Policy recommendations
            """)
        
        # Store best model
        st.session_state.best_model = best_model_name
        st.session_state.best_results = best_results
    
    # ==================== INDIVIDUAL RISK ASSESSMENT ====================
    if hasattr(st.session_state, 'best_results'):
        st.markdown("---")
        st.markdown("### 👤 Individual Risk Assessment (Using Best Model)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Household Characteristics**")
            wealth = st.selectbox("Wealth Index", ["Poorest", "Poor", "Middle", "Rich", "Richest"])
            hh_size = st.slider("Household Size", 1, 15, 4)
            urban_rural = st.selectbox("Location", ["Urban", "Rural"])
        
        with col2:
            st.markdown("**Dietary & Nutrition**")
            dietary_diversity = st.slider("Dietary Diversity Score", 0, 12, 5)
            dietary_diversity_score = st.slider("HH Dietary Score", 0, 100, 50)
            received_vitamin_a = st.selectbox("Child Vitamin A", ["Yes", "No"])
        
        with col3:
            st.markdown("**WASH & Health**")
            improved_water = st.selectbox("Improved Water Source", ["Yes", "No"])
            improved_sanitation = st.selectbox("Improved Sanitation", ["Yes", "No"])
            age_months = st.slider("Child Age (months)", 6, 59, 24)
        
        if st.button("🔍 Assess Individual Risk", type="primary", use_container_width=True):
            try:
                # Create input feature vector
                input_data = pd.DataFrame({
                    'wealth_index': [wealth],
                    'urban_rural': [urban_rural],
                    'improved_water': [1 if improved_water == 'Yes' else 0],
                    'improved_sanitation': [1 if improved_sanitation == 'Yes' else 0],
                    'age_months': [age_months],
                    'dietary_diversity': [dietary_diversity],
                    'dietary_diversity_score': [dietary_diversity_score],
                    'received_vitamin_a': [1 if received_vitamin_a == 'Yes' else 0],
                    'hh_size': [hh_size]
                })
                
                # Encode categorical variables
                for col in ['wealth_index', 'urban_rural']:
                    if col in st.session_state.label_encoders:
                        le = st.session_state.label_encoders[col]
                        input_data[col] = le.transform([input_data[col].iloc[0]])[0]
                
                # Scale and predict
                input_scaled = st.session_state.best_results['scaler'].transform(input_data)
                risk_probability = st.session_state.best_results['model'].predict_proba(input_scaled)[0, 1]
                
                # Display results
                st.markdown("#### Risk Assessment Result")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if risk_probability > 0.7:
                        st.error(f"🚨 **HIGH RISK** - {risk_probability:.1%}")
                        st.warning("""
                        **Immediate Interventions Required:**
                        - Nutritional supplementation (micronutrients)
                        - WASH improvements and education
                        - Livelihood and cash transfer support
                        - Monthly growth monitoring
                        - Maternal nutrition programs
                        """)
                    elif risk_probability > 0.4:
                        st.warning(f"⚠️ **MEDIUM RISK** - {risk_probability:.1%}")
                        st.info("""
                        **Recommended Interventions:**
                        - Nutrition education programs
                        - Dietary diversification support
                        - Quarterly health check-ups
                        - Community nutrition programs
                        """)
                    else:
                        st.success(f"✅ **LOW RISK** - {risk_probability:.1%}")
                        st.info("""
                        **Preventive Measures:**
                        - Continue current nutrition practices
                        - Regular monitoring (every 3-6 months)
                        - Maintain dietary diversity
                        - Ensure access to WASH services
                        """)
                
                with col2:
                    # Risk gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=risk_probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Risk Score"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error in risk assessment: {e}")
                import traceback
                st.error(traceback.format_exc())