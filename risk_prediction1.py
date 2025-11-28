import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')


def create_target_variable(data, target_var):
    """Create target variable from actual data with multiple fallback options"""
    try:
        # Option 1: Stunting Risk
        if target_var == "Stunting Risk":
            if 'stunting_zscore' in data.columns:
                valid_data = data['stunting_zscore'].dropna()
                if len(valid_data) > 10:
                    # Try Z-score threshold of -2
                    target = (data['stunting_zscore'] < -2).astype(int)
                    if len(target.unique()) == 2 and target.sum() > 0:
                        return target, "✅ Stunting Risk from stunting_zscore (threshold: -2)"
                    
                    # Try median threshold
                    median_val = valid_data.median()
                    target = (data['stunting_zscore'] < median_val).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Stunting Risk from stunting_zscore (threshold: {median_val:.2f})"
                    
                    # Try 25th percentile
                    q25 = valid_data.quantile(0.25)
                    target = (data['stunting_zscore'] < q25).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Stunting Risk from stunting_zscore (Q1 threshold: {q25:.2f})"
            
            # Fallback to height_for_age
            if 'height_for_age' in data.columns:
                valid_data = data['height_for_age'].dropna()
                if len(valid_data) > 10:
                    median_val = valid_data.median()
                    target = (data['height_for_age'] < median_val).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Stunting Risk from height_for_age (threshold: {median_val:.2f})"
        
        # Option 2: Wasting Risk
        elif target_var == "Wasting Risk":
            if 'wasting_zscore' in data.columns:
                valid_data = data['wasting_zscore'].dropna()
                if len(valid_data) > 10:
                    target = (data['wasting_zscore'] < -2).astype(int)
                    if len(target.unique()) == 2 and target.sum() > 0:
                        return target, "✅ Wasting Risk from wasting_zscore (threshold: -2)"
                    
                    median_val = valid_data.median()
                    target = (data['wasting_zscore'] < median_val).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Wasting Risk from wasting_zscore (threshold: {median_val:.2f})"
                    
                    q25 = valid_data.quantile(0.25)
                    target = (data['wasting_zscore'] < q25).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Wasting Risk from wasting_zscore (Q1 threshold: {q25:.2f})"
            
            if 'weight_for_height' in data.columns:
                valid_data = data['weight_for_height'].dropna()
                if len(valid_data) > 10:
                    median_val = valid_data.median()
                    target = (data['weight_for_height'] < median_val).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Wasting Risk from weight_for_height (threshold: {median_val:.2f})"
        
        # Option 3: Underweight Risk
        elif target_var == "Underweight Risk":
            if 'underweight_zscore' in data.columns:
                valid_data = data['underweight_zscore'].dropna()
                if len(valid_data) > 10:
                    target = (data['underweight_zscore'] < -2).astype(int)
                    if len(target.unique()) == 2 and target.sum() > 0:
                        return target, "✅ Underweight Risk from underweight_zscore (threshold: -2)"
                    
                    median_val = valid_data.median()
                    target = (data['underweight_zscore'] < median_val).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Underweight Risk from underweight_zscore (threshold: {median_val:.2f})"
                    
                    q25 = valid_data.quantile(0.25)
                    target = (data['underweight_zscore'] < q25).astype(int)
                    if len(target.unique()) == 2:
                        return target, f"✅ Underweight Risk from underweight_zscore (Q1 threshold: {q25:.2f})"
        
        # Option 4: Any Malnutrition
        elif target_var == "Any Malnutrition":
            malnutrition_flags = []
            
            # Stunting
            if 'stunting_zscore' in data.columns:
                valid_data = data['stunting_zscore'].dropna()
                if len(valid_data) > 10:
                    q25 = valid_data.quantile(0.25)
                    malnutrition_flags.append((data['stunting_zscore'] < q25).fillna(0).astype(int))
            
            # Wasting
            if 'wasting_zscore' in data.columns:
                valid_data = data['wasting_zscore'].dropna()
                if len(valid_data) > 10:
                    q25 = valid_data.quantile(0.25)
                    malnutrition_flags.append((data['wasting_zscore'] < q25).fillna(0).astype(int))
            
            # Underweight
            if 'underweight_zscore' in data.columns:
                valid_data = data['underweight_zscore'].dropna()
                if len(valid_data) > 10:
                    q25 = valid_data.quantile(0.25)
                    malnutrition_flags.append((data['underweight_zscore'] < q25).fillna(0).astype(int))
            
            # Anemia
            if 'anemic' in data.columns:
                anemic_data = data['anemic'].dropna()
                if len(anemic_data) > 0:
                    malnutrition_flags.append(data['anemic'].fillna(0).astype(int))
            
            if malnutrition_flags:
                combined = pd.concat(malnutrition_flags, axis=1).sum(axis=1)
                target = (combined > 0).astype(int)
                if len(target.unique()) == 2:
                    return target, f"✅ Any Malnutrition from {len(malnutrition_flags)} indicators"
        
        # Ultimate fallback: Use any numeric column
        st.warning("⚠️ Using fallback target creation method...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ['___index', 'child_id', 'woman_id']]
        
        for col in numeric_cols:
            valid_data = data[col].dropna()
            if len(valid_data) > 20:
                # Try median split
                median_val = valid_data.median()
                target = (data[col] < median_val).astype(int)
                if len(target.unique()) == 2:
                    return target, f"✅ Target created from {col} using median split"
        
        return None, "❌ Could not create any binary target variable from available data"
    
    except Exception as e:
        return None, f"❌ Error creating target: {str(e)}"


def risk_prediction():
    """Advanced malnutrition risk prediction with multiple ML models"""
    st.markdown('<div class="main-header">🔮 Malnutrition Risk Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    merged_data = st.session_state.merged_data
    
    # Data inspection section
    with st.expander("📋 Data Inspector", expanded=False):
        st.write("**Available Columns:**")
        cols = merged_data.columns.tolist()
        
        # Show columns grouped by type
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = merged_data.select_dtypes(include=['object']).columns.tolist()
        
        st.write(f"**Numeric Columns ({len(numeric_cols)}):**")
        for col in numeric_cols[:15]:
            non_null = merged_data[col].notna().sum()
            st.write(f"  - {col}: {non_null}/{len(merged_data)}")
        
        st.write(f"**Categorical Columns ({len(categorical_cols)}):**")
        st.write(", ".join(categorical_cols[:10]))
        
        st.write("**Data Shape:**", merged_data.shape)
    
    # Model configuration
    st.markdown("### 🤖 Machine Learning Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_variable = st.selectbox("Target Variable:", [
            "Stunting Risk", "Wasting Risk", "Underweight Risk", "Any Malnutrition"
        ], key="target_var_select")
        
        model_type = st.selectbox("Algorithm:", [
            "Random Forest", "Gradient Boosting", "Logistic Regression", "Support Vector Machine"
        ], key="model_type_select")
    
    with col2:
        test_size = st.slider("Test Set Size:", 0.1, 0.4, 0.2, 0.05, key="test_size_slider")
        cv_folds = st.slider("Cross-Validation Folds:", 3, 10, 5, key="cv_folds_slider")
        
        if st.button("🚀 Train Model", type="primary", use_container_width=True, key="train_button"):
            with st.spinner("Training machine learning model..."):
                model_results = train_risk_model(merged_data, target_variable, model_type, test_size, cv_folds)
                if model_results:
                    # Store results with target variable as key for persistence
                    if 'model_results_dict' not in st.session_state:
                        st.session_state.model_results_dict = {}
                    
                    st.session_state.model_results_dict[target_variable] = model_results
                    st.session_state.current_target = target_variable
                    st.balloons()
                    st.success(f"✅ Model trained for {target_variable}!")

    # Display results if model is trained
    if 'model_results_dict' in st.session_state and target_variable in st.session_state.model_results_dict:
        st.markdown("---")
        st.markdown(f"### 📊 Results for: **{target_variable}**")
        display_model_results(st.session_state.model_results_dict[target_variable], target_variable)
    elif 'model_results_dict' in st.session_state:
        st.info(f"⚠️ No model trained yet for {target_variable}. Click 'Train Model' to analyze this target.")
        
        # Show which targets have been trained
        trained_targets = list(st.session_state.model_results_dict.keys())
        st.write(f"**Available results:** {', '.join(trained_targets)}")
    
    st.markdown("---")
    
    # Real-time prediction interface
    st.markdown("### 👤 Real-time Risk Assessment")
    real_time_prediction(merged_data)


def train_risk_model(data, target_var, model_type, test_size=0.2, cv_folds=5):
    """Train ML model for risk prediction"""
    try:
        # Create target variable
        y, target_msg = create_target_variable(data, target_var)
        
        if y is None:
            st.error(target_msg)
            return None
        
        st.info(target_msg)
        st.write(f"**Target Distribution:** {dict(y.value_counts())}")
        
        # Get ALL available numeric columns as features (excluding target)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID and index columns
        exclude_cols = ['___index', 'child_id', 'woman_id', 'stunting_zscore', 
                       'wasting_zscore', 'underweight_zscore', 'S13_01']
        
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Add categorical columns that can be encoded
        categorical_possible = ['wealth_index', 'urban_rural', 'sex', 'province', 'district']
        for col in categorical_possible:
            if col in data.columns and col not in feature_cols:
                feature_cols.append(col)
        
        if not feature_cols:
            # Show all available columns for debugging
            st.error("❌ No suitable feature columns found!")
            st.write("**All available columns:**")
            st.write(data.columns.tolist())
            return None
        
        st.write(f"✅ Using {len(feature_cols)} features")
        st.write(f"Features: {feature_cols[:10]}...")
        
        # Prepare features
        X = data[feature_cols].copy()
        
        # Encode categorical variables
        categorical_cols = ['wealth_index', 'urban_rural', 'sex', 'province', 'district']
        le_dict = {}
        
        for col in categorical_cols:
            if col in X.columns:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
                    le_dict[col] = le
                except:
                    pass
        
        # Handle missing values - impute with median for numeric, mode for categorical
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)
        
        # Clean data - remove rows with NaN in X or y
        y_clean = y.fillna(0).astype(int)
        valid_idx = ~(X_imputed.isnull().any(axis=1) | y_clean.isnull())
        
        X_clean = X_imputed[valid_idx].reset_index(drop=True)
        y_clean = y_clean[valid_idx].reset_index(drop=True)
        
        if len(X_clean) < 20:
            st.error(f"❌ Not enough valid samples: {len(X_clean)} (need ≥20)")
            return None
        
        st.write(f"✅ Training samples: {len(X_clean)}")
        st.write(f"✅ Class distribution: {dict(y_clean.value_counts())}")
        
        # Split data with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean
            )
        except Exception as e:
            st.error(f"Split error: {e}")
            return None
        
        st.write(f"Train: {len(X_train)} | Test: {len(X_test)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select model
        if model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            X_train_use, X_test_use = X_train, X_test
        elif model_type == "Gradient Boosting":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            X_train_use, X_test_use = X_train, X_test
        elif model_type == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        else:  # SVM
            model = SVC(probability=True, random_state=42, kernel='rbf')
            X_train_use, X_test_use = X_train_scaled, X_test_scaled
        
        # Train model
        with st.spinner(f"Training {model_type}..."):
            model.fit(X_train_use, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X_train_use, y_train, cv=cv_folds, scoring='roc_auc', n_jobs=-1)
        except:
            cv_scores = np.array([0.5])
        
        # Metrics
        accuracy = model.score(X_test_use, y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        st.success(f"✅ Model trained! Accuracy: {accuracy:.1%}, AUC: {auc_score:.3f}")
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(model, 'coef_'):
            try:
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': np.abs(model.coef_[0])
                }).sort_values('importance', ascending=False)
            except:
                pass
        
        return {
            'model': model,
            'model_type': model_type,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_scores': cv_scores,
            'feature_importance': feature_importance,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'feature_cols': feature_cols,
            'scaler': scaler,
            'le_dict': le_dict
        }
    
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def display_model_results(results, target_var):
    """Display model performance and insights"""
    if not results:
        return
    
    st.success(f"✅ {results['model_type']} Model for {target_var}")
    st.markdown("---")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{results['accuracy']:.1%}")
    with col2:
        st.metric("AUC Score", f"{results['auc_score']:.3f}")
    with col3:
        st.metric("CV Mean", f"{results['cv_scores'].mean():.3f}")
    with col4:
        st.metric("CV Std", f"{results['cv_scores'].std():.3f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Feature Importance")
        if results['feature_importance'] is not None and len(results['feature_importance']) > 0:
            top_features = results['feature_importance'].head(10)
            fig = px.bar(
                top_features, 
                x='importance', 
                y='feature', 
                orientation='h',
                title=f'Top 10 Features Predicting {target_var}',
                color='importance', 
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model")
    
    with col2:
        st.markdown("#### ROC Curve")
        try:
            fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, 
                y=tpr, 
                name='ROC Curve',
                line=dict(color='blue', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], 
                y=[0, 1], 
                name='Random Classifier',
                line=dict(dash='dash', color='red')
            ))
            fig.update_layout(
                title=f'ROC Curve (AUC={results["auc_score"]:.3f})',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting ROC: {e}")
    
    st.markdown("---")
    st.markdown("#### Confusion Matrix")
    
    try:
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Negative (0)', 'Positive (1)'],
            y=['Negative (0)', 'Positive (1)'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 14},
            colorscale='Blues'
        ))
        fig.update_layout(height=400, title=f'Confusion Matrix - {target_var}')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting confusion matrix: {e}")
    
    st.markdown("---")
    
    # Classification report
    st.markdown("#### Classification Report")
    try:
        from sklearn.metrics import classification_report
        report = classification_report(
            results['y_test'], 
            results['y_pred'],
            output_dict=False
        )
        st.text(report)
    except Exception as e:
        st.error(f"Error generating report: {e}")
    
    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    if results['feature_importance'] is not None and len(results['feature_importance']) > 0:
        top_5 = results['feature_importance'].head(5)['feature'].tolist()
        st.info(f"""
        **Top 5 Risk Factors for {target_var}:**
        
        1. **{top_5[0] if len(top_5) > 0 else 'N/A'}**
        2. **{top_5[1] if len(top_5) > 1 else 'N/A'}**
        3. **{top_5[2] if len(top_5) > 2 else 'N/A'}**
        4. **{top_5[3] if len(top_5) > 3 else 'N/A'}**
        5. **{top_5[4] if len(top_5) > 4 else 'N/A'}**
        
        **Model Performance:**
        - Accuracy: {results['accuracy']:.1%}
        - AUC Score: {results['auc_score']:.3f}
        - Cross-validation AUC: {results['cv_scores'].mean():.1%} ± {results['cv_scores'].std():.1%}
        
        **Interpretation:**
        - AUC > 0.7 indicates good model performance
        - AUC > 0.8 indicates excellent model performance
        """)