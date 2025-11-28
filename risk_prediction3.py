import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import joblib
import plotly.express as px
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score

MODEL_PATH = "saved_model.joblib"
MODEL_INFO_PATH = "saved_model_info.json"

def risk_prediction(): # Main function for the page
    """Malnutrition risk prediction models"""
    st.markdown('<div class="main-header">🔮 Malnutrition Risk Prediction</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Dashboard page")
        return
    
    # Load saved model on first run of the page
    if 'model_loaded_from_disk' not in st.session_state:
        load_saved_model()
        st.session_state.model_loaded_from_disk = True

    merged_data = st.session_state.merged_data
    
    # Show data overview
    with st.expander("📊 Data Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(merged_data))
        with col2:
            st.metric("Available Features", len(merged_data.columns))
        with col3:
            if 'any_malnutrition' in merged_data.columns:
                malnutrition_rate = merged_data['any_malnutrition'].mean() * 100
                st.metric("Malnutrition Rate", f"{malnutrition_rate:.1f}%")
            else:
                st.metric("Malnutrition Rate", "N/A")
    
    # Model training section
    st.markdown("### 🛠️ Manual Model Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_variable = st.selectbox("Target Variable", [
            "Stunting Risk", "Wasting Risk", "Any Malnutrition",
            "Low Dietary Diversity", "Household Food Insecurity"
        ])
        
        # Map target variable to actual columns
        target_map = {
            "Stunting Risk": "stunting_risk",
            "Wasting Risk": "wasting_risk", 
            "Any Malnutrition": "any_malnutrition",
            "Low Dietary Diversity": "low_dietary_diversity_risk",
            "Household Food Insecurity": "food_insecurity_risk"
        }
        
        target_col = target_map[target_variable]
        
        # Show target distribution
        if target_col in merged_data.columns:
            target_counts = merged_data[target_col].value_counts()
            pos_count = target_counts.get(1, 0)
            neg_count = target_counts.get(0, len(merged_data) - pos_count)
            st.write(f"**Target Distribution:** {pos_count} positive, {neg_count} negative")
            st.write(f"**Prevalence:** {pos_count/(pos_count+neg_count):.1%}")
        else:
            st.warning(f"⚠️ Target column '{target_col}' not found in data")
            # Create demo target for testing
            merged_data[target_col] = np.random.choice([0, 1], size=len(merged_data), p=[0.7, 0.3])
            st.info("Created demo target variable for testing")
    
    with col2:
        model_type = st.selectbox("Model Algorithm", [
            "Random Forest", "XGBoost", "Gradient Boosting", "Neural Network", "Logistic Regression"
        ])
        
        # Model parameters based on type
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", 50, 200, 100)
            max_depth = st.slider("Max Depth", 3, 20, 10)
        elif model_type in ["Gradient Boosting", "XGBoost"]:
            n_estimators = st.slider("Number of Estimators", 50, 200, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        elif model_type == "Neural Network":
            hidden_layer_neurons = st.slider("Neurons in Hidden Layer", 20, 200, 100)
            alpha = st.select_slider("Regularization (alpha)", [0.0001, 0.001, 0.01, 0.1, 1.0], value=0.001)

        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        
    # --- Manual Feature Selection ---
    st.markdown("#### Select Features for Modeling")
    
    # Get all potential features (exclude IDs and all potential targets)
    potential_targets = list(target_map.values())
    exclude_cols = ['___index', 'child_id', 'woman_id'] + potential_targets
    all_features = [col for col in merged_data.columns if col not in exclude_cols]
    
    # Default features to pre-select
    default_features = [feat for feat in ['wealth_index', 'fcs', 'rcsi', 'age_months', 'dietary_diversity', 'education_level'] if feat in all_features]
    
    selected_features_manual = st.multiselect("Choose features:", all_features, default=default_features)
    
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        if not selected_features_manual:
            st.warning("Please select at least one feature to train the model.")
        else:
            with st.spinner("Preparing data and training model..."):
                try:
                    model_data = prepare_model_data(merged_data, target_col, selected_features_manual)
                    
                    if model_data is not None:
                        X_train, X_test, y_train, y_test, feature_names = model_data
                        
                        # Train selected model with parameters
                        if model_type == "Random Forest":
                            model = RandomForestClassifier(
                                n_estimators=n_estimators, 
                                max_depth=max_depth, 
                                random_state=42,
                                n_jobs=-1
                            )
                        elif model_type == "Logistic Regression":
                            model = LogisticRegression(random_state=42, max_iter=1000)
                        elif model_type == "Gradient Boosting":
                            model = GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                random_state=42
                            )
                        elif model_type == "XGBoost":
                            model = xgb.XGBClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                random_state=42,
                                use_label_encoder=False,
                                eval_metric='logloss'
                            )
                        elif model_type == "Neural Network":
                            model = MLPClassifier(
                                hidden_layer_sizes=(hidden_layer_neurons,),
                                alpha=alpha,
                                random_state=42,
                                max_iter=500
                            )
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Store everything in session state with safe defaults
                        st.session_state.current_model = {
                            'model': model,
                            'feature_names': feature_names,
                            'selected_features': selected_features_manual,
                            'target_variable': target_variable,
                            'model_type': model_type,
                            'X_test': X_test,
                            'y_test': y_test,
                            'feature_importance': get_feature_importance_dict(model, feature_names),
                            'training_time': pd.Timestamp.now()
                        }
                        
                        # Evaluate model
                        evaluate_model(model, X_test, y_test, model_type, feature_names, selected_features_manual)
                        
                        st.success("✅ Model trained and ready for predictions!")
                        
                except Exception as e:
                    st.error(f"❌ Error training model: {e}")

def save_model_and_info(model_info):
    """Saves the model object and its metadata."""
    try:
        # Separate model object from JSON-serializable info
        model_to_save = model_info.pop('model', None)
        
        if model_to_save:
            joblib.dump(model_to_save, MODEL_PATH)
            
            # Clean up non-serializable items for JSON
            info_to_save = model_info.copy()
            info_to_save.pop('X_test', None)
            info_to_save.pop('y_test', None)
            info_to_save.pop('training_time', None)
            info_to_save.pop('feature_importance', None)

            with open(MODEL_INFO_PATH, 'w') as f:
                json.dump(info_to_save, f)
            
            st.success(f"✅ Model saved as `{MODEL_PATH}`. It will now be used for Quick Assessment.")
        else:
            st.error("❌ No model object found to save.")
            
    except Exception as e:
        st.error(f"Error saving model: {e}")

def load_saved_model():
    """Loads a saved model and its info into session state."""
    if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_INFO_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            with open(MODEL_INFO_PATH, 'r') as f:
                model_info = json.load(f)
            model_info['model'] = model
            st.session_state.current_model = model_info
            st.toast(f"✅ Loaded saved model: **{model_info.get('model_type')}**")
        except Exception as e:
            st.error(f"Failed to load saved model: {e}")

def prepare_model_data(merged_data, target_variable, selected_features):
    """Prepare data for model training using manually selected features."""
    try:
        # Check if target variable exists
        if target_variable not in merged_data.columns:
            st.warning(f"Target variable '{target_variable}' not found in data")
            return None
        
        if not selected_features:
            st.error("❌ No features were selected for modeling.")
            return None
        
        st.write(f"🔄 Preparing data with {len(selected_features)} selected features...")
        
        # Create working copy
        model_data = merged_data[selected_features + [target_variable]].copy()
        
        # Remove rows with missing target
        model_data = model_data.dropna(subset=[target_variable])
        
        if len(model_data) == 0:
            st.warning("No data available after cleaning target variable")
            return None
        
        st.write(f"📊 Samples available: {len(model_data)}")
        
        # Separate features and target
        X = model_data[selected_features]
        y = model_data[target_variable]
        
        # Remove rows with all features missing
        X = X.dropna(how='all')
        y = y.loc[X.index]
        
        if len(X) == 0:
            st.warning("No data available after handling missing features")
            return None
        
        # Preprocess the selected features
        X_processed, feature_names = preprocess_features(X)
        
        if X_processed.shape[1] == 0:
            st.error("❌ No features remaining after preprocessing. Some features might have only one unique value.")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        st.write(f"📚 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except Exception as e:
        st.error(f"Error in data preparation: {e}")
        return None

def preprocess_features(X):
    """Automatically preprocess features"""
    processed_features = []
    feature_names = []
    
    for col in X.columns:
        try:
            col_data = X[col].copy()
            
            # Handle missing values
            if col_data.isna().any():
                if pd.api.types.is_numeric_dtype(col_data):
                    col_data = col_data.fillna(col_data.median())
                else:
                    col_data = col_data.fillna(col_data.mode()[0] if len(col_data.mode()) > 0 else 'Unknown')
            
            # Encode based on data type
            if pd.api.types.is_numeric_dtype(col_data):
                # Scale numeric features
                if col_data.nunique() > 1:
                    if col_data.std() > 0:
                        col_data = (col_data - col_data.mean()) / col_data.std()
                    processed_features.append(col_data.values)
                    feature_names.append(f"{col}_numeric")
            else:
                # One-hot encode categorical features
                encoded, _ = pd.factorize(col_data)
                if len(np.unique(encoded)) > 1:
                    # Normalize encoded values
                    if np.std(encoded) > 0:
                        encoded = (encoded - np.mean(encoded)) / np.std(encoded)
                    processed_features.append(encoded)
                    feature_names.append(f"{col}_categorical")
                    
        except Exception as e:
            continue  # Skip problematic features
    
    if processed_features:
        X_processed = np.column_stack(processed_features)
        return X_processed, feature_names
    else:
        return np.array([]), []

def get_feature_importance_dict(model, feature_names):
    """Get feature importance as dictionary"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    return {}

def evaluate_model(model, X_test, y_test, model_type, feature_names, selected_features):
    """Comprehensive model evaluation for trained models"""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Handle AUC calculation
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = 0.5
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Display metrics
        st.markdown("#### 📈 Model Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("AUC Score", f"{auc_score:.3f}")
        with col3:
            st.metric("Precision", f"{precision:.1%}")
        with col4:
            st.metric("Recall", f"{recall:.1%}")
        with col5:
            st.metric("F1 Score", f"{f1:.1%}")
        
        # Show selected features
        st.markdown("#### 🔍 Features Used in Model")
        st.write(f"**Total features used:** {len(selected_features)}")
        
        # Display features in a nice format
        cols = 3
        features_per_col = (len(selected_features) + cols - 1) // cols
        
        feature_cols = st.columns(cols)
        for i, col in enumerate(feature_cols):
            start_idx = i * features_per_col
            end_idx = min((i + 1) * features_per_col, len(selected_features))
            with col:
                for feature in selected_features[start_idx:end_idx]:
                    st.write(f"• {feature}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            # Clean feature names for display
            clean_feature_names = [f.split('_')[0] for f in feature_names]
            importance_df = pd.DataFrame({
                'Feature': clean_feature_names,
                'Importance': model.feature_importances_
            })
            # Group by original feature name and sum importance
            importance_data = importance_df.groupby('Feature')['Importance'].sum().reset_index()
            importance_data = importance_data.nlargest(10, 'Importance')
            
            st.markdown("#### 📊 Top Feature Importance")
            fig = px.bar(importance_data, x='Importance', y='Feature', 
                         orientation='h', title='Most Important Features',
                         color='Importance', color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("#### 🎯 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, 
                          labels=dict(x="Predicted", y="Actual", color="Count"),
                          x=['No Risk', 'Risk'],
                          y=['No Risk', 'Risk'],
                          title="Confusion Matrix",
                          color_continuous_scale='Blues',
                          text_auto=True)
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Model insights
        st.markdown("#### 💡 Model Insights")
        if accuracy > 0.8:
            st.success("**Excellent model performance!** This model can reliably predict malnutrition risk.")
        elif accuracy > 0.7:
            st.info("**Good model performance.** The model provides useful predictions for risk assessment.")
        elif accuracy > 0.6:
            st.warning("**Moderate model performance.** Consider trying different algorithms or more data.")
        else:
            st.error("**Poor model performance.** The data may not contain strong predictors, or more feature engineering is needed.")
        
    except Exception as e:
        st.error(f"Error evaluating model: {e}")

def show_model_results():
    """Display model results with safe key access"""
    if not st.session_state.get('current_model'):
        return
        
    model_info = st.session_state.current_model
    
    # Safe key access with defaults
    selected_features = model_info.get('selected_features', [])
    feature_names = model_info.get('feature_names', [])
    target_variable = model_info.get('target_variable', 'Unknown')
    model_type = model_info.get('model_type', 'Unknown')
    
    st.markdown("---")
    st.markdown("### 📈 Model Results")
    
    # Add a button to save the current model
    if st.button("💾 Save this Model for Quick Assessment"):
        save_model_and_info(model_info.copy())

    # Model summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"**Model Type:** {model_type}")
        st.info(f"**Target:** {target_variable}")
    with col2:
        st.info(f"**Features Used:** {len(selected_features)}")
        st.info(f"**Manually Selected:** Yes")
    with col3:
        if 'X_test' in model_info:
            st.info(f"**Test Samples:** {len(model_info['X_test'])}")
        if 'training_time' in model_info:
            st.info(f"**Trained:** {model_info['training_time'].strftime('%H:%M:%S')}")
    
    # Show selected features safely
    if selected_features:
        with st.expander("🔍 View Selected Features"):
            cols = 3
            features_per_col = (len(selected_features) + cols - 1) // cols
            
            feature_cols = st.columns(cols)
            for i, col in enumerate(feature_cols):
                start_idx = i * features_per_col
                end_idx = min((i + 1) * features_per_col, len(selected_features))
                with col:
                    for feature in selected_features[start_idx:end_idx]:
                        st.write(f"• {feature}")
    else:
        st.warning("No feature information available")

def individual_prediction_interface():
    """Interface for individual risk prediction using the trained model"""
    st.markdown("---")
    st.markdown("### 👤 Individual Risk Assessment")

    tab1, tab2 = st.tabs(["🤖 AI Model Prediction", "📋 Quick Assessment"])

    with tab1:
        if not st.session_state.get('current_model'):
            st.info("🔮 **No AI model available**. Train a model in the section above or ensure a `saved_model.joblib` file exists in your project directory to enable AI predictions.")
        else:
            model_info = st.session_state.current_model
            selected_features = model_info.get('selected_features', [])
            
            st.markdown("#### 🔍 Make a Prediction with the AI Model")
            st.info(f"Enter the details below. The prediction will be made using the loaded **{model_info.get('model_type')}** model.")
            
            # Create input form
            input_data = create_prediction_input_form(selected_features, st.session_state.merged_data)
            
            if st.button("🎯 Predict Risk with AI Model", type="primary"):
                try:
                    # 1. Create a DataFrame from the user's input
                    input_df = pd.DataFrame([input_data])
                    
                    # 2. Ensure all selected features are present, even if not in the form
                    for col in selected_features:
                        if col not in input_df.columns:
                            # Use a default/median value from the original dataset
                            input_df[col] = st.session_state.merged_data[col].median() if pd.api.types.is_numeric_dtype(st.session_state.merged_data[col]) else st.session_state.merged_data[col].mode()[0]

                    # 3. Preprocess the input DataFrame exactly like the training data
                    input_processed, _ = preprocess_features(input_df[selected_features])

                    if input_processed.shape[1] == 0:
                        st.error("Could not process inputs for prediction.")
                    else:
                        # 4. Predict the probability
                        model = model_info['model']
                        risk_probability = model.predict_proba(input_processed)[0, 1]
                        
                        # Display results
                        display_model_prediction(risk_probability, model_info)
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

    with tab2:
        show_quick_assessment()

def get_important_features_for_input(model_info):
    """Get the most important features for user input with safe access"""
    # Common features that are usually important
    common_features = [
        'age_months', 'hh_size', 'dietary_diversity', 'wealth_index',
        'urban_rural', 'education_level', 'has_improved_water', 
        'has_improved_sanitation', 'bmi', 'anemic'
    ]
    
    # If we have feature importance, use top features
    feature_importance = model_info.get('feature_importance', {})
    if feature_importance:
        top_features = sorted(feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        important_features = [feat.split('_')[0] for feat, imp in top_features]
    else:
        important_features = common_features
    
    return important_features

def create_prediction_input_form(features, source_data):
    """Create input form based on important features"""
    st.markdown("Enter individual characteristics for risk prediction:")
    
    col1, col2, col3 = st.columns(3)
    input_data = {}
    
    form_features = [f for f in features if f in source_data.columns]
    
    # Map features to input widgets
    feature_widgets = {
        'age_months': ("Child Age (months)", "slider", (6, 59, 24)),
        'hh_size': ("Household Size", "slider", (1, 10, 4)),
        'dietary_diversity': ("Dietary Diversity Score", "slider", (0, 10, 5)),
        'wealth_index': ("Wealth Index", "select", ["Poorest", "Poor", "Middle", "Rich", "Richest"]),
        'urban_rural': ("Location", "select", ["Urban", "Rural"]),
        'education_level': ("Maternal Education", "select", ["None", "Primary", "Secondary", "Higher"]),
        'has_improved_water': ("Improved Water Source", "select", ["Yes", "No"]),
        'has_improved_sanitation': ("Improved Sanitation", "select", ["Yes", "No"]),
        'bmi': ("BMI", "slider", (15, 35, 22)),
        'anemic': ("Anemic", "select", ["Yes", "No"])
    }
    
    # Distribute features across columns
    features_per_col = (len(form_features) + 2) // 3  # Round up division
    
    for i, feature in enumerate(form_features):
        if feature in feature_widgets: # Use a predefined widget
            label, widget_type, params = feature_widgets[feature]
            
            if i < features_per_col:
                col = col1
            elif i < 2 * features_per_col:
                col = col2
            else:
                col = col3
                
            with col:
                if widget_type == "slider":
                    min_val, max_val, default_val = params
                    input_data[feature] = st.slider(label, min_val, max_val, default_val)
                elif widget_type == "select":
                    input_data[feature] = st.selectbox(label, params)
        else: # Dynamically create a widget
            col_data = source_data[feature]
            if i < features_per_col: col = col1
            elif i < 2 * features_per_col: col = col2
            else: col = col3

            with col:
                if pd.api.types.is_numeric_dtype(col_data) and col_data.nunique() > 10:
                    min_val, max_val, mean_val = col_data.min(), col_data.max(), col_data.mean()
                    input_data[feature] = st.slider(f"{feature}", float(min_val), float(max_val), float(mean_val))
                else:
                    options = col_data.unique()
                    options = [opt for opt in options if pd.notna(opt)]
                    if options:
                        input_data[feature] = st.selectbox(f"{feature}", options)
    
    return input_data

def display_model_prediction(risk_probability, model_info):
    """Display prediction results from the trained model"""
    st.markdown("---")
    st.markdown("### 📊 AI Model Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk classification
        if risk_probability > 0.7:
            st.error(f"🔴 HIGH RISK: {risk_probability:.1%}")
            st.warning("""
            **Immediate Actions Recommended:**
            • Nutritional supplementation
            • Medical assessment
            • WASH interventions
            • Food assistance
            • Regular monitoring
            """)
        elif risk_probability > 0.4:
            st.warning(f"🟡 MEDIUM RISK: {risk_probability:.1%}")
            st.info("""
            **Preventive Measures:**
            • Nutrition education
            • Dietary diversification
            • Growth monitoring
            • Health check-ups
            """)
        else:
            st.success(f"🟢 LOW RISK: {risk_probability:.1%}")
            st.info("""
            **Maintenance Actions:**
            • Continue healthy practices
            • Regular check-ups
            • Balanced diet
            • Hygiene maintenance
            """)
    
    with col2:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Probability", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}
            }
        ))
        fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)
    
    # Model information
    st.markdown("#### 🔍 Model Information")
    selected_features = model_info.get('selected_features', [])
    st.info(f"""
    **Model Type:** {model_info.get('model_type', 'Unknown')}
    **Target Variable:** {model_info.get('target_variable', 'Unknown')}
    **Features Used:** {len(selected_features)} features
    **Prediction Basis:** Machine learning model trained on your data
    """)

def show_quick_assessment():
    """Show quick risk assessment when no model is trained"""
    st.markdown("#### 📋 Quick Risk Assessment")
    st.info("This basic assessment uses common risk factors. For more accurate predictions, train a model above.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        wealth = st.selectbox("Wealth Index", ["Poorest", "Poor", "Middle", "Rich", "Richest"])
        hh_size = st.slider("Household Size", 1, 10, 4)
        child_age = st.slider("Child Age (months)", 6, 59, 24)
        dietary_diversity = st.slider("Dietary Diversity", 0, 10, 5)
    
    with col2:
        urban_rural = st.selectbox("Area Type", ["Urban", "Rural"])
        maternal_education = st.selectbox("Mother's Education", ["None", "Primary", "Secondary", "Higher"])
        improved_water = st.selectbox("Clean Water", ["Yes", "No"])
        improved_sanitation = st.selectbox("Good Sanitation", ["Yes", "No"])
    
    if st.button("Quick Risk Assessment"):
        risk_score = calculate_quick_risk(
            wealth, hh_size, child_age, dietary_diversity,
            urban_rural, maternal_education, improved_water, improved_sanitation
        )
        display_quick_risk_results(risk_score)

def calculate_quick_risk(wealth, hh_size, child_age, dietary_diversity,
                        urban_rural, maternal_education, improved_water, improved_sanitation):
    """Calculate quick risk score"""
    risk_score = 0.3
    
    # Wealth impact
    wealth_weights = {"Poorest": 0.4, "Poor": 0.3, "Middle": 0.1, "Rich": 0.05, "Richest": 0.0}
    risk_score += wealth_weights.get(wealth, 0.1)
    
    # Other factors
    if hh_size > 6: risk_score += 0.15
    elif hh_size > 4: risk_score += 0.08
    
    if child_age < 24: risk_score += 0.2
    elif child_age < 36: risk_score += 0.1
    
    if dietary_diversity < 4: risk_score += 0.25
    elif dietary_diversity < 6: risk_score += 0.1
    
    if urban_rural == "Rural": risk_score += 0.1
    
    education_weights = {"None": 0.2, "Primary": 0.1, "Secondary": 0.05, "Higher": 0.0}
    risk_score += education_weights.get(maternal_education, 0.1)
    
    if improved_water == "No": risk_score += 0.1
    if improved_sanitation == "No": risk_score += 0.1
    
    return min(risk_score, 1.0)

def display_quick_risk_results(risk_score):
    """Display quick risk assessment results"""
    st.markdown("### 📊 Quick Assessment Results")
    
    if risk_score > 0.7:
        st.error(f"🔴 High Risk: {risk_score:.1%}")
    elif risk_score > 0.4:
        st.warning(f"🟡 Medium Risk: {risk_score:.1%}")
    else:
        st.success(f"🟢 Low Risk: {risk_score:.1%}")

def predictive_risk_mapping():
    """
    Use the trained model to predict risk across the dataset and map it.
    """
    st.markdown("---")
    st.markdown("### 🗺️ Predictive Risk Mapping")

    if not st.session_state.get('current_model'):
        st.info("Train a model above to generate a predictive risk map for the entire dataset.")
        return

    if 'district' not in st.session_state.merged_data.columns:
        st.warning("⚠️ A 'district' column is required in the data to create a risk map.")
        return

    if st.button("Generate Predictive Risk Map", type="primary"):
        with st.spinner("Generating predictions and rendering map..."):
            try:
                model_info = st.session_state.current_model
                model = model_info['model']
                selected_features = model_info['selected_features']
                target_variable = model_info['target_variable']

                # 1. Prepare the full dataset for prediction
                data_to_predict = st.session_state.merged_data.copy()
                X_full = data_to_predict[selected_features]

                # Use the same preprocessing pipeline
                X_processed, _ = preprocess_features(X_full)

                if X_processed.shape[1] == 0:
                    st.error("❌ Feature preprocessing failed for the full dataset.")
                    return

                # 2. Make predictions (risk probabilities)
                if hasattr(model, 'predict_proba'):
                    risk_probabilities = model.predict_proba(X_processed)[:, 1]
                else:
                    # Fallback for models without predict_proba
                    risk_probabilities = model.predict(X_processed)

                data_to_predict['predicted_risk'] = risk_probabilities

                # 3. Aggregate risk by district
                district_risk = data_to_predict.groupby('district')['predicted_risk'].mean().reset_index()
                district_risk.rename(columns={'predicted_risk': 'Average Predicted Risk'}, inplace=True)
                district_risk['Average Predicted Risk'] = district_risk['Average Predicted Risk'] * 100 # As percentage

                st.markdown(f"#### Predicted Risk of '{target_variable}' by District")

                # 4. Load GeoJSON for Rwanda districts
                try:
                    with open('rwanda_districts.geojson', 'r') as f:
                        geojson_data = json.load(f)
                except FileNotFoundError:
                    st.error("❌ `rwanda_districts.geojson` not found. Please add it to your project directory to display the map.")
                    st.write("Displaying data as a bar chart instead:")
                    fig_bar = px.bar(
                        district_risk.sort_values('Average Predicted Risk', ascending=False),
                        x='district', y='Average Predicted Risk',
                        title=f"Average Predicted Risk of {target_variable}",
                        labels={'Average Predicted Risk': 'Predicted Risk (%)'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    return

                # 5. Create and display the choropleth map
                fig_map = px.choropleth_mapbox(
                    district_risk,
                    geojson=geojson_data,
                    locations='district',
                    featureidkey="properties.ADM2_EN", # Key in GeoJSON to match with 'district' column
                    color='Average Predicted Risk',
                    color_continuous_scale="Reds",
                    range_color=(0, district_risk['Average Predicted Risk'].max()),
                    mapbox_style="carto-positron",
                    zoom=7.5,
                    center={"lat": -1.9403, "lon": 29.8739},
                    opacity=0.6,
                    labels={'Average Predicted Risk': 'Avg. Risk (%)'}
                )
                fig_map.update_layout(
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=600
                )
                st.plotly_chart(fig_map, use_container_width=True)
                st.success("✅ Predictive risk map generated successfully!")

            except Exception as e:
                st.error(f"An error occurred while generating the map: {e}")

def early_warning_simulation():
    """
    Simulate risk changes based on seasonal or other factors.
    """
    st.markdown("---")
    st.markdown("### 🚨 Early Warning System Simulation")

    if not st.session_state.get('current_model'):
        st.info("Train a model above to run a simulation.")
        return

    # Identify potential seasonal/temporal features
    potential_features = ['season', 'month', 'survey_month']
    sim_feature = None
    for feature in potential_features:
        if feature in st.session_state.merged_data.columns:
            sim_feature = feature
            break
    
    if not sim_feature:
        st.warning(f"⚠️ To run a simulation, your data needs a column like 'season' or 'month'.")
        return

    st.markdown(f"This tool simulates how risk might change based on the **'{sim_feature}'** feature. It uses the trained model to predict outcomes under different scenarios.")

    # Get unique values for the simulation feature
    sim_values = st.session_state.merged_data[sim_feature].unique()
    sim_values = [v for v in sim_values if pd.notna(v)] # Remove NaNs

    if len(sim_values) < 2:
        st.warning(f"Not enough unique values in '{sim_feature}' to run a meaningful simulation.")
        return

    # UI for simulation
    scenario_value = st.selectbox(f"Select a scenario for '{sim_feature}':", sim_values)

    if st.button(f"🚀 Run Simulation for '{scenario_value}'", type="primary"):
        with st.spinner(f"Simulating risk for scenario: '{sim_feature}' = '{scenario_value}'..."):
            try:
                model_info = st.session_state.current_model
                model = model_info['model']
                selected_features = model_info['selected_features']
                target_variable = model_info['target_variable']

                # Create a copy of the data for simulation
                sim_data = st.session_state.merged_data.copy()
                
                # Apply the scenario: change the simulation feature for all rows
                sim_data[sim_feature] = scenario_value

                # Prepare data for prediction
                X_sim = sim_data[selected_features]
                X_sim_processed, _ = preprocess_features(X_sim)

                if X_sim_processed.shape[1] == 0:
                    st.error("❌ Feature preprocessing failed for the simulation dataset.")
                    return

                # Predict risk probabilities for the simulated data
                if hasattr(model, 'predict_proba'):
                    sim_risk_probs = model.predict_proba(X_sim_processed)[:, 1]
                else:
                    sim_risk_probs = model.predict(X_sim_processed)

                # Calculate and display the overall simulated risk
                overall_sim_risk = np.mean(sim_risk_probs) * 100

                st.metric(
                    label=f"Simulated Average Risk for '{scenario_value}'",
                    value=f"{overall_sim_risk:.1f}%"
                )
                st.success("✅ Simulation complete. The metric above shows the predicted average risk for the entire population under this scenario.")

            except Exception as e:
                st.error(f"An error occurred during the simulation: {e}")

def compare_models():
    """
    Train multiple models and compare their performance side-by-side.
    """
    st.markdown("---")
    st.markdown("### ⚖️ Model Comparison")

    if not st.session_state.get('current_model'):
        st.info("Train a model in the 'Manual Model Training' section first to enable model comparison.")
        return

    st.markdown("Compare the performance of top algorithms on the currently selected feature set and target variable.")

    if st.button("🚀 Compare Top 3 Models (RF, XGBoost, NN)", type="primary"):
        with st.spinner("Training and comparing models... This may take a moment."):
            try:
                model_info = st.session_state.current_model
                X_train = model_info['X_train']
                y_train = model_info['y_train']
                X_test = model_info['X_test']
                y_test = model_info['y_test']

                models_to_compare = {
                    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                    "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss'),
                    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), alpha=0.001, random_state=42, max_iter=500)
                }

                results = []

                for name, model in models_to_compare.items():
                    # Train model
                    model.fit(X_train, y_train)

                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

                    # Calculate metrics
                    accuracy = (y_pred == y_test).mean()
                    try:
                        auc_score = roc_auc_score(y_test, y_pred_proba)
                    except ValueError:
                        auc_score = 0.5 # Handle cases with only one class in y_test
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)

                    results.append({
                        "Model": name,
                        "Accuracy": accuracy,
                        "AUC": auc_score,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1
                    })

                results_df = pd.DataFrame(results)

                st.markdown("#### Performance Comparison")
                st.dataframe(results_df.style.format({
                    "Accuracy": "{:.2%}", "AUC": "{:.3f}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.2%}"
                }).highlight_max(axis=0, subset=pd.IndexSlice[:, ['Accuracy', 'AUC', 'F1 Score']], color='lightgreen'))

                # Visualize comparison
                results_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
                fig = px.bar(results_melted, x='Metric', y='Score', color='Model',
                             barmode='group', title="Model Performance Metrics Comparison",
                             labels={'Score': 'Metric Score', 'Metric': 'Performance Metric'})
                fig.update_layout(yaxis_title="Score", xaxis_title="Metric")
                st.plotly_chart(fig, use_container_width=True)

                st.success("✅ Model comparison complete!")

            except Exception as e:
                st.error(f"An error occurred during model comparison: {e}")

# Initialize session state if not exists
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Flag to ensure model is loaded only once per session
if 'model_loaded_from_disk' not in st.session_state:
    st.session_state.model_loaded_from_disk = False

# Show model results if available
show_model_results()

# Show prediction interface
individual_prediction_interface()

# Show predictive mapping interface
predictive_risk_mapping()

# Show early warning simulation interface
early_warning_simulation()

# Show model comparison interface
compare_models()