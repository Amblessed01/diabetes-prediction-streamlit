# app.py - Diabetes Prediction ML Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import time

# Model imports
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier, 
                              BaggingClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction ML Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #34495e;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .highlight-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: #e8f4fc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3498db;
        text-align: center;
        margin: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header"> Diabetes Prediction Machine Learning Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/diabetes.png", width=100)
    st.markdown("###  Navigation")
    
    sections = st.radio(
        "Select Analysis Section:",
        [" Dataset Overview", 
         " Data Quality Analysis",
         " Exploratory Visualizations",
         " Data Preprocessing",
         " Feature Engineering",
         " Model Training & Comparison",
         " Best Model Analysis",
         " Model Deployment"]
    )
    
    st.markdown("---")
    
    # Dataset info in sidebar
    st.markdown("###  Dataset Info")
    st.info("""
    - **100,000 records** with **9 features**
    - **Target:** Diabetes (binary)
    - **Features:** Age, BMI, HbA1c, Blood Glucose, etc.
    """)
    
    st.markdown("---")
    
    # Download processed data
    st.markdown("###  Export Data")
    if st.button("Export Processed Dataset"):
        st.info("Dataset will be available for download after processing")

# Initialize session state
if 'thankgod_israel' not in st.session_state:
    st.session_state.thankgod_israel = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# Function to load and process data
@st.cache_data
def load_data():
    try:
        thankgod_israel = pd.read_csv('C:/Users/User/Desktop/OSIRI UNIVERSITY Files/diabetes folder/diabetes_prediction_dataset.csv')
        st.session_state.thankgod_israel = thankgod_israel
        return thankgod_israel
    except:
        st.error("Dataset not found at specified path. Please check the file location.")
        return None

# Load data automatically
if st.session_state.thankgod_israel is None:
    thankgod_israel = load_data()
else:
    thankgod_israel = st.session_state.thankgod_israel

# Dataset Overview Section
if sections == " Dataset Overview":
    st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)
    
    if thankgod_israel is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Total Records", f"{len(thankgod_israel):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Features", f"{len(thankgod_israel.columns)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            missing_values = thankgod_israel.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_values:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Show data sample
        st.markdown('<div class="subsection-header"> First 5 Rows</div>', unsafe_allow_html=True)
        st.dataframe(thankgod_israel.head(), use_container_width=True)
        
        # Data types
        st.markdown('<div class="subsection-header"> Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            'Column': thankgod_israel.columns,
            'Data Type': thankgod_israel.dtypes.values,
            'Non-Null Count': thankgod_israel.count().values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Summary statistics
        st.markdown('<div class="subsection-header"> Summary Statistics</div>', unsafe_allow_html=True)
        with st.expander("View Descriptive Statistics"):
            st.dataframe(thankgod_israel.describe(), use_container_width=True)
    else:
        st.warning("Please ensure the dataset is available at the specified path.")

# Data Quality Analysis Section
elif sections == " Data Quality Analysis":
    st.markdown('<h2 class="section-header">Data Quality Analysis</h2>', unsafe_allow_html=True)
    
    if thankgod_israel is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.subheader("❓ Missing Values")
            missing_values = thankgod_israel.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Count': missing_values.values,
                'Percentage': (missing_values.values / len(thankgod_israel)) * 100
            })
            st.dataframe(missing_df, use_container_width=True)
            
            if missing_values.sum() == 0:
                st.success(" No missing values found!")
            else:
                st.warning("Missing values detected")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
            st.subheader(" Duplicate Records")
            duplicates = thankgod_israel.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicates:,}")
            
            if duplicates > 0:
                st.warning(f" {duplicates:,} duplicate rows found")
                if st.button("Remove Duplicates Now"):
                    initial_shape = thankgod_israel.shape
                    thankgod_israel = thankgod_israel.drop_duplicates()
                    st.session_state.thankgod_israel = thankgod_israel
                    final_shape = thankgod_israel.shape
                    st.success(f" Removed {initial_shape[0] - final_shape[0]:,} duplicates")
                    st.info(f"New shape: {final_shape}")
            else:
                st.success("No duplicates found")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Class distribution analysis
        st.markdown('<div class="subsection-header"> Class Distribution Analysis</div>', unsafe_allow_html=True)
        
        if 'diabetes' in thankgod_israel.columns:
            class_dist = thankgod_israel['diabetes'].value_counts()
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ['#3498db', '#e74c3c']
                bars = ax.bar(['Non-Diabetic (0)', 'Diabetic (1)'], class_dist.values, color=colors)
                ax.set_ylabel('Count', fontsize=12)
                ax.set_title('Diabetes Class Distribution', fontsize=14, fontweight='bold')
                
                # Add labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 100,
                           f'{height:,}', ha='center', va='bottom', fontsize=11)
                
                st.pyplot(fig)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Non-Diabetic", f"{class_dist[0]:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Diabetic", f"{class_dist[1]:,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Imbalance ratio
            imbalance_ratio = class_dist[0] / class_dist[1]
            st.info(f"**Class Imbalance Ratio:** {imbalance_ratio:.1f}:1 (Non-Diabetic:Diabetic)")

# Exploratory Visualizations Section
elif sections == "Exploratory Visualizations":
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    if thankgod_israel is not None:
        # Visualization type selector
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Numerical Distributions", "Categorical Distributions", 
             "Correlation Matrix", "Feature Relationships", "Scatter Matrix"]
        )
        
        if viz_type == "Numerical Distributions":
            col1, col2 = st.columns(2)
            
            with col1:
                selected_num = st.selectbox(
                    "Select Numerical Feature:",
                    ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
                )
            
            with col2:
                plot_type = st.radio(
                    "Plot Type:",
                    ["Histogram", "Box Plot", "Violin Plot"]
                )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if plot_type == "Histogram":
                ax.hist(thankgod_israel[selected_num], bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel(selected_num)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_num}')
            
            elif plot_type == "Box Plot":
                ax.boxplot(thankgod_israel[selected_num].dropna())
                ax.set_ylabel(selected_num)
                ax.set_title(f'Box Plot of {selected_num}')
            
            else:  # Violin Plot
                ax.violinplot(thankgod_israel[selected_num].dropna())
                ax.set_ylabel(selected_num)
                ax.set_title(f'Violin Plot of {selected_num}')
            
            st.pyplot(fig)
        
        elif viz_type == "Categorical Distributions":
            selected_cat = st.selectbox(
                "Select Categorical Feature:",
                ['gender', 'smoking_history', 'hypertension', 'heart_disease']
            )
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Bar plot
            value_counts = thankgod_israel[selected_cat].value_counts()
            ax1.bar(range(len(value_counts)), value_counts.values)
            ax1.set_xticks(range(len(value_counts)))
            ax1.set_xticklabels(value_counts.index, rotation=45)
            ax1.set_title(f'{selected_cat} Distribution')
            ax1.set_ylabel('Count')
            
            # Pie chart
            ax2.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax2.set_title(f'{selected_cat} Percentage')
            
            st.pyplot(fig)
        
        elif viz_type == "Correlation Matrix":
            numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                             'hypertension', 'heart_disease', 'diabetes']
            
            corr_matrix = thankgod_israel[numerical_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, ax=ax)
            ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
            
            st.pyplot(fig)
            
            # Highlight strong correlations
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.3:
                        strong_corrs.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': f"{corr:.3f}"
                        })
            
            if strong_corrs:
                st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
                st.subheader("Strong Correlations (|r| > 0.3)")
                st.dataframe(pd.DataFrame(strong_corrs), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Data Preprocessing Section
elif sections == "Data Preprocessing":
    st.markdown('<h2 class="section-header">Data Preprocessing Pipeline</h2>', unsafe_allow_html=True)
    
    if thankgod_israel is not None:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Preprocessing Steps")
        
        preprocessing_steps = st.multiselect(
            "Select preprocessing steps to apply:",
            ["Remove Duplicates", "Handle Outliers", "Encode Categorical", 
             "Scale Features", "Balance Classes"],
            default=["Remove Duplicates", "Encode Categorical", "Scale Features"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Initialize processed data
        processed_data = thankgod_israel.copy()
        
        # Process steps
        if st.button("Apply Selected Preprocessing", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            steps_completed = []
            
            # Step 1: Remove duplicates
            if "Remove Duplicates" in preprocessing_steps:
                status_text.text("Removing duplicates...")
                initial_rows = len(processed_data)
                processed_data = processed_data.drop_duplicates()
                removed = initial_rows - len(processed_data)
                steps_completed.append(f"Removed {removed:,} duplicates")
                progress_bar.progress(20)
            
            # Step 2: Handle outliers
            if "Handle Outliers" in preprocessing_steps:
                status_text.text("Capping outliers...")
                numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
                for col in numerical_cols:
                    Q1 = processed_data[col].quantile(0.25)
                    Q3 = processed_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    processed_data[col] = processed_data[col].clip(lower_bound, upper_bound)
                steps_completed.append("Outliers capped using IQR method")
                progress_bar.progress(40)
            
            # Step 3: Encode categorical
            if "Encode Categorical" in preprocessing_steps:
                status_text.text("Encoding categorical variables...")
                # Encode gender
                processed_data['gender'] = processed_data['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
                
                # One-hot encode smoking history
                smoking_dummies = pd.get_dummies(processed_data['smoking_history'], 
                                                  prefix='smoking', drop_first=False)
                processed_data = pd.concat([processed_data, smoking_dummies], axis=1)
                processed_data = processed_data.drop('smoking_history', axis=1)
                
                steps_completed.append("Categorical variables encoded")
                progress_bar.progress(60)
            
            # Step 4: Scale features
            if "Scale Features" in preprocessing_steps:
                status_text.text("Scaling numerical features...")
                scaler = StandardScaler()
                numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
                processed_data[numerical_features] = scaler.fit_transform(processed_data[numerical_features])
                steps_completed.append("Features scaled (StandardScaler)")
                progress_bar.progress(80)
            
            # Step 5: Balance classes
            if "Balance Classes" in preprocessing_steps and 'diabetes' in processed_data.columns:
                status_text.text("Balancing classes with SMOTE...")
                X = processed_data.drop('diabetes', axis=1)
                y = processed_data['diabetes']
                
                # Split data first
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
                )
                
                # Apply SMOTE
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                steps_completed.append("Classes balanced with SMOTE")
                progress_bar.progress(100)
            
            status_text.text("Preprocessing complete!")
            
            # Show completion summary
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.subheader("Preprocessing Completed")
            for step in steps_completed:
                st.write(f"- {step}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store processed data in session state
            st.session_state.processed_data = processed_data
            st.session_state.processed = True
            
            # Show processed data sample
            with st.expander("View Processed Data Sample"):
                st.dataframe(processed_data.head(), use_container_width=True)
                st.write(f"**Shape:** {processed_data.shape}")

# Feature Engineering Section
elif sections == "Feature Engineering":
    st.markdown('<h2 class="section-header">Feature Engineering</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed:
        processed_data = st.session_state.processed_data
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Create New Features")
        
        # Feature engineering options
        eng_features = st.multiselect(
            "Select features to create:",
            ["Interaction Terms", "Polynomial Features", "Risk Score", "Binning"],
            default=["Interaction Terms", "Risk Score"]
        )
        
        if st.button("Engineer Features", type="primary"):
            progress_bar = st.progress(0)
            
            # Create copies for engineering
            X_train_eng = processed_data.copy()
            if 'diabetes' in X_train_eng.columns:
                X_train_eng = X_train_eng.drop('diabetes', axis=1)
            
            features_created = []
            
            # Interaction terms
            if "Interaction Terms" in eng_features:
                X_train_eng['age_bmi_interaction'] = X_train_eng['age'] * X_train_eng['bmi']
                X_train_eng['hba1c_glucose_interaction'] = X_train_eng['HbA1c_level'] * X_train_eng['blood_glucose_level']
                features_created.append("age_bmi_interaction")
                features_created.append("hba1c_glucose_interaction")
                progress_bar.progress(25)
            
            # Polynomial features
            if "Polynomial Features" in eng_features:
                X_train_eng['age_squared'] = X_train_eng['age'] ** 2
                X_train_eng['bmi_squared'] = X_train_eng['bmi'] ** 2
                features_created.append("age_squared")
                features_created.append("bmi_squared")
                progress_bar.progress(50)
            
            # Risk score
            if "Risk Score" in eng_features:
                X_train_eng['diabetes_risk_score'] = (
                    X_train_eng['age'] * 0.3 + 
                    X_train_eng['bmi'] * 0.2 + 
                    X_train_eng['HbA1c_level'] * 0.4 + 
                    X_train_eng['blood_glucose_level'] * 0.1
                )
                features_created.append("diabetes_risk_score")
                progress_bar.progress(75)
            
            progress_bar.progress(100)
            
            # Store engineered features
            st.session_state.X_engineered = X_train_eng
            st.session_state.features_created = features_created
            
            # Show results
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.subheader("Feature Engineering Complete")
            st.write(f"**New features created:** {len(features_created)}")
            st.write(f"**Features:** {', '.join(features_created)}")
            st.write(f"**New shape:** {X_train_eng.shape}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display new features
            with st.expander("View Engineered Features"):
                st.dataframe(X_train_eng[features_created].head(), use_container_width=True)
    else:
        st.warning("Please complete data preprocessing first!")

# Model Training Section
elif sections == " Model Training & Comparison":
    st.markdown('<h2 class="section-header">Model Training & Comparison</h2>', unsafe_allow_html=True)
    
    if st.session_state.processed and 'X_engineered' in st.session_state:
        X = st.session_state.X_engineered
        y = st.session_state.processed_data['diabetes']
        
        # Model selection
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Select Models to Train")
        
        model_options = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "LightGBM": LGBMClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=42)
        }
        
        selected_models = st.multiselect(
            "Choose models to train:",
            list(model_options.keys()),
            default=["Random Forest", "XGBoost", "Logistic Regression", "Gradient Boosting"]
        )
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        with col2:
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Train models button
        if st.button("Train Selected Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes"):
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size/100, random_state=42, stratify=y
                )
                
                # Apply SMOTE to training data
                smote = SMOTE(random_state=42)
                X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                
                # Store for later use
                st.session_state.X_train = X_train_balanced
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train_balanced
                st.session_state.y_test = y_test
                
                # Train models
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Training {model_name}...")
                    
                    try:
                        model = model_options[model_name]
                        
                        # Train model
                        start_time = time.time()
                        model.fit(X_train_balanced, y_train_balanced)
                        training_time = time.time() - start_time
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                        
                        # Calculate metrics
                        metrics = {
                            'Model': model_name,
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred),
                            'Recall': recall_score(y_test, y_pred),
                            'F1-Score': f1_score(y_test, y_pred),
                            'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0,
                            'Training Time': training_time
                        }
                        
                        results.append(metrics)
                        
                        # Store best model
                        if st.session_state.best_model is None or metrics['F1-Score'] > st.session_state.best_model['F1-Score']:
                            st.session_state.best_model = {
                                'name': model_name,
                                'model': model,
                                'metrics': metrics
                            }
                        
                    except Exception as e:
                        st.warning(f"Failed to train {model_name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                status_text.text("Training complete!")
                
                # Store results
                st.session_state.results_df = pd.DataFrame(results)
                st.session_state.models_trained = True
                
                # Display results
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.subheader("Model Training Complete")
                st.write(f"Trained {len(results)} models successfully")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show results table
                results_display = st.session_state.results_df.copy()
                results_display = results_display.sort_values('F1-Score', ascending=False)
                
                st.dataframe(
                    results_display.style.format({
                        'Accuracy': '{:.3f}',
                        'Precision': '{:.3f}',
                        'Recall': '{:.3f}',
                        'F1-Score': '{:.3f}',
                        'ROC-AUC': '{:.3f}',
                        'Training Time': '{:.2f}s'
                    }).background_gradient(subset=['F1-Score'], cmap='YlOrRd'),
                    use_container_width=True
                )
                
                # Visualize comparison
                st.markdown('<div class="subsection-header"> Model Performance Comparison</div>', unsafe_allow_html=True)
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time']
                titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Training Time (s)']
                
                for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
                    ax = axes[idx//3, idx%3]
                    sorted_results = results_display.sort_values(metric, ascending=False)
                    bars = ax.barh(sorted_results['Model'][:5], sorted_results[metric][:5])
                    ax.set_title(f'Top 5 by {title}')
                    ax.set_xlabel(title)
                    
                    # Add value labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width, bar.get_y() + bar.get_height()/2,
                               f'{width:.3f}' if metric != 'Training Time' else f'{width:.1f}s',
                               ha='left', va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
    else:
        st.warning("Please complete feature engineering first!")

# Best Model Analysis Section
elif sections == "Best Model Analysis":
    st.markdown('<h2 class="section-header">Best Model Analysis</h2>', unsafe_allow_html=True)
    
    if st.session_state.models_trained and st.session_state.best_model:
        best_model_info = st.session_state.best_model
        best_model = best_model_info['model']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Best Model", best_model_info['name'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{best_model_info['metrics']['F1-Score']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{best_model_info['metrics']['Accuracy']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.metric("ROC-AUC", f"{best_model_info['metrics']['ROC-AUC']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Detailed analysis
        tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curve", "Classification Report", "Feature Importance"])
        
        with tab1:
            y_pred = best_model.predict(st.session_state.X_test)
            cm = confusion_matrix(st.session_state.y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-Diabetic', 'Diabetic'],
                       yticklabels=['Non-Diabetic', 'Diabetic'])
            ax.set_title(f'Confusion Matrix - {best_model_info["name"]}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        with tab2:
            y_pred_proba = best_model.predict_proba(st.session_state.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'{best_model_info["name"]} (AUC = {best_model_info["metrics"]["ROC-AUC"]:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {best_model_info["name"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab3:
            report = classification_report(st.session_state.y_test, y_pred,
                                          target_names=['Non-Diabetic', 'Diabetic'])
            st.text(report)
        
        with tab4:
            if hasattr(best_model, 'feature_importances_'):
                feature_names = st.session_state.X_engineered.columns
                importances = best_model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                bars = ax.barh(importance_df['Feature'][:15][::-1], 
                              importance_df['Importance'][:15][::-1])
                ax.set_xlabel('Importance')
                ax.set_title(f'Top 15 Feature Importance - {best_model_info["name"]}')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)
                
                st.dataframe(importance_df.head(10), use_container_width=True)
            else:
                st.info("Feature importance not available for this model type")
        
        # Hyperparameter tuning
        st.markdown('<div class="subsection-header"> Hyperparameter Tuning</div>', unsafe_allow_html=True)
        
        if st.button("🔧 Tune Best Model Hyperparameters"):
            with st.spinner("Performing hyperparameter tuning..."):
                # Define parameter grid based on model type
                if best_model_info['name'] == "Random Forest":
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 20, 30, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                
                grid_search = GridSearchCV(
                    estimator=best_model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(st.session_state.X_train, st.session_state.y_train)
                
                st.success(f" Best Parameters: {grid_search.best_params_}")
                st.success(f" Best F1-Score: {grid_search.best_score_:.3f}")
                
                # Update best model with tuned version
                st.session_state.best_model['model'] = grid_search.best_estimator_

# Model Deployment Section
elif sections == " Model Deployment":
    st.markdown('<h2 class="section-header">Model Deployment & Prediction</h2>', unsafe_allow_html=True)
    
    if st.session_state.models_trained and st.session_state.best_model:
        best_model_info = st.session_state.best_model
        best_model = best_model_info['model']
        
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.subheader("Export Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Model to Disk"):
                try:
                    joblib.dump(best_model, 'best_diabetes_model.pkl')
                    st.success("Model saved as 'best_diabetes_model.pkl'")
                except Exception as e:
                    st.error(f"Failed to save model: {e}")
        
        with col2:
            if st.button("Download Model"):
                model_bytes = joblib.dumps(best_model)
                st.download_button(
                    label="Download Model File",
                    data=model_bytes,
                    file_name="diabetes_model.pkl",
                    mime="application/octet-stream"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Live prediction interface
        st.markdown('<div class="subsection-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            st.write("Enter patient information for diabetes prediction:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.slider("Age", 0.0, 80.0, 45.0, 0.1)
                bmi = st.slider("BMI", 10.0, 50.0, 27.3, 0.1)
            
            with col2:
                hba1c = st.slider("HbA1c Level", 3.5, 9.0, 5.5, 0.1)
                glucose = st.slider("Blood Glucose Level", 80, 300, 138)
            
            with col3:
                hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                gender = st.selectbox("Gender", [0, 1, 2], format_func=lambda x: ["Female", "Male", "Other"][x])
            
            # Create input array
            submit = st.form_submit_button("Predict Diabetes Risk", type="primary")
            
            if submit:
                # Prepare input features
                input_features = np.array([[
                    age, hypertension, heart_disease, bmi, hba1c, glucose, gender,
                    0, 0, 0, 0, 0, 0  # Placeholder for smoking history columns
                ]])
                
                try:
                    # Make prediction
                    prediction = best_model.predict(input_features)[0]
                    probability = best_model.predict_proba(input_features)[0][1]
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        if prediction == 1:
                            st.metric("Prediction", "DIABETIC", delta="High Risk")
                        else:
                            st.metric("Prediction", "NON-DIABETIC", delta="Low Risk")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                        st.metric("Risk Probability", f"{probability:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Risk interpretation
                    if probability > 0.7:
                        st.error(" **High Risk:** Patient shows strong indicators of diabetes risk")
                    elif probability > 0.4:
                        st.warning(" **Moderate Risk:** Patient shows some indicators of diabetes risk")
                    else:
                        st.success(" **Low Risk:** Patient shows minimal indicators of diabetes risk")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    else:
        st.warning("Please train models first!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>🩺 Diabetes Prediction ML Dashboard | Built with Streamlit & Scikit-learn</p>
    <p>Dataset: Diabetes Prediction Dataset | 100,000 records | 9 features</p>
</div>
""", unsafe_allow_html=True)