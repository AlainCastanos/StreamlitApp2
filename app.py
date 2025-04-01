import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc
import io
import base64

# Set page configuration
st.set_page_config(page_title="ML Model Trainer", layout="wide")

# Initialize session state for storing data and models
if "data" not in st.session_state:
    st.session_state.data = None
if "model" not in st.session_state:
    st.session_state.model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None
if "metrics" not in st.session_state:
    st.session_state.metrics = {}
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
if "is_classification" not in st.session_state:
    st.session_state.is_classification = False
if "trained" not in st.session_state:
    st.session_state.trained = False
if "encoded_columns" not in st.session_state:
    st.session_state.encoded_columns = {}
if "label_encoders" not in st.session_state:
    st.session_state.label_encoders = {}

# Title and introduction
st.title("Machine Learning Model Trainer")
st.markdown("""
This application allows you to train machine learning models on various datasets.
You can select features, choose different models, configure parameters, and visualize the results.
""")

# Sidebar for dataset selection and file upload
with st.sidebar:
    st.header("Data Selection")
    data_source = st.radio("Select data source:", ["Seaborn datasets", "Upload your own CSV"])
    
    if data_source == "Seaborn datasets":
        dataset_name = st.selectbox(
            "Choose a dataset:", 
            ["tips", "iris", "titanic", "diamonds", "mpg", "penguins"]
        )
        
        @st.cache_data
        def load_seaborn_data(name):
            return sns.load_dataset(name)
        
        try:
            data = load_seaborn_data(dataset_name)
            st.session_state.data = data
            st.success(f"Loaded {dataset_name} dataset with {data.shape[0]} rows and {data.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success(f"Uploaded dataset with {data.shape[0]} rows and {data.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")

# Main content area
# Data preview tab
tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Model Configuration", "Training", "Results"])

with tab1:
    if st.session_state.data is not None:
        st.header("Dataset Preview")
        st.dataframe(st.session_state.data.head())
        
        st.header("Dataset Statistics")
        st.write(st.session_state.data.describe())
        
        st.header("Data Types")
        st.write(st.session_state.data.dtypes)
        
        # Missing values
        st.header("Missing Values")
        missing_values = st.session_state.data.isnull().sum()
        st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")
    else:
        st.info("Please select a dataset or upload a CSV file")

# Model configuration tab
with tab2:
    if st.session_state.data is not None:
        st.header("Feature Selection and Model Configuration")
        
        @st.fragment
        def feature_selection_fragment():
            st.subheader("Feature Selection")
            
            # Get numerical and categorical columns
            numerical_cols = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Target variable selection
            target_variable = st.selectbox("Select target variable:", st.session_state.data.columns.tolist())
            
            # Feature selection
            st.subheader("Select features")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Numerical Features")
                numerical_features = [col for col in numerical_cols if col != target_variable]
                selected_numerical = st.multiselect(
                    "Select numerical features:",
                    numerical_features,
                    default=numerical_features[:min(len(numerical_features), 3)]
                )
            
            with col2:
                st.write("Categorical Features")
                categorical_features = [col for col in categorical_cols if col != target_variable]
                selected_categorical = st.multiselect(
                    "Select categorical features:",
                    categorical_features,
                    default=categorical_features[:min(len(categorical_features), 2)]
                )
            
            # Determine if classification or regression
            if target_variable in categorical_cols or st.session_state.data[target_variable].nunique() < 10:
                st.session_state.is_classification = True
                st.info(f"Detected a classification problem (target: {target_variable})")
            else:
                st.session_state.is_classification = False
                st.info(f"Detected a regression problem (target: {target_variable})")
            
            return target_variable, selected_numerical, selected_categorical
        
        target_variable, selected_numerical, selected_categorical = feature_selection_fragment()
        
        # Model selection and parameters
        with st.form(key='model_config_form'):
            st.subheader("Model Configuration")
            
            # Model selection
            if st.session_state.is_classification:
                model_type = st.selectbox(
                    "Select model type:",
                    ["Logistic Regression", "Random Forest Classifier"]
                )
            else:
                model_type = st.selectbox(
                    "Select model type:",
                    ["Linear Regression", "Random Forest Regressor"]
                )
            
            # Train-test split parameters
            test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random state:", 0, 100, 42)
            
            # Model specific parameters
            if model_type in ["Random Forest Classifier", "Random Forest Regressor"]:
                n_estimators = st.slider("Number of trees:", 10, 200, 100, 10)
                max_depth = st.slider("Maximum depth:", 1, 20, 5, 1)
                min_samples_split = st.slider("Minimum samples to split:", 2, 10, 2, 1)
            
            elif model_type == "Logistic Regression":
                C = st.slider("Regularization strength (C):", 0.01, 10.0, 1.0, 0.01)
                max_iter = st.slider("Maximum iterations:", 100, 1000, 100, 100)
            
            # Submit button
            submit_button = st.form_submit_button(label="Save Configuration")
            
            if submit_button:
                st.session_state.model_config = {
                    "target_variable": target_variable,
                    "numerical_features": selected_numerical,
                    "categorical_features": selected_categorical,
                    "model_type": model_type,
                    "test_size": test_size,
                    "random_state": random_state
                }
                
                if model_type in ["Random Forest Classifier", "Random Forest Regressor"]:
                    st.session_state.model_config.update({
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split
                    })
                elif model_type == "Logistic Regression":
                    st.session_state.model_config.update({
                        "C": C,
                        "max_iter": max_iter
                    })
                
                st.success("Configuration saved!")
    else:
        st.info("Please select a dataset or upload a CSV file")

# Training tab
with tab3:
    if st.session_state.data is not None and "model_config" in st.session_state:
        st.header("Model Training")
        
        # Function to prepare data
        def prepare_data():
            config = st.session_state.model_config
            data = st.session_state.data.copy()
            
            # Handle missing values
            for col in config["numerical_features"] + [config["target_variable"]]:
                if col in data.columns and data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].median(), inplace=True)
            
            for col in config["categorical_features"]:
                if col in data.columns and data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].mode()[0], inplace=True)
            
            # Encode categorical features
            X = data[config["numerical_features"]].copy()
            st.session_state.label_encoders = {}
            
            for col in config["categorical_features"]:
                if col in data.columns:
                    le = LabelEncoder()
                    data[f"{col}_encoded"] = le.fit_transform(data[col])
                    X[f"{col}_encoded"] = data[f"{col}_encoded"]
                    st.session_state.label_encoders[col] = le
                    st.session_state.encoded_columns[col] = f"{col}_encoded"
            
            # Encode target if classification
            y = data[config["target_variable"]].copy()
            if st.session_state.is_classification and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.session_state.label_encoders['target'] = le
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config["test_size"],
                random_state=config["random_state"]
            )
            
            # Scale numerical features
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            for col in config["numerical_features"]:
                if col in X_train.columns:
                    X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
                    X_test_scaled[col] = scaler.transform(X_test[[col]])
            
            return X_train_scaled, X_test_scaled, y_train, y_test
        
        # Function to train model
        def train_model(X_train, y_train):
            config = st.session_state.model_config
            
            if config["model_type"] == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
            
            elif config["model_type"] == "Random Forest Regressor":
                model = RandomForestRegressor(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_split=config["min_samples_split"],
                    random_state=config["random_state"]
                )
                model.fit(X_train, y_train)
            
            elif config["model_type"] == "Logistic Regression":
                model = LogisticRegression(
                    C=config["C"],
                    max_iter=config["max_iter"],
                    random_state=config["random_state"]
                )
                model.fit(X_train, y_train)
            
            elif config["model_type"] == "Random Forest Classifier":
                model = RandomForestClassifier(
                    n_estimators=config["n_estimators"],
                    max_depth=config["max_depth"],
                    min_samples_split=config["min_samples_split"],
                    random_state=config["random_state"]
                )
                model.fit(X_train, y_train)
            
            return model
        
        # Function to evaluate model
        def evaluate_model(model, X_test, y_test):
            config = st.session_state.model_config
            
            # Make predictions
            predictions = model.predict(X_test)
            st.session_state.predictions = predictions
            
            # Calculate metrics
            metrics = {}
            
            if not st.session_state.is_classification:
                # Regression metrics
                metrics["Mean Squared Error"] = mean_squared_error(y_test, predictions)
                metrics["Root Mean Squared Error"] = np.sqrt(metrics["Mean Squared Error"])
                metrics["R-squared"] = r2_score(y_test, predictions)
            else:
                # Classification metrics
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, predictions)
                metrics["Confusion Matrix"] = cm
                
                # Classification report
                report = classification_report(y_test, predictions, output_dict=True)
                metrics["Classification Report"] = report
                
                # ROC curve and AUC
                if y_pred_proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    metrics["ROC"] = {"fpr": fpr, "tpr": tpr}
                    metrics["AUC"] = auc(fpr, tpr)
            
            # Feature importance
            if hasattr(model, "feature_importances_"):
                feature_names = list(X_test.columns)
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                st.session_state.feature_importance = feature_importance
            elif hasattr(model, "coef_"):
                feature_names = list(X_test.columns)
                coefficients = model.coef_[0] if st.session_state.is_classification else model.coef_
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': np.abs(coefficients)
                }).sort_values('importance', ascending=False)
                st.session_state.feature_importance = feature_importance
            
            return metrics
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training in progress..."):
                try:
                    # Prepare data
                    X_train, X_test, y_train, y_test = prepare_data()
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    # Train model
                    model = train_model(X_train, y_train)
                    st.session_state.model = model
                    
                    # Evaluate model
                    metrics = evaluate_model(model, X_test, y_test)
                    st.session_state.metrics = metrics
                    
                    st.session_state.trained = True
                    st.success("Model trained successfully!")
                except Exception as e:
                    st.error(f"Error during training: {e}")
    else:
        st.info("Please configure the model in the 'Model Configuration' tab")

# Results tab
with tab4:
    if st.session_state.trained:
        st.header("Model Results")
        
        # Show model metrics
        st.subheader("Model Performance Metrics")
        
        if not st.session_state.is_classification:
            # Regression metrics
            metrics_df = pd.DataFrame({
                'Metric': list(st.session_state.metrics.keys()),
                'Value': list(st.session_state.metrics.values())
            })
            st.dataframe(metrics_df)
            
            # Residuals plot
            st.subheader("Residual Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            residuals = st.session_state.y_test - st.session_state.predictions
            
            # Histogram
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')
            ax.set_title('Residual Distribution')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Actual vs Predicted
            st.subheader("Actual vs Predicted Values")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(st.session_state.y_test, st.session_state.predictions, alpha=0.5)
            
            # Add the perfect prediction line
            min_val = min(st.session_state.y_test.min(), st.session_state.predictions.min())
            max_val = max(st.session_state.y_test.max(), st.session_state.predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted Values')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        else:
            # Classification metrics
            st.subheader("Classification Report")
            if "Classification Report" in st.session_state.metrics:
                report = st.session_state.metrics["Classification Report"]
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            if "Confusion Matrix" in st.session_state.metrics:
                cm = st.session_state.metrics["Confusion Matrix"]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
            
            # ROC Curve
            if "ROC" in st.session_state.metrics:
                st.subheader("ROC Curve")
                fig, ax = plt.subplots(figsize=(8, 6))
                fpr = st.session_state.metrics["ROC"]["fpr"]
                tpr = st.session_state.metrics["ROC"]["tpr"]
                auc_score = st.session_state.metrics["AUC"]
                
                ax.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC Curve')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
        # Feature Importance
        if st.session_state.feature_importance is not None:
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, max(6, len(st.session_state.feature_importance) * 0.3)))
            
            # Only show top 15 features if there are many
            fi_df = st.session_state.feature_importance.head(15) if len(st.session_state.feature_importance) > 15 else st.session_state.feature_importance
            
            sns.barplot(x='importance', y='feature', data=fi_df, ax=ax)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Model Export
        st.subheader("Export Model")
        
        # Function to download the model summary
        def get_model_summary():
            summary = io.StringIO()
            summary.write("# ML Model Training Results\n\n")
            
            # Model configuration
            summary.write("## Model Configuration\n\n")
            for key, value in st.session_state.model_config.items():
                summary.write(f"- **{key}**: {value}\n")
            
            # Performance metrics
            summary.write("\n## Performance Metrics\n\n")
            if not st.session_state.is_classification:
                # Regression metrics
                for metric, value in st.session_state.metrics.items():
                    summary.write(f"- **{metric}**: {value:.4f}\n")
            else:
                # Classification metrics
                if "AUC" in st.session_state.metrics:
                    summary.write(f"- **AUC**: {st.session_state.metrics['AUC']:.4f}\n")
                
                if "Classification Report" in st.session_state.metrics:
                    report = st.session_state.metrics["Classification Report"]
                    summary.write("\n### Classification Report\n\n")
                    summary.write("| Class | Precision | Recall | F1-Score | Support |\n")
                    summary.write("|-------|-----------|--------|----------|--------|\n")
                    
                    for class_name, metrics in report.items():
                        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                            summary.write(f"| {class_name} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1-score']:.3f} | {metrics['support']} |\n")
            
            # Feature importance
            if st.session_state.feature_importance is not None:
                summary.write("\n## Feature Importance\n\n")
                summary.write("| Feature | Importance |\n")
                summary.write("|---------|------------|\n")
                
                for _, row in st.session_state.feature_importance.iterrows():
                    summary.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
            
            return summary.getvalue()
        
        if st.download_button(
            label="Download Report (Markdown)",
            data=get_model_summary(),
            file_name="model_summary.md",
            mime="text/markdown"
        ):
            st.success("Download started!")
    else:
        st.info("Please train a model in the 'Training' tab to see results")

# Footer
st.markdown("---")
st.markdown("ML Model Trainer App - Created with Streamlit")