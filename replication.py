import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Step 0: Load and preprocess the COMPAS dataset
def load_and_preprocess_compas_data(file_path='compas-scores-raw.csv'):
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"{file_path} not found. Ensure the file path is correct.")
    
    # Display the first few rows and the columns for verification
    print("Data Preview:")
    print(data.head())
    print("Columns in DataFrame:")
    print(data.columns)
    
    # Filter data for 'Risk of Recidivism'
    compas_recid = data[data['DisplayText'] == 'Risk of Recidivism'].copy()
    # Convert 'ScoreText' to binary target variable
    compas_recid['Risk'] = compas_recid['ScoreText'].apply(lambda x: 1 if x == 'High' else 0)

    # Extract relevant features
    features = ['Sex_Code_Text', 'Ethnic_Code_Text', 'Age', 'MaritalStatus', 'RawScore', 'DecileScore']
    
    # Parse dates explicitly
    compas_recid['DateOfBirth'] = pd.to_datetime(compas_recid['DateOfBirth'], format='%m/%d/%y')
    compas_recid['Screening_Date'] = pd.to_datetime(compas_recid['Screening_Date'], format='%m/%d/%y %H:%M')
    compas_recid['Age'] = compas_recid['Screening_Date'].dt.year - compas_recid['DateOfBirth'].dt.year

    # Select and process relevant features
    compas_recid = compas_recid[features + ['Risk']]
    compas_recid = pd.get_dummies(compas_recid, columns=['Sex_Code_Text', 'Ethnic_Code_Text', 'MaritalStatus'], drop_first=True)
    
    return compas_recid

# Step 1: Train a classifier
def train_classifier(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# Step 2: Generate explanations using LIME and SHAP
def generate_lime_explanations(model, X_test, num_features=10):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test.values, feature_names=X_test.columns, class_names=['Low', 'High'], 
        discretize_continuous=True)
    
    lime_explanations = [explainer.explain_instance(X_test.values[i], model.predict_proba, num_features=num_features) 
                         for i in range(X_test.shape[0])]
    return lime_explanations

def generate_shap_explanations(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return shap_values

# Helper function to calculate sensitivity
def calculate_sensitivity(explanations, sensitive_feature):
    explanations_with_sensitive_feature_top = 0
    for explanation in explanations:
        if isinstance(explanation, list):  # for SHAP values
            top_features = [feature[0] for feature in sorted(enumerate(explanation), key=lambda x: -abs(x[1]))[:3]]
            if sensitive_feature in top_features:
                explanations_with_sensitive_feature_top += 1
        else:  # for LIME explanations
            if sensitive_feature in [feat[0] for feat in explanation.as_list()]:
                explanations_with_sensitive_feature_top += 1
    
    return explanations_with_sensitive_feature_top / len(explanations)

# Main script
if __name__ == "__main__":
    data = load_and_preprocess_compas_data()
    
    # Ensure we have the target variable
    if 'Risk' in data.columns:
        X = data.drop('Risk', axis=1)
        y = data['Risk']
    else:
        raise KeyError("Column 'Risk' is missing from the dataset.")
    
    # Create DataFrame to retain column names for LIME and SHAP
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)

    model = train_classifier(X_train_df, y_train)
    
    # Generate explanations
    lime_explanations = generate_lime_explanations(model, X_test_df)
    shap_explanations = generate_shap_explanations(model, X_test_df)

    # Sensitivity analysis
    sensitive_feature = 'Ethnic_Code_Text_African-American'
    lime_sensitivity = calculate_sensitivity(lime_explanations, sensitive_feature)
    shap_sensitivity = calculate_sensitivity(shap_explanations, sensitive_feature)
    
    # Placeholder for OOD evaluation
    f1_scores = np.linspace(0, 1, 10)  # Placeholder for actual F1 scores, to be computed
    lime_sensitivities = [lime_sensitivity] * 10  # Example placeholder
    shap_sensitivities = [shap_sensitivity] * 10  # Example placeholder
    
    plt.plot(f1_scores, lime_sensitivities, label='LIME Sensitivity')
    plt.plot(f1_scores, shap_sensitivities, label='SHAP Sensitivity')
    plt.xlabel('F1 Score of OOD Classifier')
    plt.ylabel('% Explanation with "race" as Top Feature')
    plt.legend()
    plt.show()