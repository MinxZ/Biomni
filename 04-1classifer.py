import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

# Load the cleaned data
input_filename = 'demo_20240610_chembl34_extraxt_fup_human_1c1d_asdemo.csv'  # Replace with your actual file name
df = pd.read_csv(input_filename)

# Ensure 'MaxEStateIndex' and 'fup_converted' columns exist
if 'MaxEStateIndex' not in df.columns:
    raise ValueError("The input CSV does not contain the 'MaxEStateIndex' column.")
if 'fup_converted' not in df.columns:
    raise ValueError("The input CSV does not contain the 'fup_converted' column.")

# Select descriptor columns starting from 'MaxEStateIndex'
descriptor_columns = df.columns[df.columns.get_loc('MaxEStateIndex'):]
X = df[descriptor_columns]  # Features (descriptors)
y = df['fup_converted']     # Target variable

# Drop rows with NaNs in descriptors
data = pd.concat([X, y], axis=1).dropna()
X_clean = data[descriptor_columns]
y_clean = data['fup_converted']

# --------------------------------------------------------------
# Convert fup_converted to Binary Classification Based on Threshold 0.05
# --------------------------------------------------------------
threshold = 0.05
y_class = (y_clean > threshold).astype(int)  # 1 if > 0.05, else 0

# --------------------------------------------------------------
# Detect and Remove Outliers in Descriptors
# --------------------------------------------------------------

# Check for infinity and extremely large values
infinity_check = X_clean.isin([float('inf'), float('-inf')]).sum()
large_value_threshold = 1e10
large_value_check = (X_clean > large_value_threshold).sum()

# Remove problematic descriptors
problematic_descriptors = set(infinity_check[infinity_check > 0].index.tolist() +
                              large_value_check[large_value_check > 0].index.tolist())

if problematic_descriptors:
    print(f"\nWarning: Removed descriptors due to infinity or large values:\n{problematic_descriptors}")
    X_clean = X_clean.drop(columns=problematic_descriptors)

# --------------------------------------------------------------
# Proceed to Model Training
# --------------------------------------------------------------
if X_clean.shape[1] == 0:
    print("\nError: No descriptors left for model training after removing problematic columns.")
else:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_class, test_size=0.2, random_state=42, stratify=y_class)

    # Train Random Forest Classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_classifier.predict(X_test)

    # --------------------------------------------------------------
    # Classification Results
    # --------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nClassification Results (Threshold = {threshold}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Detailed Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # --------------------------------------------------------------
    # Confusion Matrix Visualization
    # --------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['<= 0.05', '> 0.05'], yticklabels=['<= 0.05', '> 0.05'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # --------------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------------
    importances = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X_clean.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    # Display top 10 most important features
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Most Important Features')
    plt.xlabel('importance Score')
    plt.ylabel('Feature')
    plt.show()

    from sklearn.metrics import accuracy_score, cohen_kappa_score

    # --------------------------------------------------------------
    # Classification Results with Accuracy and Kappa
    # --------------------------------------------------------------
    # Predict on test data
    y_pred = rf_classifier.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(y_test, y_pred)

    # Display results
    print(f"\nClassification Results (Threshold = {threshold}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

