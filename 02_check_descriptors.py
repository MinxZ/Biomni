import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the cleaned data from Step 1
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

# Drop rows with NaNs in descriptors (from previous step)
data = pd.concat([X, y], axis=1).dropna()
X_clean = data[descriptor_columns]
y_clean = data['fup_converted']

# --------------------------------------------------------------
# Check for Infinity and Extremely Large Values Before Training
# --------------------------------------------------------------

# 1. Check for infinity values
infinity_check = X_clean.isin([float('inf'), float('-inf')]).sum()
print("Number of infinity values in each descriptor column:")
print(infinity_check[infinity_check > 0])  # Display only columns with issues

# 2. Check for extremely large values (> 1e10)
large_value_threshold = 1e10  # You can adjust this threshold if needed
large_value_check = (X_clean > large_value_threshold).sum()
print("\nNumber of extremely large values (> 1e10) in each descriptor column:")
print(large_value_check[large_value_check > 0])  # Display only columns with issues

# --------------------------------------------------------------
# Remove Problematic Descriptors and Notify the User
# --------------------------------------------------------------
# Combine problematic columns (infinity and large values)
problematic_descriptors = set(infinity_check[infinity_check > 0].index.tolist() +
                              large_value_check[large_value_check > 0].index.tolist())

if problematic_descriptors:
    print(f"\nWarning: The following descriptors have been removed due to infinity or extremely large values:\n{problematic_descriptors}")
    # Remove problematic descriptors from X_clean
    X_clean = X_clean.drop(columns=problematic_descriptors)
else:
    print("\nNo infinity or extremely large values detected. Proceeding to model training.")

# --------------------------------------------------------------
# Proceed to Model Training After Cleaning
# --------------------------------------------------------------
# Check again to ensure no problematic columns remain
if X_clean.shape[1] == 0:
    print("\nError: No descriptors left for model training after removing problematic columns.")
else:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

    # Train Random Forest for Feature Importance and Regression
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = rf_model.predict(X_test)

    # --------------------------------------------------------------
    # Regression Results
    # --------------------------------------------------------------
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRegression Results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")

    # --------------------------------------------------------------
    # Feature Importance
    # --------------------------------------------------------------
    importances = rf_model.feature_importances_
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
    plt.title('Top 10 Most Important Features for fup_converted')
    plt.xlabel('importance Score')
    plt.ylabel('Feature')
    plt.show()

    # --------------------------------------------------------------
    # Actual vs Predicted Plot
    # --------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line of perfect prediction
    plt.xlabel('Actual fup_converted')
    plt.ylabel('Predicted fup_converted')
    plt.title('Actual vs Predicted Values')
    plt.show()
