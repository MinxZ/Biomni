import pandas as pd

# Step 1: Load CSV file
input_filename = 'demo_20240610_chembl34_extraxt_fup_human_1c1d_asdemo.csv'  # Replace with your actual file name
df = pd.read_csv(input_filename)

# Display the first few rows to understand the structure
print("First few rows of the original data:")
print(df.head())

# Step 2: Identify descriptor columns (starting from 'MaxEStateIndex')
if 'MaxEStateIndex' not in df.columns:
    raise ValueError("The input CSV file does not contain the 'MaxEStateIndex' column.")

# Get all columns starting from 'MaxEStateIndex' as descriptors
descriptor_columns = df.columns[df.columns.get_loc('MaxEStateIndex'):]
print(f"\nIdentified descriptor columns: {list(descriptor_columns)}")

# Step 3: Check for missing values (NaN) in descriptor columns
missing_values_count = df[descriptor_columns].isna().sum()
print("\nMissing values in each descriptor column:")
print(missing_values_count)

# Total missing values in descriptor columns
total_missing = missing_values_count.sum()
print(f"\nTotal missing values in descriptor columns: {total_missing}")

# Step 4: Remove rows with NaNs in descriptor columns (keep 0 values)
df_clean = df.dropna(subset=descriptor_columns)

# Show row counts before and after removing NaNs
original_count = df.shape[0]
cleaned_count = df_clean.shape[0]

print(f"\nOriginal number of rows: {original_count}")
print(f"Number of rows after removing NaNs in descriptors: {cleaned_count}")
print(f"Number of rows removed: {original_count - cleaned_count}")
