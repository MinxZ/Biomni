import os

import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem

# Initialize Mordred calculator with all available descriptors
calc = Calculator(descriptors, ignore_3D=True)  # Set ignore_3D=True if 3D descriptors are not required

# Load CSV file
input_filename = 'data/data/20240610_chembl34_extraxt_fup_human_1c1d.csv'  # Replace with your file name
df = pd.read_csv(input_filename)

# Check if the canonical_smiles column exists
if 'canonical_smiles' not in df.columns:
    raise ValueError("The input CSV does not contain a 'canonical_smiles' column.")

# Function to calculate Mordred descriptors
def calculate_mordred(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return pd.Series([None] * calc.descriptors_number(), index=[str(d) for d in calc.descriptors])
    return calc(mol)

# Calculate descriptors for each SMILES
descriptor_df = df['canonical_smiles'].apply(calculate_mordred)

# Append descriptor columns to the original dataframe
result_df = pd.concat([df, descriptor_df], axis=1)

# Save to a new CSV file with 'mordred_' prefix
output_filename = f"mordred_{os.path.basename(input_filename)}"
result_df.to_csv(output_filename, index=False)

print(f"Descriptors calculated and saved to {output_filename}")

# Print the number of descriptors generated
num_descriptors = calc.descriptors_number()
print(f"Total number of descriptors generated: {num_descriptors}")
