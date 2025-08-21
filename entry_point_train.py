import argparse
import io
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from math import floor, log10

import boto3
import cloudpickle
import joblib
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import (QED, AllChem, Crippen, DataStructs, Descriptors,
                        FilterCatalog, rdMolDescriptors)
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm


def calculate_descriptors(mol):
    if mol is None:
        return None
    
    descriptors = []
    for name, function in Descriptors._descList:
        try:
            value = function(mol)
            descriptors.append(value)
        except:
            descriptors.append(np.nan)  # Treat errors as NaN
    
    return descriptors

def compute_descriptors_parallel(mols, n_jobs=-1):
    descriptors_list = Parallel(n_jobs=n_jobs)(delayed(calculate_descriptors)(mol) for mol in tqdm(mols))
    
    # Remove None values if any
    return [desc for desc in descriptors_list if desc is not None]


def calculate_rdkit_descriptors(mols, n_jobs=-1):
    """
    Calculate all RDKit descriptors for a given list of RDKit molecule objects and remove any NaN values.
    
    Args:
        mols (list): List of RDKit molecule objects.
    
    Returns:
        numpy.ndarray: Array containing all descriptors.
        list: List of descriptor names.
    """
    descriptors_list = []
    
    if n_jobs == 1:
        print("Calculating descriptors in serial mode")
        for mol in tqdm(mols):
            if mol is None:
                continue
            
            # Calculate all descriptors
            descriptors = []
            for name, function in Descriptors._descList:
                try:
                    value = function(mol)
                    descriptors.append(value)
                except:
                    descriptors.append(np.nan)  # Treat errors as NaN
            
            descriptors_list.append(descriptors)
    else:
        print("Calculating descriptors in parallel mode")
        descriptors_list = Parallel(n_jobs=n_jobs)(delayed(calculate_descriptors)(mol) for mol in tqdm(mols))
    
    fea_name = [name for name, function in Descriptors._descList]    
    # Convert to numpy array
    descriptors_array = np.array(descriptors_list)
    
    # Remove columns with any NaN values
    # valid_columns = ~np.isnan(descriptors_array).any(axis=0)
    # valid_columns = ~np.isinf(descriptors_array).any(axis=0)
    # replace nan with 0
    descriptors_array = np.nan_to_num(descriptors_array, nan=0.0, posinf=1, neginf=-1)
    # descriptors_array = descriptors_array[:, valid_columns]
    # fea_name = list(np.array(fea_name)[valid_columns])
    ceil = 3e+38
    descriptors_array[descriptors_array > ceil] = ceil
    
    return descriptors_array, fea_name

def round_to_significant(x, sig=3):
    if x == 0:
        return 0
    return round(x, sig - int(floor(log10(abs(x)))) - 1)

def round_floats(obj, sig=3):
    """
    Recursively round all float values in a nested data structure to a specified number of significant figures.
    This function traverses through dictionaries, lists, and other iterables, rounding any float
    values it encounters. It leaves other data types unchanged.
    Args:
        obj: The object to process. Can be a float, dict, list, or any other type.
        sig (int): The number of significant figures to round floats to. Defaults to 3.
    Returns:
        The input object with all floats rounded to the specified number of significant figures.
        If the input is not a float, dict, or list, it is returned unchanged.
    Examples:
        >>> round_floats(3534.412397)
        3530.0
        >>> round_floats(0.2397)
        0.24
        >>> round_floats({'a': 1.23456, 'b': [2.3456, 3.4567]})
        {'a': 1.23, 'b': [2.35, 3.46]}
    """
    # If the object is a float, round it to the specified number of significant figures
    if isinstance(obj, float):
        return round_to_significant(obj, sig)
    
    # If the object is a dictionary, apply round_floats to each of its values
    elif isinstance(obj, dict):
        return {k: round_floats(v, sig) for k, v in obj.items()}
    
    # If the object is a list, apply round_floats to each of its elements
    elif isinstance(obj, list):
        return [round_floats(v, sig) for v in obj]
    
    # If the object is neither a float, dict, nor list, return it unchanged
    return obj


def mol2svg(mol):
    """
    Convert RDKit molecule to SVG string.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
    
    Returns:
        str: SVG representation of the molecule.
    """
    d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def mol2fp(mol, radi=2, nBits=1024):
    """
    Convert RDKit molecule to Morgan fingerprint.
    
    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.
        radi (int): Radius for Morgan fingerprint. Default is 2.
        nBits (int): Number of bits in the fingerprint. Default is 1024.
    
    Returns:
        numpy.ndarray: Morgan fingerprint as a numpy array.
    """
    arr = np.zeros((1,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radi, nBits=nBits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def train_and_evaluate_model(x, y, smiles=None, n_estimators=100, random_state=42, n_splits=5):
    """
    Train and evaluate the PropertyRegressor model.
    
    Args:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        smiles (list): List of SMILES strings. Default is None.
    
    Returns:
        tuple: Trained model and evaluation results.
    """
    regressor = PropertyRegressor(n_estimators=n_estimators, random_state=random_state, n_splits=n_splits)
    regressor.load_data(x, y, smiles)
    mse = regressor.train_and_evaluate()
    return regressor

def prepare_true_vs_predicted_data(regressor):
    """
    Prepare data for True vs Predicted plot.
    
    Args:
        regressor (PropertyRegressor): Trained regressor model.
    
    Returns:
        dict: Data for True vs Predicted plot.
    """

    return {
        'true': regressor.ys,
        'pred': regressor.y_preds,
        'smiles': regressor.smiles
    }

def prepare_residual_data(regressor):
    """
    Prepare data for Residual plot.
    
    Args:
        regressor (PropertyRegressor): Trained regressor model.
    
    Returns:
        dict: Data for Residual plot.
    """
    return {
        'pred': regressor.y_preds,
        'residuals': regressor.residuals,
        'smiles': regressor.smiles
    }

def prepare_feature_importance_data(regressor, fea_name):
    """
    Prepare data for Feature Importance plot.
    
    Args:
        regressor (PropertyRegressor): Trained regressor model.
        fea_name (list): List of feature names.
    
    Returns:
        dict: Data for Feature Importance plot.
    """
    return {
        'Feature': fea_name,
        'importance': regressor.feature_importances.tolist()
    }

def prepare_model_results_data(regressor):
    """
    Prepare data for Model Results table and plots.
    
    Args:
        regressor (PropertyRegressor): Trained regressor model.
    
    Returns:
        dict: Data for Model Results table and plots.
    """
    return {
        'Dataset': ['Train', 'Test'],
        'MSE': [regressor.mse_train, regressor.mse],
        'R2': [regressor.r2_train, regressor.r2]
    }

def train_and_get_results(smiles, y, n_estimators, random_state, n_splits):
    mols = [Chem.MolFromSmiles(smi)for smi in smiles ]
    x, fea_name = calculate_rdkit_descriptors(mols)

    regressor = train_and_evaluate_model(x, y, smiles, n_estimators=n_estimators, random_state=random_state, n_splits=n_splits)

    # Prepare data for various plots and visualizations
    true_vs_pred_data = prepare_true_vs_predicted_data(regressor)
    residual_data = prepare_residual_data(regressor)
    feature_importance_data = prepare_feature_importance_data(regressor, fea_name)
    model_results_data = prepare_model_results_data(regressor)
    return regressor, true_vs_pred_data, residual_data, feature_importance_data, model_results_data


class PropertyRegressor:
    """
    A class to handle QSPR regression tasks using cross-validation.

    This class encapsulates the workflow of loading data, training a RandomForest model
    using cross-validation, making predictions, and evaluating results for QSPR modeling.
    """

    def __init__(self, n_estimators=100, random_state=42, n_splits=5):
        """
        Initialize the QSPRRegressor.

        Args:
            n_estimators (int): Number of trees in the RandomForest model.
            random_state (int): Random seed for reproducibility.
            n_splits (int): Number of splits for K-Fold cross-validation.
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_splits = n_splits
        self.models = []
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.X, self.y = None, None
        self.feature_importances = None

    def load_data(self, X, y, smiles=None):
        """
        Load the dataset.

        Args:
            X (np.array): Feature matrix.
            y (np.array): Target vector.
        """
        self.X = X
        self.y = y
        self.smiles_ori = np.array(smiles)

    def train_and_evaluate(self):
        """
        Train the RandomForest model using cross-validation and evaluate.

        Returns:
            float: Combined MSE for all validations.
        """
        mse_scores = []
        mse_scores_train = []
        r2_scores = []
        r2_scores_train = []

        feature_importances = []
        val_indexs = []
        ys = []
        y_preds = []
        smiles = []
        residuals = []

        y_train = []
        y_train_preds = []
        for fold, (train_index, val_index) in enumerate(self.kf.split(self.X), 1):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]
            val_indexs.append(val_index)

            model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
            model.fit(X_train, y_train)
            
            self.models.append(model)
            
            y_train_preds.extend(list(model.predict(X_train)))
            y_pred = model.predict(X_val)

            mse = mean_squared_error(y_val, y_pred)
            mse_train = mean_squared_error(y_train, model.predict(X_train))
            mse_scores.append(mse)
            mse_scores_train.append(mse_train)

            r2 = r2_score(y_val, y_pred)
            r2_train = r2_score(y_train, model.predict(X_train))
            r2_scores.append(r2)
            r2_scores_train.append(r2_train)

            feature_importances.append(model.feature_importances_)
            print(f"Fold {fold} MSE: {mse:.4f}")

            ys.extend(list(y_val))
            y_preds.extend(list(y_pred))
            if self.smiles_ori is not None:
                smiles.extend(self.smiles_ori[val_index])
            residuals.extend(list(y_val - y_pred))

        self.mse = np.mean(mse_scores)
        self.mse_train = np.mean(mse_scores_train)
        self.r2 = np.mean(r2_scores)
        self.r2_train = np.mean(r2_scores_train)
        print(f"Combined MSE: {self.mse:.4f}")

        self.feature_importances = np.mean(feature_importances, axis=0)
        self.val_indexs = val_indexs
        self.ys = ys
        self.y_preds = y_preds
        self.residuals = residuals
        self.smiles = smiles

        return self.mse


    def get_feature_importance(self):
        """
        Get the average feature importance across all folds.

        Returns:
            np.array: Average feature importance.
        """
        return self.feature_importances

    def predict(self, X):
        """
        Make predictions using all trained models and average the results.

        Args:
            X (np.array): Feature matrix to predict on.

        Returns:
            np.array: Average predicted values across all models.
        """
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)


    def load_models(self, filename_prefix):
        """
        Load trained models from files.

        Args:
            filename_prefix (str): Prefix for the filenames to load the models.
        """
        self.models = cloudpickle.load("{filename_prefix}.pkl")

    def save_models(self, filename_prefix):
        """
        Save all trained models to files.

        Args:
            filename_prefix (str): Prefix for the filenames to save the models.
        """
        with open(f"{filename_prefix}.pkl", "wb") as f:
            cloudpickle.dump(self.models, f)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators for the Random Forest')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of splits for cross-validation')
    parser.add_argument('--bucket-name', type=str, help='S3 bucket name for saving model artifacts')
    parser.add_argument('--link1', type=str, help='S3 bucket link')
    parser.add_argument('--link2', type=str, help='S3 bucket link')
    parser.add_argument('--y_name', type=str, help='y target')
    parser.add_argument('--job_name', type=str, help='job_name')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    return parser.parse_args()

if __name__ == '__main__':

    s3_client = boto3.client('s3', region_name='ap-south-1')

    #     parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    args = parse_args()

    # Update the script to use the parsed arguments
    n_estimators = args.n_estimators
    random_state = args.random_state
    n_splits = args.n_splits
    bucket_name = args.bucket_name  # You can also hardcode it if you don't want to pass it dynamically
    link1 = args.link1
    link2 = args.link2
    link = link1 + link2
    y_name = args.y_name
    job_name = args.job_name

    df = pd.read_csv(link)
    data_size = len(df)
    
    y = df[y_name].iloc[:100]  # Target vector
    smiles = df['_id'].iloc[:100]  # SMILES strings of molecules

    regressor, true_vs_pred_data, residual_data, feature_importance_data, model_results_data = \
        train_and_get_results(smiles, y, n_estimators=n_estimators, random_state=random_state, n_splits=n_splits)

    # Prepare the response data
    response_data_full = {
        'true_vs_pred_data': true_vs_pred_data,
        'residual_data': residual_data,
        'feature_importance_data': feature_importance_data,
        'model_results_data': model_results_data,
        'data_size': len(smiles),
    }
    response_data = round_floats(response_data_full)

    randomly_select = np.random.choice(len(smiles), 100, replace=False)
    true_vs_pred_data_select = {
        'true': np.array(true_vs_pred_data['true'])[randomly_select].tolist(),
        'pred': np.array(true_vs_pred_data['pred'])[randomly_select].tolist(),
       'smiles': np.array(true_vs_pred_data['smiles'])[randomly_select].tolist()
    }
    residual_data_select = {
        'pred': np.array(residual_data['pred'])[randomly_select].tolist(),
       'residuals': np.array(residual_data['residuals'])[randomly_select].tolist(),
       'smiles': np.array(residual_data['smiles'])[randomly_select].tolist()
    }
    response_data = {
        'true_vs_pred_data': true_vs_pred_data_select,
        'residual_data': residual_data_select,
        'feature_importance_data': feature_importance_data,
        'model_results_data': model_results_data,
        'data_size': len(smiles),
    }
    response_data = round_floats(response_data)

    stat_file_full = f'{job_name}_results_full.json'
    with open(stat_file_full, "w") as f:
        json.dump(response_data_full, f, indent=4)

    stat_file = f'{job_name}_results.json'
    with open(stat_file, "w") as f:
        json.dump(response_data, f, indent=4)

    model_file = f'{job_name}_model.joblib'
    with open(model_file, "wb") as f:
        cloudpickle.dump(regressor, f)

    s3_key = model_file
    s3_client.upload_file(model_file, bucket_name, s3_key)
    s3_url = f's3://{bucket_name}/{s3_key}'
    print(f"\nqspr model uploaded to {s3_url}")

    s3_key = stat_file
    s3_client.upload_file(stat_file, bucket_name, s3_key)
    s3_url = f's3://{bucket_name}/{s3_key}'
    print(f"\nqspr stat uploaded to {s3_url}")

    s3_key = stat_file_full
    s3_client.upload_file(stat_file_full, bucket_name, s3_key)
    s3_url = f's3://{bucket_name}/{s3_key}'
    print(f"\nqspr stat uploaded to {s3_url}")