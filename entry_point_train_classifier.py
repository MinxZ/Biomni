import argparse
import json
from math import floor, log10

import boto3
import cloudpickle
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
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
    Train and evaluate the PropertyClassifier model.
    
    Args:
        x (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        smiles (list): List of SMILES strings. Default is None.
    
    Returns:
        tuple: Trained model and evaluation results.
    """
    regressor = PropertyClassifier(n_estimators=n_estimators, random_state=random_state, n_splits=n_splits)
    regressor.load_data(x, y, smiles)
    _ = regressor.train_and_evaluate()
    return regressor

def prepare_true_vs_predicted_data(regressor):
    """
    Prepare data for True vs Predicted plot.
    
    Args:
        regressor (PropertyClassifier): Trained regressor model.
    
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
        regressor (PropertyClassifier): Trained regressor model.
    
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
        regressor (PropertyClassifier): Trained regressor model.
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
        regressor (PropertyClassifier): Trained regressor model.
    
    Returns:
        dict: Data for Model Results table and plots.
    """
    return {
        'Dataset': ['Train', 'Test'],
        'Accuracy': [regressor.accuracy_train, regressor.accuracy],
        'F1': [regressor.f1_train, regressor.f1],
        'Recall': [regressor.recall_train, regressor.recall],
        'Precision': [regressor.precision_train, regressor.precision],
        'ROC AUC': [regressor.roc_auc_train, regressor.roc_auc]
    }

def train_and_get_results(smiles, y, n_estimators, random_state, n_splits):
    mols = [Chem.MolFromSmiles(smi)for smi in smiles ]
    x, fea_name = calculate_rdkit_descriptors(mols)

    regressor = train_and_evaluate_model(x, y, smiles, n_estimators=n_estimators, random_state=random_state, n_splits=n_splits)

    # Prepare data for various plots and visualizations
    true_vs_pred_data = prepare_true_vs_predicted_data(regressor)
    feature_importance_data = prepare_feature_importance_data(regressor, fea_name)
    model_results_data = prepare_model_results_data(regressor)
    return regressor, true_vs_pred_data, feature_importance_data, model_results_data


class PropertyClassifier:
    """
    A class to handle QSPR classification tasks using cross - validation.

    This class encapsulates the workflow of loading data, training a RandomForest model
    using cross - validation, making predictions, and evaluating results for QSPR modeling.
    """

    def __init__(self, n_estimators=100, random_state=42, n_splits=5):
        """
        Initialize the QSPRClassifier.

        Args:
            n_estimators (int): Number of trees in the RandomForest model.
            random_state (int): Random seed for reproducibility.
            n_splits (int): Number of splits for K - Fold cross - validation.
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
        Train the RandomForest model using cross - validation and evaluate.

        Returns:
            float: Combined accuracy for all validations.
        """
        accuracy_scores = []
        accuracy_scores_train = []

        f1_scores = []
        f1_scores_train = []

        recall_scores = []
        recall_scores_train = []

        precision_scores = []
        precision_scores_train = []

        roc_auc_scores = []
        roc_auc_scores_train = []

        feature_importances = []
        val_indexs = []
        ys = []
        y_preds = []
        smiles = []

        y_train = []
        y_train_preds = []
        for fold, (train_index, val_index) in enumerate(self.kf.split(self.X), 1):
            X_train, X_val = self.X[train_index], self.X[val_index]
            y_train, y_val = self.y[train_index], self.y[val_index]
            val_indexs.append(val_index)

            model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1)
            model.fit(X_train, y_train)

            self.models.append(model)

            y_train_preds.extend(list(model.predict(X_train)))
            y_pred = model.predict(X_val)

            accuracy = model.score(X_val, y_val)
            accuracy_train = model.score(X_train, y_train)
            accuracy_scores.append(accuracy)
            accuracy_scores_train.append(accuracy_train)

            f1_val = f1_score(y_val, y_pred, average='weighted')
            f1_train = f1_score(y_train, model.predict(X_train), average='weighted')
            f1_scores.append(f1_val)
            f1_scores_train.append(f1_train)

            recall_score_val = recall_score(y_val, y_pred, average='weighted')
            recall_score_train = recall_score(y_train, model.predict(X_train), average='weighted')
            recall_scores.append(recall_score_val)
            recall_scores_train.append(recall_score_train)

            

            precision_score_val = precision_score(y_val, y_pred, average='weighted')
            precision_score_train = precision_score(y_train, model.predict(X_train), average='weighted')
            
            precision_scores.append(precision_score_val)
            precision_scores_train.append(precision_score_train)
            
            roc_auc_score_val = roc_auc_score(y_val, y_pred, average='weighted')
            roc_auc_scores.append(roc_auc_score_val)
            roc_auc_score_train = roc_auc_score(y_train, model.predict(X_train), average='weighted')
            roc_auc_scores_train.append(roc_auc_score_train)

            feature_importances.append(model.feature_importances_)
            print(f"Fold {fold} Accuracy: {accuracy:.4f}")

            ys.extend(list(y_val))
            y_preds.extend(list(y_pred))
            if self.smiles_ori is not None:
                smiles.extend(self.smiles_ori[val_index])
            # misclassifications.extend([y_val[i]!= y_pred[i] for i in range(len(y_val))])

        self.accuracy = np.mean(accuracy_scores)
        self.accuracy_train = np.mean(accuracy_scores_train)
        self.f1 = np.mean(f1_scores)
        self.f1_train = np.mean(f1_scores_train)
        self.recall = np.mean(recall_scores)
        self.recall_train = np.mean(recall_scores_train)
        self.precision = np.mean(precision_scores)
        self.precision_train = np.mean(precision_scores_train)
        self.roc_auc = np.mean(roc_auc_scores)
        self.roc_auc_train = np.mean(roc_auc_scores_train)
        print(f"Combined Accuracy: {self.accuracy:.4f}")

        self.feature_importances = np.mean(feature_importances, axis=0)
        self.val_indexs = val_indexs
        self.ys = ys
        self.y_preds = y_preds
        # self.misclassifications = misclassifications
        self.smiles = smiles

        return self.accuracy


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
    parser.add_argument('--bucket-name', type=str, default='1', help='S3 bucket name for saving model artifacts')
    parser.add_argument('--link1', type=str, default='1', help='S3 bucket link')
    parser.add_argument('--link2', type=str, default='1', help='S3 bucket link')
    parser.add_argument('--y_name', type=str, default='is_active',help='y target')
    parser.add_argument('--job_name', type=str, default='1',help='job_name')
    parser.add_argument('--output-data', type=str, default='/opt/ml/processing/output')
    return parser.parse_args()


if __name__ == '__main__':
    #     parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    s3_client = boto3.client('s3', region_name='ap-south-1')
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

    # df = pd.read_csv(link)
    df = pd.read_csv('/Users/z/Downloads/temp_a7cfa88fd5.csv')
    # random shuffle df 
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    false_count = (df[y_name] == False).sum()
    true_count = (df[y_name] == True).sum()

    # Determine the smaller count
    min_count = min(false_count, true_count)

    # Sample equal number of False and True instances
    df_false = df[df[y_name] == False].sample(n=min(min_count//4, false_count), random_state=42)
    df_true = df[df[y_name] == True].sample(n=min_count, random_state=42)

    balanced_df = pd.concat([df_false, df_true])

    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    y = balanced_df[y_name].iloc[:1000]  # Target vector
    smiles = balanced_df['_id'].iloc[:1000]  # SMILES strings of molecules

    regressor, true_vs_pred_data, feature_importance_data, model_results_data = \
        train_and_get_results(smiles, y, n_estimators=n_estimators, random_state=random_state, n_splits=n_splits)

    randomly_select = np.random.choice(len(smiles), 100, replace=False)
    true_vs_pred_data_select = {
        'true': np.array(true_vs_pred_data['true'])[randomly_select].tolist(),
        'pred': np.array(true_vs_pred_data['pred'])[randomly_select].tolist(),
       'smiles': np.array(true_vs_pred_data['smiles'])[randomly_select].tolist()
    }
    true_vs_pred_data_select['pred'] = [1 if i else 0 for i in true_vs_pred_data_select['pred']]
    true_vs_pred_data_select['true'] = [1 if i else 0 for i in true_vs_pred_data_select['true']]

    # Prepare the response data
    response_data = {
        'true_vs_pred_data': true_vs_pred_data_select,
        'feature_importance_data': feature_importance_data,
        'model_results_data': model_results_data,
        'data_size': len(smiles),
    }

    response_data = round_floats(response_data)
    print(response_data)

    stat_file = f'{job_name}_results.json'
    response_data 

    with open(stat_file, "w") as f:
        json.dump(response_data, f, indent=4)

    model_file = f'{job_name}_model.joblib'
    with open(model_file, "wb") as f:
        cloudpickle.dump(regressor, f)

    # s3_key = model_file
    # s3_client.upload_file(model_file, bucket_name, s3_key)
    # s3_url = f's3://{bucket_name}/{s3_key}'
    # print(f"\nqspr model uploaded to {s3_url}")

    # s3_key = stat_file
    # s3_client.upload_file(stat_file, bucket_name, s3_key)
    # s3_url = f's3://{bucket_name}/{s3_key}'
    # print(f"\nqspr stat uploaded to {s3_url}")
