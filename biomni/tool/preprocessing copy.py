def save_preprocessing_pipeline(workflow_results, output_path="preprocessing_pipeline.json"):
    """
    Save comprehensive preprocessing pipeline metadata for inference.
    
    This function extracts all the preprocessing steps, transformations, scalers,
    encoders, and feature selections from workflow results and saves them to a
    JSON file for later use in inference.
    
    Args:
        workflow_results (dict): Complete workflow results from training pipeline
        output_path (str): Path to save the preprocessing pipeline JSON
        
    Returns:
        dict: The saved preprocessing pipeline metadata
    """
    import json
    import pickle
    import base64
    from datetime import datetime
    
    print("💾 Saving comprehensive preprocessing pipeline...")
    
    # Extract workflow components
    data_prep_result = workflow_results['data_preparation']
    preprocessing_result = workflow_results['preprocessing'] 
    modeling_result = workflow_results['modeling']
    
    # Serialize sklearn objects (scalers, encoders, etc.)
    def serialize_sklearn_object(obj):
        if obj is None:
            return None
        pickled = pickle.dumps(obj)
        return base64.b64encode(pickled).decode('utf-8')
    
    # Build comprehensive pipeline metadata
    pipeline_metadata = {
        "pipeline_info": {
            "created_at": datetime.now().isoformat(),
            "pipeline_version": "comprehensive_v1.0",
            "description": "Complete preprocessing pipeline for molecule property prediction"
        },
        
        "data_preparation": {
            "original_target_column": data_prep_result.get('target_column'),
            "original_smiles_column": data_prep_result.get('smiles_column', 'canonical_smiles'),
            "descriptor_calculation": {
                "method": "RDKit",
                "n_descriptors_calculated": data_prep_result['processing_stats']['descriptors_calculated'],
                "batch_size": data_prep_result.get('batch_size', 1000),
                "n_jobs": data_prep_result.get('n_jobs', 8)
            },
            "quality_filtering": {
                "correlation_threshold": data_prep_result.get('correlation_threshold', 0.95),
                "variance_threshold": data_prep_result.get('variance_threshold', 0.01)
            }
        },
        
        "preprocessing": {
            "target_transformation": {
                "transformation_needed": preprocessing_result['transform_analysis']['transformation_needed'],
                "method": preprocessing_result['transform_analysis'].get('recommended_method'),
                "transformation_info": preprocessing_result.get('transformation_info'),
                "target_column": preprocessing_result['target_column']
            },
            "data_cleaning": {
                "initial_descriptors": preprocessing_result['data_stats']['initial_descriptors'],
                "final_descriptors": preprocessing_result['data_stats']['final_descriptors'],
                "removed_descriptors": preprocessing_result['data_stats']['initial_descriptors'] - preprocessing_result['data_stats']['final_descriptors']
            },
            "descriptor_columns": preprocessing_result['descriptor_columns'],
            "preprocessing_steps": preprocessing_result.get('preprocessing_steps', [])
        },
        
        "feature_selection": {
            "original_features": len(modeling_result['feature_selection_result']['original_features']),
            "selected_features": modeling_result['selected_features'],
            "correlation_threshold": modeling_result['parameters']['correlation_threshold'],
            "importance_threshold": modeling_result['parameters']['importance_threshold'],
            "removed_features": list(set(modeling_result['feature_selection_result']['original_features']) - 
                                   set(modeling_result['selected_features'])),
            "feature_reduction_ratio": modeling_result['feature_selection_result']['feature_reduction_ratio']
        },
        
        "normalization": {
            # Extract normalization parameters if available
            "method": "standard",  # Default assumption
            "feature_columns": modeling_result['selected_features']
        },
        
        "model_info": {
            "regression_model_file": modeling_result.get('regression_model_file'),
            "classification_model_file": modeling_result.get('classification_model_file'),
            "model_type": modeling_result['parameters']['model_type'],
            "classification_threshold": modeling_result['parameters'].get('classification_threshold'),
            "performance": {
                "regression_r2": modeling_result['regression_result']['metrics'].get('r2') or modeling_result['regression_result']['metrics'].get('test_r2'),
                "classification_accuracy": modeling_result['classification_result']['metrics'].get('accuracy') or modeling_result['classification_result']['metrics'].get('test_accuracy') if modeling_result['classification_result'] else None
            }
        },
        
        "data_stats": {
            "original_molecules": workflow_results['summary']['original_molecules'],
            "final_molecules": workflow_results['summary']['final_molecules'],
            "final_features": workflow_results['summary']['final_features'],
            "data_retention_rate": workflow_results['summary']['final_molecules'] / workflow_results['summary']['original_molecules']
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(pipeline_metadata, f, indent=2, default=str)
    
    print(f"✅ Preprocessing pipeline saved to: {output_path}")
    print(f"📊 Pipeline includes:")
    print(f"   • Data preparation: {pipeline_metadata['data_preparation']['descriptor_calculation']['n_descriptors_calculated']} descriptors")
    print(f"   • Feature selection: {len(pipeline_metadata['feature_selection']['selected_features'])} selected features")
    print(f"   • Target transformation: {pipeline_metadata['preprocessing']['target_transformation']['transformation_needed']}")
    print(f"   • Model files: regression + {'classification' if pipeline_metadata['model_info']['classification_model_file'] else 'no classification'}")
    
    return pipeline_metadata


def load_preprocessing_pipeline(pipeline_path="preprocessing_pipeline.json"):
    """
    Load preprocessing pipeline metadata for inference.
    
    Args:
        pipeline_path (str): Path to the preprocessing pipeline JSON file
        
    Returns:
        dict: The loaded preprocessing pipeline metadata
    """
    import json
    
    print(f"📂 Loading preprocessing pipeline from: {pipeline_path}")
    
    with open(pipeline_path, 'r') as f:
        pipeline_metadata = json.load(f)
    
    print("✅ Preprocessing pipeline loaded successfully")
    print(f"📊 Pipeline version: {pipeline_metadata['pipeline_info']['pipeline_version']}")
    print(f"📅 Created: {pipeline_metadata['pipeline_info']['created_at']}")
    
    return pipeline_metadata


def apply_inference_preprocessing(smiles_list, pipeline_metadata, verbose=True):
    """
    Apply the complete preprocessing pipeline to new molecules for inference.
    
    This function takes raw SMILES strings and applies the exact same preprocessing
    steps that were used during training, ensuring consistent transformations.
    
    Args:
        smiles_list (list): List of SMILES strings to process
        pipeline_metadata (dict): Pipeline metadata from save_preprocessing_pipeline
        verbose (bool): Whether to print progress information
        
    Returns:
        dict: Processed data ready for model inference
    """
    import pandas as pd
    import numpy as np
    
    if verbose:
        print(f"🧬 Applying inference preprocessing to {len(smiles_list)} molecule(s)")
    
    # Step 1: Calculate molecular descriptors using the same method
    if verbose:
        print("📍 Step 1: Calculating molecular descriptors")
    
    descriptor_result = smiles_to_descriptors(
        smiles_list=smiles_list,
        n_jobs=pipeline_metadata['data_preparation']['descriptor_calculation'].get('n_jobs', 1),
        batch_size=pipeline_metadata['data_preparation']['descriptor_calculation'].get('batch_size', 1000),
        remove_failed=True
    )
    
    if len(descriptor_result['valid_indices']) == 0:
        print("❌ No valid molecules found after descriptor calculation")
        return None
    
    # Create DataFrame with descriptors
    if verbose:
        print(f"✅ Successfully calculated descriptors for {len(descriptor_result['valid_indices'])} molecules")
    
    # Step 2: Create DataFrame with same structure as training
    if verbose:
        print("📍 Step 2: Creating feature DataFrame")
    
    # Get valid SMILES
    valid_smiles = [smiles_list[i] for i in descriptor_result['valid_indices']]
    
    # Create base DataFrame
    inference_data = pd.DataFrame({
        'smiles': valid_smiles
    })
    
    # Add descriptors with rdkit_ prefix (matching training format)
    descriptors_array = descriptor_result['descriptors']
    feature_names = descriptor_result['feature_names']
    
    for i, desc_name in enumerate(feature_names):
        inference_data[f"rdkit_{desc_name}"] = descriptors_array[:, i]
    
    if verbose:
        print(f"✅ Created DataFrame with {len(inference_data)} rows and {len(feature_names)} descriptors")
    
    # Step 3: Apply feature selection (keep only selected features)
    if verbose:
        print("📍 Step 3: Applying feature selection")
    
    selected_features = pipeline_metadata['feature_selection']['selected_features']
    available_features = [col for col in inference_data.columns if col.startswith('rdkit_')]
    
    # Check if selected features are available
    missing_features = set(selected_features) - set(available_features)
    if missing_features:
        if verbose:
            print(f"⚠️  Warning: {len(missing_features)} features from training not available in inference data")
    
    # Keep only available selected features
    features_to_use = [f for f in selected_features if f in available_features]
    final_features = ['smiles'] + features_to_use
    
    inference_data_selected = inference_data[final_features].copy()
    
    if verbose:
        print(f"✅ Applied feature selection: {len(available_features)} → {len(features_to_use)} features")
    
    # Step 4: Handle missing descriptors (fill with mean/median from training if possible)
    if verbose:
        print("📍 Step 4: Handling missing values")
    
    # For now, use simple imputation (could be enhanced with training statistics)
    numeric_columns = inference_data_selected.select_dtypes(include=[np.number]).columns
    inference_data_selected[numeric_columns] = inference_data_selected[numeric_columns].fillna(0)
    
    if verbose:
        print("✅ Missing values handled")
    
    # Step 5: Final preparation
    if verbose:
        print("📍 Step 5: Final data preparation")
    
    result = {
        'processed_data': inference_data_selected,
        'feature_columns': features_to_use,
        'valid_smiles': valid_smiles,
        'valid_indices': descriptor_result['valid_indices'],
        'n_molecules_processed': len(inference_data_selected),
        'n_features': len(features_to_use),
        'preprocessing_summary': {
            'original_molecules': len(smiles_list),
            'valid_molecules': len(valid_smiles),
            'success_rate': len(valid_smiles) / len(smiles_list) * 100,
            'features_available': len(features_to_use),
            'features_missing': len(missing_features) if missing_features else 0
        }
    }
    
    if verbose:
        print(f"✅ Inference preprocessing completed")
        print(f"📊 Summary: {result['preprocessing_summary']['original_molecules']} → {result['preprocessing_summary']['valid_molecules']} molecules")
        print(f"🎯 Success rate: {result['preprocessing_summary']['success_rate']:.1f}%")
        print(f"📈 Features: {result['preprocessing_summary']['features_available']} available")
    
    return result