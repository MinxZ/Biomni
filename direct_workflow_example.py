"""
Direct Function Usage Example: ChEMBL fup Modeling Workflow

This example demonstrates direct usage of the new preprocessing and modeling functions
for a complete cheminformatics workflow without using the agent interface.

Dataset: data/data/20240610_chembl34_extraxt_fup_human_1c1d.csv
SMILES Column: canonical_smiles  
Target: fup_converted
"""

import os
from datetime import datetime

from biomni.tool.modeling import (analyze_feature_importance,
                                  create_classification_plots,
                                  create_regression_plots, save_model,
                                  select_best_features_regression,
                                  train_classification_model,
                                  train_regression_model)
# Import our new functions including the enhanced descriptor calculation
from biomni.tool.preprocessing import (load_and_inspect_data,
                                       smiles_to_descriptors)


def main_workflow():
    """Execute the complete cheminformatics workflow."""
    
    print("🧪 Direct ChEMBL fup Modeling Workflow")
    print("="*60)
    
    # Configuration
    data_path = "data/data/20240610_chembl34_extraxt_fup_human_1c1d.csv"
    smiles_column = "canonical_smiles"
    target_column = "fup_converted"
    
    # Create image folder for saving plots
    image_folder = "image"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"📁 Created image folder: {image_folder}")
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return None
    
    # ============================================================================
    # Phase 1: Data Loading and Inspection
    # ============================================================================
    print("\n📍 Phase 1: Data Loading and Inspection")
    print("-" * 40)
    
    # Load and inspect data
    data_info = load_and_inspect_data(data_path)
    if data_info is None:
        print("❌ Failed to load data")
        return None
    
    data = data_info['data']
    inspection = data_info['inspection']
    
    print(f"✅ Loaded dataset: {data.shape}")
    print(f"📊 Missing values: {sum(inspection['missing_values'].values())}")
    print(f"🔄 Duplicates: {inspection['duplicates']}")
    
    # ============================================================================
    # Phase 2: Molecular Descriptor Calculation
    # ============================================================================
    print("\n📍 Phase 2: Molecular Descriptor Calculation")
    print("-" * 40)
    
    descriptor_file = "fup_dataset_with_descriptors.csv"
    
    # Check if descriptors already exist
    if os.path.exists(descriptor_file):
        print(f"🔄 Loading existing descriptors from {descriptor_file}")
        try:
            import pandas as pd
            data_with_descriptors = pd.read_csv(descriptor_file)
            print(f"✅ Descriptors loaded from file, dataset shape: {data_with_descriptors.shape}")
            
            # Verify the loaded data has the expected columns
            if smiles_column in data_with_descriptors.columns and target_column in data_with_descriptors.columns:
                data = data_with_descriptors
                print("✅ Descriptor file contains required columns")
            else:
                print("⚠️  Descriptor file missing required columns, recalculating...")
                raise ValueError("Missing required columns")
                
        except Exception as e:
            print(f"⚠️  Error loading descriptor file: {e}")
            print("🔄 Will recalculate descriptors...")
            # Fall through to calculation
            calculate_descriptors = True
        else:
            calculate_descriptors = False
    else:
        print("📊 Descriptor file not found, calculating descriptors...")
        calculate_descriptors = True
    
    # Calculate molecular descriptors using new enhanced functions
    if calculate_descriptors:
        try:
            # Extract SMILES from the dataset
            smiles_list = data[smiles_column].tolist()
            
            # Use the new enhanced descriptor calculation function
            descriptor_result = smiles_to_descriptors(
                smiles_list,
                n_jobs=8,  # Use 8 threads for faster processing
                batch_size=1000,  # Process in batches for memory efficiency
                remove_failed=True  # Remove molecules that failed parsing
            )
            
            print(f"✅ Descriptors calculated successfully")
            print(f"📊 Processing stats: {descriptor_result['stats']}")
            
            # Analyze descriptor quality
            quality_analysis = analyze_descriptor_quality(
                descriptor_result['descriptors'],
                descriptor_result['feature_names']
            )
            
            # Filter descriptors based on quality
            filtered_descriptors, filtered_names, removal_stats = filter_descriptors_by_quality(
                descriptor_result['descriptors'],
                descriptor_result['feature_names'],
                quality_analysis
            )
            
            print(f"📊 Quality filtering: {removal_stats}")
            
            # Merge with original data to preserve target column and other metadata
            # First, create a mapping of valid SMILES to target values
            import pandas as pd

            # Get valid rows from original data
            valid_data = data.iloc[descriptor_result['valid_indices']].copy()
            
            # Add descriptors to the valid data with rdkit_ prefix
            for i, desc_col in enumerate(filtered_names):
                valid_data[f"rdkit_{desc_col}"] = filtered_descriptors[:, i]
            
            data_with_descriptors = valid_data
            
            # Save the descriptor dataset
            data_with_descriptors.to_csv(descriptor_file, index=False)
            print(f"✅ Descriptors saved to {descriptor_file}, dataset shape: {data_with_descriptors.shape}")
            
            # Update our working data
            data = data_with_descriptors
            
        except ImportError as e:
            print(f"⚠️  Warning: {e}")
            print("   Continuing with existing descriptors if available...")
            
            # Check if we already have descriptors in the dataset
            descriptor_cols = [col for col in data.columns if col.startswith(('rdkit_', 'MaxEStateIndex'))]
            if descriptor_cols:
                print(f"✅ Found {len(descriptor_cols)} existing descriptor columns")
            else:
                print("❌ No descriptors found and cannot calculate new ones")
                return None
    
    # ============================================================================
    # Phase 3: Target Property Analysis and Transformation
    # ============================================================================
    print("\n📍 Phase 3: Target Property Analysis")
    print("-" * 40)
    
    # Analyze target distribution and detect transformation need
    transform_analysis = detect_log_transformation_need(
        data, 
        target_column,
        plot=False  # Disable plot to avoid display
    )
    
    # Apply transformation if recommended
    if transform_analysis['transformation_needed']:
        method = transform_analysis['recommended_method']
        print(f"🔄 Applying {method} transformation...")
        
        data_transformed, _ = apply_log_transformation(
            data, 
            target_column,
            method=method
        )
        
        # Update target column name if needed
        if method == 'log1p':
            target_column_transformed = target_column
        else:
            target_column_transformed = target_column
            
        data = data_transformed
        print(f"✅ Applied {method} transformation")
    else:
        print("✅ No transformation needed")
        target_column_transformed = target_column
    
    # ============================================================================
    # Phase 4: Preprocessing Pipeline
    # ============================================================================
    print("\n📍 Phase 4: Preprocessing Pipeline")
    print("-" * 40)
    
    # Identify descriptor columns
    try:
        descriptor_columns = identify_descriptor_columns(data, 'rdkit_MaxEStateIndex')
    except ValueError:
        # Try rdkit prefix if rdkit_MaxEStateIndex not found
        descriptor_columns = [col for col in data.columns if col.startswith('rdkit_')]
        if not descriptor_columns:
            print("❌ No descriptor columns found")
            return None
        print(f"📋 Using {len(descriptor_columns)} RDKit descriptor columns")
    
    # ============================================================================
    # Phase 4.1: Clean Error objects from RDKit descriptors
    print("\n📍 Phase 4.1: Cleaning Error Objects")
    print("-" * 40)
    
    data = data_with_descriptors
    data = data[descriptor_columns + [target_column_transformed]]
    mask = ~(data[descriptor_columns].isnull().any(axis=1) | data[target_column_transformed].isnull())
    data_clean = data[mask]
    clean_descriptor_columns = [col for col in descriptor_columns if not data_clean[col].isnull().any()]

    # Clean Error objects that might have resulted from failed descriptor calculations
    data_cleaned, _ = clean_error_objects(data, descriptor_columns)
    data = data_cleaned  # Update data with cleaned version
    
    # ============================================================================
    # Phase 4.2: Detect Problematic Values
    print("\n📍 Phase 4.2: Detecting Problematic Values") 
    print("-" * 40)
    
    # Detect problematic values
    problematic_analysis = detect_infinity_and_large_values(data, descriptor_columns)
    
    # Remove problematic columns
    if problematic_analysis['problematic_columns']:
        data_clean = remove_problematic_columns(data, problematic_analysis['problematic_columns'])
        # Update descriptor columns list
        descriptor_columns = [col for col in descriptor_columns 
                            if col not in problematic_analysis['problematic_columns']]
    else:
        data_clean = data
    
    # Keep only descriptor columns and target property
    columns_to_keep = descriptor_columns + [target_column_transformed]
    data_clean = data_clean[columns_to_keep]
    
    # Remove descriptor columns with NaN values and rows with NaN target
    initial_descriptors = len(descriptor_columns)
    
    # Find descriptor columns that have any NaN values
    nan_descriptor_columns = [col for col in descriptor_columns if data_clean[col].isnull().any()]
    clean_descriptor_columns = [col for col in descriptor_columns if not data_clean[col].isnull().any()]
    
    # Remove rows with NaN target values only
    initial_rows = len(data_clean)
    data_clean = data_clean.dropna(subset=[target_column_transformed])
    final_rows = len(data_clean)
    
    # Update descriptor columns list to only include clean descriptors
    descriptor_columns = clean_descriptor_columns
    
    # Keep only clean descriptor columns and target
    columns_to_keep = descriptor_columns + [target_column_transformed]
    data_clean = data_clean[columns_to_keep]
    
    print(f"✅ Cleaned dataset: {initial_rows} → {final_rows} rows")
    print(f"📊 Descriptors: {initial_descriptors} → {len(descriptor_columns)} (removed {len(nan_descriptor_columns)} with NaN)")
    if nan_descriptor_columns:
        print(f"🗑️  Removed descriptors with NaN: {nan_descriptor_columns[:5]}{'...' if len(nan_descriptor_columns) > 5 else ''}")
    
    # Save cleaned data for reuse
    cleaned_data_file = "cleaned_data_intermediate.csv"
    data_clean.to_csv(cleaned_data_file, index=False)
    print(f"💾 Saved cleaned data to {cleaned_data_file}")
    
    # ============================================================================
    # Phase 5: Feature Selection
    # ============================================================================
    print("\n📍 Phase 5: Feature Selection")
    print("-" * 40)
    
    # Comprehensive feature selection (regression-optimized)
    feature_selection_result = select_best_features_regression(
        data_clean,
        target_column_transformed,
        feature_columns=descriptor_columns,
        correlation_threshold=0.95,
        importance_threshold=0.001,
        model_type='random_forest'
    )
    
    # Visualize feature selection results
    # visualize_feature_selection_results(feature_selection_result, save_path=f"{image_folder}/feature_selection_results.png")
    
    # Get final dataset
    final_data = feature_selection_result['final_data']
    selected_features = feature_selection_result['selected_features']
    
    print(f"✅ Feature selection completed")
    print(f"📊 Features: {len(descriptor_columns)} → {len(selected_features)}")
    print(f"📊 Reduction ratio: {feature_selection_result['feature_reduction_ratio']:.2%}")
    
    # Save feature-selected data for reuse
    feature_selected_file = "feature_selected_data.csv"
    final_data.to_csv(feature_selected_file, index=False)
    print(f"💾 Saved feature-selected data to {feature_selected_file}")
    
    # Save selected features list
    import json
    selected_features_file = "selected_features.json"
    with open(selected_features_file, 'w') as f:
        json.dump(selected_features, f, indent=2)
    print(f"💾 Saved selected features list to {selected_features_file}")
    
    # ============================================================================
    # Phase 6: Regression Modeling
    # ============================================================================
    print("\n📍 Phase 6: Regression Modeling")
    print("-" * 40)
    
    # Train regression model
    regression_result = train_regression_model(
        final_data,
        target_column_transformed,
        feature_columns=selected_features,
        model_type='random_forest',
        random_state=42
    )
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(regression_result, top_n=15, plot=True, save_path=f"{image_folder}/regression_feature_importance.png")
    
    # Create diagnostic plots
    create_regression_plots(regression_result, save_path=f"{image_folder}/regression_diagnostics.png")
    
    # Save regression model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    r2_score = regression_result['metrics']['test_r2']
    regression_model_name = f"fup_regression_model_r2_{r2_score:.3f}_{timestamp}.pkl"
    save_model(regression_result, regression_model_name)
    
    print(f"✅ Regression model saved: {regression_model_name}")
    
    # ============================================================================
    # Phase 7: Classification Modeling
    # ============================================================================
    print("\n📍 Phase 7: Classification Modeling")
    print("-" * 40)
    
    # Train classification model (threshold = 0.05)
    classification_result = train_classification_model(
        final_data,
        target_column_transformed,
        feature_columns=selected_features,
        model_type='random_forest',
        threshold=0.05,
        random_state=42
    )
    
    # Analyze feature importance for classification
    analyze_feature_importance(classification_result, top_n=15, plot=True, save_path=f"{image_folder}/classification_feature_importance.png")
    
    # Create diagnostic plots
    create_classification_plots(classification_result, save_path=f"{image_folder}/classification_diagnostics.png")
    
    # Save classification model
    accuracy = classification_result['metrics']['test_accuracy']
    classification_model_name = f"fup_classification_model_acc_{accuracy:.3f}_{timestamp}.pkl"
    save_model(classification_result, classification_model_name)
    
    print(f"✅ Classification model saved: {classification_model_name}")
    
    # ============================================================================
    # Phase 8: Results Summary
    # ============================================================================
    print("\n📍 Phase 8: Results Summary")
    print("-" * 40)
    
    print("\n🎯 WORKFLOW SUMMARY")
    print("="*60)
    
    print(f"📊 Dataset Statistics:")
    print(f"   Original shape: {data_info['data'].shape}")
    print(f"   Final shape: {final_data.shape}")
    print(f"   Data retention: {len(final_data)/len(data_info['data']):.1%}")
    
    print(f"\n🔬 Feature Engineering:")
    print(f"   Original features: {len(descriptor_columns)} descriptors")
    print(f"   Selected features: {len(selected_features)}")
    print(f"   Feature reduction: {feature_selection_result['feature_reduction_ratio']:.1%}")
    
    print(f"\n📈 Model Performance:")
    print(f"   Regression R²: {r2_score:.4f}")
    print(f"   Regression MAE: {regression_result['metrics']['test_mae']:.4f}")
    print(f"   Classification Accuracy: {accuracy:.4f}")
    print(f"   Classification F1-Score: {classification_result['metrics']['f1_score']:.4f}")
    
    if transform_analysis['transformation_needed']:
        print(f"\n🔄 Transformation Applied:")
        print(f"   Method: {transform_analysis['recommended_method']}")
        print(f"   Confidence: {transform_analysis['confidence']:.1%}")
    
    print(f"\n💾 Generated Files:")
    print(f"   • fup_dataset_with_descriptors.csv (cached descriptors)")
    print(f"   • cleaned_data_intermediate.csv (cleaned data)")
    print(f"   • feature_selected_data.csv (final dataset)")
    print(f"   • selected_features.json (feature list)")
    print(f"   • {regression_model_name}")
    print(f"   • {classification_model_name}")
    print(f"   • {image_folder}/feature_selection_results.png")
    print(f"   • {image_folder}/regression_feature_importance.png")
    print(f"   • {image_folder}/regression_diagnostics.png")
    print(f"   • {image_folder}/classification_feature_importance.png")
    print(f"   • {image_folder}/classification_diagnostics.png")
    
    print(f"\n💡 Recommendations:")
    print(f"   • Use regression model for continuous fup prediction")
    print(f"   • Use classification model for high/low fup screening")
    print(f"   • Feature importance suggests focusing on: {importance_df.head(3)['Feature'].tolist()}")
    
    return {
        'final_data': final_data,
        'selected_features': selected_features,
        'regression_result': regression_result,
        'classification_result': classification_result,
        'feature_selection_result': feature_selection_result,
        'transform_analysis': transform_analysis
    }


if __name__ == "__main__":
    print("🧬 Direct ChEMBL fup Modeling Workflow")
    print("This example demonstrates direct function usage for cheminformatics modeling")
    print("="*80)
    
    try:
        results = main_workflow()
        
        if results:
            print("\n✅ Workflow completed successfully!")
            print("Check the generated files and plots for detailed results.")
        else:
            print("\n❌ Workflow failed. Check error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # print("\n🔧 Function Usage Demonstrated:")
    # print("   ✓ load_and_inspect_data() - Data loading and inspection")
    # print("   ✓ calculate_rdkit_descriptors() - RDKit descriptor calculation")
    # print("   ✓ detect_log_transformation_need() - Statistical transformation analysis")
    # print("   ✓ apply_log_transformation() - Target variable transformation")
    # print("   ✓ detect_infinity_and_large_values() - Data quality checking")
    # print("   ✓ remove_problematic_columns() - Data cleaning")
    # print("   ✓ select_best_features_regression() - Regression-optimized feature selection pipeline")
    # print("   ✓ train_regression_model() - Regression modeling")
    # print("   ✓ train_classification_model() - Classification modeling")
    # print("   ✓ analyze_feature_importance() - Feature importance analysis")
    # print("   ✓ create_regression_plots() - Regression diagnostics")
    # print("   ✓ create_classification_plots() - Classification diagnostics")
    # print("   ✓ save_model() - Model persistence")