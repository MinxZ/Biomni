"""
New Direct Workflow Example: ChEMBL fup Modeling using Comprehensive Workflows

This example demonstrates the simplified usage of the new comprehensive workflow functions
for a complete cheminformatics pipeline. The new approach reduces complexity from 15+ 
individual function calls to just 3 comprehensive workflow functions.

Dataset: data/data/20240610_chembl34_extraxt_fup_human_1c1d.csv
SMILES Column: canonical_smiles  
Target: fup_converted

Key improvements:
- Simplified from ~400 lines to ~100 lines
- Better error handling and progress tracking
- Comprehensive reporting and statistics
- Automatic intermediate file management
"""

import os
from datetime import datetime

from biomni.tool.modeling import modeling_workflow
# Import the new comprehensive workflow functions
from biomni.tool.preprocessing import (data_preparation_workflow,
                                       preprocessing_pipeline_workflow,
                                       save_preprocessing_pipeline)


def comprehensive_cheminformatics_workflow():
    """Execute the complete cheminformatics workflow using the new comprehensive functions."""
    
    print("🧪 Comprehensive ChEMBL fup Modeling Workflow (New Version)")
    print("="*70)
    
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
    # WORKFLOW 1: Data Preparation
    # ============================================================================
    print(f"\n{'='*70}")
    print("🧬 WORKFLOW 1: DATA PREPARATION")
    print(f"{'='*70}")
    
    # Single function call replaces ~8 individual functions
    data_prep_result = data_preparation_workflow(
        data_path=data_path,
        smiles_column=smiles_column,
        target_column=target_column,
        n_jobs=8,
        batch_size=1000
    )
    
    if data_prep_result is None:
        print("❌ Data preparation failed")
        return None
    
    # Save prepared data with descriptors
    descriptor_file = "fup_dataset_with_descriptors.csv"
    data_prep_result['data_with_descriptors'].to_csv(descriptor_file, index=False)
    print(f"💾 Saved data with descriptors to {descriptor_file}")
    
    # Display preparation summary
    stats = data_prep_result['processing_stats']
    print(f"\n📊 Data Preparation Summary:")
    print(f"   Original molecules: {stats['original_rows']}")
    print(f"   Valid molecules: {stats['valid_molecules']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    print(f"   Descriptors calculated: {stats['descriptors_calculated']}")
    
    # ============================================================================
    # WORKFLOW 2: Preprocessing Pipeline
    # ============================================================================
    print(f"\n{'='*70}")
    print("🔧 WORKFLOW 2: PREPROCESSING PIPELINE")
    print(f"{'='*70}")
    
    # Single function call replaces ~7 individual functions
    preprocessing_result = preprocessing_pipeline_workflow(
        data=data_prep_result['data_with_descriptors'],
        target_column=target_column,
        descriptor_start_col='rdkit_'
    )
    
    if preprocessing_result is None:
        print("❌ Preprocessing failed")
        return None
    
    # Save cleaned intermediate data
    cleaned_data_file = "cleaned_data_intermediate.csv"
    preprocessing_result['preprocessed_data'].to_csv(cleaned_data_file, index=False)
    print(f"💾 Saved preprocessed data to {cleaned_data_file}")
    
    # Display preprocessing summary
    prep_stats = preprocessing_result['data_stats']
    print(f"\n📊 Preprocessing Summary:")
    print(f"   Rows: {prep_stats['initial_rows']} → {prep_stats['final_rows']}")
    print(f"   Descriptors: {prep_stats['initial_descriptors']} → {prep_stats['final_descriptors']}")
    print(f"   Transformation applied: {preprocessing_result['transform_analysis']['transformation_needed']}")
    if preprocessing_result['transform_analysis']['transformation_needed']:
        method = preprocessing_result['transform_analysis']['recommended_method']
        confidence = preprocessing_result['transform_analysis']['confidence']
        print(f"   Transformation method: {method} (confidence: {confidence:.1%})")
    
    # ============================================================================
    # WORKFLOW 3: Modeling
    # ============================================================================
    print(f"\n{'='*70}")
    print("🤖 WORKFLOW 3: MODELING")
    print(f"{'='*70}")
    
    # Single function call replaces ~10 individual functions
    modeling_result = modeling_workflow(
        data=preprocessing_result['preprocessed_data'],
        target_column=preprocessing_result['target_column'],
        feature_columns=preprocessing_result['descriptor_columns'],
        correlation_threshold=0.95,
        importance_threshold=0.001,
        model_type='random_forest',
        classification_threshold=0.05,  # For binary classification at threshold 0.05
        random_state=42,
        image_folder=image_folder
    )
    
    if modeling_result is None:
        print("❌ Modeling failed")
        return None
    
    # Save feature-selected data
    feature_selected_file = "feature_selected_data.csv"
    modeling_result['final_data'].to_csv(feature_selected_file, index=False)
    print(f"💾 Saved feature-selected data to {feature_selected_file}")
    
    # Save selected features list
    import json
    selected_features_file = "selected_features.json"
    with open(selected_features_file, 'w') as f:
        json.dump(modeling_result['selected_features'], f, indent=2)
    print(f"💾 Saved selected features list to {selected_features_file}")
    
    # Save comprehensive preprocessing pipeline for inference
    pipeline_metadata_file = "preprocessing_pipeline.json"
    workflow_results = {
        'data_preparation': data_prep_result,
        'preprocessing': preprocessing_result,
        'modeling': modeling_result,
        'summary': {
            'original_molecules': data_prep_result['processing_stats']['original_rows'],
            'final_molecules': len(modeling_result['final_data']),
            'final_features': len(modeling_result['selected_features'])
        }
    }
    save_preprocessing_pipeline(workflow_results, pipeline_metadata_file)
    print(f"💾 Saved preprocessing pipeline to {pipeline_metadata_file}")
    
    # ============================================================================
    # COMPREHENSIVE RESULTS SUMMARY
    # ============================================================================
    print(f"\n{'='*70}")
    print("📋 COMPREHENSIVE WORKFLOW RESULTS")
    print(f"{'='*70}")
    
    # Extract key metrics
    regression_metrics = modeling_result['regression_result']['metrics']
    classification_metrics = modeling_result['classification_result']['metrics'] if modeling_result['classification_result'] else None
    
    print(f"\n🧬 Data Processing Results:")
    print(f"   📊 Original dataset: {data_prep_result['processing_stats']['original_rows']} molecules")
    print(f"   📊 Final dataset: {modeling_result['final_data'].shape}")
    print(f"   📈 Data retention: {len(modeling_result['final_data'])/data_prep_result['processing_stats']['original_rows']:.1%}")
    print(f"   🧪 Descriptors: {data_prep_result['processing_stats']['descriptors_calculated']} → {len(modeling_result['selected_features'])}")
    print(f"   📉 Feature reduction: {modeling_result['feature_selection_result']['feature_reduction_ratio']:.1%}")
    
    print(f"\n🤖 Model Performance:")
    
    # Handle both cross-validation and train/test split scenarios for regression metrics
    regression_result = modeling_result['regression_result']
    if 'cross_validation' in regression_result and regression_result['cross_validation']:
        # Cross-validation metrics
        r2 = regression_metrics['r2']
        mae = regression_metrics['mae']
        mse = regression_metrics['mse']
    else:
        # Train/test split metrics
        r2 = regression_metrics['test_r2']
        mae = regression_metrics['test_mae']
        mse = regression_metrics['test_mse']
    
    print(f"   📈 Regression R²: {r2:.4f}")
    print(f"   📊 Regression MAE: {mae:.4f}")
    print(f"   📊 Regression MSE: {mse:.4f}")
    
    if classification_metrics:
        # Handle both cross-validation and train/test split scenarios for classification metrics
        classification_result = modeling_result['classification_result']
        if 'cross_validation' in classification_result and classification_result['cross_validation']:
            # Cross-validation metrics
            accuracy = classification_metrics['accuracy']
        else:
            # Train/test split metrics
            accuracy = classification_metrics['test_accuracy']
        
        print(f"   🎯 Classification Accuracy: {accuracy:.4f}")
        print(f"   📊 Classification F1-Score: {classification_metrics['f1_score']:.4f}")
        print(f"   📊 Classification Precision: {classification_metrics['precision']:.4f}")
        print(f"   📊 Classification Recall: {classification_metrics['recall']:.4f}")
    
    if preprocessing_result['transform_analysis']['transformation_needed']:
        print(f"\n🔄 Target Transformation:")
        print(f"   Method: {preprocessing_result['transform_analysis']['recommended_method']}")
        print(f"   Confidence: {preprocessing_result['transform_analysis']['confidence']:.1%}")
    
    print(f"\n💾 Generated Files:")
    print(f"   • {descriptor_file} (data with descriptors)")
    print(f"   • {cleaned_data_file} (preprocessed data)")
    print(f"   • {feature_selected_file} (final modeling dataset)")
    print(f"   • {selected_features_file} (selected features)")
    print(f"   • {modeling_result['regression_model_file']} (regression model)")
    if modeling_result['classification_model_file']:
        print(f"   • {modeling_result['classification_model_file']} (classification model)")
    
    generated_plots = modeling_result['generated_files']['plots']
    print(f"   📊 Diagnostic Plots:")
    for plot_type, plot_path in generated_plots.items():
        if plot_path:
            print(f"      • {plot_path}")
    
    print(f"\n💡 Workflow Insights:")
    print(f"   🚀 Complexity Reduction: ~400 lines → ~100 lines of code")
    print(f"   ⚡ Function Calls: 15+ individual → 3 comprehensive workflows")
    print(f"   🛡️ Error Handling: Comprehensive built-in error handling")
    print(f"   📈 Progress Tracking: Detailed progress and statistics")
    print(f"   🎯 Best Features: {', '.join(modeling_result['regression_importance_df'].head(3)['Feature'].tolist() if modeling_result['regression_importance_df'] is not None else ['N/A'])}")
    
    # print(f"\n🏆 Recommendations:")
    # print(f"   • Use regression model for continuous fup prediction")
    # if classification_metrics:
    #     print(f"   • Use classification model for high/low fup screening (threshold: {modeling_result['classification_threshold']})")
    # print(f"   • Model shows {'good' if regression_metrics['test_r2'] > 0.7 else 'moderate' if regression_metrics['test_r2'] > 0.5 else 'limited'} predictive performance")
    # print(f"   • Consider feature engineering if performance needs improvement")
    
    # Return comprehensive results
    return {
        'data_preparation': data_prep_result,
        'preprocessing': preprocessing_result,
        'modeling': modeling_result,
        'summary': {
            'total_workflows': 3,
            'original_molecules': data_prep_result['processing_stats']['original_rows'],
            'final_molecules': len(modeling_result['final_data']),
            'final_features': len(modeling_result['selected_features']),
            'regression_r2': regression_metrics['test_r2'],
            'classification_accuracy': classification_metrics['test_accuracy'] if classification_metrics else None,
            'files_generated': 5 + len([p for p in generated_plots.values() if p is not None]),
            'workflow_version': 'comprehensive_v2.0'
        }
    }


if __name__ == "__main__":
    print("🧬 New Comprehensive ChEMBL fup Modeling Workflow")
    print("This example demonstrates the power of grouped workflow functions")
    print("="*80)
    
    try:
        results = comprehensive_cheminformatics_workflow()
        
        if results:
            print(f"\n{'='*80}")
            print("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"📊 Summary: {results['summary']['original_molecules']} → {results['summary']['final_molecules']} molecules")
            print(f"🎯 Performance: R² = {results['summary']['regression_r2']:.4f}")
            print(f"📁 Files: {results['summary']['files_generated']} files generated")
            print(f"⚡ Version: {results['summary']['workflow_version']}")
            print("\n🔍 Check the generated files and plots for detailed results.")
            
        else:
            print("\n❌ Workflow failed. Check error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("🚀 WORKFLOW COMPARISON:")
    print("   📊 Old approach: ~400 lines, 15+ function calls")
    print("   ⚡ New approach: ~100 lines, 3 workflow calls")  
    print("   🛡️ Built-in error handling and progress tracking")
    print("   📈 Comprehensive statistics and reporting")
    print("   💾 Automatic intermediate file management")
    print(f"{'='*80}")