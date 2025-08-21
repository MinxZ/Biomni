"""
Example: Minimal Agent Configuration for Preprocessing Only

This example shows how to configure the Biomni agent to use only 
preprocessing tools without default data lake and packages.
"""

from biomni.agent.a1 import A1


def create_minimal_preprocessing_agent():
    """Create an agent configured only for preprocessing tasks."""
    
    # Initialize agent without downloading default data lake
    agent = A1(
        use_tool_retriever=True,
        download_data_lake=False, llm='azure-gpt-4o'
    )
    
    # Clear default data lake to avoid distractions
    agent.data_lake_dict = {}
    
    # Keep only essential packages for preprocessing
    essential_packages = {
        'pandas': 'Data manipulation and analysis library',
        'numpy': 'Numerical computing library', 
        'scikit-learn': 'Machine learning library with preprocessing tools',
        'scipy': 'Scientific computing library'
    }
    agent.library_content_dict = essential_packages
    
    # Filter module2api to keep only preprocessing tools
    # preprocessing_modules = {
    #     'biomni.tool.preprocessing': agent.module2api.get('biomni.tool.preprocessing', [])
    # }
    agent.module2api = {}
    
    # Reconfigure agent with minimal setup
    agent.configure()
    
    print("✅ Configured minimal agent with:")
    print(f"   📦 {len(essential_packages)} essential packages")
    # print(f"   🔧 {len(preprocessing_modules.get('biomni.tool.preprocessing', []))} preprocessing tools")
    print(f"   📊 {len(agent.data_lake_dict)} data lake items (empty)")
    
    return agent


def example_minimal_preprocessing():
    """Example using minimal agent configuration with real data."""
    
    print("🚀 Minimal Preprocessing Agent Example")
    print("="*50)
    
    # Create minimal agent
    agent = create_minimal_preprocessing_agent()
    
    # Add real data descriptions (these files exist in data/)
    agent.data_lake_dict = {
        'my_dataset.csv': 'Patient demographics and outcomes with missing values and outliers (510 rows)',
        'experiment_data.csv': 'Laboratory experiment results with mixed data types (300 rows)',
        'survey_responses.json': 'Nested survey responses in JSON format (200 responses)'
    }
    
    # Reconfigure after adding custom data
    agent.configure()
    
    log, result = agent.go("""
    I have real datasets in the data/data/ directory that need preprocessing:
    
    1. my_dataset.csv contains:
       - patient_id: unique identifier
       - age: numeric (some missing values, outliers around 120-150)
       - gender: categorical (M/F/Other)
       - diagnosis: categorical (Hypertension, Diabetes, Heart Disease, etc.)
       - treatment_response: numeric score (0-100, has outliers 150-200)
       - follow_up_months: numeric
       - bmi: numeric (some missing values)
       - blood_pressure_systolic: numeric
       - blood_pressure_diastolic: numeric
       - Contains ~15% missing values and duplicate rows
    
    Please:
    1. Load and inspect the data from 'data/data/my_dataset.csv'
    2. Detect and report all data quality issues
    3. Handle missing values using appropriate strategies for each column type
    4. Remove outliers from treatment_response and age using IQR method
    5. Encode categorical variables (gender, diagnosis) for machine learning
    6. Normalize all numeric features using standard scaling
    7. Generate a comprehensive preprocessing report
    8. Show before/after statistics
    
    Use only preprocessing tools - load the actual file and process real data.
    """)
    
    print("📋 Agent Response:")
    print(result)
    return log, result


def example_preprocessing_only_tools():
    """Example showing how to restrict to preprocessing tools only."""
    
    print("\n🔧 Preprocessing Tools Only Example")
    print("="*50)
    
    # Create agent with minimal configuration
    agent = A1(use_tool_retriever=True, download_data_lake=False, llm='azure-gpt-4o')
    
    # Complete reset - only preprocessing
    agent.data_lake_dict = {}
    agent.library_content_dict = {
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical operations',
        'scikit-learn': 'Preprocessing utilities'
    }
    
    # Keep ONLY preprocessing module (now available after adding to utils.py)
    if 'biomni.tool.preprocessing' in agent.module2api:
        preprocessing_tools = agent.module2api['biomni.tool.preprocessing']
        agent.module2api = {'biomni.tool.preprocessing': preprocessing_tools}
        print(f"✅ Found {len(preprocessing_tools)} preprocessing tools")
    else:
        print("⚠️ Preprocessing module not found - it should be available now")
        print("Available modules:", list(agent.module2api.keys()))
        return None, None
    
    agent.configure()
    
    log, result = agent.go("""
    Show me all available preprocessing functions and create a template workflow.
    
    I want to understand:
    1. What preprocessing functions are available to me
    2. How to load data from different file formats
    3. How to handle missing values with different strategies
    4. How to detect and remove outliers
    5. How to encode categorical variables
    6. How to normalize/scale numeric features
    7. How to assess data quality issues
    8. How to create an automated preprocessing pipeline
    
    Provide working code examples for each capability.
    Focus only on preprocessing - no other analysis tools.
    """)
    
    print("📋 Agent Response:")
    print(result)
    return log, result


def example_custom_data_preprocessing():
    """Example with custom data definitions."""
    
    print("\n📊 Custom Data Preprocessing Example")
    print("="*50)
    
    # Minimal agent setup
    agent = A1(use_tool_retriever=True, download_data_lake=False, llm='azure-gpt-4o')
    
    # Define your custom datasets
    custom_datasets = {
        'clinical_trial_data.csv': 'Clinical trial patient data with demographics, biomarkers, and outcomes',
        'gene_expression.xlsx': 'RNA-seq gene expression data with sample metadata',
        'patient_surveys.json': 'Patient-reported outcome measures in structured format',
        'lab_results.parquet': 'Laboratory test results optimized for large-scale analysis'
    }
    
    # Minimal package set
    minimal_packages = {
        'pandas': 'Data manipulation and file I/O',
        'numpy': 'Numerical computations',
        'scikit-learn': 'Preprocessing and data transformation'
    }
    
    # Configure agent
    agent.data_lake_dict = custom_datasets
    agent.library_content_dict = minimal_packages
    agent.module2api = {
        'biomni.tool.preprocessing': agent.module2api.get('biomni.tool.preprocessing', [])
    }
    agent.configure()
    
    log, result = agent.go("""
    Design a flexible preprocessing system for biomedical data analysis.
    
    Requirements:
    1. Handle multiple file formats (CSV, Excel, JSON, Parquet)
    2. Detect and report data quality issues automatically
    3. Apply appropriate preprocessing based on data characteristics:
       - Missing value imputation strategies by column type
       - Outlier detection and removal for continuous variables
       - Categorical encoding for machine learning compatibility
       - Feature scaling for algorithm requirements
    
    4. Create a reusable pipeline that can process any of my datasets
    5. Include validation and error handling
    6. Generate comprehensive reports for documentation
    
    Provide a complete, production-ready preprocessing framework.
    """)
    
    print("📋 Agent Response:")
    print(result)
    return log, result


def example_step_by_step_preprocessing():
    """Example showing step-by-step preprocessing control."""
    
    print("\n📝 Step-by-Step Preprocessing Example")
    print("="*50)
    
    # Minimal setup
    agent = A1(use_tool_retriever=True, download_data_lake=False, llm='azure-gpt-4o')
    agent.data_lake_dict = {}
    agent.library_content_dict = {'pandas': 'Data analysis', 'scikit-learn': 'Preprocessing'}
    agent.module2api = {'biomni.tool.preprocessing': agent.module2api.get('biomni.tool.preprocessing', [])}
    agent.configure()
    
    log, result = agent.go("""
    Walk me through each preprocessing function step-by-step:
    
    1. load_and_inspect_data: Show how to load different file types and interpret inspection results
    2. detect_data_quality_issues: Demonstrate quality assessment with custom thresholds
    3. clean_missing_values: Compare different missing value strategies (auto, mean, median, KNN)
    4. remove_outliers: Show IQR, Z-score, and modified Z-score methods
    5. encode_categorical_variables: Compare label encoding vs one-hot encoding
    6. normalize_data: Compare standard, minmax, and robust scaling
    7. generate_preprocessing_report: Create comprehensive documentation
    8. create_data_preprocessing_pipeline: Combine everything into automated workflow
    
    For each function, provide:
    - Function signature and parameters
    - Example usage with realistic data scenarios
    - When to use each option/strategy
    - How to interpret the results
    
    Create a comprehensive preprocessing guide.
    """)
    
    print("📋 Agent Response:")
    print(result)
    return log, result


if __name__ == "__main__":
    print("🧬 Minimal Biomni Agent - Preprocessing Only")
    print("="*60)
    
    # Run examples
    examples = [
        ("Minimal Preprocessing Setup", example_minimal_preprocessing),
        # ("Preprocessing Tools Only", example_preprocessing_only_tools),
        # ("Custom Data Preprocessing", example_custom_data_preprocessing),
        # ("Step-by-Step Guide", example_step_by_step_preprocessing),
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n🔄 Running: {name}")
            log, result = example_func()
            if log and result:
                results[name] = {"status": "success"}
                print(f"✅ Completed: {name}")
            else:
                results[name] = {"status": "skipped"}
                print(f"⏭️ Skipped: {name}")
        except Exception as e:
            print(f"❌ Error in {name}: {str(e)}")
            results[name] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("📊 EXECUTION SUMMARY")
    print("="*60)
    
    for name, result in results.items():
        status_icons = {"success": "✅", "failed": "❌", "skipped": "⏭️"}
        icon = status_icons.get(result["status"], "❓")
        print(f"{icon} {name}")
        if "error" in result:
            print(f"   └─ Error: {result['error']}")
    
    print("\n🔧 Configuration Tips:")
    print("   • Set download_data_lake=False to avoid default datasets")
    print("   • Clear agent.data_lake_dict = {} to remove default data")
    print("   • Filter agent.module2api to keep only preprocessing tools")
    print("   • Limit agent.library_content_dict to essential packages")
    print("   • Call agent.configure() after making changes")
    
    print("\n💡 Usage Pattern:")
    print("   ```python")
    print("   agent = A1(download_data_lake=False, llm='azure-gpt-4o')")
    print("   agent.data_lake_dict = {}  # Clear defaults")
    print("   agent.library_content_dict = {'pandas': 'Data analysis'}")
    print("   agent.module2api = {'biomni.tool.preprocessing': tools}")
    print("   agent.configure()  # Apply changes")
    print("   ```")