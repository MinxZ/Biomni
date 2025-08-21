"""
Multithreaded Molecular Descriptor Calculation Example

This example demonstrates the new multithreaded descriptor calculation function
with progress tracking using tqdm and configurable delays for rate limiting.

Features:
- Multithreaded processing for faster computation
- Progress bar with real-time success rate tracking
- Configurable delays for rate limiting
- Chunk-based processing for memory efficiency
- Error handling and retry logic
"""

import os
import time
from biomni.tool.preprocessing import load_and_inspect_data
from biomni.tool.modeling import calculate_molecular_descriptors


def demo_multithreaded_descriptors():
    """Demonstrate multithreaded descriptor calculation with various configurations."""
    
    print("🧪 Multithreaded Molecular Descriptor Calculation Demo")
    print("="*60)
    
    # Load sample data
    data_path = "data/data/20240610_chembl34_extraxt_fup_human_1c1d.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please ensure the ChEMBL dataset is available.")
        return
    
    # Load data
    data_info = load_and_inspect_data(data_path)
    if data_info is None:
        print("❌ Failed to load data")
        return
    
    data = data_info['data']
    print(f"✅ Loaded dataset: {data.shape}")
    
    # Take a smaller sample for demonstration (first 200 molecules)
    sample_size = min(200, len(data))
    data_sample = data.head(sample_size).copy()
    print(f"📊 Using sample of {sample_size} molecules for demonstration")
    
    # ============================================================================
    # Configuration 1: Standard multithreaded processing
    # ============================================================================
    print("\n📍 Configuration 1: Standard Multithreaded Processing")
    print("-" * 50)
    
    start_time = time.time()
    
    result1 = calculate_molecular_descriptors(
        data_sample,
        smiles_column='canonical_smiles',
        descriptor_prefix='standard_',
        n_jobs=4,                    # 4 threads
        delay_seconds=0.01,          # 10ms delay
        chunk_size=25                # Process 25 molecules per chunk
    )
    
    elapsed1 = time.time() - start_time
    print(f"⏱️  Processing time: {elapsed1:.2f} seconds")
    print(f"📊 Throughput: {sample_size/elapsed1:.1f} molecules/second")
    
    # ============================================================================
    # Configuration 2: High-speed processing (more threads, less delay)
    # ============================================================================
    print("\n📍 Configuration 2: High-Speed Processing")
    print("-" * 50)
    
    start_time = time.time()
    
    result2 = calculate_molecular_descriptors(
        data_sample,
        smiles_column='canonical_smiles', 
        descriptor_prefix='fast_',
        n_jobs=8,                    # 8 threads
        delay_seconds=0.002,         # 2ms delay
        chunk_size=50                # Larger chunks
    )
    
    elapsed2 = time.time() - start_time
    print(f"⏱️  Processing time: {elapsed2:.2f} seconds")
    print(f"📊 Throughput: {sample_size/elapsed2:.1f} molecules/second")
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 1
    print(f"🚀 Speedup: {speedup:.1f}x faster than standard configuration")
    
    # ============================================================================
    # Configuration 3: Conservative processing (rate-limited)
    # ============================================================================
    print("\n📍 Configuration 3: Conservative Processing (Rate-Limited)")
    print("-" * 50)
    
    # Use smaller sample for rate-limited demo
    small_sample = data.head(50).copy()
    
    start_time = time.time()
    
    result3 = calculate_molecular_descriptors(
        small_sample,
        smiles_column='canonical_smiles',
        descriptor_prefix='conservative_',
        n_jobs=2,                    # Fewer threads
        delay_seconds=0.05,          # 50ms delay (rate limiting)
        chunk_size=10                # Small chunks
    )
    
    elapsed3 = time.time() - start_time
    print(f"⏱️  Processing time: {elapsed3:.2f} seconds")
    print(f"📊 Throughput: {len(small_sample)/elapsed3:.1f} molecules/second")
    print(f"💡 This configuration is suitable for API rate limits or resource constraints")
    
    # ============================================================================
    # Results Summary
    # ============================================================================
    print("\n📊 PROCESSING SUMMARY")
    print("="*60)
    
    configurations = [
        ("Standard (4 threads, 10ms delay)", result1, elapsed1, sample_size),
        ("High-Speed (8 threads, 2ms delay)", result2, elapsed2, sample_size),
        ("Conservative (2 threads, 50ms delay)", result3, elapsed3, len(small_sample))
    ]
    
    for config_name, result, elapsed, n_molecules in configurations:
        throughput = n_molecules / elapsed if elapsed > 0 else 0
        descriptor_count = len([col for col in result.columns if col.startswith(('standard_', 'fast_', 'conservative_'))])
        
        print(f"\n🔧 {config_name}:")
        print(f"   ⏱️  Time: {elapsed:.2f}s")
        print(f"   📊 Throughput: {throughput:.1f} molecules/s")
        print(f"   🧪 Descriptors: {descriptor_count}")
        print(f"   📈 Dataset shape: {result.shape}")
    
    # Performance recommendations
    print(f"\n💡 PERFORMANCE RECOMMENDATIONS")
    print("="*60)
    print("🚀 For maximum speed:")
    print("   • Use n_jobs=8-16 (depending on CPU cores)")
    print("   • Set delay_seconds=0.001-0.005")
    print("   • Use chunk_size=100-200")
    print("   • Ensure sufficient RAM for large datasets")
    
    print("\n⚖️  For balanced performance:")
    print("   • Use n_jobs=4-6")
    print("   • Set delay_seconds=0.01")
    print("   • Use chunk_size=50-100")
    print("   • Good for most production scenarios")
    
    print("\n🛡️  For rate-limited/conservative processing:")
    print("   • Use n_jobs=1-2")
    print("   • Set delay_seconds=0.05-0.1")
    print("   • Use chunk_size=10-25")
    print("   • Suitable for API limits or shared resources")
    
    print("\n📦 REQUIRED PACKAGES")
    print("="*60)
    print("pip install rdkit mordred tqdm pandas scikit-learn")
    
    return configurations


def benchmark_descriptor_calculation():
    """Benchmark different threading configurations."""
    
    print("🏁 Descriptor Calculation Benchmark")
    print("="*60)
    
    # Test configurations
    configs = [
        {"n_jobs": 1, "delay": 0.01, "name": "Single Thread"},
        {"n_jobs": 2, "delay": 0.01, "name": "2 Threads"},
        {"n_jobs": 4, "delay": 0.01, "name": "4 Threads"},
        {"n_jobs": 6, "delay": 0.01, "name": "6 Threads"},
        {"n_jobs": 8, "delay": 0.01, "name": "8 Threads"},
    ]
    
    # Load small sample for benchmarking
    data_path = "data/data/20240610_chembl34_extraxt_fup_human_1c1d.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    
    data_info = load_and_inspect_data(data_path)
    sample_data = data_info['data'].head(100).copy()  # 100 molecules for benchmark
    
    results = []
    
    for config in configs:
        print(f"\n🔄 Testing {config['name']}...")
        
        start_time = time.time()
        
        try:
            result = calculate_molecular_descriptors(
                sample_data,
                smiles_column='canonical_smiles',
                descriptor_prefix=f"bench_{config['n_jobs']}t_",
                n_jobs=config['n_jobs'],
                delay_seconds=config['delay'],
                chunk_size=20,
                save_path=None  # Don't save during benchmark
            )
            
            elapsed = time.time() - start_time
            throughput = len(sample_data) / elapsed
            
            results.append({
                'config': config['name'],
                'threads': config['n_jobs'],
                'time': elapsed,
                'throughput': throughput,
                'descriptors': len([col for col in result.columns if col.startswith('bench_')])
            })
            
            print(f"   ✅ Completed in {elapsed:.2f}s ({throughput:.1f} mol/s)")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    # Display benchmark results
    print(f"\n📊 BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Configuration':<15} {'Threads':<8} {'Time (s)':<10} {'Throughput':<12} {'Speedup':<8}")
    print("-" * 60)
    
    baseline_time = results[0]['time'] if results else 1
    
    for result in results:
        speedup = baseline_time / result['time']
        print(f"{result['config']:<15} {result['threads']:<8} {result['time']:<10.2f} "
              f"{result['throughput']:<12.1f} {speedup:<8.1f}x")
    
    return results


if __name__ == "__main__":
    print("🧬 Multithreaded Molecular Descriptor Calculation")
    print("This example demonstrates the new multithreaded descriptor calculation features")
    print("="*80)
    
    try:
        # Run main demo
        print("\n🎯 Running main demonstration...")
        demo_configs = demo_multithreaded_descriptors()
        
        # Ask user if they want to run benchmark
        run_benchmark = input("\n🏁 Run threading benchmark? (y/N): ").strip().lower()
        
        if run_benchmark == 'y':
            print("\n🚀 Starting benchmark...")
            benchmark_results = benchmark_descriptor_calculation()
        
        print("\n✅ Demo completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Please install required packages:")
        print("pip install rdkit mordred tqdm pandas scikit-learn")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n🔧 Key Features Demonstrated:")
    print("   ✓ Multithreaded descriptor calculation")
    print("   ✓ Progress tracking with tqdm")
    print("   ✓ Configurable delays for rate limiting")
    print("   ✓ Chunk-based processing")
    print("   ✓ Performance benchmarking")
    print("   ✓ Error handling and resilience")
    print("   ✓ Real-time success rate monitoring")