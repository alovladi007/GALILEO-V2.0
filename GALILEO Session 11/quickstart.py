#!/usr/bin/env python3
"""
Quick Start Script
==================
Sets up the benchmarking suite and runs a quick validation.
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_command(cmd, description):
    """Run command and report status."""
    print(f"➤ {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed: {e}")
        if e.stdout:
            print(f"  Output: {e.stdout}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        return False


def main():
    """Main quick start procedure."""
    print("\n" + "🔬" * 35)
    print("GEOPHYSICS BENCHMARKING SUITE - QUICK START")
    print("🔬" * 35)
    
    # Step 1: Check Python version
    print_header("Step 1: Checking Python Version")
    py_version = sys.version_info
    print(f"  Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        print("  ❌ Python 3.9 or higher required!")
        return False
    else:
        print("  ✅ Python version OK")
    
    # Step 2: Install dependencies
    print_header("Step 2: Installing Dependencies")
    if not run_command("pip install -r requirements.txt", "Installing packages"):
        print("  ⚠️  Some dependencies may have failed to install")
    
    # Step 3: Generate datasets
    print_header("Step 3: Generating Test Datasets")
    if not run_command(
        'python -c "from bench.datasets import create_sample_datasets; create_sample_datasets()"',
        "Generating regression datasets"
    ):
        print("  ❌ Dataset generation failed")
        return False
    
    # Step 4: Run quick test
    print_header("Step 4: Running Quick Validation Test")
    if not run_command("pytest tests/test_bench.py::TestRegressionDatasets::test_dataset_initialization -v", 
                      "Running validation test"):
        print("  ⚠️  Test failed, but setup may still work")
    
    # Step 5: Run example
    print_header("Step 5: Running Example Benchmark")
    if not run_command("python bench.py --suite spatial", "Running spatial benchmark suite"):
        print("  ⚠️  Benchmark failed")
    
    # Success!
    print_header("🎉 Quick Start Complete!")
    
    print("Next steps:")
    print("\n  📚 Read the documentation:")
    print("     cat docs/verification.md")
    
    print("\n  🔬 Run all benchmarks:")
    print("     python bench.py --suite all")
    
    print("\n  📊 Generate reports:")
    print("     python bench.py --suite all --report html")
    
    print("\n  🧪 Run tests:")
    print("     pytest tests/ -v")
    
    print("\n  📈 Check coverage:")
    print("     python bench.py --coverage")
    
    print("\n  💡 See examples:")
    print("     python examples/example_usage.py")
    
    print("\n" + "=" * 70)
    print("Setup complete! Happy benchmarking! 🚀")
    print("=" * 70 + "\n")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
