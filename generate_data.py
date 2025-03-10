"""
Data Generator Script for BFI Finance Indonesia HR Analytics

This script generates synthetic HR data for BFI Finance Indonesia
and saves it as CSV files for the HR analytics project.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import data generator
from data.data_generator import HRDataGenerator

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/models', exist_ok=True)

def generate_data(num_employees=1000, historical_years=3, seed=42):
    """
    Generate all datasets and save them to CSV files.
    
    Args:
        num_employees (int): Number of current employees to generate
        historical_years (int): Years of historical data to generate
        seed (int): Random seed for reproducibility
    """
    print(f"\n{'='*60}")
    print(f"Generating HR data for BFI Finance Indonesia")
    print(f"Number of current employees: {num_employees}")
    print(f"Historical years: {historical_years}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")

    # Initialize generator
    print("Initializing data generator...")
    generator = HRDataGenerator(seed=seed)

    # Generate all dataset
    print("\nGenerating datasets:")
    data_dict = generator.generate_all_data(num_employees, historical_years)

    # Save datasets
    print("\nSaving datasets to CSV files:")
    output_dir = "data/raw"
    generator.save_datasets(data_dict, output_dir)

    # Print summary
    print("\nData generation complete")
    print("\nDataset summary:")
    for name, df in data_dict.items():
        print(f"  - {name}: {len(df)} records, {len(df.columns)} columns")

    print(f"\nData files saved to: {os.path.abspath(output_dir)}")
    print(f"{'='*60}\n")

    return data_dict

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate synthetic HR data for BFI Finance Indonesia')
    parser.add_argument('--employees', type=int, default=1000, help='Number of current employees to generate')
    parser.add_argument('--years', type=int, default=3, help='Years of historical data to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Generate data
    generate_data(args.employees, args.years, args.seed)