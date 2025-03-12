"""
Data generation stage: Creates synthetic HR data using parameters from params.yaml.
"""
import os
import sys
import yaml
import importlib.util
from datetime import datetime

# Add the project root to the path to import the data generator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the data generator dynamically
spec = importlib.util.spec_from_file_location(
    "data_generator", 
    os.path.join(os.path.dirname(__file__), "../../src/data/data_generator.py")
)
data_generator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_generator_module)
HRDataGenerator = data_generator_module.HRDataGenerator

def main():
    """Generate synthetic HR data based on parameters in params.yaml."""
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Get active attrition scenario
    active_scenario = next(
        (scenario for scenario_name, scenario in params['attrition_scenarios'].items() 
         if scenario['is_active']), 
        params['attrition_scenarios']['normal']
    )
    
    print(f"Generating data with {active_scenario['description']} scenario")
    print(f"Attrition rate: {active_scenario['rate']}")
    
    # Initialize generator
    generator = HRDataGenerator(seed=params['company']['random_seed'])
    
    # Modify the generator's attrition rate
    generator.desired_attrition_rate = active_scenario['rate']
    
    # Generate data
    data_dict = generator.generate_all_data(
        num_employees=params['company']['num_employees'],
        historical_years=params['company']['historical_years'],
    )
    
    # Create output directory if it doesn't exist
    os.makedirs("data/stages/raw", exist_ok=True)
    
    # Save datasets
    generator.save_datasets(data_dict, output_dir="data/stages/raw")
    
    # Save metadata about the generation
    with open("data/stages/raw/metadata.yaml", "w") as f:
        yaml.dump({
            "company": params['company'],
            "attrition_scenario": active_scenario,
            "generation_date": datetime.fromtimestamp(os.path.getmtime("params.yaml")),
        }, f)
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()
