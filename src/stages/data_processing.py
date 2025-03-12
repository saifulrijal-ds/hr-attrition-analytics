"""
Data preprocessing stage: Splits employee data into training and testing sets based on date.
"""
import os
import sys
import yaml
import pandas as pd
from datetime import datetime

def main():
    """
    Preprocess the raw employee data and split into training and testing sets.
    Primarily uses employees.csv but can be extended to incorporate other data sources.
    """
    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # Extract split parameters
    split_params = params['split']
    train_end_date = datetime.strptime(split_params['train_end_date'], "%Y-%m-%d")
    test_start_date = datetime.strptime(split_params['test_start_date'], "%Y-%m-%d")
    
    print(f"Loading raw data...")
    
    # Create output directory
    os.makedirs("data/stages/preprocessed", exist_ok=True)
    
    # Load employee data
    employees_df = pd.read_csv("data/stages/raw/employees.csv")
    
    # Convert date columns to datetime
    date_columns = ['HireDate']
    
    for col in date_columns:
        if col in employees_df.columns:
            employees_df[col] = pd.to_datetime(employees_df[col])
    
    # Add ExitDate if it exists
    if 'ExitDate' in employees_df.columns:
        employees_df['ExitDate'] = pd.to_datetime(employees_df['ExitDate'])
    else:
        # For employees who haven't left, ExitDate is None
        employees_df['ExitDate'] = None
    
    print(f"Total employee records: {len(employees_df)}")
    
    # Now implement date-based splitting with three approaches:
    
    # APPROACH 1: Split by HireDate - employees hired before/after cutoff date
    train_df_hire = employees_df[employees_df['HireDate'] <= train_end_date]
    test_df_hire = employees_df[employees_df['HireDate'] >= test_start_date]
    
    # APPROACH 2: Temporal split that looks at employee status as of cutoff date
    # Employees who left before train_end_date go to training
    # Employees who were active as of test_start_date go to test
    train_df_status = employees_df[
        (employees_df['Attrition'] == True) & 
        (employees_df['ExitDate'] <= train_end_date) | 
        (employees_df['HireDate'] <= train_end_date)
    ]
    
    test_df_status = employees_df[
        (employees_df['HireDate'] <= test_start_date) & 
        ((employees_df['Attrition'] == False) | 
         (employees_df['ExitDate'] >= test_start_date))
    ]
    
    # APPROACH 3: Use the approach specified in params.yaml
    split_method = split_params.get('method', 'status')
    
    if split_method == 'hire_date':
        train_df = train_df_hire
        test_df = test_df_hire
    elif split_method == 'status':
        train_df = train_df_status
        test_df = test_df_status
    else:
        # Default to status-based split
        train_df = train_df_status
        test_df = test_df_status
    
    # Remove duplicate employees that might appear in both sets
    if not split_params.get('allow_duplicates', False):
        # If an employee appears in test, remove from train
        train_df = train_df[~train_df['EmployeeID'].isin(test_df['EmployeeID'])]
    
    print(f"Split method: {split_method}")
    print(f"Training set size: {len(train_df)} employees")
    print(f"Testing set size: {len(test_df)} employees")
    
    # Add metadata columns for tracking
    train_df['DataSplit'] = 'train'
    test_df['DataSplit'] = 'test'
    
    # Apply any basic cleaning steps
    # (More extensive cleaning would be in a separate pipeline stage)
    for df in [train_df, test_df]:
        # Handle extreme values
        if 'MonthlyIncome' in df.columns:
            df['MonthlyIncome'] = df['MonthlyIncome'].clip(lower=0)
        
        # Convert boolean columns to integers for compatibility with more models
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)
    
    # Save the split datasets
    train_df.to_csv("data/stages/preprocessed/train_data.csv", index=False)
    test_df.to_csv("data/stages/preprocessed/test_data.csv", index=False)
    
    # Save metadata about the split
    with open("data/stages/preprocessed/split_metadata.yaml", "w") as f:
        yaml.dump({
            "split_method": split_method,
            "train_end_date": split_params['train_end_date'],
            "test_start_date": split_params['test_start_date'],
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_attrition_rate": train_df['Attrition'].mean(),
            "test_attrition_rate": test_df['Attrition'].mean()
        }, f)
    
    print("Data preprocessing and splitting complete!")

if __name__ == "__main__":
    main()