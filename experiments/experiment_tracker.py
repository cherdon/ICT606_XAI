"""
Utility for tracking experiment results in a CSV file.
"""
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


def save_experiment_results(
    experiment_name: str,
    results: Dict,
    output_dir: str,
    csv_filename: str = "all_experiments.csv"
) -> pd.DataFrame:
    """
    Save experiment results to a CSV file. Updates existing entry if experiment
    name exists, otherwise creates a new row.
    
    Parameters
    ----------
    experiment_name : str
        Unique name of the experiment
    results : dict
        Dictionary containing metrics like 'accuracy', 'f1', 'auc_roc', etc.
    output_dir : str
        Directory to save the CSV file
    csv_filename : str
        Name of the CSV file
        
    Returns
    -------
    pd.DataFrame
        Updated dataframe with all experiments
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Prepare the row data
    row_data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # Add all metrics from results
    for key, value in results.items():
        if isinstance(value, (int, float)) and value is not None:
            row_data[key] = round(value, 4) if isinstance(value, float) else value
    
    # Load existing CSV or create new dataframe
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Check if experiment already exists
        if experiment_name in df['experiment_name'].values:
            # Update existing row
            idx = df[df['experiment_name'] == experiment_name].index[0]
            for key, value in row_data.items():
                df.loc[idx, key] = value
        else:
            # Add new row
            new_row = pd.DataFrame([row_data])
            df = pd.concat([df, new_row], ignore_index=True)
    else:
        # Create new dataframe
        df = pd.DataFrame([row_data])
    
    # Reorder columns to have experiment_name and timestamp first
    cols = ['experiment_name', 'timestamp']
    other_cols = [c for c in df.columns if c not in cols]
    df = df[cols + sorted(other_cols)]
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"\nExperiment results saved to: {csv_path}")
    
    return df


def load_experiment_results(
    output_dir: str,
    csv_filename: str = "all_experiments.csv"
) -> Optional[pd.DataFrame]:
    """
    Load experiment results from CSV file.
    
    Returns None if file doesn't exist.
    """
    csv_path = os.path.join(output_dir, csv_filename)
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None


def print_experiment_summary(output_dir: str, csv_filename: str = "all_experiments.csv"):
    """
    Print a summary of all experiments.
    """
    df = load_experiment_results(output_dir, csv_filename)
    
    if df is None:
        print("No experiments recorded yet.")
        return
    
    print("\n" + "="*80)
    print("ALL EXPERIMENT RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
