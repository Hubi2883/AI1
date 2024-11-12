import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_dataset(file_path, num_features=3, num_samples=10000):
    """
    Generates a synthetic dataset with sinusoidal features and a target variable.

    Parameters:
    - file_path (str): Path where the CSV will be saved.
    - num_features (int): Number of feature columns to generate.
    - num_samples (int): Number of data points (rows) to generate.

    The function creates a DataFrame with 'date', 'feature1' to 'featureN', and 'target'.
    """
    # Generate a date range with hourly intervals
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_samples)]

    # Initialize the data dictionary
    data = {'date': dates}

    # Generate sinusoidal features
    for i in range(1, num_features + 1):
        # Random frequency between 0.01 and 0.05
        freq = 0.003#np.random.uniform(0.01, 0.05)
        # Random amplitude between 0.5 and 1.5
        amplitude = 1#np.random.uniform(0.5, 2)
        # Random phase between 0 and 2π
        phase = 0#np.random.uniform(0, 2 * np.pi)
        data[f'feature{i}'] = amplitude * np.sin(2 * np.pi * freq * np.arange(num_samples) + phase)

    # Generate the target as a combination of features
    data['target'] = data['feature1'] + 0.5 * data['feature2'] - 0.3 * data['feature3']

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Dataset successfully saved to {file_path}")

if __name__ == "__main__":
    # Define the output file path
    output_file = "synthetic_electricity.csv"

    # Check if the file already exists to prevent overwriting
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Please remove it or choose a different name.")
    else:
        # Generate the dataset
        generate_synthetic_dataset(
            file_path=output_file,
            num_features=3,
            num_samples=10000
        )
