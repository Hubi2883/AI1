import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_dataset(file_path, num_samples=10000):
    """
    Generates a synthetic dataset with modified sinusoidal features and a target variable.

    Parameters:
    - file_path (str): Path where the CSV will be saved.
    - num_samples (int): Number of data points (rows) to generate.

    The function creates a DataFrame with 'date', 'feature1', 'feature2', 'feature3', and 'target'.
    """
    # Generate a date range with hourly intervals
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_samples)]

    # Initialize the data dictionary
    data = {'date': dates}

    # Generate a base sine wave
    freq = 0.0005
    amplitude = 7
    phase = 0
    sine_wave = amplitude * np.sin(2 * np.pi * freq * np.arange(num_samples) + phase)

    # Create features based on the sine wave
    data['1'] = np.where(sine_wave > 0, 0, sine_wave)
    data['2'] = np.where(sine_wave <= 0, 0, sine_wave)
    data['3'] = np.zeros(num_samples)

    # Calculate the target as the sum of the features
    data['OT'] = data['1'] + data['2'] + data['3']

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(file_path, index=False)
    print(f"Dataset successfully saved to {file_path}")

if __name__ == "__main__":
    # Define the output file path
    output_file = "/ceph/home/student.aau.dk/fo55xa/AI1/dataset/electricity/synthetic_electricity4.csv"

    # Check if the file already exists to prevent overwriting
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Please remove it or choose a different name.")
    else:
        # Generate the dataset
        generate_synthetic_dataset(
            file_path=output_file,
            num_samples=10000
        )
