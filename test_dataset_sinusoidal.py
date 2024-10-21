# test_dataset_sinusoidal.py

import sys
import os

# Add the project root directory to sys.path to allow module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_provider.data_loader import Dataset_Sinusoidal

def main():
    # Initialize the dataset for training
    train_dataset = Dataset_Sinusoidal(
        root_path=os.path.join(current_dir, 'dataset', 'sinusoidal'),
        flag='train',
        size=[50, 10, 10],
        features='S',
        data_path='sinusoidal.csv',
        target='value',
        scale=True,
        timeenc=1,  # Use time_features
        freq='h',
        percent=100
    )
    
    print(f"Number of training samples: {len(train_dataset)}")
    
    # Fetch a sample
    try:
        sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
        print("Sample shapes:")
        print(f"seq_x: {sample_x.shape}")
        print(f"seq_y: {sample_y.shape}")
        if sample_x_mark is not None and sample_y_mark is not None:
            print(f"seq_x_mark: {sample_x_mark.shape}")
            print(f"seq_y_mark: {sample_y_mark.shape}")
        else:
            print("No time features used.")
    except Exception as e:
        print(f"An error occurred while fetching a sample: {e}")

if __name__ == "__main__":
    main()

