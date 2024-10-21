# test_dataset_sinusoidal.py

from data_provider.data_loader import Dataset_Sinusoidal

def main():
    # Initialize the dataset for training
    train_dataset = Dataset_Sinusoidal(
        root_path='./dataset/sinusoidal/',
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
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    print("Sample shapes:")
    print(f"seq_x: {sample_x.shape}")
    print(f"seq_y: {sample_y.shape}")
    if sample_x_mark is not None and sample_y_mark is not None:
        print(f"seq_x_mark: {sample_x_mark.shape}")
        print(f"seq_y_mark: {sample_y_mark.shape}")
    else:
        print("No time features used.")

if __name__ == "__main__":
    main()

