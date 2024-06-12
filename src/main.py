import os

from data_preprocessing import load_data, preprocess_data
from data_splitting import split_data


if __name__ == "__main__":
    file_names = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]
    file_paths = [os.path.join("data", file_name) for file_name in file_names]

    # Load the data
    data = load_data(file_paths)
    
    # Preprocess the data
    data = preprocess_data(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Print shapes of the splits to verify
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
