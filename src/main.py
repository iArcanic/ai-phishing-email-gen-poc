import os

from data_preprocessing import load_data, preprocess_data
from data_splitting import split_data
from train_gpt2 import train_gpt2_model


if __name__ == "__main__":
    file_names = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]
    file_paths = [os.path.join("data", file_name) for file_name in file_names]

    # Load the data
    print("Loading the data...")
    data = load_data(file_paths)
    
    # Preprocess the data
    print("Preprocessing the data...")
    data = preprocess_data(data)
    
    # Split the data
    print("Splitting datasets...")
    X_train, X_test, y_train, y_test = split_data(data)

    # Train the generative model
    train_gpt2_model(X_train)
