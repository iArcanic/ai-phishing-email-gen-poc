import os

from data_preprocessing import load_data, preprocess_data
from data_splitting import split_data
from train_gpt2 import train_gpt2_model
from generate_text import generate_text


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
    X_train, X_test, Y_train, Y_test = split_data(data)

    # Train the generative model
    train_gpt2_model(X_train, X_test)

    # Generate synthetic data
    prompt = "Dear customer, we have noticed unusual activity in your bank account."
    generated_texts = generate_text(prompt)
    
    for i, text in enumerate(generated_texts):
        print(f"Generated text {i+1}:\n{text}\n")

