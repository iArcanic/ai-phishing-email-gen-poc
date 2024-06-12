import pandas as pd
import re


def load_data(file_paths):
    """Load multiple CSV data into a single DataFrame."""
    data_frames = [pd.read_csv(file_path) for file_path in file_paths]
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data


def preprocess_text(text):
    """Preprocess text data."""
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"\'\n", " ", text)
    return text


def preprocess_data(data):
    """Preprocess the entire dataset."""

    # Fill missing values with an empty string
    data = data.fillna("")

    # Remove duplicates
    data = data.drop_duplicates()

    # Apply the preprocessing function to the message column
    data['cleaned_message'] = data['message'].apply(preprocess_text)
    
    # Create a new feature for message length
    data['message_length'] = data['cleaned_message'].apply(len) 
    return data
