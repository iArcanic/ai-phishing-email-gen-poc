from sklearn.model_selection import train_test_split


def split_data(data):
    """Split data into training and testing sets."""
    
    # Define features and labels
    features = data[['cleaned_message', 'compound', 'neg', 'neu', 'pos', 'message_length']]
    labels = data['cleaned_message']  # Using cleaned messages as labels for synthetic generation purposes
    
    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
