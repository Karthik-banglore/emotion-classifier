import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Clean the input text by removing URLs, mentions, punctuation, numbers, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    return text

def preprocess_dataframe(df, text_column='text', label_column='label'):
    """Clean and preprocess the DataFrame, dropping rows with missing values."""
    df = df.copy()
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    df = df.dropna(subset=[text_column, label_column])
    return df

def load_and_preprocess(csv_path, text_column='text', label_column='label', test_size=0.2, random_state=42):
    """
    Load CSV, preprocess the text and labels, and split into train/test sets.
    """
    df = pd.read_csv(csv_path)
    df = preprocess_dataframe(df, text_column, label_column)
    X = df[text_column].values
    y = df[label_column].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    # Example usage
    csv_path = 'data/emotion_dataset.csv'  # Update with your actual CSV path
    X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)
    print("Training set size:", len(X_train))
    print("Test set size:", len(X_test))
    print("Sample training text:", X_train[0])
    print("Sample training label:", y_train[0])