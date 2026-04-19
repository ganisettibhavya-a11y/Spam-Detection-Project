import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

def preprocess_url(url):
    """
    Performs basic preprocessing on the URL to extract features.
    
    Args:
        url (str): The URL string.

    Returns:
        str: The preprocessed URL string.
    """
    # Remove protocol (http/https) and www.
    url = re.sub(r'https?://(?:www\.)?', '', url)
    return url

def train_phishing_model(data_path):
    """
    Loads data, trains a Random Forest model, and saves it.

    Args:
        data_path (str): Path to the dataset CSV file.
    """
    try:
        # Load the dataset
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' was not found. Please run generate_dataset.py first.")
        return

    # Apply preprocessing
    df['url'] = df['url'].apply(preprocess_url)

    # Convert labels to numerical format
    # 0 for 'legitimate', 1 for 'phishing'
    df['label'] = df['label'].apply(lambda x: 1 if x == 'phishing' else 0)

    # Split data into training and testing sets
    X = df['url']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create a TF-IDF Vectorizer to convert text data to numerical features
    vectorizer = TfidfVectorizer(max_features=2000, lowercase=True, analyzer='char')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model and the vectorizer to a file
    joblib.dump(model, 'phishing_model.pkl')
    joblib.dump(vectorizer, 'phishing_vectorizer.pkl')
    print("\nModel and vectorizer saved as 'phishing_model.pkl' and 'phishing_vectorizer.pkl'.")

if __name__ == '__main__':
    train_phishing_model('phishing_dataset.csv')
