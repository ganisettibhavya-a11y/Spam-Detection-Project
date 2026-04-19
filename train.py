import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Generated Dataset
try:
    df = pd.read_csv('spam_detection_dataset.csv')
    print(f"Dataset loaded with {df.shape[0]} rows.")
except FileNotFoundError:
    print("Error: 'spam_detection_dataset.csv' not found. Run the generation script first.")
    exit()

# Ensure the columns and data types are correct
df['text_content'] = df['text_content'].astype(str)
df['label'] = df['label'].astype(int)

# 2. Split Data into Training and Testing Sets
X = df['text_content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training on {X_train.shape[0]} samples. Testing on {X_test.shape[0]} samples.")

# 3. Create a Scikit-learn Pipeline
# This automatically handles the sequence: Vectorization -> Training
model_pipeline = Pipeline([
    # Step 1: Feature Extraction using TF-IDF
    ('vectorizer', TfidfVectorizer(
        stop_words='english',      # Remove common English stop words
        lowercase=True,            # Convert text to lowercase
        max_features=5000          # Limit the vocabulary size
    )),
    
    # Step 2: Classifier (Multinomial Naive Bayes is excellent for text)
    ('classifier', MultinomialNB(alpha=1.0)) # alpha=1.0 is Laplace smoothing
])

# 4. Train the Model
print("\nStarting model training...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")

# 5. Evaluate the Model
y_pred = model_pipeline.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['HAM (0)', 'SPAM (1)']))
print("-" * 30)

# 6. Save the Trained Model Pipeline
model_filename = 'spam_detector_pipeline.pkl'
try:
    with open(model_filename, 'wb') as file:
        pickle.dump(model_pipeline, file)
    print(f"\nTrained model pipeline saved successfully as '{model_filename}'")
except Exception as e:
    print(f"Error saving model: {e}")