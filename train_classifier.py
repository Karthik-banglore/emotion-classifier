import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data.preprocess import load_and_preprocess

# 1. Load and preprocess data
csv_path = 'data/emotion_dataset.csv'
X_train, X_test, y_train, y_test = load_and_preprocess(csv_path)

# 2. Convert text to features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 3. Train a classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)

# 4. Predict and evaluate
y_pred = clf.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Print a detailed classification report
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
print("Classification Report:\n", classification_report(
    y_test, y_pred, target_names=[label_map[i] for i in range(6)]
))

