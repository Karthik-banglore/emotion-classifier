# Emotion Classifier

A simple web app to classify text into emotions (sadness, joy, love, anger, fear, surprise) using machine learning.

## Dataset Used
- [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion): A labeled dataset containing text samples and their corresponding emotions.

## Approach Summary
- Preprocessed text data (cleaning, lowercasing, removing punctuation, etc.).
- Used TF-IDF vectorization for feature extraction.
- Trained a Logistic Regression model to classify text into one of six emotions.
- Evaluated the model using accuracy, confusion matrix, and classification report.
- Built an interactive web app using Gradio for predictions.

## Dependencies
- scikit-learn
- pandas
- gradio
- datasets