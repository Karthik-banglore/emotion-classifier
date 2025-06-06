# Emotion Classifier

A simple web app to classify text into emotions (sadness, joy, love, anger, fear, surprise) using machine learning.

[![Hugging Face Spaces](https://img.shields.io/badge/Gradio-Demo-blue?logo=gradio)](https://huggingface.co/spaces/Karthikbt/Emotion_Classification_Model)

## 🚀 Try the App

👉 [Click here to use the Emotion Classifier](https://huggingface.co/spaces/Karthikbt/Emotion_Classification_Model)

## Features

- Enter any sentence and get the predicted emotion.
- Built with scikit-learn, pandas, and Gradio.
- Trained on the dair-ai/emotion dataset.

## How it works

1. Enter your text in the input box.
2. Click "Submit".
3. See the predicted emotion instantly!

## 📊 Model Performance

**Accuracy:** 0.85

**Confusion Matrix:**

| sadness | joy  | love | anger | fear | surprise |
|---------|------|------|-------|------|----------|
|   859   |  56  |  1   |   8   |  9   |    0     |
|   21    | 1021 | 23   |   5   |  1   |    1     |
|   19    |  91  | 145  |   3   |  3   |    0     |
|   42    |  37  |  0   | 342   | 11   |    0     |
|   31    |  41  |  0   |  17   | 294  |    4     |
|   13    |  26  |  1   |   1   | 23   |   51     |

**Classification Report:**

```
              precision    recall  f1-score   support

      sadness       0.87      0.92      0.90       933
          joy       0.80      0.95      0.87      1072
         love       0.85      0.56      0.67       261
        anger       0.91      0.79      0.85       432
         fear       0.86      0.76      0.81       387
     surprise       0.91      0.44      0.60       115
    accuracy                            0.85      3200
    macro avg       0.87      0.74      0.78      3200
 weighted avg       0.85      0.85      0.84      3200
```


---

<!--
title: Emotion_Classification_Model
app_file: app.py
sdk: gradio
sdk_version: 4.44.1
-->
