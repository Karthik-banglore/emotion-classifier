import gradio as gr
import joblib
from train_classifier import vectorizer, clf, label_map
from data.preprocess import clean_text

def predict_emotion(text):
    text_clean = clean_text(text)
    X_vec = vectorizer.transform([text_clean])
    pred = clf.predict(X_vec)[0]
    return label_map[int(pred)]

iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs="text",
    title="Emotion Classifier",
    description="Enter a sentence and the model will predict the emotion."
)

if __name__ == "__main__":
    iface.launch(share=True)