from datasets import load_dataset
import pandas as pd
dataset = load_dataset("dair-ai/emotion")
df = pd.DataFrame(dataset['train'])
df.to_csv("data/emotion_dataset.csv",index=False)
print("✅ Dataset saved at data/emotion_dataset.csv")