import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset
df = pd.read_csv("Symptom2Disease.csv")

# Check actual column names
print("Columns:", df.columns)

# Remove unnamed column if exists
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Group symptoms and labels
df["text"] = df["text"].str.lower()
df["symptom_list"] = df["text"].str.split(",")
X_symptoms = df["symptom_list"]
y_disease = df["label"]

# Encode symptoms using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X_symptoms)

# Train the model
model = RandomForestClassifier()
model.fit(X_encoded, y_disease)

# Save model and encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(mlb, open("symptom_encoder.pkl", "wb"))
print("✅ Model and encoder saved successfully.")
