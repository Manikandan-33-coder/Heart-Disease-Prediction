# main.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class CHDPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None
        self.model = LogisticRegression(max_iter=1000)

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("✅ Data loaded")

    def clean_data(self):
        self.cleaned_df = self.df.dropna()
        print(f"✅ Cleaned data: {self.cleaned_df.shape[0]} rows remaining")

    def split_data(self):
        x = self.cleaned_df.iloc[:, :-1]
        y = self.cleaned_df.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42)
        print("✅ Data split into train and test sets")

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)
        print("✅ Model trained")

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("\n📊 Model Evaluation:")
        print("Accuracy: {:.2f}%".format(accuracy_score(self.y_test, y_pred) * 100))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def save_model(self, filename='chd_model.pkl'):
        joblib.dump(self.model, filename)
        print(f"✅ Model saved to {filename}")


if __name__ == "__main__":
    predictor = CHDPredictor("framingham.csv")  # <-- Make sure the CSV is in the same folder
    predictor.load_data()
    predictor.clean_data()
    predictor.split_data()
    predictor.train_model()
    predictor.evaluate_model()
    predictor.save_model()
