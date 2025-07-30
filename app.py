from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load model and accuracy
model = joblib.load("model/chd_model.pkl")
accuracy = 0.84  # Replace with actual value from training (example: 0.84 = 84%)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', accuracy=round(accuracy * 100, 2))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[key]) for key in request.form]
        df = pd.DataFrame([features], columns=[
            'male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
            'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
            'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
        ])
        prediction = model.predict(df)[0]
        result = "🔴 High risk of CHD in 10 years" if prediction == 1 else "🟢 Low risk of CHD in 10 years"
        return render_template('index.html', prediction_text=result, accuracy=round(accuracy * 100, 2))
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}", accuracy=round(accuracy * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
