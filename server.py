# This line of code without database
from flask import Flask, render_template, request
import pickle
import numpy as np
import spacy

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the model and scaler for prediction
with open('health_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define expert system rules
def expert_system_rules(ml_prediction, user_input):
    age, gender, body_temperature, pulse_rate, respiration_rate, blood_pressure, blood_oxygen, weight, blood_glucose, diet_quality = user_input[0]

    if age > 60 and blood_pressure > 140:
        return "Unhealthy"
    if body_temperature > 38 and respiration_rate > 20:
        return "Unhealthy"
    if blood_glucose > 150 and diet_quality == 2:
        return "Unhealthy"
    if pulse_rate < 60 and blood_oxygen < 90:
        return "Unhealthy"
    if weight < 50 and blood_pressure < 90:
        return "Borderline"
    return ["Healthy", "Borderline", "Unhealthy"][ml_prediction]

# NLP Code
def extract_number(text):
    doc = nlp(text)
    for token in doc:
        if token.like_num:
            return float(token.text)
    raise ValueError("No numerical value found in the input.")

def extract_gender(text):
    doc = nlp(text)
    for token in doc:
        if token.text.lower() == "male":
            return 0
        elif token.text.lower() == "female":
            return 1
    raise ValueError("Gender not specified")

def extract_diet_quality(text):
    doc = nlp(text)
    for token in doc:
        if token.text.lower() == "good":
            return 0
        elif token.text.lower() == "fair":
            return 1
        elif token.text.lower() == "poor":
            return 2
    raise ValueError("Diet quality not specified")

def predict_health(user_input):
    user_input_scaled = scaler.transform(user_input)
    ml_prediction = model.predict(user_input_scaled)[0]
    return expert_system_rules(ml_prediction, user_input)

# Recommendation system
def recommend_actions(health_status):
    if health_status == "Unhealthy":
        return "Please consult a doctor immediately. Also, consider the following actions: - Exercise regularly - Eat a balanced diet - Get enough sleep"
    elif health_status == "Borderline":
        return "Please consider the following actions: - Reduce stress - Increase physical activity - Monitor your health parameters regularly"
    else:
        return "Keep up the good work! Consider the following actions: - Continue exercising regularly - Maintain a healthy diet - Get regular check-ups"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        gender = extract_gender(request.form['gender'])
        body_temperature = float(request.form['body_temperature'])
        pulse_rate = int(request.form['pulse_rate'])
        respiration_rate = int(request.form['respiration_rate'])
        blood_pressure = float(request.form['blood_pressure'])
        blood_oxygen = int(request.form['blood_oxygen'])
        weight = float(request.form['weight'])
        blood_glucose = int(request.form['blood_glucose'])
        diet_quality = extract_diet_quality(request.form['diet_quality'])

        user_input = np.array([[age, gender, body_temperature, pulse_rate, respiration_rate, blood_pressure, 
                                blood_oxygen, weight, blood_glucose, diet_quality]])

        result = predict_health(user_input)
        recommendation = recommend_actions(result)
        
        return render_template('result.html', prediction=result, recommendation=recommendation)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='locahost', port=5000)
