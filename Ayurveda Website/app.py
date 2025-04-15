from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained AdaBoost model
with open('ada_model.pkl', 'rb') as model_file:
    ada_classifier = pickle.load(model_file)

# Define the column names as they appear in the training data
feature_columns = [
    'age', 'gender', 'bmi', 'waist', 'activity', 'urination', 'thirst', 'hunger',
    'weight_loss', 'vision', 'fatigue', 'healing', 'dry_mouth', 'tingling',
    'skin_infections', 'darkening', 'concentration', 'irritability',
    'erectile_disfunction', 'Delayed_Wound_Healing', 'sleep', 'belly_fat'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        bmi = float(request.form['bmi'])
        waist = float(request.form['waist'])
        activity = int(request.form['activity'])
        urination = int(request.form['urination'])
        thirst = int(request.form['thirst'])
        hunger = int(request.form['hunger'])
        weight_loss = int(request.form['weight_loss'])
        vision = int(request.form['vision'])
        fatigue = int(request.form['fatigue'])
        healing = int(request.form['healing'])
        dry_mouth = int(request.form['dry_mouth'])
        tingling = int(request.form['tingling'])
        skin_infections = int(request.form['skin_infections'])
        darkening = int(request.form['darkening'])
        concentration = int(request.form['concentration'])
        irritability = int(request.form['irritability'])
        erectile_dysfunction = int(request.form['erectile_disfunction'])
        delayed_wound_healing = int(request.form['Delayed_Wound_Healing'])
        sleep = int(request.form['sleep'])
        belly_fat = int(request.form['belly_fat'])

        # Create a DataFrame in the order of the training data
        input_data = pd.DataFrame([[age, gender, bmi, waist, activity, urination, thirst, hunger,
                                    weight_loss, vision, fatigue, healing, dry_mouth, tingling,
                                    skin_infections, darkening, concentration, irritability,
                                    erectile_dysfunction, delayed_wound_healing, sleep, belly_fat]],
                                  columns=feature_columns)

        # Predict using the loaded model
        prediction = ada_classifier.predict(input_data)
        prediction_proba = ada_classifier.predict_proba(input_data)

        # Extract the probability for the predicted class
        probability = prediction_proba[0][prediction[0]] * 100  # Convert to percentage

        # Process prediction result
        result_text = 'Positive for Pre-diabetes' if prediction[0] == 1 else 'Negative for Pre-diabetes'
        result = f"{result_text} with a probability of {probability:.2f}%"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
