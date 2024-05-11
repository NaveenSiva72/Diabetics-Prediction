from flask import Flask, render_template, request
import numpy as np
import joblib


app = Flask(__name__)

# Load the trained model
model = joblib.load('C:\\NAVEEN_S\\example_react\\MLT_project\\Sanjeevi\\HEALTH-CARE\\trained_model1.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        pregnancies = int(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
        age = int(request.form['age'])
        
        # Prepare input data for prediction
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert the predicted model
        result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

        return render_template('result.html', 
                               pregnancies=pregnancies,
                               glucose=glucose,
                               blood_pressure=blood_pressure,
                               skin_thickness=skin_thickness,
                               insulin=insulin,
                               bmi=bmi,
                               diabetes_pedigree_function=diabetes_pedigree_function,
                               age=age,
                               result=result)

if __name__ == '__main__':
    app.run()
