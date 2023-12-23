from flask import Flask, jsonify, request,  render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load(open('model_final.joblib', 'rb'))

@app.route("/")
def Home():
    return render_template("index.html", prediction_text="")

@app.route('/predict', methods=['POST'])

def predict():
   # try:
        age = float(request.form['age'])
        stress_level = float(request.form['stressLevel'])
        heart_rate = float(request.form['heartRate'])
        occupation = int(request.form['occupation'])
        bmi_category = float(request.form['bmiCategory'])
        diastolic = float(request.form['diastolic'])
        systolic = float(request.form['systolic'])

        if not (1 <= age <= 99):
            return render_template("index.html", prediction_text='Invalid age. Please enter a valid age between 1 and 99.')
       
        def calculate_blood_pressure_category(systolic, diastolic):
            if systolic < 120 and diastolic < 80:
                return 0
            elif systolic < 129 and diastolic < 80:
                return 1
            elif systolic < 130 or diastolic < 89:
                return 2
            elif systolic >= 140 or diastolic >= 90:
                return 3
            else:
                return 4
        blood_pressure=calculate_blood_pressure_category(systolic,diastolic)


        features = np.array([age, occupation, stress_level, bmi_category, heart_rate, blood_pressure]).reshape(1, -1)
        predictions = model.predict(features)

        return render_template("index.html", prediction_text='Prediction: {}'.format(predictions[0]))
   # except Exception as e:
    #    app.logger.error('An error occurred during prediction: %s', str(e))
     #   return render_template("index.html", prediction_text='An error occurred during prediction. Please try again.')

if __name__ == '__main__':
    app.run(debug=True)