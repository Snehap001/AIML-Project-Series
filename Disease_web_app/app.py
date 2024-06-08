from flask import Flask, render_template, request, redirect, url_for
from training_models.heart import *
from training_models.diabetes import *
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():

    print("Predict Disease button was clicked.")
    return render_template('predict_disease.html')

@app.route('/symptom_analysis', methods=['POST'])
def symptom_analysis():
    # Here, you would add your code for symptom analysis
    # For now, let's just redirect to the homepage and print something on the console
    print("Symptom Analysis button was clicked.")
    return redirect(url_for('home'))
@app.route('/heart',methods=['POST'])
def heart():
    return render_template('heart.html')
@app.route('/submit_form_heart',methods=['POST'])
def predict_heart_disease():
    age = request.form['age']
    sex = request.form['sex']
    chestPain = request.form['chestPain']
    restingBP = request.form['RestingBP']
    fastingBP=request.form['fastingBP']
    restingECG=request.form['restingECG']
    cholesterol = request.form['cholesterol']
    maxHR = request.form['maxHR']
    oldPeak = request.form['OldPeak']
    exerciseAngina = request.form['exerciseAngina']
    st_slope = request.form['st_slope']
    disease=predict(sex,chestPain,restingECG,exerciseAngina,st_slope,age,cholesterol,fastingBP,maxHR,restingBP,oldPeak)
    result=""
    if disease:
        result="Based on your input you are likely to have a disease"
    else:
        result="Based on your input it is estimated that you DO NOT have the disease"
    show_div=True
    return render_template('heart.html',disease=result,show_div=show_div)

@app.route('/diabetes', methods=['POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route('/submit_diabetes',methods=['POST'])
def predict_diabetes():
    gender=request.form["gender"]
    smoking_history=request.form["smoking_history"]
    age=request.form["age"]
    hypertension=request.form["hypertension"]
    heart_disease=request.form["heart_disease"]
    bmi=request.form["bmi"]
    HbA1c_level=request.form["HbA1c_level"]
    blood_glucose_level=request.form["blood_glucose_level"]
    

    disease=diabetes_predict(gender,smoking_history,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level)
    show_div=True
    if disease:
        result="Based on your input you are likely to have a disease"
    else:
        result="Based on your input it is estimated that you DO NOT have the disease"
    return render_template('diabetes.html',disease=result,show_div=show_div)
if __name__ == '__main__':
    app.run(debug=True)
