from flask import Flask, render_template, request, redirect, url_for
from training_models.heart import *
from training_models.diabetes import *
from training_models.cancer import *
from training_models.hepatitis import *
from training_models.kidney import *
from training_models.lung_cancer import *

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
        result="Need a Checkup!"
        show_div=1
    else:
        result="You can relax!"
        show_div=2
    
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
    
    if disease:
        result="Need a Checkup!"
        show_div=1
    else:
        result="You can relax!"
        show_div=2
    return render_template('diabetes.html',disease=result,show_div=show_div)

@app.route('/cancer',methods=['POST'])
def cancer():
    return render_template('cancer.html')
@app.route('/submit_form_cancer',methods=['POST'])
def predict_cancer_disease():
    AirPollution=request.form["AirPollution"]
    Alcoholuse=request.form["Alcoholuse"]
    DustAllergy=request.form["DustAllergy"]
    OccuPationalHazards=request.form["OccuPationalHazards"]
    GeneticRisk=request.form["GeneticRisk"]
    chronicLungDisease=request.form["chronicLungDisease"]
    BalancedDiet=request.form["BalancedDiet"]
    Obesity=request.form["Obesity"]
    Smoking=request.form["Smoking"]
    PassiveSmoker=request.form["PassiveSmoker"]
    ChestPain=request.form["ChestPain"]
    CoughingofBlood=request.form["CoughingofBlood"]
    Fatigue=request.form["Fatigue"]
    WeightLoss=request.form["WeightLoss"]
    ShortnessofBreath=request.form["ShortnessofBreath"]
    Wheezing=request.form["Wheezing"]
    SwallowingDifficulty=request.form["SwallowingDifficulty"]
    ClubbingofFingerNails=request.form["ClubbingofFingerNails"]
    FrequentCold=request.form["FrequentCold"]
    DryCough=request.form["DryCough"]
    disease=cancer_predict(AirPollution,Alcoholuse,DustAllergy,OccuPationalHazards,GeneticRisk,chronicLungDisease,BalancedDiet,Obesity,Smoking,PassiveSmoker,ChestPain,CoughingofBlood,Fatigue,WeightLoss,ShortnessofBreath,Wheezing,SwallowingDifficulty,ClubbingofFingerNails,FrequentCold,DryCough)
    result=""
    if disease=="high" or disease=="medium":
        result="Need a Checkup!"
        show_div=1
    else:
        result="You can relax!"
        show_div=2
   
    return render_template('cancer.html',disease=result,show_div=show_div)

@app.route('/hepatitis',methods=['POST'])
def hepatitis():
    return render_template('hepatitis.html')
@app.route('/submit_form_hepatitis',methods=['POST'])
def predict_hepatitis_disease():
    histology=request.form["histology"]

    steroid=request.form["steroid"]

    malaise=request.form["malaise"]
    anorexia=request.form["anorexia"]
    liver_big=request.form["liver_big"]
    spleen_palpable=request.form["spleen_palpable"]
    spiders=request.form["spiders"]
    ascites=request.form["arcites"]
    varices=request.form["varices"]
    age=request.form["age"]
    bilirubin=request.form["bilirubin"]
    alk_phosphate=request.form["alk_phosphate"]
    sgot=request.form["sgot"]
    albumin=request.form["albumin"]
    protime=request.form["protime"]

    disease=hepatitis_predict(histology,steroid,malaise,anorexia,liver_big,spleen_palpable,spiders,ascites,varices,age,bilirubin,alk_phosphate,sgot,albumin,protime)
    result=""
    if disease=="live":
        result="Need a Checkup!"
        show_div=1
    else:
        result="You can relax!"
        show_div=2
    return render_template('hepatitis.html',disease=result,show_div=show_div)

@app.route('/kidney',methods=['POST'])
def kidney():
    return render_template('kidney.html')
@app.route('/submit_form_kidney',methods=['POST'])
def predict_kidney_disease():
    rbc=request.form["rbc"]
    pc=request.form["pc"]
    pcc=request.form["pcc"]	
    ba=request.form["ba"]
    htn=request.form["htn"]
    dm=request.form["dm"]
    cad=request.form["cad"]	
    appet=request.form["appet"]
    pe=request.form["pe"]	
    ane=request.form["ane"]
    age=request.form["age"]
    bp=request.form["bp"]
    sg=request.form["sg"]
    al=request.form["al"]
    su=request.form["su"]
    bgr=request.form["bgr"]
    bu=request.form["bu"]
    sc=request.form["sc"]
    sod=request.form["sod"]
    pot=request.form["pot"]
    hemo=request.form["hemo"]
    pcv=request.form["pcv"]

    wc=request.form["wc"]	
    rc=request.form["rc"]

    
    disease=kidney_predict(rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane,age,bp,sg,al,su,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc)
    result=""
    if disease=="ckd":
        result="Need a Checkup!"
        show_div=1
    else:
        result="You can relax!"
        show_div=2
    return render_template('kidney.html',disease=result,show_div=show_div)

@app.route('/lung_cancer',methods=['POST'])
def lung_cancer():
    return render_template('lung_cancer.html')
@app.route('/submit_form_lung_cancer',methods=['POST'])
def predict_lung_cancer_disease():
    GENDER = request.form['GENDER']
    AGE = request.form['AGE']
    YELLOW_FINGERS = request.form['YELLOW_FINGERS']
    ANXIETY = request.form['ANXIETY']
    PEER_PRESSURE=request.form['PEER_PRESSURE']
    CHRONIC_DISEASE=request.form['CHRONIC_DISEASE']
    FATIGUE = request.form['FATIGUE']
    ALLERGY = request.form['ALLERGY']
    WHEEZING = request.form['WHEEZING']
    ALCOHOL_CONSUMING = request.form['ALCOHOL_CONSUMING']
    COUGHING = request.form['COUGHING']
    SHORTNESS_OF_BREATH = request.form['SHORTNESS_OF_BREATH']
    SWALLOWING_DIFFICULTY = request.form['SWALLOWING_DIFFICULTY']
    CHEST_PAIN = request.form['CHEST_PAIN']
    disease=lung_predict(GENDER,AGE,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY ,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN)
    result=""
    if disease=="YES":
        result="Need a Checkup!"
        show_div=1
    else:
        result="You can relax!"
        show_div=2
    return render_template('lung_cancer.html',disease=result,show_div=show_div)
if __name__ == '__main__':
    app.run(debug=True)
