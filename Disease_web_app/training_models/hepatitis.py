import numpy as np
import pickle
with open('./model/hepatitis/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# If needed, load the column transformer and standard scaler


with open('./model/hepatitis/standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)
with open('./model/hepatitis/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def hepatitis_predict(histology,steroid,malaise,anorexia,liver_big,spleen_palpable,spiders,ascites,varices,age,bilirubin,alk_phosphate,sgot,albumin,protime):

    new_data = np.array([[histology,steroid,malaise,anorexia,liver_big,spleen_palpable,spiders,ascites,varices,age,bilirubin,alk_phosphate,sgot,albumin,protime]])
  
    print(new_data)
   
  
    new_data_scaled = sc.transform(new_data)
    # Predict the output using the trained classifier
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

# Print the predicted disease name
    return(predicted_label[0])

# histology=0

# steroid=0

# malaise=0
# anorexia=0
# liver_big=0
# spleen_palpable=0
# spiders=0
# ascites=0
# varices=0
# age=30
# bilirubin=1
# alk_phosphate=86
# sgot=18
# albumin=3.5
# protime=90

# print(hepatitis_predict(histology,steroid,malaise,anorexia,liver_big,spleen_palpable,spiders,ascites,varices,age,bilirubin,alk_phosphate,sgot,albumin,protime))