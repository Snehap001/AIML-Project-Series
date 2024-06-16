import numpy as np
import pickle
with open('./model/hepatitis/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)



with open('./model/hepatitis/standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)
with open('./model/hepatitis/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def hepatitis_predict(histology,steroid,malaise,anorexia,liver_big,spleen_palpable,spiders,ascites,varices,age,bilirubin,alk_phosphate,sgot,albumin,protime):

    new_data = np.array([[histology,steroid,malaise,anorexia,liver_big,spleen_palpable,spiders,ascites,varices,age,bilirubin,alk_phosphate,sgot,albumin,protime]])
  
    print(new_data)
   
  
    new_data_scaled = sc.transform(new_data)
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

    return(predicted_label[0])

