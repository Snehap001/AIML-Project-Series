import numpy as np
import pickle
with open('./model/heart/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('./model/heart/column_transformer.pkl', 'rb') as file:
    ct = pickle.load(file)

with open('./model/heart/standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)
with open('./model/heart/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def predict(sex,chestPain,restingECG,exerciseAngina,st_slope,age,cholesterol,fastingBP,maxHR,restingBP,oldPeak):

    new_data = np.array([[sex,chestPain,restingECG,exerciseAngina,st_slope,age,cholesterol,fastingBP,maxHR,restingBP,oldPeak]])
  
    
    new_data = np.array(ct.transform(new_data))
  
    new_data_scaled = sc.transform(new_data)
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

    return(predicted_label[0])

