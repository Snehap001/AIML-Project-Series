import numpy as np
import pickle
with open('./model/kidney/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('./model/kidney/column_transformer.pkl', 'rb') as file:
    ct = pickle.load(file)

with open('./model/kidney/standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
with open('./model/kidney/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def kidney_predict(rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane,age,bp,sg,al,su,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc):

    new_data = np.array([[rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane,age,bp,sg,al,su,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc]])
  
    
    new_data = np.array(ct.transform(new_data))

   
    new_data_scaled = scaler.transform(new_data)

    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

    
    return(predicted_label[0])

