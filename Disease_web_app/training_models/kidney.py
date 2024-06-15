import numpy as np
import pickle
with open('./model/kidney/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# If needed, load the column transformer and standard scaler
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

    # Predict the output using the trained classifier
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

# Print the predicted disease name
    
    return(predicted_label[0])

# rbc="normal"
# pc="normal"
# pcc="notpresent"	
# ba="notpresent"
# htn="no"
# dm="yes"
# cad="no"	
# appet="poor"
# pe="no"	
# ane="yes"
# age=62	
# bp=80	
# sg=1.01	
# al=2	
# su=3	
# bgr=423	
# bu=53	
# sc=1.8	
# sod=111
# pot=2.5		
# hemo=9.6
# pcv=31

# wc=7500		
# rc=5.3

# disease=kidney_predict(rbc,pc,pcc,ba,htn,dm,cad,appet,pe,ane,age,bp,sg,al,su,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc)
# print(disease)