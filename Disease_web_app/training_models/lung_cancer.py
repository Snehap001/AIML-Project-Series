import numpy as np
import pickle
with open('./model/lung_cancer/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# If needed, load the column transformer and standard scaler
with open('./model/lung_cancer/column_transformer.pkl', 'rb') as file:
    ct = pickle.load(file)

with open('./model/lung_cancer/standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)
with open('./model/lung_cancer/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def lung_predict(GENDER,AGE,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY ,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN):

    new_data = np.array([[GENDER,AGE,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY ,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]])
  
    
    new_data = np.array(ct.transform(new_data))
  
    new_data_scaled = sc.transform(new_data)
    # Predict the output using the trained classifier
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

# Print the predicted disease name
    return(predicted_label[0])

