import numpy as np
import pickle
with open('./model/diabetes/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)

# If needed, load the column transformer and standard scaler
with open('./model/diabetes/column_transformer.pkl', 'rb') as file:
    ct = pickle.load(file)

with open('./model/diabetes/standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)
with open('./model/diabetes/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)
def diabetes_predict(gender,smoking_history,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level):

    new_data = np.array([[gender,smoking_history,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level]])
    print(new_data)
    
   

   
        
    new_data = np.array(ct.transform(new_data))
    print(new_data)
    new_data_scaled = sc.transform(new_data)
    # Predict the output using the trained classifier
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

# Print the predicted disease name
    return(predicted_label[0])

# gender="Female"
# smoking_history="never"
# age="65"
# hypertension="0"
# heart_disease="1"
# bmi="25"
# HbA1c_level="6.6"
# blood_glucose_level="180"

# print(diabetes_predict(gender,smoking_history,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level))
    