import numpy as np
import pickle
with open('./model/cancer/svm_classifier.pkl', 'rb') as file:
    classifier = pickle.load(file)


with open('./model/cancer/standard_scaler.pkl', 'rb') as file:
    sc = pickle.load(file)
with open('./model/cancer/label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def cancer_predict(AirPollution,Alcoholuse,DustAllergy,OccuPationalHazards,GeneticRisk,chronicLungDisease,BalancedDiet,Obesity,Smoking,PassiveSmoker,ChestPain,CoughingofBlood,Fatigue,WeightLoss,ShortnessofBreath,Wheezing,SwallowingDifficulty,ClubbingofFingerNails,FrequentCold,DryCough):

    new_data = np.array([[AirPollution,Alcoholuse,DustAllergy,OccuPationalHazards,GeneticRisk,chronicLungDisease,BalancedDiet,Obesity,Smoking,PassiveSmoker,ChestPain,CoughingofBlood,Fatigue,WeightLoss,ShortnessofBreath,Wheezing,SwallowingDifficulty,ClubbingofFingerNails,FrequentCold,DryCough]])
  
    
    
    print(new_data)
    new_data_scaled = sc.transform(new_data)
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

    return(predicted_label[0])

