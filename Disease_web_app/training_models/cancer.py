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
    # Predict the output using the trained classifier
    predicted_output = classifier.predict(new_data_scaled)
    predicted_label = le.inverse_transform(predicted_output)

# Print the predicted disease name
    return(predicted_label[0])

# AirPollution=1
# Alcoholuse=1
# DustAllergy=1
# OccuPationalHazards=1
# GeneticRisk=1
# chronicLungDisease=1
# BalancedDiet=1
# Obesity=1
# Smoking=1
# PassiveSmoker=1
# ChestPain=1
# CoughingofBlood=1
# Fatigue=1
# WeightLoss=1
# ShortnessofBreath=1
# Wheezing=1
# SwallowingDifficulty=1
# ClubbingofFingerNails=1
# FrequentCold=1
# DryCough=1
# print(cancer_predict(AirPollution,Alcoholuse,DustAllergy,OccuPationalHazards,GeneticRisk,chronicLungDisease,BalancedDiet,Obesity,Smoking,PassiveSmoker,ChestPain,CoughingofBlood,Fatigue,WeightLoss,ShortnessofBreath,Wheezing,SwallowingDifficulty,ClubbingofFingerNails,FrequentCold,DryCough))