import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./static/csv_files/cancer patient data sets.csv')
dataset=dataset.dropna()
dataset = dataset.drop(['Snoring', 'Age','Gender'], axis=1)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', class_weight='balanced', random_state = 0)
classifier.fit(X_train, y)

import pickle

# Save the classifier
with open('./model/cancer/svm_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Optionally, save the column transformer and standard scaler


with open('./model/cancer/standard_scaler.pkl', 'wb') as file:
    pickle.dump(sc, file)

with open('./model/cancer/label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)