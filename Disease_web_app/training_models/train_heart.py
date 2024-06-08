import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('./static/csv_files/heart.csv')
dataset=dataset.dropna()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3,4])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y)

import pickle

# Save the classifier
with open('./model/heart/svm_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)

# Optionally, save the column transformer and standard scaler
with open('./model/heart/column_transformer.pkl', 'wb') as file:
    pickle.dump(ct, file)

with open('./model/heart/standard_scaler.pkl', 'wb') as file:
    pickle.dump(sc, file)

with open('./model/heart/label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)