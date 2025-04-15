#adaboost.py:
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

import pickle
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('encoded_data.csv')
x = df.drop('Prediabetes Status', axis=1)
y = df['Prediabetes Status']

smote = SMOTE(random_state=42)
x_amplified, y_amplified = smote.fit_resample(x, y) 

X_train, X_test, y_train, y_test = train_test_split(x_amplified, y_amplified, test_size=0.2, random_state=42)

base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)  
ada_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=500, random_state=42)


ada_classifier.fit(X_train, y_train)
with open('ada_model.pkl', 'wb') as model_file:
    pickle.dump(ada_classifier, model_file)
