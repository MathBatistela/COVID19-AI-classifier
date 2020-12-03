#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

    
X_normal = np.load('data/features/fold-1_patches-1.npy')
X_covid = np.load('data/features/fold-2_patches-1.npy')
X = np.concatenate((X_normal,X_covid))

y_normal = np.full(len(X_normal),0)
y_covid = np.full(len(X_covid),1)
y = np.concatenate((y_normal,y_covid))

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

pca = PCA(n_components=64)
X_pca = pca.fit_transform(X_norm)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

#K-nn -------------------------------------------------------------------------------------------      
    
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

knn.fit(X_train, y_train)

knn_dict = {
    "classifier": knn,
    "predicted": knn.predict(X_test),
    "proba": knn.predict_proba(X_test)
}
    
#MLP -------------------------------------------------------------------------------------------    
        
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(256,128,64,32), random_state=1)

mlp.fit(X_train, y_train)

mlp_dict = {
    "classifier": mlp,
    "predicted": mlp.predict(X_test),
    "proba": mlp.predict_proba(X_test)
}

#Random Forest ---------------------------------------------------------------------------------   
        
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=1000, random_state=42)

rf.fit(X_train, y_train)

rf_dict = {
    "classifier": rf,
    "predicted": rf.predict(X_test),
    "proba": rf.predict_proba(X_test)
}

#SVM -------------------------------------------------------------------------------------------  
            
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

C_range = 2. ** np.arange(-5,15,2)
gamma_range = 2. ** np.arange(3,-15,-2)
k = [ 'rbf']

srv = svm.SVC(probability=True, kernel='rbf')
ss = StandardScaler()
pipeline = Pipeline([ ('scaler', ss), ('svm', srv) ])

param_grid = {
    'svm__C' : C_range,
    'svm__gamma' : gamma_range
}

grid = GridSearchCV(pipeline, param_grid, n_jobs=-1, verbose=False)
grid.fit(X_train, y_train)

model = grid.best_estimator_
    
svm_dict = {
    "classifier": model,
    "predicted": model.predict(X_test),
    "proba": model.predict_proba(X_test)
}

#Data dict ---------------------------------------------------------------------------------------

data = {
    "knn": knn_dict,
    "mlp": mlp_dict,
    "rf": rf_dict,
    "svm": svm_dict
}

#Combination -------------------------------------------------------------------------------------       

sum_1 = np.add(data["knn"]["proba"],data["mlp"]["proba"])

sum_2 = np.add(data["rf"]["proba"],data["svm"]["proba"])

total_sum = np.add(sum_1,sum_2)

predicted_combination = []
for obj_proba in total_sum:
    predicted_combination.append(np.argmax(obj_proba))        
predicted_combination = np.array(predicted_combination)
    
combination_dict = {
    "classifier": "Classifier's Combination",
    "predicted": predicted_combination
}

data['combination'] = combination_dict
    
#Printing Data -------------------------------------------------------------------------------------          
for key in data.keys():    
    print(f"----{data[key]['classifier']}----")
    print(f"\n - No rounded accuracy: {metrics.accuracy_score(y_test, data[key]['predicted'])}\n")
    print(data[key]['predicted'])
    print()
    print(metrics.classification_report(y_test, data[key]['predicted']))
    print(f"Confusion matrix:\n{metrics.confusion_matrix(y_test, data[key]['predicted'])}\n")