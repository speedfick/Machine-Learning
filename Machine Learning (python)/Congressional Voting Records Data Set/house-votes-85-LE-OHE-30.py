# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


#-----------------------------------------------------------------------------------#

dataBase = pd.read_excel('house-votes-84.xlsx')
pd.DataFrame(dataBase)

#Nan & Null
nan_values = dataBase[dataBase.isna().any(axis=1)]
null_values = dataBase[dataBase.isnull().any(axis=1)]

#splitting Data: Descriptive and Target

target = dataBase.iloc[:,0].values
descriptive = dataBase.iloc[:,1:18].values

#-----------------------------------------------------------------------------------#
# Label Encoder 
count = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

labelEncoder = LabelEncoder()

for i in count:
    descriptive[:,i] = labelEncoder.fit_transform(descriptive[:,i])


#one-hot encoding
column_transformer = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])], remainder = 'passthrough')
descriptive = np.array(column_transformer.fit_transform(descriptive))

#----------------------------------------------------------------------------------#
#splits data for training and testing
descriptive_train, descriptive_test, target_train, target_test = train_test_split(descriptive, target, test_size = 0.30, random_state = 1)

#----------------------------------------------------------------------------------#
# call Naive Bayes Algorithm
classifierNB = GaussianNB()
classifierNB.fit(descriptive_train, target_train)

predictionNB = classifierNB.predict(descriptive_test)

# accuracy and Matrix
accuracyNB = accuracy_score(target_test, predictionNB)
matrixNB = confusion_matrix(target_test, predictionNB)

#----------------------------------------------------------------------------------#
# call Decision tree Algorithm
classifierDT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifierDT.fit(descriptive_train, target_train) 

predictionDT = classifierDT.predict(descriptive_test)

# accuracy and Matrix
accuracyDT = accuracy_score(target_test, predictionDT)
matrixDT = confusion_matrix(target_test, predictionDT)

#----------------------------------------------------------------------------------#
# call Random Forest Algorithm
classifierRF = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifierRF.fit(descriptive_train, target_train) 

predictionRF = classifierRF.predict(descriptive_test)

# accuracy and Matrix
accuracyRF = accuracy_score(target_test, predictionRF)
matrixRF = confusion_matrix(target_test, predictionRF)

#----------------------------------------------------------------------------------#
# call kNN Algorithm
classifierKnn = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
classifierKnn.fit(descriptive_train,target_train)

predictionKnn = classifierKnn.predict(descriptive_test)

# accuracy and Matrix
accuracyKnn = accuracy_score(target_test, predictionKnn)
matrixKnn = confusion_matrix(target_test, predictionKnn)

#----------------------------------------------------------------------------------#
