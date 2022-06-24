# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


adult_data = pd.read_excel("adult.data.xlsx")
adult_data.describe()
columns = adult_data.columns

"Nan & Null"
nan_values = adult_data[adult_data.isna().any(axis=1)]
null_values = adult_data[adult_data.isnull().any(axis=1)]

"Target & Descriptive"
target = adult_data.iloc[:,-1]
descriptive = adult_data.iloc[:,0:-1].values


"Verificar valores negativos e range de valores altos "

"LabelEncoder - converte strings em numericos - https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html"
"             - tem o problema de haver muitos numeros iguais (0,1)"


count = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]

labelEncoder = LabelEncoder()

for i in count:
    descriptive[:,i] = labelEncoder.fit_transform(descriptive[:,i])


"One-Hot encoding"
column_transformer1 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
column_transformer2 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
column_transformer3 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [6])], remainder = 'passthrough')
column_transformer4 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [7])], remainder = 'passthrough')
column_transformer5 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [8])], remainder = 'passthrough')
column_transformer6 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [9])], remainder = 'passthrough')
column_transformer7 = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [13])], remainder = 'passthrough')
descriptive = np.array(column_transformer1.fit_transform(descriptive))
descriptive = np.array(column_transformer2.fit_transform(descriptive))
descriptive = np.array(column_transformer3.fit_transform(descriptive))
descriptive = np.array(column_transformer4.fit_transform(descriptive))
descriptive = np.array(column_transformer5.fit_transform(descriptive))
descriptive = np.array(column_transformer6.fit_transform(descriptive))
descriptive = np.array(column_transformer7.fit_transform(descriptive))

"train test split"

descriptive_train, descriptive_test, target_train, target_test = train_test_split(descriptive, target, test_size = 0.15, random_state = 0)


"getting a simple tree"

#from sklearn.tree import DecisionTreeClassifier, export

# classifier_Tree = DecisionTreeClassifier(criterion = 'entropy')
# classifier_Tree.fit(descriptive, target)    # modleo criado
# print(classifier_Tree.feature_importances_)
# prediction = classifier_Tree.predict([[2,2,1,1,2,2,1,1,2,2,1,1,2,1]])

# export.export_graphviz(classifier_Tree,
#                         out_file = 'decisionTree3.dot',
#                         feature_names = ['age', 'workclass', 'fnlwgt', 'education',
#                                          'education-num', 'marital-status', 'occupation', 'relationship',
#                                          'race', 'sex', 'capital-gain', 'capital-loss',
#                                          'hours-per-week', 'native-country'],
#                         class_names = ['No','Yes'],
#                         filled = True,
#                         leaves_parallel = True)


"Naive Bayes Algorithm"

classifier_Naive = GaussianNB()
classifier_Naive.fit(descriptive_train, target_train) # modleo criado

prediction_Naive = classifier_Naive.predict(descriptive_test)

accuracy_Naive = accuracy_score(target_test, prediction_Naive)
matrix_Naive = confusion_matrix(target_test, prediction_Naive)

"decision treen algorithm"

classifier_Tree2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_Tree2.fit(descriptive_train, target_train) 

predict_tree = classifier_Tree2.predict(descriptive_test)

accuracy_tree = accuracy_score(target_test, predict_tree)
matrix_tree = confusion_matrix(target_test, predict_tree)

"random forest algorithm"

# n_estimators => arvores criadas

classifier_Forest = RandomForestClassifier(n_estimators=10, criterion="entropy",random_state=0)
classifier_Forest.fit(descriptive_train, target_train)

predict_forest = classifier_Forest.predict(descriptive_test)

accuracy_forest = accuracy_score(target_test, predict_forest)
matrix_forest = confusion_matrix(target_test, predict_forest)


"KNN algorithm"

from sklearn.neighbors import KNeighborsClassifier

classifier_knn = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
classifier_knn.fit(descriptive_train, target_train)

predict_knn = classifier_knn.predict(descriptive_test)

accuracy_knn = accuracy_score(target_test, predict_knn)
matrix_knn = confusion_matrix(target_test, predict_knn)













