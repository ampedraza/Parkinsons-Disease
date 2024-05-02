# -*- coding: utf-8 -*-
"""
Alison Pedraza
Project
Predicting Parkinson's Disease through Voice
KNN
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn import preprocessing




def plot_confus_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




# Bring in data
df = pd.read_csv('Parkinson disease.csv')
df_subset = df[["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)",  'MDVP:Flo(Hz)',
 'MDVP:Jitter(%)',
 'MDVP:Jitter(Abs)',
 'MDVP:RAP',
 'MDVP:PPQ',
 'Jitter:DDP',
 'MDVP:Shimmer',
 'MDVP:Shimmer(dB)',
 'Shimmer:APQ3',
 'Shimmer:APQ5',
 'MDVP:APQ',
 'Shimmer:DDA', 'NHR',
 'HNR',
 'status']].sample(frac=1)


# Get X values and turn into array
X = df_subset[["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)",  'MDVP:Flo(Hz)',
 'MDVP:Jitter(%)',
 'MDVP:Jitter(Abs)',
 'MDVP:RAP',
 'MDVP:PPQ',
 'Jitter:DDP',
 'MDVP:Shimmer',
 'MDVP:Shimmer(dB)',
 'Shimmer:APQ3',
 'Shimmer:APQ5',
 'MDVP:APQ',
 'Shimmer:DDA', 'NHR',
 'HNR',
 'status']].values


# Preprocession
    # check for NAs
na = df_subset.isna().sum()
print(na)
null = df_subset.isnull().sum()
print(null)
    # all data is numeric so no need to convert categorical variables
    
    # Scaling
X_scaled = (X.values - np.mean(X.values)) / np.std(X.values) 

    # Normalize the x values since there are Hz and %
#X = preprocessing.normalize(X)

#X = preprocessing.StandardScaler().fit(X).transform(X)

y = df_subset[['status']].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=4)
print('KNN Train set:', X_train.shape, y_train.shape)
print('KNN Test set:', X_test.shape, y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k=5
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train, y_train.ravel())
yhat = neigh.predict(X_test)

   #Accuracy Evaluation
from sklearn import metrics
from sklearn.metrics import f1_score

print("Knn Train set Accuracy:", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Knn Test Set Accuracy:", metrics.accuracy_score(y_test, yhat))
print('F1 Score:', f1_score(y_test, yhat, average='weighted') )

    # Confusion Matrix
conf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()
plot_confus_matrix(conf_matrix, classes=['parkinsons = 1','healthy = 0'],normalize= False,  title='Confusion matrix')


TP = conf_matrix[0][0]
FN = conf_matrix[1][0]
TN = conf_matrix[1][1]
FP = conf_matrix[0][1]

# Sensitivity, hit rate, recall, or true positive rate
TPR = (TP/(TP+FN))
print ('TPR', TPR)
# Specificity or true negative rate
TNR = (TN/(TN+FP))
print('TNR', TNR  )










