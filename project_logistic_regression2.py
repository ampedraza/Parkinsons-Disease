

# -*- coding: utf-8 -*-
"""
Alison Pedraza
Project
Logistic Regression

"""


import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler



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

# Preprocession
    # check for NAs
na = df_subset.isna().sum()
print(na)
null = df_subset.isnull().sum()
print(null)

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
 'HNR']].values


#X = preprocessing.StandardScaler().fit(X).transform(X)
X = StandardScaler().fit_transform(X)

y = df_subset[['status']].values



# Train/Test dataset
from sklearn.model_selection import train_test_split
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X,y,test_size = 0.2, random_state=4)

print('Train set:', X_train_log.shape, y_train_log.shape)
print('Test set: ', X_test_log.shape, y_test_log.shape)



# MODELING LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C = 0.01, solver = 'liblinear').fit(X_train_log, y_train_log)
LR      # THE MODEL

# PREDICTING USING OUR MODEL: Now we can predict using our test set:
yhat = LR.predict(X_test_log)


  #Accuracy Evaluation
from sklearn import metrics
from sklearn.metrics import f1_score

print("Logistic Regression Train set Accuracy:", metrics.accuracy_score(y_train_log, LR.predict(X_train_log)))
print("Logistic Regression Test Set Accuracy:", metrics.accuracy_score(y_test_log, yhat))
print('F1 Score:', f1_score(y_test_log, yhat, average='weighted') )

    # Confusion Matrix
conf_matrix = confusion_matrix(y_test_log, yhat, labels=[1,0])
conf_matrix
np.set_printoptions(precision=2)
plt.figure()
plot_confus_matrix(conf_matrix, classes=['parkinsons = 1','healthy=0'],normalize= True,  title='Confusion matrix')
     

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


#from sklearn import svm


# Linear SVM
#svm_l = svm.SVC(kernel='linear')
#svm_l.fit(X_train_log, y_train_log) 

# Use to Predict new values
#yhat_linear = svm_l.predict(X_test_log)
#yhat_linear [0:5]


 #Accuracy Evaluation
#from sklearn import metrics
#print("SVM Linear Train set Accuracy:", metrics.accuracy_score(y_train_log, LR.predict(X_train_log)))
#print("SVM Linear Test Set Accuracy:", metrics.accuracy_score(y_test_log, yhat))

    # Confusion Matrix
#conf_matrix = confusion_matrix(y_test_log, yhat, labels=[1,0])
#conf_matrix
#np.set_printoptions(precision=2)
#plt.figure()
#plot_confus_matrix(conf_matrix, classes=['healthy=0','parkinsons=1'],normalize= False,  title='Confusion matrix')
     
























