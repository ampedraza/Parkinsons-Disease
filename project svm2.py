
"""
Alison Pedraza
CS 677 Project
Support Vector Machine

Using SVM to predict Parkinson's patients
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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



# Read in data
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
    # check for NAs on Subset of dataframe
na = df_subset.isna().sum()
print(na)
null = df_subset.isnull().sum()
print(null)


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


# Standardize data
X = StandardScaler().fit_transform(X)


y = df_subset[['status']].values

# split to Train and Test set
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split( X, y, test_size=0.2, stratify = y)
print ('SVM Train set:', X_train_svm.shape,  y_train_svm.shape)
print ('SVM Test set:', X_test_svm.shape,  y_test_svm.shape)
#scaler = preprocessing.StandardScaler().fit(X_trainX_std = scaler.transform(X)


# Using the SVM algorithm
from sklearn import svm
clf = svm.SVC(kernel='rbf')     # choose 'rbf' function
clf.fit(X_train_svm, y_train_svm) 


# Use to Predict new values
yhat = clf.predict(X_test_svm)
yhat [0:5]

from sklearn.metrics import classification_report, confusion_matrix
import itertools
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_svm, yhat, labels=[1,0])
np.set_printoptions(precision=2)


  #Accuracy Evaluation
from sklearn import metrics
print("SVM Train set Accuracy:", metrics.accuracy_score(y_train_svm, clf.predict(X_train_svm)))
print("SVM Test Set Accuracy:", metrics.accuracy_score(y_test_svm, yhat))
from sklearn.metrics import f1_score
print('F1 Score:', f1_score(y_test_svm, yhat, average='weighted') )

# Plot non-normalized confusion matrix

conf_matrix = confusion_matrix(y_test_svm, yhat, labels=[1,0])
conf_matrix

np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(conf_matrix, classes=['parkinsons = 1','healthy=0'],normalize= False,  title='Confusion matrix')
     

TP = cnf_matrix[0][0]
FN = cnf_matrix[1][0]
TN = cnf_matrix[1][1]
FP = cnf_matrix[0][1]


TPR = (TP/(TP+FN))
print ('TPR', TPR)
# Specificity or true negative rate
TNR = (TN/(TN+FP))
print('TNR', TNR  )
