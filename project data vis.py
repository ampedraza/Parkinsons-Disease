# -*- coding: utf-8 -*-
"""
Alison Pedraza
cs 677
Project
Data Visualization

parkinson's = 1
healthy = 0
"""

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




def labeling (row):
    if row == 1:
        return "Parkinson's"
    elif row ==  0:
        return 'Healthy'


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
 'status']]


df_subset['Labeled Status']=df_subset.status.apply(labeling)

df_subset_p1 = df[0:6]

import seaborn as sns

    #box plots
#Frequency
_, axes = plt.subplots(1, sharey=True, figsize=(5, 4))
sns.boxplot(x='Labeled Status', y='MDVP:Fo(Hz)', data=df_subset, hue = 'Labeled Status')
plt.legend(title = 'Average Frequency', loc = 'center')

#Shimmer (db)
_, axes = plt.subplots(1, sharey=False, figsize=(5, 4))
sns.boxplot(x='Labeled Status', y="MDVP:Shimmer(dB)", data=df_subset, hue = 'Labeled Status')
plt.legend(title = 'Shimmer(dB)', loc = 'upper right')

#Jitter %
_, axes = plt.subplots(1, sharey=False, figsize=(5, 4))
sns.boxplot(x='Labeled Status', y="MDVP:Jitter(%)", data=df_subset, hue = 'Labeled Status')
plt.legend(title = 'Jitter(%)', loc = 'upper right')

#NHR
_, axes = plt.subplots(1, sharey=False, figsize=(5, 4))
sns.boxplot(x='Labeled Status', y="NHR", data=df_subset, hue = 'Labeled Status')
plt.legend(title = 'Noise to Harmonics Ratio', loc = 'upper right')



#APQ
_, axes = plt.subplots(1, sharey=False, figsize=(5, 4))
sns.boxplot(x='Labeled Status', y='MDVP:APQ', data=df_subset, hue = 'Labeled Status')
plt.legend(title = 'MDVP:APQ', loc = 'upper right')







#sns.boxplot(x='status', y='MDVP:Flo(Hz)', data=df_subset,legend = False)
#g = sns.boxplot(x='status', y='MDVP:Jitter(%)',palette = 'Pastel1', hue = 'status', data=df_subset)
#plt.legend(title = 'Jitter(%)', loc = 'upper right', labels = ['healthy', "Parkinson's"])
#plt.show(g)



    # violin plots

_, axes = plt.subplots(1, sharey=True, figsize=(5, 4))
sns.violinplot(x='Labeled Status', y='MDVP:Fo(Hz)', palette = 'Pastel1', hue = 'Labeled Status',data=df_subset)
plt.legend(title = 'Frequency(Hz)', loc = 'best')


_, axes = plt.subplots(1, sharey=True, figsize=(5, 4))
sns.violinplot(x='Labeled Status', y='MDVP:Shimmer(dB)', palette = 'Pastel1', hue = 'Labeled Status',data=df_subset)
plt.legend(title = 'Shimmer(dB)', loc = 'best')

_, axes = plt.subplots(1, sharey=True, figsize=(5, 4))
sns.violinplot(x='Labeled Status', y='MDVP:Jitter(%)', palette = 'Pastel1', hue = 'Labeled Status',data=df_subset)
plt.legend(title = 'Jitter(%)', loc = 'upper right')


_, axes = plt.subplots(1, sharey=True, figsize=(5, 4))
sns.violinplot(x='Labeled Status', y='NHR', palette = 'Pastel1', hue = 'Labeled Status',data=df_subset)
plt.legend(title = 'Noise to Harmonics Ratio', loc = 'best')


_, axes = plt.subplots(1, sharey=True, figsize=(5, 4))
sns.violinplot(x='Labeled Status', y='MDVP:APQ', palette = 'Pastel1', hue = 'Labeled Status',data=df_subset)
plt.legend(title = 'APQ', loc = 'best')



    # scatter plot


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='status', y='MDVP:Jitter(%)', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='status', y='MDVP:Jitter(%)', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Jitter %"); plt.show()


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='status', y='MDVP:Fo(Hz)', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='status', y='MDVP:Fo(Hz)', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Frequency (Hz)"); plt.show()


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='status', y='NHR', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='status', y='NHR', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Noise to Harmonics Ratio"); plt.show()


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='status', y='MDVP:Shimmer(dB)', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='status', y='MDVP:Shimmer(dB)', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Shimmer(dB)"); plt.show()


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='status', y='MDVP:APQ', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='status', y='MDVP:APQ', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("APQ: Measures average difference in wavelength height"); plt.show()


# Other Scatter plots
ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='NHR', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='NHR', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Ave Frequency vs NHR"); plt.show()



ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='MDVP:Fhi(Hz)', y='NHR', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='MDVP:Fhi(Hz)', y='NHR', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("High Frequency vs NHR"); plt.show()


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='MDVP:Fhi(Hz)', y='MDVP:Jitter(%)', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='MDVP:Fhi(Hz)', y='MDVP:Jitter(%)', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("High Frequency vs Jitter%"); plt.show()


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:Jitter(%)', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:Jitter(%)', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Ave Frequency vs Jitter%"); plt.show()


ax = df_subset[df_subset['status'] == 1][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:APQ', color='DarkBlue', label='Parkinson');
df_subset[df_subset['status'] == 0][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:APQ', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Ave Frequency vs APQ"); plt.show()
# amplitude pertubation quotient for shimmer: measures the differece in amplitude


 
# illustrate heat map.
sns.heatmap(df_subset.select_dtypes(include='number').corr(),
            cmap=sns.cubehelix_palette(20, light=0.95, dark=0.15))




# Two Patients
df_subset_p2 = df[30:36] # healthy patient
df_subset_p1 = df[0:6]  # parkinson's patient
two_frames = [df_subset_p1, df_subset_p2]
df_joined = pd.concat(two_frames)

ax = df_joined[df_joined['status'] == 1][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:Jitter(%)', color='DarkBlue', label='Parkinson');
df_joined[df_joined['status'] == 0][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:Jitter(%)', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Two Patients: Ave Frequency vs Jitter%"); plt.show()


ax = df_joined[df_joined['status'] == 1][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:APQ', color='DarkBlue', label='Parkinson');
df_joined[df_joined['status'] == 0][0:].plot(kind='scatter', x='MDVP:Fo(Hz)', y='MDVP:APQ', color='Red', label='Healthy', figsize = (5,4), ax=ax);
plt.title("Two Patients: Ave Frequency vs APQ"); plt.show()
