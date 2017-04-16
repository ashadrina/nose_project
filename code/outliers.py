import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing

import os 
import sys
import numpy as np
import pandas as pd
import scipy
#from scipy import stats
import copy 


def load_data(in_file):
    input_f = open(in_file, "r")
    matrix = []
    for line in input_f:
        channels = [] #get channels
        for l in line.split("|"):
            samples = l.split(";")
            channels.append([float(i) for i in samples])
        matrix.append(channels)
        del channels
    input_f.close()
    return matrix

def load_labels(in_file):
    input_f = open(in_file, "r")
    labels = []
    for line in input_f:
        if ";" in line:
            labels.append(line.replace("\n","").split(";"))
        else:
            labels.append(line.replace("\n",""))
    input_f.close()
    return labels


##########################################
    
def normalize_data(data):
    norm_matrix = []
    for block in data:
        #current_max = np.amax(block)
        norm_col = []
        for col in block:
            current_mean = np.mean(col)
            surrent_std = np.std(col)
            norm_col.append([(float(i) - current_mean)//surrent_std for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix

def detrend(x):
    import numpy as np
    import scipy.signal as sps
    import matplotlib.pyplot as plt
    x = np.asarray(x)    
    jmps = np.where(np.diff(x) < -0.5)[0]  
    for j in jmps:
        x[j+1:] += x[j] - x[j+1]    
    order = 20
    x -= sps.filtfilt([1] * order, [order], x) 
    return x
    
def patch_detrend(X_train):
    X_res = []
    for matr in X_train:
        matr_res = []
        for ch in matr:
            matr_res.append(detrend(ch))
        X_res.append(matr_res)
    return X_res
 
def fit_polynom(X_train, N):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    X_train_new = []
    for matr in X_train:
        matr_new = []
        for i in range(len(matr)):
            vec = matr[i]
            L = len(vec)            
            T = 1/250.0       
            t = np.linspace(1,L,L)*T   
            xx = np.asarray(t)
            yy = np.asarray(vec)
            z = np.asarray(np.polyfit(xx, yy, N))
            ff = np.poly1d(z)
            x_new = np.linspace(xx[0], xx[-1], len(xx))
            y_new = ff(x_new)
            matr_new.append(y_new)
        X_train_new.append(matr_new)
    return X_train_new
    
########################################## 


rng = np.random.RandomState(42)
n_samples = 40

# define two outlier detection tools to be compared
classifiers = {
    "One-Class SVM": svm.OneClassSVM(kernel="rbf", gamma=0.1),
    "Robust covariance": EllipticEnvelope(),
    "Isolation Forest": IsolationForest(max_samples=n_samples, random_state=rng)}

X_train = load_data("data/data_all.txt")
y_train = load_data("data/labels_all.txt")
X_train = normalize_data(X_train)    
X_train = patch_detrend(X_train)    
X_train = np.array(X_train)
nsamples00, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples00,nx*ny))           
X = preprocessing.scale(X_train)

plt.figure(figsize=(12, 9))
for i, (clf_name, clf) in enumerate(classifiers.items()):
    clf.fit(X)
    scores_pred = clf.decision_function(X)
    y_pred = clf.predict(X)
    print (clf_name, ": ", y_pred)
    
    for voc,res in zip(y_train,y_pred):
        print (voc, " - ", res)
'

