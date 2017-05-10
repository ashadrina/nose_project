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
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib.font_manager 

X1 = load_data("data/pairs2/data_all.txt")
y_train = load_labels("data/labels_all.txt")
X1 = normalize_data(X1)    
X1 = patch_detrend(X1)    
X1 = np.array(X1)
nsamples00, nx, ny = X1.shape
X1 = X1.reshape((nsamples00,nx*ny))           
X1 = preprocessing.scale(X1)


rng = np.random.RandomState(42)
n_samples = 40

# Define "classifiers" to be used
classifiers_ee = {
    "Empirical Covariance": EllipticEnvelope(support_fraction=1., contamination=0.261),
    "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(contamination=0.261),
    "Robust covariance (default)": EllipticEnvelope()}
classifiers_svm = {
    "One-Class NuSVM": OneClassSVM(nu=0.261, gamma=0.05),
    "One-Class SVM (rbf)": svm.OneClassSVM(kernel="rbf", gamma=0.1),
    "Isolation Forest": IsolationForest(max_samples=n_samples, random_state=rng)}
    
classifiers_all = {
    "Empirical Covariance": EllipticEnvelope(support_fraction=1., contamination=0.261),
    "Robust Covariance (Minimum Covariance Determinant)": EllipticEnvelope(contamination=0.261),
    "Robust covariance (default)": EllipticEnvelope(),
    "One-Class NuSVM": OneClassSVM(nu=0.261, gamma=0.05),
    "One-Class SVM (rbf)": svm.OneClassSVM(kernel="rbf", gamma=0.1),
    "Isolation Forest": IsolationForest(max_samples=n_samples, random_state=rng)}
    
classifiers =  classifiers_all # classifiers_ee # classifiers_svm #  
colors = ['m', 'g', 'b', 'y', 'r', 'c'] # ['m', 'g', 'b'] # ['y', 'r', 'c'] #
legend1 = {}


from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel="cosine", fit_inverse_transform=True)
#X1 = kpca.fit_transform(X1)
#X1 = kpca.inverse_transform(X1)

# Learn a frontier for outlier detection with several classifiers
xx1, yy1 = np.meshgrid(np.linspace(-1, 1, 26), np.linspace(-1, 1, 38))
print (xx1.shape, yy1.shape)
print (X1.shape)
for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(1)
    clf.fit(X1)
    #Z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    #Z1 = Z1.reshape(xx1.shape)
    #legend1[clf_name] = plt.contour(xx1, yy1, Z1, levels=[0], linewidths=2, colors=colors[i])
    
    scores_pred = clf.decision_function(X1)
    y_pred = clf.predict(X1)
    print (clf_name, ": ", y_pred)
    
    n_outliers = 0
    for voc,res in zip(y_train,y_pred):
        if res == -1:
            print (voc, " - ", res)
            n_outliers += 1
    print ("outliers: ", n_outliers)
    
#legend1_values_list = list(legend1.values())
#legend1_keys_list = list(legend1.keys())

## Plot the results (= shape of the data points cloud)
#plt.figure(1)  # two clusters
#plt.title("Outlier detection on train data")
#plt.scatter(X1[:, 0], X1[:, 1], color='black')
#bbox_args = dict(boxstyle="round", fc="0.8")
#arrow_args = dict(arrowstyle="->")
#plt.annotate("several confounded points", xy=(24, 19),
             #xycoords="data", textcoords="data",
             #xytext=(13, 10), bbox=bbox_args, arrowprops=arrow_args)
#plt.xlim((xx1.min(), xx1.max()))
#plt.ylim((yy1.min(), yy1.max()))
#plt.legend((legend1_values_list[0].collections[0],
            #legend1_values_list[1].collections[0],
            #legend1_values_list[2].collections[0],
            #legend1_values_list[3].collections[0],
            #legend1_values_list[4].collections[0],
            #legend1_values_list[5].collections[0]),
           #(legend1_keys_list[0], legend1_keys_list[1], legend1_keys_list[2],
           #legend1_keys_list[3], legend1_keys_list[4], legend1_keys_list[5]),
           #loc="best",
           #prop=matplotlib.font_manager.FontProperties(size=12))
#plt.ylabel("KPCA(1)")
#plt.xlabel("KPCA(0)")

#plt.show()

