import os 
import sys
import numpy as np
import pandas as pd
import scipy
#from scipy import stats
import copy 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

from sklearn.svm import LinearSVC, SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import PCA

#from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn import preprocessing
from numpy.linalg import svd
from sklearn.metrics import roc_curve
#from sklearn.cross_validation import train_test_split


import warnings


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

def get_fmax(X):
    X_max = []
    S_max = []
    for x in X:
        s_m = []
        for s in x:
            s_m.append(max(np.array(s)))
        S_max.append(s_m)
    for x,s_max in zip(X,S_max):        
        #print ("s_max: ", s_max)
        m = list(map(list, zip(*x))) 
        X_m = []
        for s in m:
            s_new = []
            for i,ss in zip(s,s_max):
                if i != ss:
                    s_new.append(0)
                else:
                    s_new.append(i)
            X_m.append(s_new)
        X_max.append(X_m)   
        
    return X_max
    
def get_feq(X):
    X_eq = []
    S_eq = []
    for x in X:
        s_e = []
        for s in x:
            s_e.append(s[-1])
        S_eq.append(s_e)
    for x,s_eq in zip(X,S_eq):        
        #print ("s_eq: ", s_eq)
        m = list(map(list, zip(*x))) 
        X_m = []
        for s in m:
            s_new = []
            for i,ss in zip(s,s_eq):
                if i != ss:
                    s_new.append(0)
                else:
                    s_new.append(i)
            X_m.append(s_new)
        X_eq.append(X_m)   
        
    return X_eq
  

def detrend(x):
    import numpy as np
    import scipy.signal as sps
    import matplotlib.pyplot as plt

    x = np.asarray(x)    
    #plt.plot(x, label='original')

    # detect and remove jumps
    jmps = np.where(np.diff(x) < -0.5)[0]  # find large, rapid drops in amplitdue
    for j in jmps:
        x[j+1:] += x[j] - x[j+1]    
    #plt.plot(x, label='unrolled')

    # detrend with a low-pass
    order = 20
    x -= sps.filtfilt([1] * order, [order], x)  # this is a very simple moving average filter
    #plt.plot(x, label='detrended')

    #plt.legend(loc='best')
    #plt.show()
    return x
    
def patch_detrend(X_train):
    X_res = []
    for matr in X_train:
        matr_res = []
        for ch in matr:
            matr_res.append(detrend(ch))
        X_res.append(matr_res)
    return X_res
  
def load_testing_2():
    X_train_data = load_data("data/data_train_under.txt")
    y_train_lat_labels = load_labels("data/labels_train_under.txt")
    print ("initial data: ", np.array(X_train_data).shape)
    X_train_poly = fit_polynom(X_train_data, 3)
    X_train_2 = normalize_data(X_train_poly)

    X_test_data = load_data("data/data_test.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels = ["_with_".join(i) for i in y_test_lat_labels]
    print ("initial data: ", np.array(X_test_data).shape)
    X_test_poly = fit_polynom(X_test_data, 3)
    X_test_2 = normalize_data(X_test_poly)
    
    ##########################################
    X_train_big = []
    X_train_big.extend(X_train_data)
    X_train_big.extend(X_test_data)
    X_train_big = np.array(X_train_big)

    y_train_lat_big = []
    y_train_lat_big.extend(y_train_lat_labels)
    y_train_lat_big.extend(y_test_lat_labels)
    
    y_train_lat_big_list = []
    for i in y_train_lat_big:
        y_train_lat_big_list.append([i])

    mlb2 = MultiLabelBinarizer()
    y_train_big =  mlb2.fit_transform(y_train_lat_big_list) 

    X_new_data = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new_data).shape)

    return X_train_big, y_train_big, X_new_data, mlb2
 
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
    
def svm_cl_testing(X_train, y_train, X_test, mlb1):
    # pca = PCA(n_components=1)
    # pca.fit_transform(X_train)        
    # pca.fit_transform(X_test)   
   
    svc = OneVsRestClassifier(SVC(kernel='rbf', probability=True)).fit(X_train, y_train)
    y_score =  svc.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("svm train accuracy: ", 1 - err_train)
    
    from sklearn.metrics import coverage_error
    y_test_true_labels = load_labels("true_labels.txt")
    y_new_proba = svc.predict_proba(X_test)
    
    for y_pred,y_tr in zip(y_new_proba,y_test_true_labels):
       r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb1.classes_,y_pred)]
       sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
       print (sorted_by_second_1)
       print (y_tr)
       print ("-----------------------------------")
       break
      

    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    y_test_true =  mlb1.fit_transform(y_test_true_labels) 
    y_test_true = list(y_test_true)[:-1]      

    from sklearn.metrics import coverage_error
    err1 = coverage_error(y_train, y_score)
    print ("You should predict top ",err1, " labels for train")
    err2 = coverage_error(y_test_true, y_new_proba)
    print ("You should predict top ",err2, " labels for toys")

    from sklearn.metrics import label_ranking_average_precision_score
    rap1 = label_ranking_average_precision_score(y_train, y_score) 
    print ("label_ranking_average_precision_score on train", rap1)
    rap2 = label_ranking_average_precision_score(y_test_true, y_new_proba) 
    print ("label_ranking_average_precision_score on toys", rap2)
 
def main():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    N = 3
    X_train_full, y_labels_train_full, X_new, mlb = load_testing_2()
    
    ########################################
        
    # X_train_full_max = get_fmax(X_train_full)
    # X_train_full_max = np.array(X_train_full_max)    
    # X_train_full_eq = get_feq(X_train_full)
    # X_train_full_eq = np.array(X_train_full_eq)
    
    X_train_full = normalize_data(X_train_full)
    X_train_full = patch_detrend(X_train_full)    
#    X_train_full = fit_polynom(X_train_full,N)    
       
    X_train_full = np.array(X_train_full)
    nsamples00, nx, ny = X_train_full.shape
    X_train_full_2d_init = X_train_full.reshape((nsamples00,nx*ny))        
    # nsamples0, nx, ny = X_train_full_max.shape
    # X_train_full_2d_max = X_train_full_max.reshape((nsamples0,nx*ny))    
    # nsamples1, nx, ny = X_train_full_eq.shape
    # X_train_full_2d_eq = X_train_full_eq.reshape((nsamples1,nx*ny))    
    
    #X_train_full_2d = np.hstack((X_train_full_2d_init, X_train_full_2d_eq))
    #X_train_full_2d = np.hstack((X_train_full_2d_init, X_train_full_2d_max, X_train_full_2d_eq))
    
    ########################################
    
    # X_new_max = get_fmax(X_new)
    # X_new_max = np.array(X_new_max)
    # X_new_full_eq = get_feq(X_new)
    # X_new_full_eq = np.array(X_new_full_eq)

    X_new = normalize_data(X_new)    
    X_new = patch_detrend(X_new)    
#    X_new = fit_polynom(X_new,N) 

    X_new = np.array(X_new)
    nsamples22, nx, ny = X_new.shape
    X_new_2d_init = X_new.reshape((nsamples22,nx*ny))        
    # nsamples2, nx2, ny2 = X_new_max.shape
    # X_new_full_2d_max = X_new_max.reshape((nsamples2,nx2*ny2))
    # nsamples3, nx2, ny2 = X_new_full_eq.shape
    # X_test_new_2d_eq = X_new_full_eq.reshape((nsamples3,nx2*ny2))
       
    #X_new_full_2d = np.hstack((X_new_2d_init, X_new_full_2d_max, X_test_new_2d_eq))
    
    #########################################
    
    #X_train_full_2d = preprocessing.scale(X_train_full_2d)
    #X_new_full_2d = preprocessing.scale(X_new_full_2d)    
    
    X_train_full_2d = preprocessing.scale(X_train_full_2d_init)
    X_new_full_2d = preprocessing.scale(X_new_2d_init)
  
    svm_cl_testing( X_train_full_2d_init, y_labels_train_full, X_new_2d_init, mlb)


      
if __name__ == "__main__":
    main()
