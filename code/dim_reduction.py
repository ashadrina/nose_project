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

def load_dataset():
    X_train_data = load_data("data/data_train.txt")
    y_train_lat_labels = load_labels("data/labels_train.txt")
    print ("initial data: ", np.array(X_train_data).shape)

    X_test_data = load_data("data/data_test.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels = ["_with_".join(i) for i in y_test_lat_labels]
    print ("initial data: ", np.array(X_test_data).shape)
    
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

    mlb = MultiLabelBinarizer()
    y_train_big =  mlb.fit_transform(y_train_lat_big_list) 

    X_new_data = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new_data).shape)
    return X_train_big, y_train_big, X_new_data, mlb

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

def detrend(x, order, FLAG):
    import numpy as np
    import scipy.signal as sps
    import matplotlib.pyplot as plt

    x = np.asarray(x)   
    if FLAG:
        plt.plot(x, label='original')

    # detect and remove jumps
    jmps = np.where(np.diff(x) < -0.5)[0]  # find large, rapid drops in amplitdue
    for j in jmps:
        x[j+1:] += x[j] - x[j+1] 
    if FLAG:        
        plt.plot(x, label='unrolled')

    # detrend with a low-pass
    x -= sps.filtfilt([1] * order, [order], x)  # this is a very simple moving average filter
    if FLAG:
        plt.plot(x, label='detrended')

        plt.legend(loc='best')
        plt.show()
    return x
    
def patch_detrend(X_train):
    order = 15
    START = 0#1
    print ("order: ", order)
    X_res = []
    for matr in X_train:
        matr_res = []
        for ch in matr:
            if START == 1:
                matr_res.append(detrend(ch, order, 1))
                START = 0
            else:
                matr_res.append(detrend(ch, order, 0))
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

##################################  
 
def main():
    X_train, y_train, X_new, mlb = load_dataset()
    
    #X_train = load_data("data/data_train_over.txt")
    #y_train_lat = load_labels("data/labels_train_over.txt")    
    
    # X_train = load_data("data/data_all_outliers.txt")
    # y_train_lat = load_labels("data/labels_all_outliers.txt")
    
    # y_train_lat_list = []
    # for i in y_train_lat:
        # y_train_lat_list.append([i])

    # y_train =  mlb.fit_transform(y_train_lat_list) 
 
    X_train = normalize_data(X_train)    
    #X_train = patch_detrend(X_train)    
       
    X_train = np.array(X_train)
    nsamples00, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples00,nx*ny))            
    ###################################  
    # X_new = load_data("data/data_new.txt")   
    # X_new = normalize_data(X_new)    
    # #X_new = patch_detrend(X_new)    
  
    # X_new = np.array(X_new)
    # nsamples22, nx, ny = X_new.shape
    # X_new = X_new.reshape((nsamples22,nx*ny))      

    ###################################
    #X_train = preprocessing.scale(X_train)
   # X_new = preprocessing.scale(X_new)

    print (np.array(X_train).shape)
    #print (np.array(X_new).shape)
    ###################################
    y2 = load_labels("data/labels_train.txt")
    y3 = load_labels("data/labels_test.txt")
    y3 =  ["_with_".join(i) for i in y3]
    y = []
    y.extend(y2)
    y.extend(y3)
        
    from sklearn.decomposition import KernelPCA, PCA
    kpca = KernelPCA(kernel="cosine", fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X_train)
    X_back = kpca.inverse_transform(X_kpca)
    pca = PCA()
    X_pca = pca.fit_transform(X_train)
    
   # Plot results
    colors = {'azetaldegid':"#F2072E", 'azeton':"#ECAEB9", 'benzin':"#F207A8", 'benzol':"#C923C3", 'butanol':"#AA23C9", 'butilazetat':"#4923C9", 'dioktilftalat':"#0A46F8", 
    'dioktilftalat_azetal_degid':"#0ABDF8", 'dioktilftalat_azeton':"#0AF869", 'dioktilftalat_benzol':"#71F9A8", 'dioktilftalat_etilazetat':"#E3F307",  'etilazetat':"#E6EBA0", 
    'fenol':"#F0AF0B", 'geksan':"#DAB34F", 'izobutanol':"#FC5F04", 'izopropanol':"#F9813B", 'plastizol':"#272727", 'propanol':"#6E3704", 'stirol':"#B08256", 'toluol':"#D8C0AA", 
    'dioktilftalat_with_benzol':"#EFFA2C", 'dioktilftalat_with_azetaldegid': "#2CFA41", 'dioktilftalat_with_etilazetat':"#1B5D8B", 'dioktilftalat_with_azeton':"#571212"}
    
    plt.figure()
    prev_target = ""
    for arr,target_name in zip(X_train,y):
        color = colors[target_name]
        if target_name == prev_target:
            plt.scatter(arr[0], arr[1],  color=color)
        else:
            plt.scatter(arr[0], arr[1], color=color, label=target_name)
        prev_target = target_name
    plt.title('Original data')

    plt.figure()    
    prev_target = ""
    for arr,target_name in zip(X_pca,y):
        color = colors[target_name]
        if target_name == prev_target:
            plt.scatter(arr[0], arr[1], color=color)
        else:
            plt.scatter(arr[0], arr[1], color=color, label=target_name)
        prev_target = target_name
    plt.title('Projection by PCA')

    plt.figure()
    prev_target = ""
    for arr,target_name in zip(X_kpca,y):
        color = colors[target_name]
        if target_name == prev_target:
            plt.scatter(arr[0], arr[1], color=color)
        else:
            plt.scatter(arr[0], arr[1], color=color, label=target_name)
        prev_target = target_name
    plt.title('Projection by Kernel PCA')

    plt.figure()
    prev_target = ""
    for arr,target_name in zip(X_back,y):
        color = colors[target_name]
        if target_name == prev_target:
            plt.scatter(arr[0], arr[1], color=color)
        else:
            plt.scatter(arr[0], arr[1], color=color, label=target_name)
        prev_target = target_name
    plt.title('Original space after inverse transform')
    
    plt.show()
        
    
if __name__ == "__main__":
    main()
