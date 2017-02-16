import os 
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import copy 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from numpy.linalg import svd
from sklearn.svm import LinearSVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

from sklearn import preprocessing

# from keras.preprocessing import sequence
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Masking, TimeDistributedDense
# from keras.layers.embeddings import Embedding
# from keras.layers.recurrent import LSTM
# from keras.utils.np_utils import to_categorical
# from keras.utils.visualize_util import plot

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
        current_max = np.amax(block)
        norm_col = []
        for col in block:
            norm_col.append([float(i)//current_max for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix
   
def load_train():    
    #training - compounds
    X_train_0 = np.array(load_data("data/data_train.txt"))
    lat_labels_train = load_labels("data/labels_train.txt")
    print ("initial train data: ", X_train_0.shape)
    X_train_poly = fit_polynom(X_train_0, 3)
    X_train = normalize_data(X_train_poly)
    mlb = LabelBinarizer()
    y_train = mlb.fit_transform(lat_labels_train) 
    
    #testing - mixtures
    X_test_0 = np.array(load_data("data/data_test.txt"))
    lat_labels_list = load_labels("data/labels_test.txt")
    print ("initial test data: ", np.array(X_test_0).shape)
    X_test_poly = fit_polynom(X_test_0, 3)
    X_test = normalize_data(X_test_poly)
    mlb1 = MultiLabelBinarizer()
    lat_labels_list.append(lat_labels_train)
    y_test_bin_labels = mlb1.fit_transform(lat_labels_list)
    y_test = y_test_bin_labels[:-1]

    return X_train, y_train, X_test, y_test
 
def load_testing_2():
    X_train_data = load_data("data/data_train.txt")
    y_train_lat_labels = load_labels("data/labels_train.txt")
   # print (len(set(y_train_lat_labels)))
    print ("initial data: ", np.array(X_train_data).shape)

    mlb = LabelBinarizer()
    y_train_bin_labels = mlb.fit_transform(y_train_lat_labels) 

    X_train_poly = fit_polynom(X_train_data, 3)
    X_train_2 = normalize_data(X_train_poly)

    X_test_data = load_data("data/data_test.txt")
    y_test_lat_labels = load_labels("data/labels_test.txt")
    y_test_lat_labels_2 = load_labels("data/labels_test.txt")
    print ("initial data: ", np.array(X_test_data).shape)
    
    X_test_poly = fit_polynom(X_test_data, 3)
    X_test_2 = normalize_data(X_test_poly)

    mlb1 = MultiLabelBinarizer()
    y_test_lat_labels.append(y_train_lat_labels)
    y_test_bin_labels = mlb1.fit_transform(y_test_lat_labels)
    y_test = y_test_bin_labels[:-1]
    
    ###################################
    X_train_big = []
    X_train_big.extend(X_train_2)
    X_train_big.extend(X_test_2)
    X_train_big = np.array(X_train_big)

    ll = [] #y_train
    for l in y_train_lat_labels:
        ll.append([l])

    y_train_lat_big = []
    y_train_lat_big.extend(ll)
    y_train_lat_big.extend(y_test_lat_labels_2)

    mlb2 = MultiLabelBinarizer()
    y_train_big =  mlb2.fit_transform(y_train_lat_big) 

    X_new_data = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new_data).shape)
    
    X_new_poly = fit_polynom(X_new_data, 8)
    X_new_data_2 = normalize_data(X_new_poly)    
   
    X = []
    x = []
    pca = PCA(n_components=4)
    for matr in X_train_big:
        X.append(pca.fit_transform(matr))    
    for matr in X_new_data_2:
        x.append(pca.fit_transform(matr))
    return X, y_train_big, x, mlb1
 
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

def svm_cl_training(X_train, y_train, X_test, y_test):
    svc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
    err_train = np.mean(y_train != svc.predict(X_train))
    err_test  = np.mean(y_test  != svc.predict(X_test))
    print ("svm accuracy: ", 1 - err_train, 1 - err_test)
    
def knn_cl_training(X_train, y_train, X_test, y_test):
    knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
    err_train = np.mean(y_train != knn.predict(X_train))
    err_test  = np.mean(y_test  != knn.predict(X_test))
    print ("knn accuracy: ", 1 - err_train, 1 - err_test)

def rf_cl_training(X_train, y_train, X_test, y_test):
    rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000)).fit(X_train, y_train)
    err_train = np.mean(y_train != rf.predict(X_train))
    err_test  = np.mean(y_test  != rf.predict(X_test))
    print ("rf accuracy: ", 1 - err_train, 1 - err_test)

def bayes_cl_training(X_train, y_train, X_test, y_test):
    gnb = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
    err_train = np.mean(y_train != gnb.predict(X_train))
    err_test  = np.mean(y_test  != gnb.predict(X_test))
    print ("gnb accuracy: ", 1 - err_train, 1 - err_test)	

 
def knn_cl_testing(X_train, y_train, X_test, mlb1):
    knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
    err_train = np.mean(y_train != knn.predict(X_train))
    print ("knn train accuracy: ", 1 - err_train)		
    y_new = knn.predict(X_test)
    y_labels = mlb1.inverse_transform(y_new)
    txt_outfile = open("2knn_new_labels.txt", 'w')
    for y in y_labels:
        if y:
            txt_outfile.write(";".join(y)+"\n")
        else:
            txt_outfile.write("?\n")
    txt_outfile.close()

def svm_cl_testing(X_train, y_train, X_test, mlb1):
    svc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)
    err_train = np.mean(y_train != svc.predict(X_train))
    print ("svm train accuracy: ", 1 - err_train)
    y_new = svc.predict(X_test)
    y_labels = mlb1.inverse_transform(y_new)
    txt_outfile = open("2svm_new_labels.txt", 'w')
    for y in y_labels:
        if y:
            txt_outfile.write(";".join(y)+"\n")
        else:
            txt_outfile.write("?\n")
    txt_outfile.close()
 
def rf_cl_testing(X_train, y_train, X_test, mlb1):
    rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000)).fit(X_train, y_train)
    err_train = np.mean(y_train != rf.predict(X_train))
    print ("rf train accuracy: ", 1 - err_train)
    y_new = rf.predict(X_test)
    y_labels = mlb1.inverse_transform(y_new)
    txt_outfile = open("2rf_new_labels.txt", 'w')
    for y in y_labels:
        if y:
            txt_outfile.write(";".join(y)+"\n")
        else:
            txt_outfile.write("?\n")
    txt_outfile.close()
    
    
def bayes_cl_testing(X_train, y_train, X_test, mlb1):
    gnb = OneVsRestClassifier(GaussianNB()).fit(X_train, y_train)
    err_train = np.mean(y_train != gnb.predict(X_train))
    print ("gnb train accuracy: ", 1 - err_train)
    y_new = gnb.predict(X_test)
    y_labels = mlb1.inverse_transform(y_new)
    txt_outfile = open("2gnb_new_labels.txt", 'w')
    for y in y_labels:
        if y:
            txt_outfile.write(";".join(y)+"\n")
        else:
            txt_outfile.write("?\n")
    txt_outfile.close()
 
def main():
    #learn on compouns, test on mixtures
    X_train, y_train, X_test, y_test = load_train()
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    nsamples, nx, ny = X_train.shape
    X_train_2d = X_train.reshape((nsamples,nx*ny))    
    nsamples2, nx2, ny2 = X_test.shape
    X_test_2d = X_test.reshape((nsamples2,nx2*ny2))
    svm_cl_training(X_train_2d, y_train, X_test_2d, y_test)
    knn_cl_training(X_train_2d, y_train, X_test_2d, y_test)
    rf_cl_training(X_train_2d, y_train, X_test_2d, y_test)
    bayes_cl_training(X_train_2d, y_train, X_test_2d, y_test)
    #############################################
    print ("#############################################")
    #learn on compounds and mixtures and try toys
    X_train_full, y_labels_train_full, X_new,mlb = load_testing_2()
    X_train_full = np.array(X_train_full)
    X_new = np.array(X_new)
    nsamples, nx, ny = X_train_full.shape
    X_train_full_2d = X_train_full.reshape((nsamples,nx*ny))    
    nsamples2, nx2, ny2 = X_new.shape
    X_test_full_2d = X_new.reshape((nsamples2,nx2*ny2))
    
    X_train_full_2d = preprocessing.scale(X_train_full_2d)
    X_test_full_2d = preprocessing.scale(X_test_full_2d)
    
    svm_cl_testing(X_train_full_2d, y_labels_train_full, X_test_full_2d,mlb) 
    knn_cl_testing(X_train_full_2d, y_labels_train_full, X_test_full_2d,mlb) 
    rf_cl_testing(X_train_full_2d, y_labels_train_full, X_test_full_2d,mlb) 
    bayes_cl_testing(X_train_full_2d, y_labels_train_full, X_test_full_2d,mlb) 
    
if __name__ == "__main__":
    main()
