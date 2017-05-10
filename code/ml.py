# -*- coding: utf-8 -*-

import os 
import sys
import numpy as np
import pandas as pd
import scipy
#from scipy import stats
import copy 

import collections


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
            labels.append(set(line.replace("\n","").split(";")))
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
        if type(i) is list:
            y_train_lat_big_list.append(i)
        else:
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
def svm_mlb(X_train, y_train, X_new, mlb):
    from sklearn.svm import SVC, NuSVC
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("------------------------------------------")
    print ("--------------------SVM-------------------")
    print ("------------------------------------------")
    
   # clf = OneVsRestClassifier( NuSVC(cache_size=200, class_weight='balanced', coef0=0.0, 
   #      decision_function_shape='ovo', degree=1, gamma=1, kernel='rbf',
   #      max_iter=-1, nu=0.001, probability=True, random_state=None, shrinking=False, tol=0.001, verbose=False) ) 

    clf = OneVsRestClassifier( SVC(class_weight='balanced', probability=True, kernel='rbf') )
    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("svm train accuracy: ", 1 - err_train)
    
    y_new_proba = clf.predict_proba(X_new)
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    for y_pred,y_tr,i in zip(y_new_proba,y_test_true_labels,range(len(X_new))):
        print (i+1)
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,y_pred)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        print (sorted_by_second_1[:6])
        print (set(y_tr))
        print ("------------------")

    compute_av_metrics(y_train, y_score, y_new_proba, mlb)
    compute_ind_metrics(y_train, y_score, X_new, clf, mlb,  "svm")
    ##predicted_vs_real_label(y_new_proba, mlb, "svm")
    
    #y_train2 = load_labels("data/new/labels_only_other2.txt")
    #clf = OneVsRestClassifier( SVC(class_weight='balanced', probability=True, 
        #kernel='linear', coef0=0.01) )
    #clf.fit(X_train, y_train2)
    #y_score =  clf.predict(X_train) 
    #err_train = np.mean(y_train2 != y_score)
    #print ("svm train accuracy: ", 1 - err_train)
    
    #from sklearn.metrics import confusion_matrix
    #y_train2 = load_labels("data/new/labels_only_other2.txt")
    #cnf_matrix_train = confusion_matrix(y_score, y_train)    

    #np.set_printoptions(precision=2)
    #class_names = list(set(y_train))

    #plt.figure()
    #plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
    #plt.show()

def bagging_rf_mlb(X_train, y_train, X_new, mlb):
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("------------------------------------------")
    print ("------------Bagging with RF---------------")
    print ("------------------------------------------")
    
    clf = OneVsRestClassifier(BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, 
                class_weight='balanced', criterion='gini',
                max_depth=None, max_features=2, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=1500, n_jobs=1,
                oob_score=False, random_state=3, verbose=0,
                warm_start=False), max_samples=0.2, n_estimators=35))
    #y_train2 = load_labels("data/new/labels_only_other2.txt")

    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("bagging train accuracy: ", 1 - err_train)

    y_new_proba = clf.predict_proba(X_new)
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels = y_test_true_labels[-38:]
    i = 37
    #for y_pred,y_tr,i in zip(y_new_proba,y_test_true_labels,range(len(X_new))):
    for y_pred,y_tr in zip(y_new_proba,y_test_true_labels):
        print (i+1)
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,y_pred)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        print (sorted_by_second_1[:6])
        print (set(y_tr))
        print ("------------------")
    
    compute_av_metrics(y_train, y_score, y_new_proba, mlb)
    compute_ind_metrics(y_train, y_score, X_new, clf, mlb, "bagging_rf")
    #predicted_vs_real_label(y_new_proba, mlb, "bagging_rf")
    
    #from sklearn.metrics import confusion_matrix
    #y_train = load_labels("data/new/labels_only_other2.txt")
    #cnf_matrix_train = confusion_matrix(y_score, y_train)    

    #np.set_printoptions(precision=2)
    #class_names = list(set(y_train))
    #print (class_names)

    #plt.figure()
    #plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
    #plt.show()

  
def rf_mlb(X_train, y_train, X_new, mlb):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

    print ("------------------------------------------")
    print ("--------------------RF-------------------")
    print ("------------------------------------------")
    
    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1500))
    #y_train2 = load_labels("data/new/labels_only_other2.txt")
    clf.fit(X_train, y_train)
    y_score =  clf.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("random forest train accuracy: ", 1 - err_train)

    y_new_proba = clf.predict_proba(X_new)
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    for y_pred,y_tr,i in zip(y_new_proba,y_test_true_labels,range(len(X_new))):
        print (i+1)
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,y_pred)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        print (sorted_by_second_1[:6])
        print (set(y_tr))
        print ("------------------")
        
    compute_av_metrics(y_train, y_score, y_new_proba, mlb)
    compute_ind_metrics(y_train, y_score, X_new, clf, mlb, "rf")
    #predicted_vs_real_label(y_new_proba, mlb, "rf")
    
    #from sklearn.metrics import confusion_matrix
    #y_train = load_labels("data/labels_only_other.txt")
    #clf.fit(X_train, y_train)
    #y_score =  clf.predict(X_train)
    #cnf_matrix_train = confusion_matrix(y_score, y_train)    

    #np.set_printoptions(precision=2)
    #class_names = list(set(y_train))
    #print (class_names)

    #plt.figure()
    #plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
    #plt.show()
    
def mlp_mlb(X_train, y_train, X_new, mlb):
    from sklearn.neural_network import MLPClassifier

    print ("------------------------------------------")
    print ("--------------------MLP-------------------")
    print ("------------------------------------------")

    #alphas = list(np.arange(95.0, 98.0, 0.5))
    alphas = [97.5]
    names = []
    for i in alphas:
        names.append('alpha ' + str(i))

    classifiers = []
    for i in alphas:
        classifiers.append(MLPClassifier(alpha=i, random_state=42)) 
        
    print (classifiers)
    print (names)
    #y_train2 = load_labels("data/new/labels_only_other2.txt")
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        y_score =  clf.predict(X_train)
        print ("...................")
        print ("score: ", score)
        print ("alpha: ", name)
        
        #from sklearn.metrics import confusion_matrix
        #y_train = load_labels("data/new/labels_only_other2.txt")
        #cnf_matrix_train = confusion_matrix(y_score, y_train)    

        #np.set_printoptions(precision=2)
        #class_names = list(set(y_train))
        #print (class_names)

        #plt.figure()
        #plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
        #plt.show()

        y_new_proba = clf.predict_proba(X_new)
        y_test_true_labels = load_labels("true_labels.txt")
        for y_pred,y_tr in zip(y_new_proba,y_test_true_labels):
            r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,y_pred)]
            sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
            print (sorted_by_second_1)#[:6])
            print (y_tr)
            print ("------------------")
        compute_av_metrics(y_train, y_score, y_new_proba, mlb)
        compute_ind_metrics(y_train, y_score, X_new,  clf,  mlb, "mlp")
        #break
##################################
def predicted_vs_real_label(y_pred, mlb, folder_name):
    y_true = load_labels("true_labels.txt")
    y_true = [[x.replace('dioktilftalat','dof') for x in l] for l in y_true]
    nums = range(1,76)
   # folder_name = "graphs/proba/real/"+folder_name
   # folder_name = "graphs/proba/outliers/"+folder_name
    folder_name = "output/"+folder_name

    for x_test,y_tr,num in zip(y_pred,y_true, nums):
        y_tr = filter(None, y_tr)
        pred = dict([(c.replace('dioktilftalat','dof'),float("{:.3f}".format(yy))) for c,yy in zip(mlb.classes_,x_test)])      

        vocs = set(pred.keys())
        true = dict([(key, 0.0) for key in vocs])
        for tr in y_tr:
            true[tr] = 1.0

        df_pred = pd.DataFrame.from_dict(pred, orient='index')
        df_pred.columns=["Predicted"]
        df_true = pd.DataFrame.from_dict(true, orient='index')
        df_true.columns=["True"]
        df_result = pd.concat([df_true, df_pred], axis=1)
        df_result =  df_result.sort('Predicted', ascending=False)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        df_result["Predicted"].plot(kind='bar', ax=ax, color = 'teal', label="pred", alpha = 0.7, rot=90)
        df_result["True"].plot(kind='bar',ax=ax, color = 'red',  label="true",  alpha = 0.2)
        ax.set_xticklabels(df_result.index)
        plt.legend()
        plt.tight_layout()
        plt.title(num)
        #plt.show()

        try:
         os.stat(folder_name)
        except:
         os.mkdir(folder_name) 
        plt.savefig(folder_name+"/"+str(num)+".png", dpi=100)
        plt.close('all')

def compute_av_metrics(y_train, y_score, y_new_proba, mlb):
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    #y_test_true_labels.append(['other'])
    y_test_true =  mlb.fit_transform(y_test_true_labels) 
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
  
def compute_ind_metrics(y_train, y_score, X_new, clf, mlb, algo):
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    #y_test_true_labels.append(['other'])
    y_test_true =  mlb.fit_transform(y_test_true_labels) 
    y_test_true = list(y_test_true)[:-1]      
    
    from sklearn.metrics import coverage_error
    #err1 = coverage_error(y_train, y_score)
    #print ("You should predict top ",err1, " labels for train")
    from sklearn.metrics import label_ranking_average_precision_score
    #rap1 = label_ranking_average_precision_score(y_train, y_score)
    #print ("label_ranking_average_precision_score on train", rap1)

    true = {}
    pred = {}
    y_new = []
    for i in range(len(X_new)):
        #print (i+1)
        y_new_proba = clf.predict_proba([X_new[i]])  
        y_new.append(y_new_proba)
        err2 = coverage_error([y_test_true[i]], y_new_proba)
        #print ("Compounds in real solution: ", len(y_test_true_labels[i]))
        #print ("You should predict top ",err2, " labels for toys")
        rap2 = label_ranking_average_precision_score([y_test_true[i]], y_new_proba)
        #print ("label_ranking_average_precision_score on toys", rap2)
        #print ("--------------------------------------------------------------")
        true[i] = float(len(y_test_true_labels[i]))
        pred[i] = err2
        
    df_pred = pd.DataFrame.from_dict(pred, orient='index')
    df_pred.columns=["Predicted"]
    df_true = pd.DataFrame.from_dict(true, orient='index')
    df_true.columns=["True"]
    df_result = pd.concat([df_true, df_pred], axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    df_result["Predicted"].plot(kind='line', ax=ax, color = 'teal', label="pred", alpha = 0.7, rot=90, lw=3)
    df_result["True"].plot(kind='line',ax=ax, color = 'red',  label="true",  alpha = 0.7, lw=3, ls='dotted')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.title("Coverage error VS true labell number")
    #plt.show()
    #plt.savefig("graphs/metrics/outliers/"+algo+".png", dpi=100)
    #plt.savefig("graphs/metrics/real/"+algo+".png", dpi=100)
    plt.savefig("output/"+algo+".png", dpi=100)
    compute_area_between_curves(df_result)

   
def compute_area_between_curves(df_result):
    y1 = df_result["Predicted"].values
    y2 = df_result["True"].values
    x = df_result.index
    z = y1-y2
    dx = x[1:] - x[:-1]
    cross_test = np.sign(z[:-1] * z[1:])
    x_intersect = x[:-1] - dx / (z[1:] - z[:-1]) * z[:-1]
    dx_intersect = - dx / (z[1:] - z[:-1]) * z[:-1]
    areas_pos = abs(z[:-1] + z[1:]) * 0.5 * dx # signs of both z are same
    areas_neg = 0.5 * dx_intersect * abs(z[:-1]) + 0.5 * (dx - dx_intersect) * abs(z[1:])
    areas = np.where(cross_test < 0, areas_neg, areas_pos)
    total_area = np.sum(areas)
    print ("Area between curves: ", total_area)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    X_train, y_train, X_new, mlb = load_dataset()
     
    #remove outliers
    #X_train = load_data("data/data_all_outliers.txt")
    #y_train = load_labels("data/labels_all_outliers.txt")
    
    #detect only DOF
    #X_train = load_data("data/pairs2/data_all.txt")
    #y_train = load_labels("data/labels_dof_vs_other.txt")
    
    #detect without DOF
    #X_train = load_data("data/data_all_dof.txt")
    #y_train = load_labels("data/labels_all_dof.txt")
    
    #detect without rare classes (OTHER)
    #X_train = load_data("data/pairs2/data_all.txt")
    #y_train = load_labels("data/labels_all_vs_other.txt")
    
    #detect rare classes (OTHER)
    #X_train = load_data("data/data_only_other.txt")
    #y_train = load_labels("data/labels_only_other.txt")
    
    #X_train = load_data("data/new/data_all.txt")
    #y_train = load_labels("data/new/labels_only_other2.txt")
    
    X_train = load_data("data/new/data_all.txt")
    y_train = load_labels("data/new/labels_av_vs_other.txt")
    
    y_train_lat_list = []
    for i in y_train:
        y_train_lat_list.append([i])

    y_train =  mlb.fit_transform(y_train_lat_list) 
 
    X_train = normalize_data(X_train)    
    X_train = patch_detrend(X_train)
       
    X_train = np.array(X_train)
    nsamples00, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples00,nx*ny))        
    
   # from sklearn.decomposition import KernelPCA, PCA
    #kpca = KernelPCA(n_components=2, kernel="cosine", fit_inverse_transform=True)

  #  X_train = kpca.fit_transform(X_train)
  #  X_train = kpca.inverse_transform(X_train)
    
    ###################################  
    X_new = load_data("data/data_new.txt")
    X_new = normalize_data(X_new)    
    X_new = patch_detrend(X_new)    

    X_new = np.array(X_new)
    nsamples22, nx, ny = X_new.shape
    X_new = X_new.reshape((nsamples22,nx*ny))      

  #  X_new = kpca.fit_transform(X_new)    
 #  X_new = kpca.inverse_transform(X_new)    
    ###################################
    X_train = preprocessing.scale(X_train)
    X_new = preprocessing.scale(X_new)
    
    print (np.array(X_train).shape)
    print (np.array(X_new).shape)
    
    svm_mlb(X_train, y_train, X_new, mlb)
    print ("------------------------------------------")
    bagging_rf_mlb(X_train, y_train, X_new, mlb) 
    print ("------------------------------------------")
    rf_mlb(X_train2, y_train, X_new2, mlb)
    print ("------------------------------------------")
    mlp_mlb(X_train, y_train, X_new, mlb) 
    print ("------------------------------------------")  


if __name__ == "__main__":
    main()
