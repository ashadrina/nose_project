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
    
def bagging_rf_mlb(X_train, y_train, new_data, mlb):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    
    bc = BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, 
                class_weight=None, criterion='gini',
                max_depth=None, max_features=10, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False), max_samples=0.2, n_estimators=55)

    bc.fit(X_train, y_train)
    y_score =  bc.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("bagging train accuracy: ", 1 - err_train)
        
    y_test_true_labels = load_labels("true_labels.txt")
    y_new_proba = bc.predict_proba(new_data)  
    for x_test,y_tr in zip(y_new_proba,y_test_true_labels[:3]):
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,x_test)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        # print (sorted_by_second_1)
        # print (y_tr)
        # print ("-----------------------------------")
        
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
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
    
def mlp_mlb(X_train, y_train, X_new, mlb):
    from sklearn.neural_network import MLPClassifier
    alphas = np.logspace(-5, 3, 5)
    names = []
    for i in alphas:
        names.append('alpha ' + str(i))

    classifiers = []
    for i in alphas:
        classifiers.append(MLPClassifier(alpha=i, random_state=1))
        
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        print ("score: ", score)
        
        y_new_proba = clf.predict_proba(X_new)
        
        y_test_true_labels = load_labels("true_labels.txt")
        y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
        y_test_true_labels.append(['benzin'])
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
               
def svm_mlb(X_train, y_train, X_test, mlb):
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    
    svc = OneVsRestClassifier(SVC(kernel='rbf', probability=True))
    svc.fit(X_train, y_train)
    y_score =  svc.predict(X_train) 
    err_train = np.mean(y_train != y_score)
    print ("svm train accuracy: ", 1 - err_train)
    
    y_test_true_labels = load_labels("true_labels.txt")
    y_new_proba = svc.predict_proba(X_test)
    for x_test,y_tr in zip(y_new_proba,y_test_true_labels):
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,x_test)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        #print (sorted_by_second_1)
       # print (y_tr)
       # print ("-----------------------------------")
        
    y_test_true_labels = load_labels("true_labels.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
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
   
    #y_new = svc.predict(X_test)
    #array_np = np.asarray(y_new)
    #low_values_indices = array_np < 0.01  # Where values are low
    #array_np[low_values_indices] = 0  # All low values set to 0
    #y_labels = mlb.inverse_transform(y_new)
    #txt_outfile = open("2svm_new_labels_prob.txt", 'w')
    #txt_outfile.write(";".join(mlb.classes_)+"\n")
    #for y in y_new:
        # # if y and type(y) is tuple:
        #txt_outfile.write(";".join("{:.3f}".format(i)  for i in list(y))+"\n")
        # # elif y and type(y) is str:
           # # txt_outfile.write(y+"\n")
        # # else:
            # # txt_outfile.write("?\n")
    # txt_outfile.close() 
   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
   # X_train, y_train, X_new, mlb = load_dataset()
   # print ("initial data: ", np.array(X_new).shape)
   
    X_train = load_data("data/data_all_outliers.txt")
    y_train_lat= load_labels("data/labels_all_outliers.txt")
    
    y_train = []
    for i in y_train_lat:
        y_train.append([i])

    mlb = MultiLabelBinarizer()
    y_train =  mlb.fit_transform(y_train) 

    X_train = normalize_data(X_train)    
    X_train = patch_detrend(X_train)    

    X_train = np.array(X_train)
    nsamples00, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples00,nx*ny))        
    ################################
    X_new = load_data("data/data_new.txt")

    X_new = normalize_data(X_new)    
    X_new = patch_detrend(X_new)    

    X_new = np.array(X_new)
    nsamples22, nx, ny = X_new.shape
    X_new = X_new.reshape((nsamples22,nx*ny))        
    ################################
    X_train = preprocessing.scale(X_train)
    X_new = preprocessing.scale(X_new)
    ################################
    print (np.array(X_train).shape)
    print (np.array(X_new).shape)
    
    svm_mlb(X_train, y_train, X_new, mlb) 
    #bagging_rf_mlb(X_train, y_train, X_new, mlb) #do not work
    #mlp_mlb(X_train, y_train, X_new, mlb) #do not work
    
      
if __name__ == "__main__":
    main()
    
    
