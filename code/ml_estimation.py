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

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing

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
    
########################################## 
 
def model_evaluation(X_train, y_train): 
    kfold = KFold(n=len(X_train), shuffle=True, n_folds=7, random_state=42)

    # # Prepare some basic models
    # models = []
    # models.append(('LogRegr', OneVsRestClassifier(LogisticRegression())))
    # models.append(('Knn', OneVsRestClassifier(KNeighborsClassifier())))
    # models.append(('NaiveBayess', OneVsRestClassifier(GaussianNB())))
    # models.append(('RFC', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_features=10))))
    # models.append(('SVM', OneVsRestClassifier(SVC(probability=True))))
    # models.append(('ExtraTrees', OneVsRestClassifier(ExtraTreesClassifier())))
    # models.append(('AdaBoost', OneVsRestClassifier(AdaBoostClassifier())))

    # Evaluate each model in turn
    # for name, model in models:
        # cv_results = cross_val_score(model, X_train, y_train, cv=kfold)#, scoring=scoring, n_jobs=1)
        # print("{0}: ({1:.3f}) +/- ({2:.3f})".format(name, cv_results.mean(), cv_results.std()))
        
    return kfold

def estimation_log_reg(X_train, y_train, kfold):
    print ("parameter estimation for Logistic Regression")
    from sklearn.grid_search import GridSearchCV
    lr_grid = GridSearchCV(
        estimator = LogisticRegression(random_state=5),
        param_grid = { 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 1, 10, 100] }, 
        cv = kfold, 
        n_jobs = 1)

    lr_grid.fit(X_train, y_train)

    print("best score: ", lr_grid.best_score_)
    print("best params: ", lr_grid.best_params_)

#best score:  0.9235836627140975
#best params:  {'penalty': 'l2', 'C': 1}

def run_log_reg(X_train, y_train, X_new, y_new):
	clf = LogisticRegression(random_state=5, C=1, penalty='l2')
	clf.fit(X_train, y_train)
	y_score = clf.predict(X_new)
	err_train = np.mean(y_new != y_score)
	print ("LR train accuracy: ", 1 - err_train)
	
	#from sklearn.metrics import confusion_matrix
	#cnf_matrix_train = confusion_matrix(y_new, y_score)    

	#np.set_printoptions(precision=2)
	#class_names = list(set(y_new))

	#plt.figure()
	#plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
	#plt.show()

def estimation_svm(X_train, y_train, kfold):    
    print ("parameter estimation for SVM")  
    from sklearn.svm import SVC, NuSVC  
    from sklearn.grid_search import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    Gammas = [0.001, 0.01, 1, 10]
    kernels = ['rbf','linear', 'sigmoid']
    decision_function = ['ovo', 'ovr']
    coefs = [0.0, 0.5, 1.0, 10.0, 100.0]
    class_weight = ['balanced', 'auto']
    Cs = [0.001, 0.01, 1, 10]

   # svc_grid = GridSearchCV(estimator = OneVsRestClassifier(NuSVC(nu=0.001, probability=True)), 
    #    param_grid = {'estimator__C': Cs, 'estimator__gamma': Gammas, 'estimator__kernel': kernels, 
    #    'estimator__decision_function_shape': decision_function,
    #    'estimator__coef0': coefs, 'estimator__class_weight': class_weight },
    #    cv = kfold, n_jobs = 1)    
        
    svc_grid = GridSearchCV(estimator = OneVsRestClassifier(SVC(probability=True)), 
        param_grid = {'estimator__C': Cs, 'estimator__gamma': Gammas, 'estimator__kernel': kernels, 
        'estimator__decision_function_shape': decision_function,
        'estimator__coef0': coefs, 'estimator__class_weight': class_weight },
        cv = kfold, n_jobs = 1)
 
    svc_grid.fit(X_train, y_train)

    print ("best score:", svc_grid.best_score_)
    print ("best params: ", svc_grid.best_params_)
    
def estimation_rf(X_train, y_train, kfold):    
    print ("parameter estimation for RF")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    n_est = [7, 15, 25, 50, 101, 1001]#, 3000, 5000, 7000]
    criterion = ['gini', 'entropy']
    

    svc_grid = GridSearchCV(OneVsRestClassifier(estimator = RandomForestClassifier(class_weight='balanced')), 
        param_grid = {'estimator__n_estimators': n_est, 'estimator__criterion' : criterion },
        cv = kfold, n_jobs = 1)
 
    svc_grid.fit(X_train, y_train)

    print ("best score:", svc_grid.best_score_)
    print ("best params: ", svc_grid.best_params_)
    
def estimation_bagging(X_train, y_train, kfold):    
    print ("parameter estimation for Bagging")
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.grid_search import GridSearchCV
    from sklearn.multiclass import OneVsRestClassifier
    n_est = [7, 15, 25, 50, 100]

    svc_grid = GridSearchCV(estimator = BaggingClassifier(base_estimator=RandomForestClassifier()), 
        param_grid = {'estimator__n_estimators': n_est },
        cv = kfold, n_jobs = 1)
 
    svc_grid.fit(X_train, y_train)

    print ("best score:", svc_grid.best_score_)
    print ("best params: ", svc_grid.best_params_)


def run_svm(X_train, y_train, X_new, y_new):
	clf = SVC(random_state=7, C=1, gamma=0.001,kernel='rbf')
	clf.fit(X_train, y_train)
	y_score = clf.predict(X_new)
	err_train = np.mean(y_new != y_score)
	print ("LR train accuracy: ", 1 - err_train)
	
	#from sklearn.metrics import confusion_matrix
	#cnf_matrix_train = confusion_matrix(y_new, y_score)    

	#np.set_printoptions(precision=2)
	#class_names = list(set(y_new))

	#plt.figure()
	#plot_confusion_matrix(cnf_matrix_train, classes=class_names, title='Confusion matrix train')
	#plt.show()

   
 
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
    
   # X_train = load_data("data/new/data_all.txt")
   # y_train = load_labels("data/new/labels_av_vs_other.txt")
   
    # X_train = load_data("data/new/data_all.txt")
    # y_train = load_labels("data/new/labels_only_other2.txt")
    
    # y_train_lat_list = []
    # for i in y_train:
        # y_train_lat_list.append([i])

    # y_train =  mlb.fit_transform(y_train_lat_list) 
 

    # X_train = normalize_data(X_train)    
    # X_train = patch_detrend(X_train)    

    # X_train = np.array(X_train)
    # nsamples00, nx, ny = X_train.shape
    # X_train = X_train.reshape((nsamples00,nx*ny))        

    # X_new = normalize_data(X_new)    
    # X_new = patch_detrend(X_new)    

    # X_new = np.array(X_new)
    # nsamples22, nx, ny = X_new.shape
    # X_new = X_new.reshape((nsamples22,nx*ny))        

    # X_train = preprocessing.scale(X_train)
    # X_new = preprocessing.scale(X_new)

    # print (np.array(X_train).shape)
    # print (np.array(X_new).shape)

    #y_train_lat_labels = load_labels("data/labels_train.txt")
    #y_test_lat_labels = load_labels("data/labels_test.txt")
    #y_test_lat_labels = ["_with_".join(i) for i in y_test_lat_labels]    

    #y_train = []
    #y_train.extend(y_train_lat_labels)
    #y_train.extend(y_test_lat_labels)
    
    #########################
    
    #y_train_over = load_labels("data/labels_train_over.txt")
    #X_train_over = load_data("data/data_train_over.txt")
    #X_train_over = np.array(X_train_over)
    #nsamples00, nx, ny = X_train_over.shape
    #X_train_over = X_train_over.reshape((nsamples00,nx*ny))   
    #print (np.array(X_train_over).shape)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1, whiten=True)

    X_train2 = []
    for X in X_train:
        pca.fit(X)
        X2 = pca.transform(X)
        X_train2.append(X2)    
         
    X_new2 = []
    for X in X_new:
        pca.fit(X)
        X2 = pca.transform(X)
        X_new2.append(X2)
    
    X_train2 = np.array(X_train2)
    nsamples00, nx, ny = X_train2.shape
    X_train2 = X_train2.reshape((nsamples00,nx*ny))        
    
    X_new2 = np.array(X_new2)
    nsamples00, nx, ny = X_new2.shape
    X_new2 = X_new2.reshape((nsamples00,nx*ny))      
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage = 'auto')
    y_train2 = load_labels("data/labels_all.txt")
    clf.fit_transform(X_train2, y_train2)
    
    print ("Parameter estimation for train:")
    kfold = model_evaluation(X_train2, y_train2)  
    estimation_svm(X_train2, y_train2, kfold)
    #estimation_rf(X_train, y_train, kfold)
    #y_test_true_labels = load_labels("true_labels_other.txt")
    #y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    ##y_test_true_labels.append(['benzin'])
    ##y_test_true_labels.append(['other2'])
    #y_test_true =  mlb.transform(y_test_true_labels) 
    #y_new = list(y_test_true)#[:-1]  
        
    #X = []
    #X.extend(list(X_train))
    #X.extend(list(X_new))
    #Y = []
    #Y.extend(list(y_train))
    #Y.extend(list(y_new))
    #print ("Parameter estimation for all:")
   ## X = np.array(X)
   ## Y = np.array(Y)
   ## print (np.array(X).shape, np.array(Y).shape)
    #kfold = model_evaluation(X, Y)  
    ##estimation_log_reg(X_train_over, y_train_over, kfold)
    #estimation_svm(X, Y, kfold)
    #estimation_rf(X, Y, kfold)
   ## estimation_bagging(X, Y, kfold)

    ##run_log_reg(X_train_over, y_train_over, X_train, y_train)
    ##run_svm(X_train_over, y_train_over, X_train, y_train)
    
if __name__ == "__main__":
    main()
