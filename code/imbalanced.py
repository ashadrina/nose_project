import os 
import sys
import numpy as np
import copy 
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
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

def load_dataset(voc):
    X_train = load_data("data/pairs2/"+voc+".txt")
    y = load_labels("data/pairs2/"+voc+"_labels.txt")
    print ("initial data: ", np.array(X_train).shape)
    print ("initial data: ", Counter(y))
    X_train = normalize_data(X_train)
    X_train = patch_detrend(X_train) 
    return X_train, y

def load_new_objects():
    X_new = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new).shape)
    X_new = normalize_data(X_new)
    X_new = patch_detrend(X_new) 
    return X_new

def normalize_data(data):
    norm_matrix = []
    for block in data:
        norm_col = []
        for col in block:
            current_mean = np.mean(col)
            surrent_std = np.std(col)
            norm_col.append([(float(i) - current_mean)//surrent_std for i in col])
        norm_matrix.append(norm_col)
    return norm_matrix

def bagging_rf(X,y):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import train_test_split as tts
    from sklearn.utils import shuffle
    bc = BaggingClassifier(base_estimator=RandomForestClassifier(bootstrap=True, 
                class_weight=None, criterion='gini',
                max_depth=None, max_features=10, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False), max_samples=0.2, n_estimators=27)

    X_train, X_test, y_train, y_test = tts(X, y, train_size=0.6, random_state=42)
    bc.fit(X_train, y_train)
    y_new = bc.predict(X_test)
    
    #for x,xx in zip(y_test, y_new):
    #    print (x, xx)
    err_train = np.mean(y_test != y_new)
    print ("Bagging with Random Forest accuracy: ", 1 - err_train)
    
def bagging_rf_testing(X,y, new_data, pairs):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.cross_validation import train_test_split as tts
    from sklearn.utils import shuffle
    from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
    
    bc = BaggingClassifier(base_estimator=OneVsRestClassifier(RandomForestClassifier(bootstrap=True, 
                class_weight=None, criterion='gini',
                max_depth=None, max_features=10, max_leaf_nodes=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)), max_samples=0.2, n_estimators=55)
    #bc = BaggingClassifier(base_estimator=SVC(C=0.1, kernel='linear', degree=3, 
                        #gamma='auto', coef0=0.0, shrinking=False, 
                        #probability=True, tol=0.001, cache_size=200, 
                        #class_weight=None, verbose=False, max_iter=-1, 
                        #decision_function_shape=None, random_state=42), 
                        #max_samples=0.2, n_estimators=27)

    bc.fit(X, y)
    print(bc.classes_)  
    
    y_new = bc.predict_proba(new_data)  
    for x in y_new:
        print (x)
        #res = []
        #for p,v in zip(x, pairs):
        #    res.append((v[0],"%.4f" % p))
        #sorted_by_second_1 = sorted(res, key=lambda tup: tup[1], reverse=True)
        #print (sorted_by_second_1)
        
        
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
    

def resampling_data(X, y):
    from imblearn.over_sampling import ADASYN 
    RANDOM_STATE = 42
    ada = ADASYN(random_state=RANDOM_STATE)
    X_res, y_res = ada.fit_sample(X, y)
    print('Resampled dataset shape {}'.format(Counter(y_res)))
    return X_res, y_res 

def viz_svm_2d(X_init,y, voc):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn import svm
    
    pca = PCA(n_components=2)
    X = pca.fit_transform(X_init)
    
    C = 1.0  # SVM regularization parameter
    h = 4  # step size in the mesh

    print ("linear")
    svc = svm.SVC(kernel='linear', C=C).fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
    print ("rbf")
    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
    print ("poly")
    bagging_rf = svm.LinearSVC(C=C).fit(X, y)

    # create a mesh to plot in
    print (np.array(X).shape)
    x_min, x_max = X[:, 0].min() - 15, X[:, 0].max() + 15
    y_min, y_max = X[:, 1].min() - 15, X[:, 1].max() + 15
    print (x_min, x_max)
    print (y_min, y_max)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    print ("ready to plot")
    for i, clf in enumerate((svc, bagging_rf, rbf_svc, poly_svc)):
        print ("doiing "+titles[i]+" graph")
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
        plt.xlabel('PCA1')
        plt.ylabel('PCA2')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
        print ("doing "+titles[i]+"- DONE")
    plt.suptitle(voc)
    plt.show() 

def svm_cl_testing(X_train, y_train, X_test):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MultiLabelBinarizer
    
    pca = PCA(n_components=121)
    pca.fit_transform(X_train)        
    pca.fit_transform(X_test)   
    
    mlb = MultiLabelBinarizer()
    print (y_train)
    y_train_tmp = []
    for i in y_train:
        y_train_tmp.append([i])
    y_train_bin = mlb.fit_transform(y_train_tmp)
    
    svc = OneVsRestClassifier(SVC(kernel='rbf')).fit(X_train, y_train_bin)
    y_score =  svc.predict(X_train) 
    err_train = np.mean(y_train_bin != y_score)
    print ("svm train accuracy: ", 1 - err_train)
    
    y_true_nums = encode_labels(pairs, "true_labels.txt")
    #y_true_words = decode_labels(pairs,y_true_nums)
    
    y_new_proba = svc.predict(X_test)
    for i in y_new_proba:
        print (i)
    #for y_new,y_tr in zip(y_new_proba,y_true_nums):
        #r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,y_new)]
        #sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        #print (sorted_by_second_1)
    #    print (mlb.inverse_transform(y_new), " - ", y_tr)

def encode_labels(pairs, path_to_file):
    true_labels = load_labels(path_to_file)
    d = {'dioktilftalat':0}
    for item in pairs:
        if "dof_dof" in item[0]:
            v = item[0].split("_")
            voc = "dioktilftalat_with_"+str(v[2])
        else:
            voc = item[0].split("_")[1]
        d[voc] = item[1]
        
    labels_encoded = []
    for item in true_labels:
        item = list(filter(None, item))
        if len(item) == 1:
            labels_encoded.append([d[item[0]]])
        else:
            s = []
            for v in item:
                s.append(d[v])
            labels_encoded.append(s)
    return labels_encoded
    
def decode_labels(pairs,nums):
    d = {0:'dioktilftalat'}
    for item in pairs:
        if "dof_dof" in item[0]:
            v = item[0].split("_")
            voc = "dioktilftalat_with_"+str(v[2])
        else:
            voc = item[0].split("_")[1]
        d[item[1]] = voc
    labels_decoded = []
    for item in nums:
        if type(item) is int:
            labels_decoded.append(d[item])
        else:
            s = []
            for v in item:
                s.append(d[v])
            labels_decoded.append(s)
    return labels_decoded

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


def data_to_file(X_train):
    X = X_train.reshape(X_train.shape[0], 121,8)
    to_write = []
    for matr in X:
        new_matr = []
        matr = list(zip(*matr)) 
        for item in matr:
            new_matr.append(";".join([str("%.2f" % i) for i in item]))
        to_write.append("|".join(new_matr))
        
    txt_outfile = open("data/data_train_over.txt", 'w')
    for res in to_write: 
        txt_outfile.write(res+"\n")
    txt_outfile.close()

def labels_to_file(y):
    txt_outfile = open("data/labels_train_over.txt", 'w')
    for res in y: 
        txt_outfile.write(res+"\n")
    txt_outfile.close()    

def main():
    cols = ["t"+str(i) for i in range(0,968)]
    cols.append("y")
    generated_data = pd.DataFrame(columns=cols)
    label_index = 1
    first = 1
    global pairs
    pairs = []
    dfs = []
    for dirname, dirnames, filenames in os.walk('data/pairs2/'):
        for filename in filenames:
        #   if "_labels" not in filename and "dof" in filename:
           if "_labels" not in filename:
                voc = filename.split(".")[0]
                print (voc)
                pairs.append((voc,label_index))
                X_train_full, y_labels_train_full = load_dataset(voc)
                X_train_full = np.array(X_train_full)
                nsamples00, nx, ny = X_train_full.shape
                X_train_full_2d_init = X_train_full.reshape((nsamples00,nx*ny))        
                print (np.array(X_train_full_2d_init).shape)
                X_res, y_res = resampling_data(X_train_full_2d_init, y_labels_train_full)
                X_res = preprocessing.scale(X_res)
                #viz_svm_2d(X_res, y_res, voc)
                data = np.hstack((X_res, y_res.reshape(np.array(y_res).shape[0], 1)))
                if first == 1:
                    df1 = pd.DataFrame(data,columns=cols)
                    df1.loc[df1['y']=='0'] = 0
                    df1.loc[df1['y']=='1'] = 1
                    dfs.append(df1)
                    label_index = 2
                    first = 0
                else:
                    df = pd.DataFrame(data,columns=cols)
                    df1 = df.loc[df['y']=='1']
                    df1.loc[df1['y']=='1'] = label_index
                    dfs.append(df1)
                    label_index += 1
                
    generated_data = pd.concat(dfs)
    y_train = list(generated_data['y'].values)
    X_train = list(generated_data.drop('y', axis = 1).values)
    
    #X_train = preprocessing.scale(X_train)
    
    #print (pairs)
    #print (y_train)
    
    y_train_lat = decode_labels(pairs,y_train)
    
    #data_to_file(X_train)
    #labels_to_file(y_train_lat)
    
    #bagging_rf(X_train,y_train)
        
    X_new = load_new_objects()
    X_new = np.array(X_new)
    nsamples00, nx, ny = X_new.shape
    X_new_2d = X_new.reshape((nsamples00,nx*ny)) 
    #X_new_2d = preprocessing.scale(X_new_2d)
    print (np.array(X_train).shape)       
    print (np.array(X_new_2d).shape)
    
    d = [('dioktilftalat', 0)]
    for item in pairs:
        if "dof_dof" in item[0]:
            v = item[0].split("_")
            voc = "dioktilftalat_with_"+str(v[2])
        else:
            voc = item[0].split("_")[1]
        d.append((voc, item[1]))
    
    print (d)
    bagging_rf_testing(X_train, y_train, X_new_2d, d)
    # #svm_cl_testing(X_train, y_train, X_new_2d)

if __name__ == "__main__":
    main()
