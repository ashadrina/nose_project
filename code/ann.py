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

from sklearn import preprocessing

from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding

#from keras.utils.visualize_util import plot
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils

from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder


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
    ##########################################
    X_new_data = load_data("data/data_new.txt")
    print ("initial data: ", np.array(X_new_data).shape)
    return X_train_big, y_train_lat_big, X_new_data

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
def create_model_fully_connected():
    model = Sequential()
    model.add(Dense(968, input_dim=968, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7)) 
    model.add(Dense(484, activation='sigmoid'))
    model.add(Dropout(0.5)) 
    model.add(Dense(121, activation='sigmoid'))
    model.add(Dropout(0.5)) 
    model.add(Dense(60, activation='sigmoid'))
    model.add(Dropout(0.1)) 
    model.add(Dense(fully_n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    return model
  
def parameter_est_fully_connected(model, X_train, y_train):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)
    batch_size = [4, 10, 20, 40]
    epochs = [10, 30, 50, 70, 100]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    print (np.array(X_train).shape, np.array(dummy_y).shape)
    grid_result = grid.fit(X_train, dummy_y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
def run_model_fully_connected(model, X_train, y_train, X_new):
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y)    

    print ("Training model...")
    model.fit(X_train, dummy_y, nb_epoch=100, batch_size=fully_b_size, verbose=2, shuffle=False)
    y_score = model.predict_classes(X_train)
    print ("Training model... DONE")
    
    #model performance estimation
    train_loss, train_acc = model.evaluate(X_train, dummy_y, batch_size=fully_b_size, verbose=0)
    print ("Train score: "+str(train_acc)+", loss: "+str(train_loss))

    y_new_proba = model.predict_proba(X_new)

    compute_av_metrics(y_train, encoder.inverse_transform(y_score), y_new_proba)
    compute_ind_metrics(y_train, encoder.inverse_transform(y_score), X_new, y_new_proba, model,"fcn") 

##########################################
def get_lookback(a, lsize):
    a=a.tolist()
    new_a = []
    for item in a:
        st=a.index(item)
        if st-lsize < 0:
            arr = []
            for i in range(0, lsize):
                arr.append(a[st-lsize+i])
        else:
            arr=a[st-lsize:st]
        new_a.append(arr)
    return np.asarray(new_a)

def create_model_lstm(): 
    #define model
    model = Sequential()
    model.add(LSTM(121, return_sequences=True, batch_input_shape=(lstm_b_size, lstm_look_back, 968), stateful=True))  # returns a sequence of vectors
   # model.add(LSTM(121, return_sequences=True, batch_input_shape=(lstm_b_size, lstm_look_back, 968), stateful=False))  # returns a sequence of vectors
    model.add(BatchNormalization())
    model.add(Dropout(0.7)) 
    model.add(LSTM(121, batch_input_shape=(lstm_b_size, lstm_look_back, 968), stateful=True, dropout_W=0.4, dropout_U=0.4))  # return a single vector 
   # model.add(LSTM(121, batch_input_shape=(lstm_b_size, lstm_look_back, 968), stateful=False, dropout_W=0.4, dropout_U=0.4))  # return a single vector 
    model.add(Dropout(0.6)) 
    model.add(Dense(lstm_n_classes, activation='sigmoid'))

    print ("Compiling LSTM model...")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ("Compiling LSTM model... DONE")    
    return model
    
def run_lstm_rnn(model, X_train, y_train, X_new):
    epoches = 300
    print ("Preparing data...")
    X_train = get_lookback(X_train, lstm_look_back)
    X_new =  get_lookback(X_new, lstm_look_back)

    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    dummy_y = np_utils.to_categorical(encoded_Y, lstm_n_classes)
    
    print (X_train.shape, dummy_y.shape, X_new.shape)
    print ("Training LSTM model...")
    for i in range(epoches):
        model.fit(X_train, dummy_y, nb_epoch=1, batch_size=lstm_b_size, verbose=2, shuffle=False)
        model.reset_states()
    print ("Training LSTM model... DONE")

    #model performance estimation
    train_loss, train_acc = model.evaluate(X_train, dummy_y, batch_size=lstm_b_size, verbose=0)
    model.reset_states()
    print ("Train score: "+str(train_acc)+", loss: "+str(train_loss))

    #predictions for training and test
    y_score = model.predict_classes(X_train, batch_size=lstm_b_size) 
    y_new_proba = model.predict_proba(X_new, batch_size=lstm_b_size)

    compute_av_metrics(y_train, encoder.inverse_transform(y_score), y_new_proba)
    compute_ind_metrics(y_train, encoder.inverse_transform(y_score), X_new, y_new_proba, model,"lstm") 
##########################################
def predicted_vs_real_label(y_pred, mlb, folder_name):
    import pandas
    y_true = load_labels("true_labels_other.txt")
    y_true = [[x.replace('dioktilftalat','dof') for x in l] for l in y_true]
    nums = range(1,76)
    #folder_name = "graphs/proba/real/"+folder_name
    #folder_name = "graphs/proba/outliers/"+folder_name
    folder_name = "output/"+folder_name
   
    for x_test,y_tr,num in zip(y_pred,y_true, nums):
        y_tr = filter(None, y_tr)
        pred = dict([(c.replace('dioktilftalat','dof'),float("{:.3f}".format(yy))) for c,yy in zip(mlb.classes_,x_test)])      
        vocs = set(pred.keys())
        true = dict([(key, 0.0) for key in vocs])
        for tr in y_tr:
            true[tr] = 1.0

        df_pred = pandas.DataFrame.from_dict(pred, orient='index')
        df_pred.columns=["Predicted"]
        df_true = pandas.DataFrame.from_dict(true, orient='index')
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
        # plt.show()

        try:
         os.stat(folder_name)
        except:
         os.mkdir(folder_name) 
        plt.savefig(folder_name+"/"+str(num)+".png", dpi=100)
        plt.close('all')

def compute_av_metrics(y_train, y_score, y_new_proba):        
    ytrain = []
    for i in y_train:
        ytrain.append([i])   
        
    yscore = []
    for i in y_score:
        yscore.append([i])    

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    ytrain =  mlb.fit_transform(ytrain)
    yscore = mlb.transform(yscore) 
    
    y_test_true_labels = load_labels("true_labels_other.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    for y_pred,y_tr,i in zip(y_new_proba,y_test_true_labels, range(len(y_new_proba))):
        print (i+1)
        r1 = [(c,"{:.3f}".format(yy)) for c,yy in zip(mlb.classes_,y_pred)]
        sorted_by_second_1 = sorted(r1, key=lambda tup: tup[1], reverse=True)
        print (sorted_by_second_1[:6])
        print (set(y_tr))
        print ("------------------")

    y_test_true_labels = load_labels("true_labels_other.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
   # y_test_true_labels.append(['other2'])
    y_test_true = mlb.fit_transform(y_test_true_labels)       
    y_test_true = list(y_test_true)[:-1]
    
    from sklearn.metrics import coverage_error
    err1 = coverage_error(ytrain, yscore)
    print ("You should predict top ",err1, " labels for train")
    err2 = coverage_error(y_test_true, y_new_proba)
    print ("You should predict top ",err2, " labels for toys")

    from sklearn.metrics import label_ranking_average_precision_score
    rap1 = label_ranking_average_precision_score(ytrain, yscore) 
    print ("label_ranking_average_precision_score on train", rap1)
    rap2 = label_ranking_average_precision_score(y_test_true, y_new_proba) 
    print ("label_ranking_average_precision_score on toys", rap2)
  
def compute_ind_metrics(y_train, y_score, X_new, probabilities, clf, algo):
    ytrain = []
    for i in y_train:
        ytrain.append([i])   
        
    yscore = []
    for i in y_score:
        yscore.append([i])    

    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    ytrain =  mlb.fit_transform(ytrain) 
    yscore = mlb.transform(yscore) 
    
    y_test_true_labels = load_labels("true_labels_other.txt")
    y_test_true_labels = [list(filter(None, lab)) for lab in y_test_true_labels]
    y_test_true_labels.append(['benzin'])
    #y_test_true_labels.append(['other'])
    y_test_true = mlb.fit_transform(y_test_true_labels) 
    y_test_true = list(y_test_true)[:-1]

    from sklearn.metrics import coverage_error
    #err1 = coverage_error(ytrain, yscore)
    #print ("You should predict top ",err1, " labels for train")
    from sklearn.metrics import label_ranking_average_precision_score
    #rap1 = label_ranking_average_precision_score(ytrain, yscore)
    #print ("label_ranking_average_precision_score on train", rap1)
        
    true = {}
    pred = {}
    y_new = []
    for i in range(len(X_new)):
       # print (i+1)
        y_new_proba = probabilities[i]
        y_new.append(y_new_proba)
        err2 = coverage_error([y_test_true[i]], [y_new_proba])
       # print ("Compounds in real solution: ", len(set(y_test_true_labels[i])))
       # print ("You should predict top ",err2, " labels for toys")
        rap2 = label_ranking_average_precision_score([y_test_true[i]], [y_new_proba])
      #  print ("label_ranking_average_precision_score on toys", rap2)
      #  print ("--------------------------------------------------------------")
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
    plt.savefig("graphs/metrics/real/"+algo+".png", dpi=100)
    #plt.savefig("graphs/metrics/outliers/"+algo+".png", dpi=100)
    plt.savefig("output/"+algo+".png", dpi=100)
    compute_area_between_curves(df_result)
    #predicted_vs_real_label(y_new, mlb, algo)
    
    #predicted_vs_real_label(y_new_proba, mlb, algo)

    
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
    
def main():
    X_train, y_train, X_new = load_dataset()
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
    
    X_train = load_data("data/new/data_all.txt")
    y_train = load_labels("data/new/labels_only_other2.txt")
    
    #X_train = load_data("data/new/data_all.txt")
    #y_train = load_labels("data/new/labels_av_vs_other.txt")

    X_train = normalize_data(X_train)    
    X_train = patch_detrend(X_train)    
    
    X_train = np.array(X_train)
    nsamples00, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples00,nx*ny))       
    
    ###################################  
    X_new = load_data("data/data_new.txt")
    X_new = normalize_data(X_new)    
    X_new = patch_detrend(X_new)    
  
    X_new = np.array(X_new)
    nsamples22, nx, ny = X_new.shape
    X_new = X_new.reshape((nsamples22,nx*ny))      

    ###################################
    X_train = preprocessing.scale(X_train)
    X_new = preprocessing.scale(X_new)

    print (np.array(X_train).shape)
    print (np.array(X_new).shape)
    ###################################
    global fully_b_size
    #fully_b_size = 5
    fully_b_size = 5 #detect without DOF
    global fully_n_classes
    fully_n_classes = len(set(y_train))
    
    model = create_model_fully_connected()
    #parameter_est_fully_connected(model, X_train, y_train)
    run_model_fully_connected(model, X_train, y_train, X_new)

    global lstm_look_back
    lstm_look_back = 5
    global lstm_b_size
    #lstm_b_size = 5
    lstm_b_size = 5 #detect without DOF
    global lstm_n_classes
    lstm_n_classes = len(set(y_train))
    
    model = create_model_lstm()
    run_lstm_rnn(model, X_train, y_train, X_new)
    
if __name__ == "__main__":
    main()
