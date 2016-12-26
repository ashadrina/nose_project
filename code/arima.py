import os 
import sys
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

from statsmodels.graphics.api import qqplot

import statsmodels.api as sm    
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import ARResults

import scipy
from statsmodels.iolib.table import SimpleTable

def load_data(in_file):
    input_f = open(in_file, "r")
    matrix = []
    for line in input_f:
        channels = [] #get channels
        for l in line.split("|"):
            samples = l.split(";")
            channels.append([float(i) for i in samples])
        matrix.append(channels)
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
    
def labels_to_int(labels):
    new_labels = []
    for label in labels:
        new_labels.append(int((label.tolist()).index(1)+1))
    return new_labels

def norm_test(X_train, labels, folder_name):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    res = {}
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        plt.clf()
        plt.cla()
        fig, axes = plt.subplots(4, 2)
        fig.tight_layout()
        fig.set_size_inches(18.5, 10.5)
        for i,row in zip(range(0,len(matr),2), axes):
            s1 = matr[i]
            s2 = matr[i+1]
            stats.normaltest(s1)
            stats.normaltest(s2)          
            qqplot(s1, line='q', ax=row[0], fit=True)
            qqplot(s2, line='q', ax=row[1], fit=True)
            row[0].set_title(sensors[i])
            row[1].set_title(sensors[i+1])
        plt.suptitle("Normality test for "+m_name, size=16)
        #plt.show()
        print (folder_name+"/"+m_name+"_"+str(ind)+".png")
        plt.savefig(folder_name+"/"+m_name+"_"+str(ind)+".png", dpi=100)
        plt.close('all')

def jarque_bera_test(X_train, labels):
    res_dict = {}
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        for y in matr:
            lab = m_name+"_"+str(ind)
            jb_test = sm.stats.stattools.jarque_bera(y)
            if lab not in list(res_dict.keys()):
                res_dict[lab] = [list(jb_test)]
            else:
                res_dict[lab].append(list(jb_test))
        
    row =  [u'Chemical', "Sensor", u'JB', u'p-value', u'skew', u'kurtosis'] 
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]  
    chemicals = list(res_dict.keys())
    txt_outfile = open("jarque_bera.txt", 'w')
    txt_outfile.write(";".join(row)+"\n")
    res_dict_ordered = OrderedDict(sorted(res_dict.items(), key=lambda t: t[0]))
    for values,name in zip(list(res_dict_ordered.values()),chemicals):
        for r,s in zip(values,sensors):
            out = [name, s]
            out.extend(r)
            out = [str(o) for o in out]
            txt_outfile.write(";".join(out)+"\n")
    txt_outfile.close()

def jarque_bera_test_analisys(in_file):
    df = pd.read_csv(in_file,";",header=0)
    if len(df.loc[df['p-value'] > 0.05]) > 0:
        print ("There are "+str(len(df.loc[df['p-value'] > 0.05]))+" time series with not normal distribution!")
    
def fit_distribution(X_train, labels, folder_name):
    size = 120
    x = scipy.arange(size)
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        for y in matr:
            h = plt.hist(y, bins=range(120), color='b')
            dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

            for dist_name in dist_names:
                dist = getattr(stats, dist_name)
                param = dist.fit(y)
                pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
                plt.plot(pdf_fitted, label=dist_name)
                plt.xlim(-1,120)
            plt.legend(loc='upper right')
            plt.show()

def autocorr(X_train, labels, folder_name):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        for i in range(len(matr)):
            plt.clf()
            plt.cla()
            fig, axes = plt.subplots(2, 1)
            fig.tight_layout()
            fig.set_size_inches(18.5, 10.5)
            s = matr[i]
            sm.graphics.tsa.plot_acf(s, lags=120, ax=axes[0])
            sm.graphics.tsa.plot_pacf(s, lags=120, ax=axes[1])
            print (folder_name+"/"+m_name+"_"+str(ind)+"_"+sensors[i]+"_"+".png")
            try:
                os.stat(folder_name+"/"+m_name+"_"+str(ind))
            except:
                os.mkdir(folder_name+"/"+m_name+"_"+str(ind)) 
            plt.savefig(folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+"_"+".png", dpi=100)
            plt.close('all')

def test_stationarity(X_train, labels, folder_name):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        for i in range(len(matr)):
            plt.clf()
            plt.cla()
            #Determing rolling statistics
            rolmean = pd.rolling_mean(matr[i], window=12)
            rolstd = pd.rolling_std(matr[i], window=12)
            #Plot rolling statistics:
            fig = plt.figure(figsize=(12, 8))
            orig = plt.plot(matr[i], color='blue',label='Original')
            mean = plt.plot(rolmean, color='red', label='Rolling Mean')
            std = plt.plot(rolstd, color='black', label = 'Rolling Std')
            plt.legend(loc='best')
            plt.title('Rolling Mean & Standard Deviation')
            #plt.show()
            print (folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+"_"+".png",)
            try:
                os.stat(folder_name+"/"+m_name+"_"+str(ind))
            except:
                os.mkdir(folder_name+"/"+m_name+"_"+str(ind)) 
            plt.savefig(folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+"_"+".png", dpi=100)
            plt.close('all')
            #stats_to_file(folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+".txt", dfoutput)

    
def a_dickey_fully_test(X_train, labels, out_file, stat_flag):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    stat = []
    txt_outfile = open(out_file, 'w')
    txt_outfile.write("Chemical;Sensor;UnitRoot;Stationarity\n")
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        if stat_flag == 1: new_matr = [] 
        for i in range(len(matr)):            
            dftest = adfuller(matr[i], autolag='AIC')
            if dftest[0] > dftest[4]['5%']: 
                if stat_flag == 0: print (m_name+": unit roots - YES, stationarity - NO"+"\n")
                adf_res = [m_name+"_"+str(ind+1), sensors[i], "1", "0"]
                if stat_flag == 1: new_matr.append(stationarize(matr[i]))
            else:
                if stat_flag == 0: print (m_name+": unit root - NO, stationarity - YES"+"\n")
                adf_res = [m_name+"_"+str(ind+1), sensors[i], "0", "1"]
                if stat_flag == 1: new_matr.append(matr[i])
            txt_outfile.write(";".join(adf_res)+"\n")
        if stat_flag == 1: stat.append(new_matr)
    txt_outfile.close()
    return stat
  
def a_dickey_fully_test_analisys(in_file):
    df = pd.read_csv(in_file,";",header=0)
    #print (df.loc[df['Stationarity'] != 1 ])
    if len(df.loc[df['Stationarity'] != 1 ]) > 0:
        print ("There are "+str(len(df.loc[df['Stationarity'] != 1]))+" non-stationary time series!")
        return 1
    else:
        return 0
        
def stationarize(vec):
    r = list(np.log(vec))
    return r

def arima_find_best(X_train, labels):
    best_dict = {}
    e = 1
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    print ("ARIMA parameters estimation...")
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        print (m_name)
        for i in range(len(matr)):    
            armodel = ARIMA(matr[i], (0,0,0)).fit(solver='newton')
            best = [(0,0,0), armodel.aic]
            for p in range(0,4) :
                for d in range(0,4) :
                    for q in range(0,4) :
                        try:
                            armodel = ARIMA(matr[i], (p,d,q)).fit(solver='newton')
                            if armodel.aic < best[1]:
                                best = [(p,d,q), armodel.aic]
                        except ValueError:
                            e = 1
                            #print ("AR or MA non stationary")
                        except np.linalg.linalg.LinAlgError:
                            e = 1
                            #print ("Singular matrix")
            k = m_name+"_"+str(ind+1)+sensors[i]
            best_dict[k] = best
            #print ("BEST: ", best)
    return best_dict

def data_to_file(OUT_FILE, lat_labels,  maxlist):
    txt_outfile = open(OUT_FILE, 'w')
    for lab, ll in zip(lat_labels, maxlist):
        rr = [lab, ":", " ".join([str(l) for l in ll]), "\n"]
        res = ' '.join(str(i) for i in rr)
        txt_outfile.write(res)
    txt_outfile.close()

def stats_to_file(OUT_FILE, output):
    txt_outfile = open(OUT_FILE, 'w')
    txt_outfile.write(str(output))
    txt_outfile.close()
    
###########################	
# python clustering.py 
#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/ 
#http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.tsa.arima_model.ARIMA.html
#http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/

#https://university.prognoz.ru/biu/ru/%D0%90%D0%B2%D1%82%D0%BE%D1%80%D0%B5%D0%B3%D1%80%D0%B5%D1%81%D1%81%D0%B8%D1%8F
#https://university.prognoz.ru/biu/ru/Special:History?topicVersionId=807&topic=%D0%90%D0%9A%D0%A4_%D0%B8_%D0%A7%D0%90%D0%9A%D0%A4
#http://www.legalmanager.ru/lems-835-1.html
###########################

def main():
    X_train = np.array(load_data("data/data_train.txt"))
    lat_labels = np.array(load_labels("data/labels_train.txt"))
    # #rus_labels = np.array(load_labels("data/rus/labels_train.txt"))
    # #print (len(set(lat_labels)), len(set(rus_labels)))
    print ("initial data: ", np.array(X_train).shape)
    
    #norm_test(X_train, lat_labels, "graphs/norm")
    #jarque_bera_test(X_train, lat_labels)
    #jarque_bera_test_analisys("jarque_bera.txt")
    ##fit_distribution(X_train[:3], lat_labels, "graphs/distr")
    
    #test_stationarity(X_train, lat_labels, "graphs/stat")
    #print (np.array(X_train).shape)
    #X_train_stat = a_dickey_fully_test(X_train, lat_labels, "adf_protocol.txt", 1)
    #ret = a_dickey_fully_test_analisys("adf_protocol.txt")
    #if ret == 1:
        #print (np.array(X_train_stat).shape)
        #r2 = a_dickey_fully_test(X_train_stat, lat_labels, "adf_protocol_stat.txt", 0)
        #ret2 = a_dickey_fully_test_analisys("adf_protocol_stat.txt")
        #print (ret2)
     
    #autocorr(X_train, lat_labels, "graphs/auto")    
    res = arima_find_best(X_train, lat_labels)
    for k,v in res.items():
        print (k, " - ", v)
    ###################################3 
    #X_test = np.array(load_data("data/data_test.txt"))
    #lat_labels_test = load_labels("data/labels_test.txt")
    #rus_labels_test = np.array(load_labels("data/rus/labels_train.txt"))
    #print (len(set(lat_labels_test)), len(set(rus_labels_test)))
    #print ("initial data: ", np.array(X_test).shape)

    # mlb1 = MultiLabelBinarizer()
    # lat_labels_test.append(lat_labels)
    # bin_labels_test = mlb1.fit_transform(lat_labels_test)
    # y_test = bin_labels_test[:-1]

    # print ("Test: ", X_test.shape, np.array(lat_labels_test).shape)	    
    #  regr_coeff_test = regr(X_test, y_test)
    #  data_to_file("output/regr/test.txt", y_test, regr_coeff_test)
    #############################################   
    #   X_new = np.array(load_data("data/data_new.txt"))
    #   lat_labels_new = np.array(load_labels("data/test_names.txt"))
    #   print ("initial data: ", np.array(X_new).shape, np.array(lat_labels_new).shape)
    #
    #    print ("New: ", X_new.shape)	
    #    regr_coeff_new = regr(X_new)
    #   # data_to_file("output/regr/new.txt", lat_labels_new, newmax)

if __name__ == "__main__":
    main()
