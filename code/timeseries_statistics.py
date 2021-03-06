import os 
import sys
import numpy as np
import pandas as pd
import scipy
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict

import statsmodels.api as sm  
import statsmodels.tsa.stattools
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA

import networkx


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
        try:
            os.stat(folder_name)
        except:
            os.mkdir(folder_name) 
        print (folder_name+"/"+m_name+"_"+str(ind)+".png")
        plt.savefig(folder_name+"/"+str(ind)+"_"+m_name+".png", dpi=100)
        plt.close('all')

def jarque_bera_test(X_train, labels,outname):
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
    txt_outfile = open(outname, 'w')
    txt_outfile.write(";".join(row)+"\n")
    res_dict_ordered = OrderedDict(sorted(res_dict.items()))
    for name,values in res_dict_ordered.items():
        for r,s in zip(values,sensors):
            out = [name, s]
            out.extend(r)
            out = [str(o) for o in out]
            txt_outfile.write(";".join(out)+"\n")
    txt_outfile.close()

def jarque_bera_test_analisys(in_file):
    df = pd.read_csv(in_file,";",header=0, encoding="cp1251")
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
            print (folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+".png")
            try:
                os.stat(folder_name)
            except:
                os.mkdir(folder_name) 
                
            try:
                os.stat(folder_name+"/"+m_name+"_"+str(ind))
            except:
                os.mkdir(folder_name+"/"+m_name+"_"+str(ind)) 
            plt.savefig(folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+".png", dpi=100)
            plt.close('all')

def autocorr_radius(X_train, labels, folder_name):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    row =  [u'Chemical', "Sensor", u'Radius'] 
    outfile = open(folder_name, 'w')
    outfile.write(";".join(row)+"\n")
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        for i in range(len(matr)):
            autocorr=statsmodels.tsa.stattools.acf(matr[i], nlags=120)   
            radius = 0
            for item,num in zip(autocorr,range(1,len(autocorr))):
                if (item > 0):
                    continue
                else:
                    radius = num - 1
                    break
            res = m_name+"_"+str(ind)+";"+sensors[i]+";"+str(radius) + "\n"
            outfile.write(res)
    outfile.close()       
     
def cross_corr(X_train, labels, folder_name):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        plt.clf()
        plt.cla()
        corr = np.corrcoef(matr)
        graph = networkx.from_numpy_matrix(corr, create_using=networkx.Graph())
        
        labels = {}
        for node,o_name in zip(graph.nodes(),sensors):
            labels[node] = o_name

        pos = networkx.circular_layout(graph)

        networkx.draw_networkx_nodes(graph, pos, node_size=800, node_color='r', alpha=0.8)
        elarge = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] > 0.9]
        emid = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] > 0.3 and  d['weight'] <= 0.9 ]
        esmall = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] > 0 and d['weight'] <= 0.3]
        enlarge = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] >= -0.3 and d['weight'] < 0]
        enmid = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] >= -0.9 and  d['weight'] < -0.3 ]
        ensmall = [(u,v) for (u,v,d) in graph.edges(data=True) if d['weight'] < -0.9]

        #plot edges
        networkx.draw_networkx_edges(graph, pos, edgelist = elarge, width = 6, edge_color = '#8B0000')
        networkx.draw_networkx_edges(graph, pos, edgelist = emid, width = 5, edge_color = '#FF4500')
        networkx.draw_networkx_edges(graph, pos, edgelist = esmall, width = 4, alpha = 0.5, edge_color='#FF7F50')
        networkx.draw_networkx_edges(graph, pos, edgelist = enlarge, width = 3, edge_color = '#40E0D0')
        networkx.draw_networkx_edges(graph, pos, edgelist = enmid, width = 2, edge_color = '#0000FF')
        networkx.draw_networkx_edges(graph, pos, edgelist = ensmall, width = 1, alpha = 0.5, edge_color='#000080')

        # labels
        networkx.draw_networkx_labels(graph,pos,labels,font_size=20,font_color='white')

        plt.title(str(m_name)+", "+str(ind), fontsize=14, fontweight='bold')
        plt.axis('off')

        max_pos_patch = mpatches.Patch(color='#8B0000')
        mid_pos_patch = mpatches.Patch(color='#FF4500')
        min_pos_patch = mpatches.Patch(color='#FF7F50')
        max_neg_patch = mpatches.Patch(color='#000080')
        mid_neg_patch = mpatches.Patch(color='#0000FF')
        min_neg_patch = mpatches.Patch(color='#40E0D0')

        lgd = plt.legend([max_pos_patch, mid_pos_patch, min_pos_patch, min_neg_patch, mid_neg_patch, max_neg_patch], ['corr > 0.9','0.3 < corr <= 0.9','0 < corr <= 0.3','-0.3 <= corr < 0', '-0.9 <= corr < -0.3', 'corr < -0.9'], loc='center left', bbox_to_anchor=(1, 0.5))
        print (folder_name+"/"+m_name+"_"+str(ind)+".jpeg")
        try:
            os.stat(folder_name)
        except:
            os.mkdir(folder_name) 
        plt.savefig(folder_name+"/"+str(ind)+"_"+m_name+".png", dpi=300, format='jpeg', bbox_extra_artists=(lgd,), bbox_inches='tight')
        #plt.show()
        plt.close('all')

def fit_polynom(X_train, labels, N, folder_name):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        plt.clf()
        plt.cla()
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
            plt.plot (y_new, lw=2)
        plt.legend(sensors, loc='best') 
        plt.title(m_name+"_"+str(ind)) 
        plt.xlabel('Time (sec)')
        plt.ylabel('dF')
        print (folder_name+"/"+m_name+"_"+str(ind)+".png")
        try:
            os.stat(folder_name)
        except:
            os.mkdir(folder_name) 
        plt.savefig(folder_name+"/"+str(ind)+"_"+m_name+".png", dpi = 300, bbox_inches='tight')#, pad_inches=0)
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
            print (folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+".png")
            try:
                os.stat(folder_name)
            except:
                os.mkdir(folder_name) 
            try:
                os.stat(folder_name+"/"+m_name+"_"+str(ind))
            except:
                os.mkdir(folder_name+"/"+m_name+"_"+str(ind)) 
            plt.savefig(folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+".png", dpi=100)
            plt.close('all')
            #stats_to_file(folder_name+"/"+m_name+"_"+str(ind)+"/"+sensors[i]+".txt", dfoutput)

    
def a_dickey_fully_test(X_train, labels, out_file, stat_flag):
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    #stat = []
    txt_outfile = open(out_file, 'w')
    txt_outfile.write("Chemical;Sensor;UnitRoot;Stationarity\n")
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        if stat_flag == 1: new_matr = [] 
        for i in range(len(matr)):            
            dftest = adfuller(matr[i], autolag='AIC')
            if dftest[0] > dftest[4]['5%']: 
                if stat_flag == 0: print (m_name+": unit roots - YES, stationarity - NO"+"\n")
                adf_res = [m_name+"_"+str(ind), sensors[i], "1", "0"]
                if stat_flag == 1: new_matr.append(stationarize(matr[i]))
            else:
                if stat_flag == 0: print (m_name+": unit root - NO, stationarity - YES"+"\n")
                adf_res = [m_name+"_"+str(ind), sensors[i], "0", "1"]
                if stat_flag == 1: new_matr.append(matr[i])
            txt_outfile.write(";".join(adf_res)+"\n")
        if stat_flag == 1: stat.append(new_matr)
    txt_outfile.close()
    #return stat
  
def a_dickey_fully_test_analisys(in_file):
    df = pd.read_csv(in_file,";",header=0, encoding="cp1251")
    if len(df.loc[df['Stationarity'] != 1 ]) > 0:
        print ("There are "+str(len(df.loc[df['Stationarity'] != 1]))+" non-stationary time series!")
        return 1
    else:
        return 0
        
def stationarize(vec):
    r = list(np.log(vec))
    return r

def arima_find_best(X_train, labels, out_file):
    e = 0
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    print ("ARIMA parameters estimation...")
    txt_outfile = open(out_file, 'w')
    txt_outfile.write("Chemical;Sensor;p;d;q;AIC\n")
    txt_outfile.close()
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        print (m_name)
        for i in range(len(matr)):    
            armodel = ARIMA(matr[i], (0,0,0)).fit(solver='newton')
            best = [(0,0,0), armodel.aic]
            del armodel
            for p in range(0,3):
                for d in range(0,2):
                    for q in range(0,3):
                        try:
                            armodel = ARIMA(matr[i], (p,d,q)).fit(solver='newton')
                            if armodel.aic < best[1]:
                                best = [[p,d,q], armodel.aic]
                        except ValueError:
                            e = 1
                            #print ("AR or MA non stationary")
                        except np.linalg.linalg.LinAlgError:
                            e = 2
                            #print ("Singular matrix")
            output = [m_name+"_"+str(ind), sensors[i], str(best[0][0]), str(best[0][1]), str(best[0][2]), str(best[1])]
            txt_outfile = open(out_file, 'a')
            txt_outfile.write(";".join(output)+"\n")
            txt_outfile.close()
            del best
            del armodel
            
def arima_radius(X_train, labels, radius_file, out_file):
    e = 0
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    print ("ARIMA parameters estimation...")
    txt_outfile = open(out_file, 'w')
    txt_outfile.write("Chemical;Sensor;p;d;q;AIC\n")
    txt_outfile.close()
    for matr,m_name,ind in zip(X_train,labels,range(len(labels))):
        print (m_name)
        for i in range(len(matr)):    
            armodel = ARIMA(matr[i], (0,0,0)).fit(solver='newton')
            best = [(0,0,0), armodel.aic]
            del armodel
            for p in range(0,3):
                for d in range(0,2):
                    for q in range(0,3):
                        try:
                            armodel = ARIMA(matr[i], (p,d,q)).fit(solver='newton')
                            if armodel.aic < best[1]:
                                best = [[p,d,q], armodel.aic]
                        except ValueError:
                            e = 1
                            #print ("AR or MA non stationary")
                        except np.linalg.linalg.LinAlgError:
                            e = 2
                            #print ("Singular matrix")
            output = [m_name+"_"+str(ind), sensors[i], str(best[0][0]), str(best[0][1]), str(best[0][2]), str(best[1])]
            txt_outfile = open(out_file, 'a')
            txt_outfile.write(";".join(output)+"\n")
            txt_outfile.close()
            del best
            del armodel
    
def generate_summary(hb, adf, rad, ar, outfile):  
    hb_df = pd.read_csv(hb,";",header=0, encoding="cp1251")
    hb_df['Normality'] = np.where(hb_df['p-value'] > 0.05, 1, 0)
    hb_df = hb_df.drop('JB', 1)
    hb_df = hb_df.drop('p-value', 1)
    hb_df = hb_df.drop('skew', 1)
    hb_df = hb_df.drop('kurtosis', 1)
    adf_df = pd.read_csv(adf,";",header=0, encoding="cp1251")
    adf_df = adf_df.drop('Chemical', 1)
    adf_df = adf_df.drop('Sensor', 1)
    adf_df = adf_df.drop('UnitRoot', 1)
    rad_df = pd.read_csv(rad,";",header=0, encoding="cp1251")
    rad_df = rad_df.drop('Chemical', 1)
    rad_df = rad_df.drop('Sensor', 1)
    ar_df = pd.read_csv(ar,";",header=0, encoding="cp1251")
    ar_df = ar_df.drop('Chemical', 1)
    ar_df = ar_df.drop('Sensor', 1)
    frames = [hb_df, adf_df, rad_df, ar_df]
    result = pd.concat(frames, axis=1, join='inner')
    result.to_csv(outfile, sep=';', encoding='utf-8')
    
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
###########################

def main():
    # X_train = np.array(load_data("data/data_train.txt"))
    # lat_labels_train = np.array(load_labels("data/labels_train.txt"))
    # print ("initial data: ", X_train.shape)
    
    # armodel = ARIMA(X_train[0][0], (43,1,0)).fit(solver='newton')
    # print ("AIC", armodel)
    #cross_corr(X_train, lat_labels_train, "graphs/crosscorr/train")
    #fit_polynom(X_train, lat_labels_train, 5, "graphs/poly/train")
    #norm_test(X_train, lat_labels_train, "graphs/norm/train")
    #jarque_bera_test(X_train, lat_labels_train, "output/stats/jarque_bera_train.txt")
    #jarque_bera_test_analisys("output/stats/jarque_bera_train.txt")   
    #a_dickey_fully_test(X_train, lat_labels_train, "output/stats/adf_protocol_train.txt", 0)
    #ret = a_dickey_fully_test_analisys("output/stats/adf_protocol_train.txt")
    #autocorr(X_train, lat_labels_train, "graphs/auto/train")    
    #autocorr_radius(X_train, lat_labels_train, "output/stats/radius_train.txt")
    #arima_find_best(X_train, lat_labels_train, "output/stats/arima_est_train.txt")
    #test_stationarity(X_train, lat_labels_train, "graphs/stat/train")
    
    ##fit_distribution(X_train[:3], lat_labels_train, "graphs/distr") #DO NOT UNCOMMENT     
 
    #generate_summary("output/stats/jarque_bera_train.txt", "output/stats/adf_protocol_train.txt", "output/stats/radius_train.txt", "output/stats/arima_est_train.txt", "output/stats/res_table_train.csv")
    ###################################
    # X_test = np.array(load_data("data/data_test.txt"))
    # lat_labels_list = load_labels("data/labels_test.txt")
    # rus_labels_test = np.array(load_labels("data/rus/labels_test.txt"))
    # print ("initial data: ", np.array(X_test).shape)

    # lat_labels_test = []
    # for item in lat_labels_list:
        # lat_labels_test.append("-".join(item))
    # cross_corr(X_test, lat_labels_test, "graphs/crosscorr/test")
    # fit_polynom(X_test, lat_labels_test, 5, "graphs/poly/test")
    # norm_test(X_test, lat_labels_test, "graphs/norm/test")
    # jarque_bera_test(X_test, lat_labels_test,"output/stats/jarque_bera_test.txt")
    # jarque_bera_test_analisys("output/stats/jarque_bera_test.txt")   
    # a_dickey_fully_test(X_test, lat_labels_test, "output/stats/adf_protocol_test.txt", 0)
    # ret = a_dickey_fully_test_analisys("output/stats/adf_protocol_test.txt")
    # autocorr(X_test, lat_labels_test, "graphs/auto/test")    
    # autocorr_radius(X_test, lat_labels_test,"output/stats/radius_test.txt")
    # arima_find_best(X_test, lat_labels_test, "output/stats/arima_est_test.txt")
    # test_stationarity(X_test, lat_labels_test, "graphs/stat/test")
    # generate_summary("output/stats/jarque_bera_test.txt", "output/stats/adf_protocol_test.txt", "output/stats/radius_test.txt", "output/stats/arima_est_test.txt", "output/stats/res_table_test.csv")

    #############################################   
    X_new = np.array(load_data("data/data_new.txt"))
    rus_labels_list = np.array(load_labels("data/labels_new.txt"))
    rus_labels_new = []
    for lab in rus_labels_list:
        rus_labels_new.append(lab.replace(" ", "_"))
        
    print ("initial data: ", np.array(X_new).shape, np.array(rus_labels_new).shape)
    print ("New: ", X_new.shape)     
    # cross_corr(X_new, rus_labels_new, "graphs/crosscorr/new")
    # fit_polynom(X_new, rus_labels_new, 5, "graphs/poly/new")
    # norm_test(X_new, rus_labels_new, "graphs/norm/new")
    # jarque_bera_test(X_new, rus_labels_new,"output/stats/jarque_bera_new.txt")
    # jarque_bera_test_analisys("output/stats/jarque_bera_new.txt")   
    # a_dickey_fully_test(X_new, rus_labels_new, "output/stats/adf_protocol_new.txt", 0)
    # ret = a_dickey_fully_test_analisys("output/stats/adf_protocol_new.txt")
    # autocorr(X_new, rus_labels_new, "graphs/auto/new")    
    # autocorr_radius(X_new, rus_labels_new, "output/stats/radius_new.txt")
    # #arima_find_best(X_new, rus_labels_new, "output/stats/arima_est_new.txt")
    # test_stationarity(X_new, rus_labels_new, "graphs/stat/new")     
    # #generate_summary("output/stats/jarque_bera_new.txt", "output/stats/adf_protocol_new.txt", "output/stats/radius_new.txt", "output/stats/arima_est_new.txt", "output/stats/res_table_new.csv")

if __name__ == "__main__":
    main()
