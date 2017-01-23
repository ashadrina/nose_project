import os 
import sys
import numpy as np
import matplotlib.pyplot as plt

def load_data(in_file):
    input_f = open(in_file, "r")
    matrix = []
    for line in input_f:
        channels = [] 
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

def data_to_file(OUT_FILE, lat_labels,  maxlist):
    txt_outfile = open(OUT_FILE, 'w')        
    for lab, ll in zip(lat_labels, maxlist):
        rr = [lab, ":", " ".join([str(l) for l in ll]), "\n"]
        res = ' '.join(str(i) for i in rr)
        txt_outfile.write(res)
    txt_outfile.close()

def vizualize_data(data, labels, folder_name):  
    sensors = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"] 
    for matr,m_name,ind in zip(data, labels, range(len(labels))):
        m = list(map(list, zip(*matr)))
        plt.clf()
        plt.cla()
        plt.figure(figsize=(12,9))
        plt.plot(m)
        plt.legend(sensors, loc='best') 
        plt.title(m_name)
        plt.xlabel('Time (sec)')
        plt.ylabel('dF')
        #plt.show()
        
        if type(m_name) is list:
            mn = "_".join(m_name)
            #print (folder_name+"/"+mn+"_"+str(ind)+".png")
            plt.savefig(folder_name+"/"+mn+"_"+str(ind)+".png")
        else:
            #print (folder_name+"/"+m_name+"_"+str(ind)+".png")
            plt.savefig(folder_name+"/"+m_name+"_"+str(ind)+".png")
        plt.close('all')
        
def main():
    X_train = np.array(load_data("data/data_train.txt"))
    lat_labels_train = np.array(load_labels("data/labels_train.txt"))
    print ("initial data: ", np.array(X_train).shape)
    print ("Train: ", X_train.shape, np.array(lat_labels_train).shape)   
    vizualize_data(X_train, lat_labels_train, "graphs/matr/train")
    ###################################3 
    X_test = np.array(load_data("data/data_test.txt"))
    lat_labels_test = load_labels("data/labels_test.txt")
    print ("initial data: ", np.array(X_test).shape)
    print ("Test: ", X_test.shape, np.array(lat_labels_test).shape)            
    vizualize_data(X_test, lat_labels_test, "graphs/matr/test")
    ############################################   
    X_new = np.array(load_data("data/data_new.txt"))
    rus_labels_list = np.array(load_labels("data/new_names.txt"))
    rus_labels_new = []
    for lab in rus_labels_list:
        rus_labels_new.append(lab.replace(" ", "_"))
    print ("initial data: ", np.array(X_new).shape, np.array(rus_labels_new).shape)
    print ("New: ", X_new.shape)     
    print (rus_labels_new)
    vizualize_data(X_new, rus_labels_new, "graphs/matr/new")

if __name__ == "__main__":
    main()
