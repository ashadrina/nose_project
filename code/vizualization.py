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
            labels.append("_".join(line.replace("\n","").split(";")))
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

def label_distribution(lat_labels):  
    from collections import Counter
    c = dict(Counter(lat_labels))
    print (c)
    new_c = {"other":0}
    for k, v in c.items():
        if v == 1:
            new_c["other"] += 1
        else:
            new_c[k] = v
    print (new_c)
    # Create a list of colors (from iWantHue)
    colors = ["#229954", "#58D68D", "#82E0AA", "#ABEBC6", "#73C6B6", "#17A589", "#AED6F1"]
    #{'dioktilftalat': 9, 'azetal_degid': 4, 'plastizol_': 2, 'azeton': 3, 'other': 14, 'benzol': 4, 'etilazetat': 4}

    # Create a pie chart
    plt.pie( list(new_c.values()), labels=list(new_c.keys()), shadow=False, colors=colors, startangle=90,  autopct='%1.1f%%')

    # View the plot drop above
    plt.axis('equal')

    # View the plot
    plt.tight_layout()
    #plt.show()
    
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

def plot_pca_lda_2d(X, y):
    colors = {'azetal_degid':"#F2072E", 'azeton':"#ECAEB9", 'benzin':"#F207A8", 'benzol':"#C923C3", 'butanol':"#AA23C9", 'butilazetat':"#4923C9", 'dioktilftalat':"#0A46F8", 
    'dioktilftalat_azetal_degid':"#0ABDF8", 'dioktilftalat_azeton':"#0AF869", 'dioktilftalat_benzol':"#71F9A8", 'dioktilftalat_etilazetat':"#E3F307",  'etilazetat':"#E6EBA0", 
    'fenol':"#F0AF0B", 'geksan':"#DAB34F", 'izobutanol':"#FC5F04", 'izopropanol':"#F9813B", 'plastizol_':"#272727", 'propanol':"#6E3704", 'stirol':"#B08256", 'toluol':"#D8C0AA"}
        
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
  #  plt.clf()
   # plt.cla()   
    
    plt.figure()
    prev_target = ""
    for arr,target_name in zip(X_pca,y):
        color = colors[target_name]
        if target_name == prev_target:
            plt.scatter(arr[0], arr[1], s=20, color=color)
        else:
            plt.scatter(arr[0], arr[1], s=20, color=color, label=target_name)
        prev_target = target_name
    plt.title('PCA of scaled dataset')
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
   # plt.legend(loc='best')

  #  plt.show()
 #   plt.savefig("graphs/pca_lda/pca_norm_detrend_scale.png", dpi = 300, bbox_inches='tight', bbox_extra_artists=(lgd,))
     
#   plt.clf()
#    plt.cla()    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit(X_pca, y).transform(X_pca)
    
    plt.figure()
    prev_target = ""
    for arr,target_name in zip(X_lda,y):
        color = colors[target_name]
        if target_name == prev_target:
            plt.scatter(arr[0], arr[1],s=20, color=color)
        else:
            plt.scatter(arr[0], arr[1],  s=20,color=color, label=target_name)
        prev_target = target_name
    plt.title('LDA of scaled dataset')
   
    lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
#    plt.savefig("graphs/pca_lda/lda_norm_detrend_scale.png", dpi = 300, bbox_inches='tight', bbox_extra_artists=(lgd,))
   
  #  plt.close('all') 
    
def plot_pca_lda_3d(X, y):
    from mpl_toolkits.mplot3d import axes3d
    colors = {'azetal_degid':"#F2072E", 'azeton':"#ECAEB9", 'benzin':"#F207A8", 'benzol':"#C923C3", 'butanol':"#AA23C9", 'butilazetat':"#4923C9", 'dioktilftalat':"#0A46F8", 
    'dioktilftalat_azetal_degid':"#0ABDF8", 'dioktilftalat_azeton':"#0AF869", 'dioktilftalat_benzol':"#71F9A8", 'dioktilftalat_etilazetat':"#E3F307",  'etilazetat':"#E6EBA0", 
    'fenol':"#F0AF0B", 'geksan':"#DAB34F", 'izobutanol':"#FC5F04", 'izopropanol':"#F9813B", 'plastizol_':"#272727", 'propanol':"#6E3704", 'stirol':"#B08256", 'toluol':"#D8C0AA"}
        
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit(X).transform(X)
        
    plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    ax = fig.gca(projection='3d')
    prev_target = ""
    for arr,target_name in zip(X_pca,y):
        color = colors[target_name]
        if target_name == prev_target:
            ax.scatter(arr[0], arr[1], arr[2],  color=color)
        else:
            ax.scatter(arr[0], arr[1], arr[2],  color=color, label=target_name)
        prev_target = target_name
    plt.title('PCA of dataset')

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(n_components=3)
    X_lda = lda.fit(X, y).transform(X)
    
    plt.figure()
    ax = fig.add_subplot(1, 1, 1, axisbg="1.0")
    ax = fig.gca(projection='3d')
    prev_target = ""
    for arr,target_name in zip(X_lda,y):
        color = colors[target_name]
        if target_name == prev_target:
            ax.scatter(arr[0], arr[1], arr[2],  color=color)
        else:
            ax.scatter(arr[0], arr[1], arr[2],  color=color, label=target_name)
        prev_target = target_name
    plt.title('LDA of dataset')
  
    plt.legend()
    plt.show()

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

    
def main():
    X_train = load_data("data/data_train.txt")
    X_train = normalize_data(X_train)
    X_train = patch_detrend(X_train) 
    X_train = np.array(X_train)
    lat_labels_train = np.array(load_labels("data/labels_train.txt"))
    print ("initial data: ", np.array(X_train).shape)
    print ("Train: ", X_train.shape, np.array(lat_labels_train).shape)   
    # vizualize_data(X_train, lat_labels_train, "graphs/matr/train")
    # ###################################3 
    X_test = load_data("data/data_test.txt")
    X_test = normalize_data(X_test)
    X_test = patch_detrend(X_test) 
    X_test = np.array(X_test)
    lat_labels_test = load_labels("data/labels_test.txt")
    print ("initial data: ", np.array(X_test).shape)
    print ("Test: ", X_test.shape, np.array(lat_labels_test).shape)      
    # vizualize_data(X_test, lat_labels_test, "graphs/matr/test")

    y = []
    y.extend(lat_labels_train)
    y.extend(lat_labels_test)
    #label_distribution(y)
    
    X = []
    X.extend(X_train)
    X.extend(X_test) 
    X = np.array(X)
    nsamples00, nx, ny = X.shape
    X = X.reshape((nsamples00,nx*ny))   
    
    from sklearn import preprocessing
    X = preprocessing.scale(X)    
    
    plot_pca_lda_2d(X, y)
    ###########################################   
    # X_new = np.array(load_data("data/data_new.txt"))
    # rus_labels_list = np.array(load_labels("data/new_names.txt"))
    # rus_labels_new = []
    # for lab in rus_labels_list:
        # rus_labels_new.append(lab.replace(" ", "_"))
    # print ("initial data: ", np.array(X_new).shape, np.array(rus_labels_new).shape)
    # print ("New: ", X_new.shape)     
    # print (rus_labels_new)
    #vizualize_data(X_new, rus_labels_new, "graphs/matr/new")

if __name__ == "__main__":
    main()
