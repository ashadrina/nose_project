import numpy as np
from sklearn.preprocessing import  MultiLabelBinarizer

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

def hamming_score(y_true, y_pred):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len( set_true.intersection(set_pred) ) / float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

if __name__ == "__main__":

    y_true = load_labels("output/true_labels_final.txt")
    
    lens = []
    for y in y_true:
        y = list(filter(None, y))
        lens.append(len(y))
        
    print (lens)
    print (max(lens))
    
    y_pred = load_labels("output/group_final.txt")
    
    mlb = MultiLabelBinarizer()
    y_true =  mlb.fit_transform(y_true) 
    y_pred =  mlb.transform(y_pred) 
    
    print ("Groups:")
    
    print (np.array(y_true).shape)
    print (np.array(y_pred).shape)
    
    print('Hamming score: {0}'.format(hamming_score(y_true, y_pred))) 

    # Subset accuracy: 
    #1 if the prediction for one sample fully matches the gold
    # 0 otherwise.
    import sklearn.metrics
    print('Subset accuracy: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred)))

    # Hamming loss (smaller is better)
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 
    
    #######################################################
    y_true = load_labels("output/true_labels_final.txt")
    y_pred = load_labels("output/dataset_final.txt")
    
    mlb = MultiLabelBinarizer()
    y_true =  mlb.fit_transform(y_true) 
    y_pred =  mlb.transform(y_pred) 
    
    print (np.array(y_true).shape)
    print (np.array(y_pred).shape)

    print ("Dataset:")
    print('Hamming score: {0}'.format(hamming_score(y_true, y_pred))) 
    
    # Subset accuracy: 
    #1 if the prediction for one sample fully matches the gold
    # 0 otherwise.
    import sklearn.metrics
    print('Subset accuracy: {0}'.format(sklearn.metrics.accuracy_score(y_true, y_pred)))

    # Hamming loss (smaller is better)
    print('Hamming loss: {0}'.format(sklearn.metrics.hamming_loss(y_true, y_pred))) 