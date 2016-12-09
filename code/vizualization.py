import os 
import sys
import numpy as np

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
    for matr,label,ind in zip(data,labels,range(len(data))):
        for x,si in zip(matr,range(len(matr))):
            plt.clf()
            plt.plot(x)
            if type(label) is list and len(label) == 2:   
                print ("yes!")
                plt.title("_".join(label)+"_"+str(ind)+" sensor "+str(si))  
                plt.savefig(folder_name+"/"+"_".join(label)+"_"+str(ind)+"_sensor_"+str(si)+".png")
            else:
                plt.title(label+"_"+str(ind)+" sensor "+str(si))  


def main():
    X_train = np.array(load_data("data/data_train.txt"))
    lat_labels = np.array(load_labels("data/labels_train.txt"))
    print ("initial data: ", np.array(X_train).shape)
    print ("Train: ", X_train.shape, np.array(y_train).shape)   
    vizualize_data(X_train, lat_labels, "graphs/data/train")
    ###################################3 
    X_test = np.array(load_data("data/data_test.txt"))
    lat_labels_test = load_labels("data/labels_test.txt")
    print ("initial data: ", np.array(X_test).shape)
    print ("Test: ", X_test.shape, np.array(lat_labels_test).shape)	    
    vizualize_data(X_test, lat_labels_test, "graphs/data/test")
    #############################################   
    #   X_new = np.array(load_data("data/data_new.txt"))
    #   lat_labels_new = np.array(load_labels("data/test_names.txt"))
    #   print ("initial data: ", np.array(X_new).shape, np.array(lat_labels_new).shape)
    #   print ("New: ", X_new.shape)	
    #   vizualize_data(X_new, lat_labels_new, "graphs/data/new")
	
if __name__ == "__main__":
	main()		
