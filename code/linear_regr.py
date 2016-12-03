import os 
import sys
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

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

#####################################
def regr(X, y):
    lr = linear_model.LinearRegression()
    for x in list(X):
        for xx in x:
            lr.fit(xx, y)
            print('Coefficients: \n', lr.coef_)

##########################

def data_to_file(OUT_FILE, lat_labels, maxlist):
    txt_outfile = open(OUT_FILE, 'w')	
    for lab, ll in zip(lat_labels, maxlist):
        rr = [lab, ":", " ".join([str(l) for l in ll]), "\n"]
        res = ' '.join(str(i) for i in rr)
        txt_outfile.write(res)
    txt_outfile.close()

###########################	
# python clustering.py 
#https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/ 
#http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.tsa.arima_model.ARIMA.html
#http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
#########################	##

def main():
    X_train = np.array(load_data("data_train.txt"))
    lat_labels = np.array(load_labels("labels_train.txt"))
    print (len(set(lat_labels)))
    print ("initial data: ", np.array(X_train).shape)

    ##transform labels
    mlb = LabelBinarizer()
    y_train = mlb.fit_transform(lat_labels) 

    print ("Train: ", X_train.shape, np.array(y_train).shape)   
    regr_coeff_train = regr(X_train, y_train)
    #   data_to_file("regr/train.txt", y_train, regr_coeff_train)
    ###################################3 
    X_test = np.array(load_data("data_val.txt"))
    lat_labels_test = np.array(load_labels("labels_val.txt"))
    print ("initial data: ", np.array(X_test).shape)

    mlb1 = MultiLabelBinarizer()
    lat_labels_test.append(lat_labels)
    bin_labels_test = mlb1.fit_transform(lat_labels_val)
    y_test = bin_labels_test[:-1]

    print ("Test: ", X_test.shape, y_test.shape)	    
    regr_coeff_test = regr(X_test, y_test)
    #  data_to_file("regr/test.txt", y_test, regr_coeff_test)
    #############################################   
    #   X_new = np.array(load_data("data_test.txt"))
    #   lat_labels_new = np.array(load_labels("test_names.txt"))
    #   print ("initial data: ", np.array(X_new).shape, np.array(lat_labels_new).shape)
    #
    #    print ("New: ", X_new.shape)	
    #    regr_coeff_new = regr(X_new)
    #   # data_to_file("regr/new.txt", lat_labels_new, newmax)
	
if __name__ == "__main__":
	main()		