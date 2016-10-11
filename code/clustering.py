import os 
import sys
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelBinarizer
from transliterate import translit #perform transliteration from russian labels to latin

from numpy.linalg import svd #data reducion

from sklearn.cluster import KMeans #n clusters NOT required

from sklearn.cluster import DBSCAN #n clusters NOT required
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from sklearn.cluster import MeanShift, estimate_bandwidth #n clusters NOT required

from sklearn.cluster import AffinityPropagation #n clusters NOT required
from itertools import cycle

#visualization
import matplotlib.pyplot as plt
 
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
		labels.append(line.replace("\n",""))
	input_f.close()
	return labels
	
	
def labels_to_int(labels):
	new_labels = []
	for label in labels:
		if ";" in label:
			l = label.split(";")
			new_labels.append([int((l[0].tolist()).index(1)+1), int((l[0].tolist()).index(1)+1)])
		else:
			new_labels.append(int((label.tolist()).index(1)+1))
	return new_labels	

def label_matching(bin_labels,labels):
	pairs = []
	for y,l in zip(bin_labels,labels):
		pairs.append((l,y))
	return pairs	
	
def normalize_data(data):
	norm_matrix = []
	for block in data:
		norm_col = []
		for col in block:
			norm_col.append([float(i)/sum(col) for i in col])
		norm_matrix.append(norm_col)
	return norm_matrix
	
def get_avg(data):
	mean_matrix = []
	for block in data:
		mean_block = []
		for col in block:
			mean_block.append(np.mean(col))
		mean_matrix.append(mean_block)
	return mean_matrix

def get_min(data):
	min_matrix = []
	for block in data:
		min_block = []
		for col in block:
			min_block.append(min(col))
		min_matrix.append(min_block)
	return min_matrix

def get_max(data):
	max_matrix = []
	for block in data:
		max_block = []
		for col in block:
			max_block.append(max(col))
		max_matrix.append(max_block)
	return max_matrix	
	
def reduce_data(data):		
	new_data = []
	for matr in data:
		new_data.append(matr[:1])
	return new_data	

#########################################	
def run_svd(data):
	new_data = []
	for matr in data:
		U, s, V = svd(np.array(matr), full_matrices=True)
		s_s = [s[0]]
		s_s_1 = [0.0] * 7
		s_s.extend(s_s_1)
		S = np.zeros((8, 121), dtype=float)	
		S[:8, :8] = np.diag(s_s)
		new_data.append(S)	
	return new_data	
	
def aff_prop(data,int_labels,lat_labels):	
	X = []
	for dat in data:
		X.extend(dat)
	X = np.array(X)
		
	af = AffinityPropagation(preference=-50).fit(X)
	cluster_centers_indices = af.cluster_centers_indices_
	labels = af.labels_
	
	n_clusters_ = len(cluster_centers_indices)

	print("* Number of classes: %d" %len(set(int_labels)))
	print('* Estimated number of clusters: %d' % n_clusters_)
	print("* Homogeneity: %0.3f" % metrics.homogeneity_score(int_labels, labels))
	print("* Completeness: %0.3f" % metrics.completeness_score(int_labels, labels))
	print("* V-measure: %0.3f" % metrics.v_measure_score(int_labels, labels))
	print("* Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(int_labels, labels))
	print("* Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(int_labels, labels))
	print("* Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean'))

	plt.close('all')
	plt.figure(1)
	plt.clf()
	
	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
		class_members = labels == k
		cluster_center = X[cluster_centers_indices[k]]
		plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
		plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)
		for x in X[class_members]:
			plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

	plt.title('Estimated number of affinity propagation clusters: %d' % n_clusters_)
#	plt.show()	  
	plt.savefig("aff_prop.png")
	
def dbscan(data,int_labels,lat_labels):	
	X = []
	for dat in data:
		X.extend(dat)
	X = np.array(X)
	
	X = StandardScaler().fit_transform(X)
	
	db = DBSCAN(eps=0.3, min_samples=10).fit(X)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	print("* Number of classes: %d" %len(set(int_labels)))
	print('* Estimated number of clusters: %d' % n_clusters_)
	print("* Homogeneity: %0.3f" % metrics.homogeneity_score(int_labels, labels))
	print("* Completeness: %0.3f" % metrics.completeness_score(int_labels, labels))
	print("* V-measure: %0.3f" % metrics.v_measure_score(int_labels, labels))
	print("* Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(int_labels, labels))	  
	print("* Adjusted Mutual Information: %0.3f"  % metrics.adjusted_mutual_info_score(int_labels, labels))
	print("* Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
	
	plt.close('all')
	plt.figure(1)
	plt.clf()
	
	# Black removed and is used for noise instead.
	unique_labels = set(labels)
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	for k, col in zip(unique_labels, colors):
		if k == -1:
			# Black used for noise.
			col = 'k'

		class_member_mask = (labels == k)

		xy = X[class_member_mask & core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)

		xy = X[class_member_mask & ~core_samples_mask]
		plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=6)

	plt.title('Estimated number of DBSCAN clusters: %d' % n_clusters_)
#	plt.show()
	plt.savefig("dbscan.png")
	
def mean_shift(data,int_labels,lat_labels):	
	X = []
	for dat in data:
		X.extend(dat)
	X = np.array(X)
	
	# The following bandwidth can be automatically detected using
	bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

	ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_

	labels_unique = np.unique(labels)
	n_clusters_ = len(labels_unique)

	print("* Number of classes: %d" %len(set(int_labels)))
	print('* Estimated number of clusters: %d' % n_clusters_)	
	print("* Homogeneity: %0.3f" % metrics.homogeneity_score(int_labels, labels))
	print("* Completeness: %0.3f" % metrics.completeness_score(int_labels, labels))
	print("* V-measure: %0.3f" % metrics.v_measure_score(int_labels, labels))
	print("* Adjusted Rand Index: %0.3f"  % metrics.adjusted_rand_score(int_labels, labels))	  
	print("* Adjusted Mutual Information: %0.3f"  % metrics.adjusted_mutual_info_score(int_labels, labels))
	print("* Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
	
	plt.close('all')
	plt.figure(1)
	plt.clf()

	colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
	for k, col in zip(range(n_clusters_), colors):
		my_members = labels == k
		cluster_center = cluster_centers[k]
		plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
		plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
				 markeredgecolor='k', markersize=14)
	plt.title('Estimated number of Mean Shift clusters: %d' % n_clusters_)
#	plt.show()
	plt.savefig("mean_shift.png")
	
###########################	
# python clustering.py train_data2.txt
#########################	##

def main():
	if len (sys.argv) == 3:
		data_file = sys.argv[1]
		labels_file = sys.argv[2]

	print (data_file, labels_file)
	data = load_data(data_file)
	lat_labels = load_labels(labels_file)

	#norm_data = normalize_data(data)
	
	#transform labels
	lb = LabelBinarizer()
	bin_labels = lb.fit_transform(lat_labels)
	int_labels = labels_to_int(bin_labels)

	print ("initial data: ", np.array(data).shape)
	ress = run_svd(data)
	new_ress = reduce_data(ress)
	print ("reduced data: ", np.array(new_ress).shape)

	print ("AFFINITY PROPAGAITION")
	aff_prop(new_ress,int_labels,lat_labels)
	print ("==================")
	print ("DBSCAN")
	dbscan(new_ress,int_labels,lat_labels)
	print ("==================")
	print ("MEAN SHIFT")
	mean_shift(new_ress,int_labels,lat_labels)
	
if __name__ == "__main__":
	main()		