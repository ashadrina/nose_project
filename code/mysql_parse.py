import os 
import sys

import numpy as np
import pymysql
from lxml import etree
import collections

import xlwt

def get_measures():
	conn = pymysql.connect(host='localhost', user='root', passwd='root', db='sniff', charset='utf8')
	cur = conn.cursor()
	cur.execute("SELECT ID FROM Measures WHERE FullLength=120")
	mids = []
	for r in cur:
		mids.append(r)
	cur.close()
	conn.close()
	return mids

def get_header(MeasureID):
	conn = pymysql.connect(host='localhost', user='root', passwd='root', db='sniff', charset='utf8')
	cur = conn.cursor()
	header = {}

	query = "SELECT Name FROM Measures WHERE ID=%s"
	cur.execute(query, (MeasureID,))
	for r in cur:
		header["name"] = r[0]
	
	query = "SELECT FullLength FROM Measures WHERE ID=%s"
	cur.execute(query, (MeasureID,))
	for r in cur:
		header["length"] = r[0]
		
	query = "SELECT StartTime FROM Measures WHERE ID=%s"
	cur.execute(query, (MeasureID,))
	for r in cur:
		header["start"] = r[0]
		
	print ("Header: ", header)	
	cur.close()
	conn.close()
	
def get_sensors(sids):
	conn = pymysql.connect(host='localhost', user='root', passwd='root', db='sniff', charset='utf8')
	cur = conn.cursor()
	sid_dict = {}
	for sid in sids:
		query = "SELECT SID,Settings FROM Sensors WHERE ID=%s"
		cur.execute(query, (sid, ))
		for r in cur:
			root = etree.fromstring(r[1])
			textelem = root.find('mainfreq')
			sid_dict[r[0]] = float(str(textelem.text).replace(",","."))
	print ("Sensors: ", sid_dict)
	cur.close()
	conn.close()	
	return sid_dict
	
def get_data(MeasureID): 
	conn = pymysql.connect(host='localhost', user='root', passwd='root', db='sniff', charset='utf8')
	cur = conn.cursor()
	query = "SELECT DISTINCT(SensorID) FROM Data WHERE MeasureID=%s"
	cur.execute(query, (MeasureID,))
	sid = []
	for r in cur:
		sid.append(int(r[0]))
	print (MeasureID, ": ", sid)
	
	data = []
	for s in sid:	
		query = "SELECT FreqValue FROM Data WHERE MeasureID=%s AND SensorID= %s"
		cur.execute(query, (MeasureID, s,))
		dd = []
		for r in cur:
			dd.append(int(r[0]))
		data.append(dd)
		
	print (np.array(data).shape)

	cur.close()
	conn.close()
	return data, sid

def generate_time_matrix(data):
	#ordrd = collections.OrderedDict(sid_dict)
	s1 = data[1][0]
	#for i in data[1][1:]:
	#	print (int(s1) - i)
	#timeseries = range(0,120)
	#data.insert(0, timeseries)
	return data
	
#def form_xls():
	

def main():
	if len (sys.argv) == 2:
		outfile = sys.argv[1]

	print (outfile)
	MeasureIDs=get_measures()
	data = []
	
	for MeasureID in MeasureIDs:
		get_header(MeasureID)
		data,sids = get_data(MeasureID)
		sid_dict = get_sensors(sids)
		data = generate_time_matrix(data)
		#print (np.array(data).shape)
		print ("_--------------_")
		
if __name__ == "__main__":
	main()
