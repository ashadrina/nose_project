import os 
import sys
import xlrd
import numpy as np
import matplotlib.pyplot as plt

def load_local_directory(local_path):
	print ("Local path: "+local_path+"...")
	dirs = os.listdir(local_path)
	return dirs	

def load_xls(xls_in):
	book = xlrd.open_workbook(xls_in)
	sheet = book.sheet_by_index(0)

	header = []
	for r in range(0, 5):
		header_row=[]
		for c in range(0,sheet.ncols):
			header_row.append(str(sheet.cell(r,c).value))
		header_row = [rr for rr in header_row if rr]
		header.append(header_row)
	
	sensors= []	
	for r in range(6, 8):
		sensor_row=[]
		for c in range(0,sheet.ncols):
			sensor_row.append(str(sheet.cell(r,c).value))
		sensors.append(sensor_row)	
		
	col_names = []
	for c in range(0,sheet.ncols):
		col_names.append(str(sheet.cell(9, c).value).replace('\u2206', 'd'))
		
	data = []
	for r in range (10, sheet.nrows):
		row = []
		for c in range(0,sheet.ncols):
			row.append(str(sheet.cell(r, c).value))
		row = [x for x in row if x]	
		data.append(row)	
	return header,sensors,col_names,data

def create_structure(data, sensors, in_file, OUT_FILE):
	in_file_name = in_file.replace(" ","_").split(".")[0]
	data = list(zip(*data)) 
	
	txt_outfile = open(OUT_FILE, 'a')
	for se,dat in zip(sensors[0][1:],data[1:]):
		str_name= in_file_name+"_"+se.split(" ")[2].replace("[","").replace("]","")
		print (str_name)
		str_data = ';'.join(list(dat))
		txt_outfile.write(str_name+";"+str_data+"\n")
	
	txt_outfile.close()

def create_labels(in_file, OUT_FILE_LABEL):
	print (in_file.split(".")[0].split(" ")[0])
	txt_outfile = open(OUT_FILE_LABEL, 'a')
	txt_outfile.write(str(in_file.split(".")[0].split(" ")[0])+"\n")
	
###########################	
# python xls_parser.py train train_data.txt
# python xls_parser.py test test_data.txt
#########################	##

def main():
	if len (sys.argv) == 3:
		local_path = sys.argv[1]
		OUT_FILE_DATA =  sys.argv[2]
		
	OUT_FILE_LABELS = "labels_"+OUT_FILE_DATA
	OUT_FILE_DATA = "data_"+OUT_FILE_DATA
	#open(OUT_FILE_DATA, 'w').close()
	open(OUT_FILE_LABELS, 'w').close()
	if os.path.isdir(local_path):
		if local_path == ".":
			local_path = os.path.abspath(os.curdir)
		local_files = load_local_directory(local_path)
		for in_file in local_files:
			header,sensors,col_names,data=load_xls(local_path+"\\\\"+in_file)
			#create_structure(data, sensors, in_file, OUT_FILE_DATA)
			create_labels(in_file, OUT_FILE_LABELS)
			
	# if len (sys.argv) == 2:
		# in_file = sys.argv[1]		
	# header,sensors,col_names,data=load_xls(in_file)
		
	
if __name__ == "__main__":
	main()		