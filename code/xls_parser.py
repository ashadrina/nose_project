import os 
import sys
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import re 

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

	str_data = []
	for dat in data[1:]:
		str_data.append(';'.join(list(dat)))
	
	res = '|'.join(list(str_data))
	txt_outfile.write(res+"\n")
	
	txt_outfile.close()

def cyrillic2latin(input):
	symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
           u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")

	tr = {ord(a): ord(b) for a, b in zip(*symbols)}
	return input.translate(tr)		
	
def create_train_labels(in_file, OUT_FILE_LABEL):
	print (in_file.split(".")[0].split(" ")[0])
	#print (header[0][1])
	txt_outfile = open(OUT_FILE_LABEL, 'a')
	cyr_label = str(in_file.split(".")[0].split(" ")[0])
	#cyr_label = str(header[0][1])
	res = cyrillic2latin(cyr_label.replace("\n",""))
	txt_outfile.write(res+"\n")
	txt_outfile.close()
    
def create_new_labels(header, OUT_FILE_LABEL):
	print (header[0][1])
	txt_outfile = open(OUT_FILE_LABEL, 'a')
	cyr_label = str(header[0][1])
	txt_outfile.write(cyr_label+"\n")
	txt_outfile.close()
	
def create_test_labels(header, OUT_FILE_LABEL):
	head_str = header[0][1]
	h1 = head_str.split("и")[0].strip()
	h2 = head_str.split("и")[1].strip().split(" ")[0]
	res = []
	if "ДОФ" in h1:
		res.append(cyrillic2latin("диоктилфталат"))
	elif "этац" in h1:
		res.append(cyrillic2latin("этилацетат"))
	elif "ац-д" in h1:
		res.append(cyrillic2latin("ацетальдегид"))
	else:
		res.append(cyrillic2latin(h1))
		
	if "ДОФ" in h2:
		res.append(cyrillic2latin("диоктилфталат"))
	elif "этац" in h2:
		res.append(cyrillic2latin("этилацетат"))
	elif "ац-д" in h2:
		res.append(cyrillic2latin("ацетальдегид"))
	else:
		res.append(cyrillic2latin(h2))
	
	txt_outfile = open(OUT_FILE_LABEL, 'a')
	txt_outfile.write(str(res[0])+";"+str(res[1])+"\n")	
	print ("create_val_labels: ", res)

###########################	
# python xls_parser.py train train.txt
# python xls_parser.py test test.txt
# python xls_parser.py val val.txt
#########################	##

def main():
    if len (sys.argv) == 3:
        local_path = sys.argv[1]
        OUT_FILE_DATA =  sys.argv[2]

    OUT_FILE_LABELS = "labels_"+OUT_FILE_DATA
    OUT_FILE_DATA = "data_"+OUT_FILE_DATA
    open(OUT_FILE_DATA, 'w').close()
    open(OUT_FILE_LABELS, 'w').close()
    if os.path.isdir(local_path):
        if local_path == ".":
            local_path = os.path.abspath(os.curdir)
        local_files = load_local_directory(local_path)
        if "new" in local_path:
            local_files = sorted(local_files, key=lambda x: (int(re.sub('\D','',x)),x))
        print (local_files)
        for in_file in local_files:
            header,sensors,col_names,data=load_xls(local_path+"\\\\"+in_file)
            print (header[0][1])
            if (np.array(data).shape) != (121,9):
                print("PANIC!!! ", (np.array(data).shape) )
            else:
                create_structure(data, sensors, in_file, OUT_FILE_DATA)
                if "train" in local_path:
                    create_train_labels(in_file, OUT_FILE_LABELS)				
                    #create_train_labels(header, OUT_FILE_LABELS)				
                if "test" in local_path:
                    create_test_labels(header, OUT_FILE_LABELS)	                
                if "new" in local_path:
                    create_new_labels(header, OUT_FILE_LABELS)	
	
	
if __name__ == "__main__":
	main()		
