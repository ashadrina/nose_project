# -*- coding: utf-8 -*-

def cyrillic2latin(input):
	symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
           u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")

	tr = {ord(a): ord(b) for a, b in zip(*symbols)}
	return input.translate(tr)		

def load_labels(in_file):
    input_f = open(in_file, "r", encoding='utf-8')
    labels = []
    for line in input_f:
        labels.append(line.replace("\n",""))
    input_f.close()
    return labels
    
labels = load_labels("misclassification.txt")
lat_labels = []
for l in labels:
    lat_labels.append(cyrillic2latin(l.replace(" ", "_")))
    
from collections import Counter

print (Counter(lat_labels))