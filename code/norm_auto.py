#!/usr/bin/python
import numpy
import sys
from scipy import signal


if __name__ == "__main__":
	if len (sys.argv) == 3:
      		in_file = sys.argv[1]
      		out_file = sys.argv[2]
	else:
		print ("Error")
		sys.exit (1)

print(in_file)
print(out_file)
#read file to list
infile1 = open(in_file, 'r')
s1 = [int(line) for line in infile1.readlines()]
infile1.close()

#list to array
s1 = numpy.array(s1)

#normalization
s1 = (s1 - numpy.mean(s1)) / (numpy.std(s1) * len(s1))

#computing normalised AUTOcorrelation
corr = numpy.correlate(s1, s1, mode='same')

#output in file
outfile = open(out_file, 'w')
i=0
for item in corr:
  string = str(float(item)) + ";" + str(i) + "\n" 
  outfile.write(string)
  i+=0.004

outfile.close()
