import numpy as np
import sys
import argparse

def main(argv):
	mainlist=argv[0]
	nsplit=int(argv[1])
	
	data=np.genfromtxt(mainlist,dtype=str,delimiter='\n')

	split=np.array_split(data,nsplit)

	i=1
	for s in split:
		print(s)
		filename='run'+str(i)+'.list'
		i+=1
		np.savetxt(filename,s,fmt='%s')

	
if __name__ == "__main__":
	main(sys.argv[1:])