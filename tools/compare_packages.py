import numpy as np

list1=np.loadtxt('local_packages.txt',dtype=str)
list2=np.loadtxt('galahad_packages.txt',dtype=str)
print('Package, galahad version, local version')
for i in range(0,list2.shape[0]):
	package2 = list2[i,0] 
	version2 = list2[i,1]
	if package2 in list1:
		ind=list(list1[:,0]).index(package2)
		package1=list1[ind,0] 
		version1=list1[ind,1] 

		print(package2+'=='+version1)