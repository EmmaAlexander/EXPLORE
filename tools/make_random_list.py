import numpy as np 
import pandas as pd

#set random seed for reproducasbility
np.random.seed(seed=0)
fraction=0.1

classes=['**','em*','ev*','lm*','ma*','ms*','pe*','v*','y*o']
picked=[]
n_arr=[]
c_arr=[]

for c in classes:
	# read in star lists of each type
	filename=c+'_all.npy'
	data=np.load(filename)
	#calculate number to pick based on fraction 
	n=int(fraction*len(data))
	sub=list(np.random.choice(data,n,replace=False))
	picked.extend(sub)
	n_arr.append(n)
	c_arr.extend([c] * n)


subset = pd.Series(picked)
subset_c=pd.Series(c_arr)

#make into one dataframe
df = pd.concat([subset,subset_c],axis=1)
df.rename(columns = {0:'name', 1:'class'}, inplace = True)
df = df.groupby('name').agg({ 'class': ', '.join}).reset_index()

#13 lots of 9010 make 117130
listlen=9010
for i in range(0,13):
	imin=i*listlen
	imax=(1+i)*listlen

	sub_df=df.iloc[imin:imax,0]

	#print(df)
	sub_df.to_csv('10percent_part_{}.list'.format(i+1),sep='\t', index=False, header=False, quoting=3,escapechar="\t")


