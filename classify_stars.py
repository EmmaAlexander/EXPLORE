import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px 
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy import units as u
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
import cmasher 
import glob as glob

label_file='10percent/10percent_labels.txt'
pyssed_files_glob='10percent/10percent_part*.dat'
pyssed_file='10percent/10percent_full.dat'

def param_update():

	plt.rcParams.update({'font.size': 10})
	plt.rcParams.update({'font.family': 'STIXGeneral'})
	plt.rcParams.update({'mathtext.fontset': 'stix'})
	pd.set_option("display.max_rows",5)
	pd.set_option("display.max_columns",None)

def glob_files(globstr):
	#write something which globbs the relevent files
	filelist=glob.glob(globstr)
	print("Files found via glob:")
	print(filelist)
	df_list=[]
	for file in filelist:
		sub_df = pd.read_table(file,skiprows=1,header=[1,2,3,4,5,6,7],low_memory=False)
		df_list.append(sub_df)
		if len(sub_df.index)<9010:
			print("Fewer than expected entries for {}: {}".format(file,len(sub_df.index) ))
	df=pd.concat(df_list)

	print("Processing data")
	# Extract text features
	cats = df.select_dtypes(exclude=np.number).columns.tolist()
	# Convert to Pandas category
	for col in cats:
	   df[col] = df[col].astype('category')
	#force numeric columns where needed
	#can probably change this to select based on column name...
	for col in  df.columns[21:222]:
	    df[col] = pd.to_numeric(df[col], errors='coerce')
	#check the above worked if needed
	#for i in range(0,239):
	#    print(i,output_df.dtypes[i])
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	features=df.replace('[]', np.nan).replace('--', np.nan)#.select_dtypes(include=numerics)
	features.to_csv('10percent/10percent_full.dat',sep='\t', index=False, quoting=3,escapechar="\t",na_rep='[]',header=True)

	return features

def read_files(filestr):
	df = pd.read_table(filestr,skiprows=1,header=[1,2,3,4,5,6,7],low_memory=False)


def main():
	param_update()

	#features=glob_files(pyssed_files_glob)
	#print("Reading label file")
	labels=pd.read_table(label_file)
	features = pd.read_table(pyssed_file,skiprows=1,header=[1,2,3,4,5,6,7],low_memory=False)
	print(len(features.index),len(labels.index))

	
	print(features.shape)
	#(n_samples, n_features), n_classes = features.shape, np.unique(labels).size
	#print(f"# classes: {n_classes}; # samples: {n_samples}; # features {n_features}")
	'''
	model_features=features.loc[:, ("Model", slice(None))]
	adopted_features=features.loc[:, ("Adopted", slice(None))]
	fitted_features=features.loc[:, ("Fitted", slice(None))]
	text_features=features.loc[:, ("Source", slice(None))]
	

	#----------------

	#do some pre-processing inc. galactic coords
	ra=features.loc[:, ("RA", slice(None))].loc[:, ("Value", slice(None))]
	dec=features.loc[:, ("Dec", slice(None))].loc[:, ("Value", slice(None))]


	coords=SkyCoord(np.asarray(ra*u.degree,np.asarray(dec*u.degree)))
	features[('Adopted','l','Value','deg','-', '-', '-')]=coords.galactic.l/u.degree
	features[('Adopted','b','Value','deg','-', '-', '-')]=coords.galactic.b/u.degree
	(n_samples, n_features), n_classes = features.shape, np.unique(labels).size

	'''


	#----------------

	'''
	
	# Split the data
	rs=0 #random state
	ts=0.2 #test size
	X=features#.iloc[:,2:] #remove RA and Dec
	# creating one hot encoder object 
	onehotencoder = OneHotEncoder()
	y=onehotencoder.fit_transform(labels.values.reshape(-1,1)).toarray()
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rs,test_size=ts)
	#random state NEEDS to be the same as above
	names_train, names_test, y_train_dummy, y_test_dummy = train_test_split(names, y, random_state=rs,test_size=ts)

	#---------------
	# Create regression matrices
	dtrain_reg = xgb.DMatrix(X_train.values, y_train, enable_categorical=True)
	dtest_reg = xgb.DMatrix(X_test.values, y_test, enable_categorical=True)

	#----------------
	params = {"objective": "reg:squarederror", "tree_method": "hist"}
	n=1024
	model = xgb.train(params=params,dtrain=dtrain_reg,num_boost_round=n,)
	preds = model.predict(dtest_reg)
	rmse = metrics.mean_squared_error(y_test, preds, squared=False)
	print("rmse = {}".format(rmse))
	accuracy = metrics.accuracy_score(y_true= np.argmax(y_test,axis=1), y_pred= np.argmax(preds,axis=1))
	print("accuracy = {}".format(accuracy))
	#f1=metrics.f1_score(np.argmax(y_test,axis=1), y_pred= np.argmax(preds,axis=1), labels=None, pos_label=1,average=None, sample_weight=None, zero_division='warn')
	#f1_list.append(f1)
	plt.rcParams["figure.figsize"] = (8,6)
	confusion_matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1), np.argmax(preds,axis=1))
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels = np.unique(labels))
	cm_display.plot()
	'''





if __name__ == "__main__":
	main()