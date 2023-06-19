from astroquery.simbad import Simbad
Simbad.ROW_LIMIT = 20000
Simbad.TIMEOUT=1200

import time
import numpy as np


directory='/Users/emma/OneDrive/Work/MLstars/SIMBAD_out/'
classes=['lm*','ma*']
#n_results=[]

for c in classes:
	print("Getting {}".format(c))
	objlist=[]
	empty=[]
	step=1
	for ra in range(0,360,step):
		try:
			qry='otypes="{}" & Kmag<99 & Plx>0 & RA>={} & RA<{}'.format(c,ra,ra+step)
			result = Simbad.query_criteria(qry)
			#print("{} {} results for {} < RA < {} degrees.".format(len(result),c,ra,ra+step))
			print("{} {}".format(ra,len(result)))
			objlist.append(list(result['MAIN_ID']))
		except:
			empty.append(ra)
		#slight time delay so simbad doesn't blacklist me...
		time.sleep(0.1)

	full_list = np.concatenate([np.array(i) for i in objlist])
	print("{} results for {} in total.".format(len(full_list),c))
	filename=c+'_all'
	#np.save(filename,full_list )
	#np.savetxt(filename+'.list',full_list,fmt='%s')