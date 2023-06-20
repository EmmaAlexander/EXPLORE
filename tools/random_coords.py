import numpy as np
import matplotlib.pyplot as plt


rng_RA=360*np.random.default_rng().random()
rng_dec=180*np.arccos(2*(np.random.default_rng().random())-1)/np.pi -90

print(rng_RA,rng_dec,'0.05')
